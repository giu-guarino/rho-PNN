import os
from scipy import io
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time

import losses
import utils
from utils import normalize, TrainGenerator, patchify, train, cumulate_EMA
from sensor import Sensor
from Utils.spectral_tools import gen_mtf
from networks import R_PNN
from torch.utils.data import DataLoader

from tools.salient_patches_extraction import patches_extractor_w_kmeans
from Metrics.evaluation import evaluation_fr, evaluation_rr

from utils import open_mat
from config_dict import config
import argparse
from pathlib import Path

def beta_adaptation(net, op, I_inp, beta_i, LSp, LSt, mtf_scope,
                    sp_ref, st_ref, thr, epsilon):

    if not os.path.exists(config['save_weights_path']):
        os.makedirs(config['save_weights_path'])

    good_b = False
    beta = beta_i
    path_prev_model = os.path.join(config['save_weights_path'], 'weights_prev_model.tar')
    path_prev_optim = os.path.join(config['save_weights_path'], 'weights_prev_optim.tar')

    torch.save(net.state_dict(), path_prev_model)
    torch.save(op.state_dict(), path_prev_optim)

    while not good_b:
        net.load_state_dict(torch.load(path_prev_model))
        op.load_state_dict(torch.load(path_prev_optim))

        op.zero_grad()
        out = net(I_inp)
        l_spec_prev = LSp(out, sp_ref[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])
        loss_st, loss_struct_no_thr = LSt(out, st_ref, thr)
        loss_t = l_spec_prev + beta * loss_st
        loss_t.backward()
        op.step()

        op.zero_grad()
        out = net(I_inp)
        l_spec_prev2 = LSp(out, sp_ref[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])
        loss_st, loss_struct_no_thr = LSt(out, st_ref, thr)
        loss_t = l_spec_prev2 + beta * loss_st
        loss_t.backward()
        op.step()

        out = net(I_inp)
        l_spec = LSp(out, sp_ref[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])

        if l_spec.item() > l_spec_prev.item() * epsilon:
            beta /= 2
            if beta <= beta_i / 8:
                beta = beta_i / 8
                good_b = True
        else:
            good_b = True

    net.load_state_dict(torch.load(path_prev_model))
    op.load_state_dict(torch.load(path_prev_optim))

    return beta



def adaptive_epochs(hs_lr, N1, nbands, eta):
    max_iter = N1 * np.ones((1, nbands), dtype=np.uint16)
    I_LR = hs_lr[0,:,:,:].numpy()
    X = I_LR.reshape(I_LR.shape[0], I_LR.shape[1] * I_LR.shape[2]).copy()
    C = np.corrcoef(X)
    c = np.diag(C, 1)
    ct = np.sum(c)

    beta = (eta * (nbands - 1) * N1) / (ct - (nbands - 1))
    max_iter[:, 1:] += (beta * (c - 1)).astype(np.uint16)
    max_iter[:, 0] = max_iter[:, 1]  # In order to avoid singularities

    return max_iter


def test_rho_pnn(args):

        # Paths and env configuration
        basepath = args.input
        out_dir = args.out_dir

        gpu_number = args.gpu_number
        use_cpu = args.use_cpu

        # Training hyperparameters

        if args.learning_rate != -1:
            learning_rate = args.learning_rate
        else:
            learning_rate = config['learning_rate']

        # Satellite configuration
        sensor = config['satellite']

        # Other params
        batch_size = config['batch_size']
        semi_width = config['semi_width']
        kernel_size = config['kernel_size']

        s = Sensor(sensor)
        mtf_scope = (kernel_size // s.ratio) // 2

        # Environment Configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

        # Devices definition
        device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")

        # Open the image

        pan, hs_lr, hs, gt, wl = open_mat(basepath)
        resolution = basepath.split('.')[-2][-2:] #Last two characters of the filename (FR or RR)
        hs_lr = normalize(hs_lr)
        hs = normalize(hs)
        pan = normalize(pan)
        nbands = wl.shape[1]

        if resolution == 'FR':
            patch_size = 120
            tr_epochs = 300
            gamma, gamma_l, beta_0, epsilon, ns, eta = config['hp_fr']
        else:
            patch_size = 90
            tr_epochs = 300
            gamma, gamma_l, beta_0, epsilon, ns, eta = config['hp_rr']

        # *************** SALIENT PATCHES SELECTION **************************************************

        if resolution == 'FR':
            patches_indexes = patches_extractor_w_kmeans(hs_lr, n_clusters=config['num_salient_patches'], patch_size=50)
            if hs_lr.shape[0] > config['num_salient_patches']:
                patches_indexes = patches_indexes[:config['num_salient_patches']]

        # **************** NETWORK ************************

        network = R_PNN(s.nbands, [48, 32, 1], s.kernels)

        # **************** LOSS AND OPTIMIZATOR ************************

        LSpec = losses.SpectralLossNocorr(gen_mtf(s.ratio, sensor, kernel_size=kernel_size, nbands=1),
                                          s.net_scope,
                                          s.ratio,
                                          device)

        LStruct = losses.StructuralLoss(s.ratio, device)

        optimizer = optim.Adam(params=network.parameters(), lr=learning_rate)


        # ******************** FINE TUNING ***********************************

        N1 = 4 * ns
        spec_limits = [gamma * gamma_l, gamma]
        spec_limits_train = spec_limits
        max_iter = adaptive_epochs(hs_lr, N1, nbands, eta).astype(np.uint16)

        sat_counter = 0
        blur_counter = 0
        tot_iter = 0

        network = network.to(device)
        LSpec = LSpec.to(device)
        LStruct = LStruct.to(device)

        history_loss = []
        history_loss_spec = []
        history_loss_struct = []
        history_min_loss = []

        history_last_spec_loss = []
        history_last_struct_loss = []

        history_best_spec_loss = []
        history_best_struct_loss = []

        if not os.path.exists(config['save_weights_path']):
            os.makedirs(config['save_weights_path'])

        path_min_loss = os.path.join(config['save_weights_path'], 'weights_RHO-PNN.tar')

        ####################### TRAIN #######################################

        train_img, train_spec = patchify(hs_lr[0,0,:,:], hs[0,0,:,:], pan[0,:,:,:], patch_size=patch_size)
        train_generator = TrainGenerator(train_img, train_spec)

        train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                                  prefetch_factor=2, persistent_workers=True)
        begin_train = time.time()

        _, path_tr_weights = train(net=network, device=device, sensor=s, LSpec=LSpec, LStruct=LStruct,
                                   optimizer=optimizer, epochs=tr_epochs, training_dataloader=train_loader,
                                   beta_limits=[0, 0.5], spec_limits=spec_limits_train,
                                   mtf_scope=mtf_scope, semi_width=semi_width, gpu_num=str(gpu_number))
        end_train = time.time()

        network.load_state_dict(torch.load(path_tr_weights))

        del path_tr_weights

        fused = []
        begin_tuning = time.time()

        pan = pan.to(device)

        for band_idx in range(nbands):

            iterations = 0
            good_beta = False

            beta_s = 0
            beta_adapt = beta_0

            ema_weights = None

            #INPUT PREPROCESS

            band_hs_lr = hs_lr[:, band_idx, None, :, :].to(device)
            band_hs = hs[:, band_idx, None, :, :].to(device)
            inp = torch.cat([band_hs, pan], dim=1).to(device)

            ################## PATCHES ##################

            inp_p = torch.cat(torch.split(inp, config['patch_dim'], -1), 0)
            inp_p = torch.cat(torch.split(inp_p, config['patch_dim'], -2), 0)

            ims_p = torch.cat(torch.split(band_hs, config['patch_dim'], -1), 0)
            ims_p = torch.cat(torch.split(ims_p, config['patch_dim'], -2), 0)

            ilr_p = torch.cat(torch.split(band_hs_lr, config['patch_dim_lr'], -1), 0)
            ilr_p = torch.cat(torch.split(ilr_p, config['patch_dim_lr'], -2), 0)

            if resolution == 'FR':
                inp_p = inp_p[patches_indexes, :, :, :].to(device)
                ims_p = ims_p[patches_indexes, :, :, :].to(device)
                ilr_p = ilr_p[patches_indexes, :, :, :]

            spec_ref = ilr_p.to(device)
            struct_ref = torch.unsqueeze(inp_p[:, -1, :, :], dim=1).to(device)

            threshold = utils.local_corr_mask(inp_p, s.ratio, s.sensor, device, semi_width)
            threshold = threshold.float()
            threshold = threshold.to(device)

            pbar = tqdm(range(max_iter[0, band_idx]), dynamic_ncols=True, initial=1)

            min_loss = np.inf
            min_spec = LSpec(ims_p, spec_ref[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])

            next_band = False
            counter = ns # Spatial iterations counter

            network.train()

            running_best_spec_loss = 0.0
            running_best_struct_loss = 0.0

            while iterations < max_iter[0, band_idx] and not next_band:

                running_loss = 0.0
                running_spec_loss = 0.0
                running_struct_loss = 0.0

                optimizer.zero_grad()

                outputs = network(inp_p)

                loss_spec = LSpec(outputs, spec_ref[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])
                loss_struct, loss_struct_no_threshold = LStruct(outputs, struct_ref, threshold)

                #HYSTERESIS
                #Hysteresis starts only after beta value has been tuned

                if good_beta:
                    if loss_spec.item() < min_spec.item() * spec_limits[0]:
                        beta_s = beta_adapt
                    elif loss_spec.item() > min_spec.item() * spec_limits[1]:
                        beta_s = 0

                loss = loss_spec + beta_s * loss_struct
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                running_spec_loss += loss_spec.item()
                running_struct_loss += loss_struct_no_threshold

                # FIND OPTIMAL BETA
                if loss_spec < min_spec * spec_limits[0] and not good_beta:
                    beta_adapt = beta_adaptation(network, optimizer, inp_p, beta_0, LSpec,
                                                 LStruct, mtf_scope, spec_ref, struct_ref, threshold, epsilon)
                    good_beta = True

                ####################### EMA #####################################
                if good_beta:
                    ema_weights = cumulate_EMA(network, ema_weights, 0.95)
                ####################### EMA #####################################

                history_loss.append(running_loss)
                history_loss_spec.append(running_spec_loss)
                history_loss_struct.append(running_struct_loss)
                history_min_loss.append(min_spec.item())

                if beta_s > 0:
                    counter -= 1
                    # When both losses are active
                    if loss_struct_no_threshold <= min_loss:
                        min_loss = loss_struct_no_threshold
                        torch.save(network.state_dict(), path_min_loss)
                        running_best_spec_loss = loss_spec.item()
                        running_best_struct_loss = loss_struct_no_threshold

                if counter == 0:
                    next_band = True

                iterations += 1

                pbar.set_postfix(
                    {'Lower bound': min_spec.item()*spec_limits[0], 'Upper bound': min_spec.item()*spec_limits[1], 'Spectral loss': running_spec_loss, 'Struct loss': running_struct_loss})
                pbar.update(1)

            tot_iter += iterations

            if counter == ns:
                sat_counter += 1
                blur_counter += 1
            elif counter > 0:
                sat_counter += 1

            history_last_spec_loss.append(running_spec_loss)
            history_last_struct_loss.append(running_struct_loss)


            # Output Folder creation
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            network.eval()

            if counter == ns:
                running_best_spec_loss = running_spec_loss
                running_best_struct_loss = running_struct_loss
            else:
                network.load_state_dict(torch.load(path_min_loss))

            history_best_spec_loss.append(running_best_spec_loss)
            history_best_struct_loss.append(running_best_struct_loss)

            if good_beta:
                network.load_state_dict(ema_weights)

            with torch.no_grad():
                output = network(inp)

            output = np.squeeze(output.cpu().numpy(), 0)
            output = output * (2 ** s.nbits)
            output = np.round(output)
            output = np.clip(output, 0, 2**s.nbits)
            output = output.astype(np.uint16)

            fused.append(output)

        fused = np.concatenate(fused, axis=0)

        end_tuning = time.time()
        duration = int(end_tuning - begin_tuning)

        name = os.path.basename(basepath).split('.')[0]
        io.savemat(os.path.join(out_dir, f"{name}.mat"), {'I_HS': fused})

        if config['compute_quality_indexes']:

            print("Computing quality indexes...")

            if resolution == 'FR':
                d_lambda, d_s, d_rho = evaluation_fr(torch.from_numpy(fused[None,:,:,:].astype(np.float32)),
                                                     torch.clone(pan.cpu()), torch.clone(hs_lr.cpu()),
                                                     torch.clone(hs_lr.cpu()), ratio=s.ratio, sensor=sensor)
                results = [d_lambda, d_s, d_rho]
            else:
                ergas, sam, q, q2n = evaluation_rr(torch.from_numpy(fused[None,:,:,:].astype(np.float32)),
                                                   torch.clone(gt), ratio=s.ratio, flag_cut=True, dim_cut=11, L=16)
                results = [ergas, sam, q2n]

            io.savemat(os.path.join(out_dir, f'quality_indexes_{name}.mat'), {'RESULTS': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='rho-PNN Training code',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='rho-PNN is an unsupervised deep learning-based hyperspectral pansharpening '
                                                 'method.',
                                     epilog='''\
Reference: 
Zero-Shot Hyperspectral Pansharpening Using Hysteresis-Based Tuning for Spectral Quality Control
G. Guarino, M. Ciotola, G. Vivone, G. Poggi, G. Scarpa 

Authors: 
- Image Processing Research Group of University of Naples Federico II ('GRIP-UNINA')
- National Research Council, Institute of Methodologies for Environmental Analysis (CNR-IMAA)
- University of Naples Parthenope

For further information, please contact the first author by email: giuseppe.guarino2[at]unina.it '''
                                     )
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required named arguments')

    required.add_argument("-i", "--input", type=str, required=True,
                          help='The path of the .mat file'
                               'For more details, please refer to the GitHub documentation.')

    optional.add_argument("-o", "--out_dir", type=str, default='Outputs',
                          help='The directory in which save the outcome.')

    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')

    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    test_rho_pnn(arguments)