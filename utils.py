import os
import math
from math import floor
import numpy as np
import torch
import torch.nn as nn
from Utils.cross_correlation import xcorr_torch
from spectral_tools import gen_mtf
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import OrderedDict
from scipy import io
import copy
from config_dict import config


def normalize(img):
    return img / 65536.0


def open_mat(path):
    # Open .mat file
    dic_file = io.loadmat(path)

    # Extract fields and convert them in float32 numpy arrays
    pan_np = dic_file['I_PAN'].astype(np.float32)
    ms_lr_np = dic_file['I_MS_LR'].astype(np.float32)
    ms_np = dic_file['I_MS'].astype(np.float32)

    if 'I_GT' in dic_file.keys():
        gt_np = dic_file['I_GT'].astype(np.float32)
        gt = torch.from_numpy(np.moveaxis(gt_np, -1, 0)[None, :, :, :])
    else:
        gt = None

    # Convert numpy arrays to torch tensors
    ms_lr = torch.from_numpy(np.moveaxis(ms_lr_np, -1, 0)[None, :, :, :])
    pan = torch.from_numpy(pan_np[None, None, :, :])
    ms = torch.from_numpy(np.moveaxis(ms_np, -1, 0)[None, :, :, :])
    wavelenghts = torch.from_numpy(dic_file['wavelengths']).float()

    return pan, ms_lr, ms, gt, wavelenghts

def cumulate_EMA(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k].cpu().detach().numpy() + (1-alpha) * current_weights_npy[k]

    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )

    return current_weights


def net_scope(kernel_size):
    """
        Compute the network scope.

        Parameters
        ----------
        kernel_size : List[int]
            A list containing the kernel size of each layer of the network.

        Return
        ------
        scope : int
            The scope of the network

        """

    scope = 0
    for i in range(len(kernel_size)):
        scope += math.floor(kernel_size[i] / 2)
    return scope


def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    I_PAN = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    I_MS = img_in[:, :-1, :, :]

    MTF_kern = gen_mtf(ratio, sensor)[:, :, 0]
    MTF_kern = np.expand_dims(MTF_kern, axis=(0, 1))
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
    pad = floor((MTF_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=MTF_kern.shape,
                          bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False
    depthconv.to(device)
    I_PAN = padding(I_PAN)
    I_PAN = depthconv(I_PAN)
    mask = xcorr_torch(I_PAN, I_MS, kernel, device)
    mask = 1.0 - mask

    return mask



class TrainGenerator(Dataset):
    def __init__(self, input_img, spec_img, ratio=6) -> None:
        self.inp  = input_img
        self.spec = spec_img
        self.ratio = ratio

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, index):
        ms_lr = self.spec[index]
        input_image = self.inp[index]

        #input_image = normalize(input_image)
        #ms_lr = normalize(ms_lr)

        #input_image = torch.from_numpy(input_image)
        #ms_lr = torch.from_numpy(ms_lr)

        return input_image, ms_lr

def model_ensemble(net, optimizer, LSpec, mtf_scope, num_models, me_epochs, training_dataloader, device, verbose=1):

    # List for saving model weights
    model_weights = []

    for model_idx in range(num_models):

        # print(f"Model {model_idx}")
        me_net = copy.deepcopy(net)
        me_optimizer = copy.deepcopy(optimizer)

        for me_epoch in range(me_epochs):

            me_net.train()
            if verbose == 1:
                me_training_loop = tqdm(training_dataloader, dynamic_ncols=True, leave=True)
            else:
                me_training_loop = training_dataloader

            for inputs, references in me_training_loop:

                if verbose == 1:
                    me_training_loop.set_description(
                        'Model Num. {:01} - Train Epoch: {:03} / {:03}'.format(model_idx, me_epoch + 1, me_epochs))

                # Preparing the data
                inputs = inputs.to(device)
                references = references.to(device)

                # Training step
                me_optimizer.zero_grad()

                outputs = me_net(inputs)
                loss_spec = LSpec(outputs, references[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])

                loss = loss_spec
                loss.backward()
                me_optimizer.step()

        model_weights.append(copy.deepcopy(me_net.state_dict()))

    averaged_state_dict = copy.deepcopy(model_weights[0])  # Copy first model

    for key in averaged_state_dict.keys():
        averaged_state_dict[key] = torch.stack([state_dict[key] for state_dict in model_weights]).mean(dim=0)

    return averaged_state_dict



# ******************** TRAINING FUNCTION ***********************************


def train(net, device, sensor, LSpec, LStruct, optimizer, epochs, training_dataloader, spec_limits, beta_limits, semi_width=18,
          save_weights_root='temp/', verbose=1, mtf_scope=5, gpu_num='0'):

    s = sensor

    # Best model checkpoint implementation
    if not os.path.exists(save_weights_root):
        os.makedirs(save_weights_root)

    path_min_loss = os.path.join(save_weights_root, 'weights_' + 'R_PNN_TR_' + gpu_num + '.tar')
    history_loss = []

    ema_weights = None
    activate_ema = False

    ############################### Model Ensemble START ###############################

    print(f"Model Ensemble Process")

    averaged_state_dict = model_ensemble(net, optimizer, LSpec, mtf_scope, config['num_models'], config['me_epochs'],
                                         training_dataloader, device, verbose)

    # Load average weights in the final model
    net.load_state_dict(averaged_state_dict)

    ############################### Model Ensemble END ###############################

    # Training Loop implementation

    for epoch in range(epochs):
        i = 0
        running_loss = 0.0
        running_spec_loss = 0.0
        running_struct_loss = 0.0

        net.train()
        if verbose == 1:
            training_loop = tqdm(training_dataloader, dynamic_ncols=True, leave=True)
        else:
            training_loop = training_dataloader

        #print('BEGIN TRAINING')

        for inputs, references in training_loop:

            if verbose == 1:
                training_loop.set_description('Train Epoch: {:03} / {:03}'.format(epoch + 1, epochs))

            # Preparing the data
            inputs = inputs.to(device)
            references = references.to(device)

            min_spec = LSpec(inputs[:,:-1,:,:], references[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])

            # Training step
            optimizer.zero_grad()

            outputs = net(inputs)

            threshold = local_corr_mask(inputs, s.ratio, s.sensor, device, semi_width)
            threshold = threshold.float()
            loss_spec = LSpec(outputs, references[:, :, mtf_scope:-mtf_scope, mtf_scope:-mtf_scope])
            loss_struct, loss_struct_no_threshold = LStruct(outputs, torch.unsqueeze(inputs[:, -1, :, :], 1), threshold)

            if loss_spec < min_spec * spec_limits[0]:
                beta_s = beta_limits[1]
                activate_ema = True
            elif loss_spec > min_spec * spec_limits[1]:
                beta_s = beta_limits[0]

            loss = loss_spec + beta_s * loss_struct
            loss.backward()
            optimizer.step()

            ####################### EMA #####################################
            if activate_ema:
                ema_weights = cumulate_EMA(net, ema_weights, 0.95)
            ####################### EMA #####################################

            running_loss += loss.item()
            running_spec_loss += loss_spec.item()
            running_struct_loss += loss_struct.item()

            i += 1
            if verbose == 1 and i < len(training_dataloader) - 1:
                training_loop.set_postfix({'Overall Loss': loss.item(), 'Spectral Loss': loss_spec.item(),
                                           'Struct Loss': loss_struct.item()})
            elif i == len(training_dataloader) - 1:
                running_loss = running_loss / len(training_dataloader)
                running_spec_loss = running_spec_loss / len(training_dataloader)
                running_struct_loss = running_struct_loss / len(training_dataloader)

                history_loss.append(running_loss)
                training_loop.set_postfix({'Overall Loss': running_loss, 'Spectral Loss': running_spec_loss,
                                           'Struct Loss': running_struct_loss})

        history = {'loss': history_loss}

    #torch.save(net.state_dict(), path_min_loss)
    net.load_state_dict(ema_weights)
    torch.save(net.state_dict(), path_min_loss)

    return history, path_min_loss


def patchify(BAND_LR, BAND, I_PAN, patch_size, ratio=6, overlap=False):

    train_img = []
    train_spec = []

    band_lr = BAND_LR[None, :, :].clone().detach()
    band = BAND[None, :, :].clone().detach()
    i_pan = I_PAN.clone().detach()

    img_size = i_pan.shape[1]
    if overlap:
        num_patch = 1 + (img_size - patch_size)//(patch_size//2)
    else:
        num_patch = img_size//patch_size
    count = 0

    #print(f"Num patches = {num_patch}")

    if overlap:
        for row in range(num_patch):
            for col in range(num_patch):
                ms_lr = band_lr[:, row * patch_size //2 //ratio:((row * patch_size//2) + patch_size) // ratio,
                        col * patch_size//2 // ratio:((col * patch_size//2) + patch_size) // ratio]
                ms = band[:, row * patch_size//2:row*patch_size//2 + patch_size, col * patch_size//2:col*patch_size//2 + patch_size]
                pan = i_pan[:, row * patch_size//2:row*patch_size//2 + patch_size, col * patch_size//2:col*patch_size//2 + patch_size]

                train_img.append(np.concatenate([ms, pan]))

                train_spec.append(ms_lr)

                count = count + 1
    else:
        for row in range(num_patch):
            for col in range(num_patch):
                ms_lr = band_lr[:, row*patch_size//ratio:(row+1)*patch_size//ratio, col*patch_size//ratio:(col+1)*patch_size//ratio]
                ms = band[:, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
                pan = i_pan[:, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]

                train_img.append(np.concatenate([ms, pan]))
                train_spec.append(ms_lr)

                count = count + 1

    return train_img, train_spec