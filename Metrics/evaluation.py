import torch
import torch.nn.functional as F
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch

from Metrics import metrics_rr as rr
from Metrics import metrics_fr as fr

import numpy as np
from scipy import io


def evaluation_rr(out_lr, ms_lr, ratio, flag_cut=True, dim_cut=11, L=16):

    if flag_cut:
        out_lr = out_lr[:, :, dim_cut-1:-dim_cut, dim_cut-1:-dim_cut]
        ms_lr = ms_lr[:, :, dim_cut-1:-dim_cut, dim_cut-1:-dim_cut]

    out_lr = torch.clip(out_lr, 0, 2 ** L)


    ergas = rr.ERGAS(ratio).to(out_lr.device)
    sam = rr.SAM().to(out_lr.device)
    q = rr.Q(out_lr.shape[1]).to(out_lr.device)
    q2n = rr.Q2n().to(out_lr.device)

    ergas_index, _ = ergas(out_lr, ms_lr)
    sam_index, _ = sam(out_lr, ms_lr)
    q_index = torch.mean(q(out_lr, ms_lr))
    q2n_index, _ = q2n(out_lr, ms_lr)

    return ergas_index.item(), sam_index.item(), q_index.item(), q2n_index.item()


def evaluation_fr_old(out, pan, ms_lr, ratio, sensor):

    sigma = ratio

    if sensor == 'PRISMA' or sensor == 'WV3':
        starting = 1
        #starting = 3
    elif sensor == 'Pavia':
        starting = 3
    else:
        starting = 3

    kernel = mtf_kernel_to_torch(gen_mtf(ratio, sensor, nbands=out.shape[1]))

    out_lp = F.conv2d(out, kernel.type(out.dtype).to(out.device), padding='same', groups=out.shape[1])

    out_lr = out_lp[:, :, starting::ratio, starting::ratio]

    ergas = rr.ERGAS(ratio).to(out.device)
    sam = rr.SAM().to(out.device)
    q = rr.Q(out.shape[1]).to(out.device)
    q2n = rr.Q2n().to(out.device)

    d_s = fr.D_s(ms_lr.shape[1], ratio).to(out.device)
    d_sr = fr.D_sR().to(out.device)
    d_rho = fr.D_rho(sigma).to(out.device)

    # Spectral Assessment
    ergas_index, _ = ergas(out_lr, ms_lr)
    sam_index, _ = sam(out_lr, ms_lr)
    q_index = torch.mean(q(out_lr, ms_lr))
    q2n_index, _ = q2n(out_lr, ms_lr)
    d_lambda_index = 1 - q2n_index

    # Spatial Assessment
    d_s_index = d_s(out, pan, ms_lr)
    d_sr_index = d_sr(out, pan)
    d_rho_index, _ = d_rho(out, pan)

    return (ergas_index.item(),
            sam_index.item(),
            q_index.item(),
            d_lambda_index.item(),
            d_s_index.item(),
            d_sr_index.item(),
            d_rho_index.item()
            )


def evaluation_fr(out, pan, ms_lr, ms, ratio, sensor):


    sigma = ratio

    if sensor == 'PRISMA' or sensor == 'WV3':
        #starting = 1
        starting = 3
    elif sensor == 'Pavia':
        starting = 3
    else:
        starting = 3

    kernel = mtf_kernel_to_torch(gen_mtf(ratio, sensor, nbands=out.shape[1]))

    filter = torch.nn.Conv2d(out.shape[1], out.shape[1], kernel_size=kernel.shape[2], groups=out.shape[1], padding='same', padding_mode='replicate', bias=False)
    filter.weight = torch.nn.Parameter(kernel.type(out.dtype).to(out.device))
    filter.weight.requires_grad = False

    #out_lp = F.conv2d(out, kernel.type(out.dtype).to(out.device), padding='same', groups=out.shape[1])
    out_lp = filter(out)
    out_lr = out_lp[:, :, starting::ratio, starting::ratio]

    #ergas = rr.ERGAS(ratio).to(out.device)
    #sam = rr.SAM().to(out.device)
    #q = rr.Q(out.shape[1]).to(out.device)
    q2n = rr.Q2n().to(out.device)

    #d_s = fr.D_s(ms_lr.shape[1], ratio).to(out.device)
    d_sr = fr.D_sR().to(out.device)
    d_rho = fr.D_rho(sigma).to(out.device)

    # Spectral Assessment
    #ergas_index, _ = ergas(out_lr, ms_lr)
    #sam_index, _ = sam(out_lr, ms_lr)
    #q_index = torch.mean(q(out_lp, ms))
    q2n_index, _ = q2n(out_lr, ms_lr)
    #d_lambda_index = 1 - q_index
    d_lambda_khan_index = 1 - q2n_index

    # Spatial Assessment
    #d_s_index = d_s(out, pan, ms_lr)
    d_sr_index = d_sr(out, pan)
    d_rho_index, _ = d_rho(out, pan)

    # Separate Spatial Assessment
    #last_band = overlap[-1]

    #d_s = fr.D_s(ms_lr[:, :last_band, :, :].shape[1], ratio).to(out.device)
    #d_sr = fr.D_sR().to(out.device)
    #d_rho = fr.D_rho(sigma).to(out.device)

    #overlapped_d_s_index = d_s(out[:, :last_band, :, :], pan, ms_lr[:, :last_band, :, :])
    #overlapped_d_sr_index = d_sr(out[:, :last_band, :, :], pan)
    #overlapped_d_rho_index, _ = d_rho(out[:, :last_band, :, :], pan)

    #d_s = fr.D_s(ms_lr[:, last_band:, :, :].shape[1], ratio).to(out.device)
    #d_sr = fr.D_sR().to(out.device)
    #d_rho = fr.D_rho(sigma).to(out.device)

    #not_overlapped_d_s_index = d_s(out[:, last_band:, :, :], pan, ms_lr[:, last_band:, :, :])
    #not_overlapped_d_sr_index = d_sr(out[:, last_band:, :, :], pan)
    #not_overlapped_d_rho_index, _ = d_rho(out[:, last_band:, :, :], pan)

    return (
            #ergas_index.item(),
            #sam_index.item(),
            #d_lambda_index.item(),
            d_lambda_khan_index.item(),
            #d_s_index.item(),
            d_sr_index.item(),
            d_rho_index.item(),
            #overlapped_d_s_index.item(),
            #overlapped_d_sr_index.item(),
            #overlapped_d_rho_index.item(),
            #not_overlapped_d_s_index.item(),
            #not_overlapped_d_sr_index.item(),
            #not_overlapped_d_rho_index.item()
            )


if __name__ == '__main__':

    resolution = 'FR'

    dataset = {
        'CAG': '20220905101901',  # Cagliari
        'UDI': '20230824100356',  # Udine
        'FCO': '20230908173127',  # Ford County
        'MCU': '20231120102229',  # Macuspana

        'VAL': '20230905163851',  # Validation
    }

    city = 'VAL'
    test_path = '/home/giuseppe.guarino/Datasets/PRISMA/' + dataset[city] + resolution + '.mat'

    #fused = torch.from_numpy(io.loadmat(f'/home/giuseppe.guarino/R_PNN_v2/results/ROLL_{city}_17.mat')['I_MS'].astype(np.float32))
    fused = torch.from_numpy(io.loadmat(f'/home/giuseppe.guarino/HS_PAN_TBOX/Results/PRISMA/20230905163851FR/TV.mat')['I_MS'].astype(np.float32))
    fused = fused[None,:,:,:]

    ref = io.loadmat(test_path)

    pan = torch.from_numpy(ref['I_PAN'].astype(np.float32))[None, None, :,:]
    ms = torch.from_numpy(np.moveaxis(ref['I_MS'].astype(np.float32), -1, 0))[None, :, :,:]
    ms_lr = torch.from_numpy(np.moveaxis(ref['I_MS_LR'].astype(np.float32), -1, 0))[None, :, :,:]

    ciccio = evaluation_fr(fused, torch.clone(pan), torch.clone(ms_lr), torch.clone(ms), ratio=6, sensor='PRISMA')

    QNR = (1 - ciccio[0])*(1-ciccio[1])

    print(f'D_Lambda = {ciccio[0]}, D_S = {ciccio[1]}, QNR = {QNR}, D_rho = {ciccio[2]}')


