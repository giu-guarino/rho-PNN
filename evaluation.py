import torch

import numpy as np
from scipy import io

from Metrics.evaluation import evaluation_fr

import argparse


def evaluation(args):


    resolution = 'FR'

    dataset = {
        'CAG': '20220905101901',  # Cagliari
        'UDI': '20230824100356',  # Udine
        'FCO': '20230908173127',  # Ford County
        'MCU': '20231120102229',  # Macuspana

        'VAL': '20230905163851',  # Validation
        'CAL': '20220905101902',  # Cagliari crop
    }

    city = args.city #'UDI'
    conf = args.conf

    test_path = '/home/giuseppe.guarino/Datasets/PRISMA/' + dataset[city] + resolution + '.mat'
    test_path = '/home/giuseppe.guarino/R_PNN_v2/results/VAL_beta05.mat'

    #fused = torch.from_numpy \
    #    (np.moveaxis(io.loadmat(f'/home/giuseppe.guarino/HS_PAN_TBOX/Results/PRISMA/20230905163851FR/BT-H.mat')['I_MS'].astype(np.float32), -1, 0))
        #(io.loadmat(f'/home/giuseppe.guarino/R_PNN_v2/results/ROLL_{city}_{conf}_R-PNN_salient.mat')['I_MS'].astype(np.float32))


    #fused = fused[None ,: ,: ,:]

    ref = io.loadmat(test_path)

    '''
    pan = torch.from_numpy(ref['I_PAN'].astype(np.float32))[None, None, : ,:]
    ms = torch.from_numpy(np.moveaxis(ref['I_MS'].astype(np.float32), -1, 0))[None, :, : ,:]
    ms_lr = torch.from_numpy(np.moveaxis(ref['I_MS_LR'].astype(np.float32), -1, 0))[None, :, : ,:]
    '''


    pan = torch.from_numpy(ref['I_PAN'].astype(np.float32))[None, :, :, :]
    ms = torch.from_numpy(ref['I_MS'].astype(np.float32))[None, :, :, :]
    ms_lr = torch.from_numpy(ref['I_MS_LR'].astype(np.float32))[None, :, :, :]

    fused = torch.from_numpy(ref['I_F_nH'].astype(np.float32))[None, :, :, :]

    ciccio = evaluation_fr(fused, torch.clone(pan), torch.clone(ms_lr), torch.clone(ms), ratio=6, sensor='PRISMA')

    QNR = (1 - ciccio[0] ) *( 1 -ciccio[1])

    print(f'D_Lambda = {ciccio[0]}, D_S = {ciccio[1]}, QNR = {QNR}, D_rho = {ciccio[2]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Lambda-PNN Test code',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     )
    optional = parser._action_groups.pop()
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-c", "--city", type=str, required=True)
    requiredNamed.add_argument("-v", "--conf", type=str, required=True)

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    evaluation(arguments)