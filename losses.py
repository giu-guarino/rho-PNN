from math import floor, ceil

import numpy as np
import torch
import torch.nn as nn

from Utils.cross_correlation import xcorr_torch as ccorr
from core import imresize as resize


class Q(nn.Module):
    def __init__(self, nbands, block_size=32):
        super(Q, self).__init__()
        self.block_size = block_size
        self.N = block_size ** 2
        filter_shape = (nbands, 1, self.block_size, self.block_size)
        kernel = torch.ones(filter_shape, dtype=torch.float32)

        self.depthconv = nn.Conv2d(in_channels=nbands,
        out_channels=nbands,
        groups=nbands,
        kernel_size=kernel.shape,
        bias=False)
        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

    def forward(self, outputs, labels):
        outputs_sq = outputs ** 2
        labels_sq = labels ** 2
        outputs_labels = outputs * labels

        outputs_sum = self.depthconv(outputs)
        labels_sum = self.depthconv(labels)

        outputs_sq_sum = self.depthconv(outputs_sq)
        labels_sq_sum = self.depthconv(labels_sq)
        outputs_labels_sum = self.depthconv(outputs_labels)

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2
        numerator = 4 * (self.N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = self.N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)
        quality_map = torch.ones(denominator.size(), device=outputs.device)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = (denominator != 0)
        quality_map[index] = numerator[index] / denominator[index]
        quality = torch.mean(quality_map, dim=(2, 3))

        return quality

class SpectralLoss(nn.Module):
    def __init__(self, mtf, net_scope, pan_shape, ratio, device, mask=None):

        # Class initialization
        super(SpectralLoss, self).__init__()
        kernel = mtf[0]
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.net_scope = net_scope
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        self.MTF_r = mtf[1]
        self.MTF_c = mtf[2]
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='sum')

        # Mask definition
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.ones((1, self.nbands, pan_shape[-2] - (self.net_scope + self.pad) * 2,
                                    pan_shape[-1] - (self.net_scope + self.pad) * 2), device=self.device)

    def forward(self, outputs, labels):

        x = self.depthconv(outputs)

        labels = labels[:, :, self.pad:-self.pad, self.pad:-self.pad]
        y = torch.zeros(x.shape, device=self.device)
        W_ = torch.zeros(x.shape, device=self.device)

        for b in range(self.nbands):
            y[:, b, self.MTF_r[b]::self.ratio, self.MTF_c[b]::self.ratio] = labels[:, b, 2::self.ratio, 2::self.ratio]
            W_[:, b, self.MTF_r[b]::self.ratio, self.MTF_c[b]::self.ratio] = self.mask[:, b, 2::self.ratio, 2::self.ratio]

        W_ = W_ / torch.sum(W_)

        x = x * W_
        y = y * W_
        L = self.loss(x, y)

        return L


class SpectralLossNocorr(nn.Module):
    def __init__(self, mtf, net_crop, ratio, device, mask=None):

        # Class initialization
        super(SpectralLossNocorr, self).__init__()
        kernel = mtf
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.net_scope = net_crop
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        #self.MTF_r = 2
        #self.MTF_c = 2
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='mean')

        # Mask definition
        #if mask is not None:
        #    self.mask = mask
        #else:
        #    self.mask = torch.ones((1, self.nbands, pan_shape[-2] - (self.net_scope + self.pad) * 2,
        #                            pan_shape[-1] - (self.net_scope + self.pad) * 2), device=self.device)

    def forward(self, outputs, labels):

        #In labels viene passata l'immagine di riferimento (MS) già croppata con "scope"

        x1 = self.depthconv(outputs)

        #Croppipamo ulteriormente l'immagine di riferimento per effetto del filtro gaussiano (MTF)
        y = labels
        #labels = labels[:, :, self.pad:-self.pad, self.pad:-self.pad]

        #y = torch.zeros(x.shape, device=self.device)
        #W_ = torch.zeros(x.shape, device=self.device)

        #for b in range(self.nbands):
            #y[:, b, self.MTF_r::self.ratio, self.MTF_c::self.ratio] = labels[:, b, 2::self.ratio, 2::self.ratio]
            #W_[:, b, self.MTF_r::self.ratio, self.MTF_c::self.ratio] = self.mask[:, b, 2::self.ratio, 2::self.ratio]


        x = x1[:, :, 3::self.ratio, 3::self.ratio]
        #x = x1[:, :, 2::self.ratio, 2::self.ratio]
        #x = x1[:, :, 1::self.ratio, 1::self.ratio]



        #W_ = W_ / torch.sum(W_)
        #print(x.shape, y.shape)
        #x = x * W_
        #y = y * W_

        L = self.loss(x, y)

        return L

class SpectralLossNocorrV(nn.Module):
    def __init__(self, mtf, net_crop, ratio, device, mask=None):

        # Class initialization
        super(SpectralLossNocorrV, self).__init__()
        kernel = mtf
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.net_scope = net_crop
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        #self.MTF_r = 2
        #self.MTF_c = 2
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='mean')
        #self.loss = nn.L2Loss(reduction='mean')

    def forward(self, outputs, labels):

        #In labels viene passata l'immagine di riferimento (MS) già croppata con "scope"
        x1 = self.depthconv(outputs)

        #Croppipamo ulteriormente l'immagine di riferimento per effetto del filtro gaussiano (MTF)
        y = labels
        L = self.loss(x1, y)

        return L


class Loss_Q(nn.Module):
    def __init__(self, mtf, net_crop, ratio, device, nbands, block_size=32):
        super(Loss_Q, self).__init__()
        self.block_size = block_size
        self.N = block_size ** 2
        filter_shape = (nbands, 1, self.block_size, self.block_size)
        kernel = torch.ones(filter_shape, dtype=torch.float32)

        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)
        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        kernel2 = mtf
        # Parameters definition
        self.net_scope = net_crop
        self.device = device
        self.ratio = ratio
        self.nbands = nbands

        # Conversion of filters in Tensor
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel2 = np.moveaxis(kernel2, -1, 0)
        kernel2 = np.expand_dims(kernel2, axis=1)

        kernel2 = torch.from_numpy(kernel2).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv2 = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel2.shape,
                                   bias=False)

        self.depthconv2.weight.data = kernel2
        self.depthconv2.weight.requires_grad = False



    def forward(self, outputs, labels):

        outputs = self.depthconv2(outputs)
        #outputs = outputs[:, :, 3::self.ratio, 3::self.ratio]

        outputs_sq = outputs ** 2
        labels_sq = labels ** 2
        outputs_labels = outputs * labels

        outputs_sum = self.depthconv(outputs)
        labels_sum = self.depthconv(labels)

        outputs_sq_sum = self.depthconv(outputs_sq)
        labels_sq_sum = self.depthconv(labels_sq)
        outputs_labels_sum = self.depthconv(outputs_labels)

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2
        numerator = 4 * (self.N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = self.N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)
        quality_map = torch.ones(denominator.size(), device=outputs.device)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = (denominator != 0)
        quality_map[index] = numerator[index] / denominator[index]
        quality = torch.mean(quality_map)

        return 1 - quality


'''
class Loss_Q(nn.Module):
    def __init__(self, mtf, net_crop, ratio, device, nbands, block_size=32):
        super(Loss_Q, self).__init__()
        self.block_size = block_size
        self.N = block_size ** 2
        filter_shape = (nbands, 1, self.block_size, self.block_size)
        kernel = torch.ones(filter_shape, dtype=torch.float32)

        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)
        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        kernel2 = mtf
        # Parameters definition
        self.net_scope = net_crop
        self.device = device
        self.ratio = ratio
        self.nbands = nbands

        # Conversion of filters in Tensor
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel2 = np.moveaxis(kernel2, -1, 0)
        kernel2 = np.expand_dims(kernel2, axis=1)

        kernel2 = torch.from_numpy(kernel2).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv2 = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel2.shape,
                                   bias=False)

        self.depthconv2.weight.data = kernel2
        self.depthconv2.weight.requires_grad = False



    def forward(self, outputs, labels):

        outputs = self.depthconv2(outputs)
        outputs = outputs[:, :, 3::self.ratio, 3::self.ratio]

        outputs_sq = outputs ** 2
        labels_sq = labels ** 2
        outputs_labels = outputs * labels

        outputs_sum = self.depthconv(outputs)
        labels_sum = self.depthconv(labels)

        outputs_sq_sum = self.depthconv(outputs_sq)
        labels_sq_sum = self.depthconv(labels_sq)
        outputs_labels_sum = self.depthconv(outputs_labels)

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2
        numerator = 4 * (self.N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = self.N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)
        quality_map = torch.ones(denominator.size(), device=outputs.device)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = (denominator != 0)
        quality_map[index] = numerator[index] / denominator[index]
        quality = torch.mean(quality_map, dim=(2, 3))

        return 1 - quality
'''

class StructuralLoss(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels, xcorr_thr):
        X_corr = torch.clamp(ccorr(outputs, labels, self.scale, self.device), min=-1)
        X = 1.0 - torch.abs(X_corr)
        #X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        worst = X.gt(xcorr_thr)
        #Y = X * worst
        Y = X
        Lxcorr = torch.mean(Y)

        return Lxcorr, Lxcorr_no_weights.item()


class StructuralLossOld(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLossOld, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels, xcorr_thr):
        X_corr = torch.clamp(ccorr(outputs, labels, self.scale, self.device), min=-1)
        #X = 1.0 - torch.abs(X_corr)
        X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        worst = X.gt(xcorr_thr)
        Y = X * worst
        #Y = X
        Lxcorr = torch.mean(Y)

        return Lxcorr, Lxcorr_no_weights.item()

class LSR(nn.Module):
    def __init__(self):
        # Class initialization
        super(LSR, self).__init__()

    def forward(self, outputs, pan):

        pan = pan.double()
        outputs = outputs.double()

        pan_flatten = torch.flatten(pan, start_dim=-2).transpose(2, 1)
        fused_flatten = torch.flatten(outputs, start_dim=-2).transpose(2, 1)
        #alpha = torch.linalg.lstsq(fused_flatten, pan_flatten).solution[:, :, :, None]
        with torch.no_grad():
            alpha = (fused_flatten.pinverse() @ pan_flatten)[:, :, :, None]
        i_r = torch.sum(outputs * alpha, dim=1, keepdim=True)

        err_reg = pan - i_r

        cd = 1 - (torch.var(err_reg, dim=(1, 2, 3)) / torch.var(pan, dim=(1, 2, 3)))

        return cd

class Ds_R(nn.Module):
    def __init__(self):
        super(Ds_R, self).__init__()
        self.metric = LSR()

    def forward(self, outputs, pan):
        lsr = torch.mean(self.metric(outputs, pan))
        return 1.0 - lsr

class Ds (nn.Module):
    def __init__(self, nbands, ratio=6, q=1, q_block_size=32):
        super(Ds, self).__init__()
        self.Q_high = Q(nbands, q_block_size)
        self.Q_low = Q(nbands, q_block_size // ratio)
        self.nbands = nbands
        self.ratio = ratio
        self.q = q
    def forward(self, outputs, pan, ms):
        pan = pan.repeat(1, self.nbands, 1, 1)
        pan_lr = resize(pan, scale=1/self.ratio)

        Q_high = self.Q_high(outputs, pan)
        Q_low = self.Q_low(ms, pan_lr)

        Ds = torch.sum(abs(Q_high - Q_low) ** self.q, dim=1)

        Ds = (Ds / self.nbands) ** (1 / self.q)

        return Ds


class SAMLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SAMLoss, self).__init__()
        self.reduction = reduction
        self.pi = np.pi
        self.eps = 1e-8
    def forward(self, outputs, labels):

        norm_outputs = torch.sum(outputs * outputs, dim=1)
        norm_labels = torch.sum(labels * labels, dim=1)
        scalar_product = torch.sum(outputs * labels, dim=1)
        norm_product = torch.sqrt(norm_outputs * norm_labels)

        mask = norm_product == 0
        scalar_product = mask * self.eps + torch.logical_not(mask) * scalar_product
        norm_product = mask * self.eps + torch.logical_not(mask) * norm_product
        # norm_product[norm_product == 0] = float('nan')
        scalar_product = torch.flatten(scalar_product, 1, 2)
        norm_product = torch.flatten(norm_product, 1, 2)
        angle = torch.sum(scalar_product / norm_product, dim=1) / norm_product.shape[1]
        loss = 1 - angle
        return torch.mean(loss)

class StructuralLossWeighted(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLossWeighted, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels, xcorr_thr):
        X_corr = torch.clamp(ccorr(outputs, labels, self.scale, self.device), min=-1)
        X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        Y = X * torch.abs(xcorr_thr)

        Lxcorr = torch.mean(Y)

        return Lxcorr, Lxcorr_no_weights.item()