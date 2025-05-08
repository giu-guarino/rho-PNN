import torch.nn as nn
import torch.nn.functional as F


# ************** NETWORK DEFINITION *********************

class R_PNN(nn.Module):
    def __init__(self, nbands, n_features, kernel_size, padding='same', padding_mode='reflect', bias=True) -> None:
        super(R_PNN, self).__init__()
        self.conv1 = nn.Conv2d(nbands + 1, n_features[0], kernel_size[0], padding=padding, padding_mode=padding_mode, bias=bias)
        self.conv2 = nn.Conv2d(n_features[0], n_features[1], kernel_size[1], padding=padding, padding_mode=padding_mode, bias=bias)
        self.conv3 = nn.Conv2d(n_features[1], n_features[2], kernel_size[2], padding=padding, padding_mode=padding_mode, bias=bias)


    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + input[:,:-1,:,:]
        return x




##########################################


from other_scripts.cbam import CBAM


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, pad='same', pad_mode='reflect', bn=False, act=nn.GELU()):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=bias, padding=pad, padding_mode=pad_mode))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class R_PNN_ATT(nn.Module):
    def __init__(self, n_channels, n_features=64, kernel_size=3, pad='same', pad_mode='reflect', bias_flag=True):
        super(R_PNN_ATT, self).__init__()

        self.conv_1 = nn.Conv2d(n_channels, n_features, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)

        self.conv_2 = nn.Conv2d(n_features, n_features, kernel_size, bias=bias_flag, padding=pad,
                                padding_mode=pad_mode)
        self.CBAM_1 = CBAM(n_features, reduction_ratio=6, spatial=True)
        self.res_block_1 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.res_block_2 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.CBAM_2 = CBAM(n_features, reduction_ratio=6, spatial=True)
        self.conv_3 = nn.Conv2d(n_features, n_channels-1, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)

    def forward(self, inp):

        x = F.relu(self.conv_1(inp))
        x = F.relu(self.conv_2(x))
        x = self.CBAM_1(x) + x
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.CBAM_2(x) + x
        x = self.conv_3(x)
        x = x + inp[:, :-1, :, :]

        return x



class R_PNN_ATT_light(nn.Module):
    def __init__(self, n_channels, n_features=64, kernel_size=3, pad='same', pad_mode='reflect', bias_flag=True):
        super(R_PNN_ATT_light, self).__init__()

        self.conv_1 = nn.Conv2d(n_channels, n_features, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)

        self.conv_2 = nn.Conv2d(n_features, n_features, kernel_size, bias=bias_flag, padding=pad,
                                padding_mode=pad_mode)
        self.CBAM_1 = CBAM(n_features, reduction_ratio=6, spatial=True)
        self.res_block_1 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.res_block_2 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.CBAM_2 = CBAM(n_features, reduction_ratio=6, spatial=True)
        self.conv_3 = nn.Conv2d(n_features, n_channels-1, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)

    def forward(self, inp):

        x = F.relu(self.conv_1(inp))
        x = F.relu(self.conv_2(x))
        x = self.CBAM_1(x) + x
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.CBAM_2(x) + x
        x = self.conv_3(x)
        x = x + inp[:, :-1, :, :]

        return x


class Pannet(nn.Module):
    # PANNET
    def __init__(self, n_feature=32,padding='same', padding_mode='reflect', bias=True) -> None:
        super(Pannet, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(2, n_feature, 3, padding=padding, padding_mode=padding_mode, bias=bias),
            nn.ReLU())

        self.res_block = ResBlock(n_feature, 3, act=nn.ReLU())
        self.res_block2 = ResBlock(n_feature, 3, act=nn.ReLU())
        self.res_block3 = ResBlock(n_feature, 3, act=nn.ReLU())
        self.res_block4 = ResBlock(n_feature, 3, act=nn.ReLU())

        self.conv_out = nn.Conv2d(n_feature, 1, 3, padding=padding, padding_mode=padding_mode, bias=bias)

    def forward(self, inp):

        #x = torch.cat((pan, hr), axis=1)
        x = self.conv_in(inp)

        x = self.res_block(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.conv_out(x)

        x = x + inp[:, :-1, :, :]
        return x


class R_PNN_2(nn.Module):
    def __init__(self, n_channels, n_features=32, kernel_size=3, pad='same', pad_mode='reflect', bias_flag=True):
        super(R_PNN_2, self).__init__()

        self.conv_1 = nn.Conv2d(n_channels, n_features, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)

        self.res_block_1 = ResBlock(n_features, 5, bias=bias_flag)
        self.res_block_2 = ResBlock(n_features, 3, bias=bias_flag)
        self.res_block_3 = ResBlock(n_features, 3, bias=bias_flag)
        self.conv_3 = nn.Conv2d(n_features, n_channels-1, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)

    def forward(self, inp):

        x = F.relu(self.conv_1(inp))
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        #x = self.res_block_3(x)
        x = self.conv_3(x)
        x = x + inp[:, :-1, :, :]

        return x