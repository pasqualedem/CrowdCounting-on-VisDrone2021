import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CC import CrowdCounterNetwork


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SASNet(CrowdCounterNetwork):
    def __init__(self, encoder, encoder_params, decoder, decoder_params, block_size):
        super().__init__(encoder, encoder_params, decoder, decoder_params)
        layer_sizes = self.encoder.get_layer_sizes()

        self.density_head = nn.ModuleList([
            DensityHeadBlock(lsize, lsize * 4) for lsize in layer_sizes
        ])

        self.confidence_head = nn.ModuleList([
            ConfidenceHeadBlock(lsize, lsize // 4) for lsize in layer_sizes
        ])

        self.block_size = block_size

    def forward(self, x):
        size = x.size()

        xenc = self.encoder(x)
        xdec = self.decoder(xenc)[::-1]

        steps = list(range(len(self.density_head)))

        # density prediction
        xden = []
        for i in steps:
            mid = self.density_head[i](xdec[i])
            # upsample the density prediction to be the same with the input size
            xden.append(F.upsample_nearest(mid, size=size[2:]))

        xconf = []
        for i in steps:
            # get patch features for confidence prediction
            mid = F.adaptive_avg_pool2d(xdec[i], output_size=(size[-2] // self.block_size, size[-1] // self.block_size))
            # confidence prediction
            mid = self.confidence_head[i](mid)
            # upsample the confidence prediction to be the same with the input size
            xconf.append(F.upsample_nearest(mid, size=size[2:]))

        # =============================================================================================================
        # soft √
        confidence_map = torch.cat(xconf, 1)
        confidence_map = torch.nn.functional.sigmoid(confidence_map)

        # use softmax to normalize
        confidence_map = torch.nn.functional.softmax(confidence_map, 1)

        density_map = torch.cat(xden, 1)
        # soft selection
        density_map *= confidence_map
        density = torch.sum(density_map, 1, keepdim=True)

        return density


class DensityHeadBlock(nn.Module):
    def __init__(self, in_branch, in_conv):
        super().__init__()
        self.block = nn.Sequential(
            MultiBranchModule(in_branch),
            Conv2d(in_conv, 1, 1, same_padding=True)
        )

    def forward(self, x):
        return self.block(x)


class ConfidenceHeadBlock(nn.Module):
    def __init__(self, in_planes, mid_planes):
        super().__init__()
        self.block = nn.Sequential(
            Conv2d(in_planes, mid_planes, 1, same_padding=True, NL='relu'),
            Conv2d(mid_planes, 1, 1, same_padding=True, NL=None)
        )

    def forward(self, x):
        return self.block(x)


class SASDecoder(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.decoder = nn.ModuleList([
            nn.Sequential(
                Conv2d(layer_sizes[i] + layer_sizes[i] // 2, layer_sizes[i], 3, same_padding=True, NL='relu'),
                Conv2d(layer_sizes[i], layer_sizes[i] // 2, 3, same_padding=True, NL='relu'),
            ) for i in range(1, len(layer_sizes))

        ])
        self.decoder.extend([
            nn.Sequential(
                Conv2d(layer_sizes[-1], layer_sizes[-1] * 2, 3, same_padding=True, NL='relu'),
                Conv2d(layer_sizes[-1] * 2, layer_sizes[-1], 3, same_padding=True, NL='relu'))
        ])

    def forward(self, xs):
        steps = list(range(len(self.decoder))[::-1])
        steps.pop(0)
        last = steps.pop()

        x = self.decoder[-1](xs[-1])
        out = [x]
        x = F.upsample_bilinear(x, size=xs[-2].size()[2:])

        for i in steps:
            x = torch.cat([xs[i], x], 1)
            x = self.decoder[i](x)
            out.append(x)
            x = F.upsample_bilinear(x, size=xs[i - 1].size()[2:])

        x = torch.cat([xs[last], x], 1)
        x = self.decoder[last](x)
        out.append(x)

        return out


# the module definition for the multi-branch in the density head
class MultiBranchModule(nn.Module):
    def __init__(self, in_channels, sync=False):
        super(MultiBranchModule, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, in_channels // 2, kernel_size=1, sync=sync)
        self.branch1x1_1 = BasicConv2d(in_channels // 2, in_channels, kernel_size=1, sync=sync)

        self.branch3x3_1 = BasicConv2d(in_channels, in_channels // 2, kernel_size=1, sync=sync)
        self.branch3x3_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=(3, 3), padding=(1, 1), sync=sync)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, in_channels // 2, kernel_size=1, sync=sync)
        self.branch3x3dbl_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=5, padding=2, sync=sync)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.branch1x1_1(branch1x1)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        outputs = [branch1x1, branch3x3, branch3x3dbl, x]
        return torch.cat(outputs, 1)


# the module definition for the basic conv module
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, sync=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        if sync:
            # for sync bn
            print('use sync inception')
            self.bn = nn.SyncBatchNorm(out_channels, eps=0.001)
        else:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
