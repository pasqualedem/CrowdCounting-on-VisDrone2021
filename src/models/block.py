import torch
import torch.nn as nn
import torch.nn.functional as F

# Helpers / wrappers
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        self.point_convs = nn.ModuleList()
        for i in range(n_stages):
            self.point_convs.append(
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for conv in self.point_convs:
            top = self.maxpool(top)
            top = conv(top)
            x = top + x
        return x


class ConvTransposeUpsampling(nn.Module):
    expansion = 1

    def __init__(self, channels, kernel_size=5, stride=4, padding=1, output_padding=1, dilation=2):
        super(ConvTransposeUpsampling, self).__init__()
        self.convtrans = nn.ConvTranspose2d(channels, channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            output_padding=output_padding)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size):
        x = self.convtrans(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.conv(x)

        return x


class ConvUpsampling(nn.Module):
    expansion = 1

    def __init__(self, channels):
        super(ConvUpsampling, self).__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=3, padding=1)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.conv(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FusionBlock(nn.Module):
    def __init__(self, conv_weight_dim, conv_adapt_dim):
        super(FusionBlock, self).__init__()
        self.conv_weight = conv1x1(*conv_weight_dim, bias=False)
        self.conv_adapt = conv1x1(*conv_adapt_dim, bias=False)

    def forward(self, x, y):
        midx = self.conv_weight(x)
        midx = nn.Upsample(size=y.size()[2:], mode='bilinear')(midx)

        midy = self.conv_adapt(y)

        out = midx + midy
        out = F.relu(out)

        return out
