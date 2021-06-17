"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block import Bottleneck, CRPBlock, conv1x1, conv3x3, FusionBlock
import torchvision.models as models


def initialize_weights(module: nn.Module):
    for name, m in module.named_children():
        if hasattr(m, 'pretrained'):
            continue
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        initialize_weights(m)


class MobileCountBase(nn.Module):

    def __init__(self, layer_sizes, encoder, decoder):
        self.layers_sizes = layer_sizes

        super(MobileCountBase, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.dropout_clf = nn.Dropout(p=0.5)
        self.clf_conv = nn.Conv2d(layer_sizes[0], 1, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def forward(self, x):
        size = x.shape[2:]

        l1, l2, l3, l4 = self.encoder(x)

        dec = self.decoder(l1, l2, l3, l4)
        dec = self.dropout_clf(dec)
        out = self.clf_conv(dec)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)

        return out

    def train(self, mode: bool = True):
        for name, m in self.named_children():
            if hasattr(m, 'pretrained'):
                m.train(False)


class Encoder(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layers[0](x)
        l2 = self.layers[1](l1)
        l3 = self.layers[2](l2)
        l4 = self.layers[3](l3)

        return l1, l2, l3, l4

    def get_layer_sizes(self):
        return self.layer_sizes


class PretrainedEncoder(Encoder):
    def __init__(self, model_name, pretrained):
        super().__init__()
        print('load the pre-trained model.')
        net = getattr(models, model_name)(pretrained)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()
        self.layers.append(net.layer1)
        self.layers.append(net.layer2)
        self.layers.append(net.layer3)
        self.layers.append(net.layer4)

        self.layer_sizes = [list(list(layer.named_children())[-1][1].named_children())[-3][1].out_channels
                            for layer in self.layers]

        if pretrained:
            for param in self.parameters():
                param.requires_grad = False
                self.pretrained = True
            self.train(False)


class LWEncoder(Encoder):
    def __init__(self, layer_sizes, input_channels):
        super().__init__()
        block = Bottleneck
        repetitions = [1, 2, 3, 4]
        expansion = [1, 6, 6, 6]
        strides = [1, 2, 2, 2]
        self.inplanes = layer_sizes[0]

        self.conv1 = nn.Conv2d(input_channels, layer_sizes[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList(
            [self._make_layer(block, layer_sizes[i], repetitions[i], strides[i], expansion[i]) for i in range(4)])
        self.layer_sizes = layer_sizes

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)


class DoubleEncoder(Encoder):
    def __init__(self, encoder_rgb: Encoder, rgb_args, encoder_tir: Encoder, tir_args):
        super().__init__()
        self.encoder_rgb = encoder_rgb(*rgb_args)
        self.encoder_tir = encoder_tir(*tir_args)
        # if self.encoder_rgb.get_layer_sizes() != self.encoder_tir.get_layer_sizes():
        #     raise Exception('The two encoders must have the same output layer sizes!')
        self.layer_sizes = self.encoder_rgb.get_layer_sizes() + self.encoder_tir.get_layer_sizes()

    def forward(self, x):
        rgb_out = self.encoder_rgb(x[:, 0:3])
        tir_out = self.encoder_tir(x[:, 3:])
        # return (mid1 + mid2 for mid1, mid2 in zip(rgb_out, tir_out))
        return (torch.hstack((mid1 + mid2)) for mid1, mid2 in zip(rgb_out, tir_out))


def _make_crp(in_planes, out_planes, stages):
    layers = [CRPBlock(in_planes, out_planes, stages)]
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.dropouts = nn.ModuleList()
        self.enlargings = nn.ModuleList()
        self.CRPs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        self.relu = nn.ReLU(inplace=True)

        self.dropouts.append(nn.Dropout(p=0.5))
        self.enlargings.append(conv1x1(layer_sizes[3], layer_sizes[1], bias=False))
        self.CRPs.append(_make_crp(layer_sizes[1], layer_sizes[1], 4))

        self.fusions.append(FusionBlock(conv_weight_dim=(layer_sizes[1], layer_sizes[0]),
                                        conv_adapt_dim=(layer_sizes[0], layer_sizes[0])))

        self.dropouts.append(nn.Dropout(p=0.5))
        self.enlargings.append(conv1x1(layer_sizes[2], layer_sizes[0], bias=False))
        self.CRPs.append(_make_crp(layer_sizes[0], layer_sizes[0], 4))

        self.fusions.append(FusionBlock(conv_weight_dim=(layer_sizes[0], layer_sizes[0]),
                                        conv_adapt_dim=(layer_sizes[0], layer_sizes[0])))

        self.enlargings.append(conv1x1(layer_sizes[1], layer_sizes[0], bias=False))
        self.CRPs.append(_make_crp(layer_sizes[0], layer_sizes[0], 4))

        self.fusions.append(FusionBlock(conv_weight_dim=(layer_sizes[0], layer_sizes[0]),
                                        conv_adapt_dim=(layer_sizes[0], layer_sizes[0])))

        self.enlargings.append(conv1x1(layer_sizes[0], layer_sizes[0], bias=False))
        self.CRPs.append(_make_crp(layer_sizes[0], layer_sizes[0], 4))

    def forward(self, l1, l2, l3, l4):
        l4 = self.dropouts[0](l4)
        x4 = self.enlargings[0](l4)
        x4 = self.relu(x4)
        x4 = self.CRPs[0](x4)

        l3 = self.dropouts[1](l3)
        x3 = self.enlargings[1](l3)
        x3 = self.fusions[0](x4, x3)
        x3 = self.CRPs[1](x3)

        x2 = self.enlargings[2](l2)
        x2 = self.fusions[1](x3, x2)
        x2 = self.CRPs[2](x2)

        x1 = self.enlargings[3](l1)
        x1 = self.fusions[2](x2, x1)
        x1 = self.CRPs[3](x1)

        return x1


class SingleEncoderMobileCount(MobileCountBase):
    def __init__(self, encoder, encoder_params):
        encoder = encoder(*encoder_params)
        decoder = Decoder(encoder.get_layer_sizes())

        super().__init__(encoder.get_layer_sizes(), encoder, decoder)


class DoubleEncoderMobileCount(MobileCountBase):
    def __init__(self, encoder_params):
        encoder = DoubleEncoder(*encoder_params)
        decoder = Decoder(encoder.layer_sizes)

        super().__init__(encoder.get_layer_sizes(), encoder, decoder)