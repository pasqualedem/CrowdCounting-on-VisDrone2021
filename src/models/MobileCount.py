"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""
import torch.nn as nn
import torch.nn.functional as F
from models.block import Bottleneck, CRPBlock, conv1x1, FusionBlock, ConvTransposeUpsampling, ConvUpsampling
from models.CC import CrowdCounterNetwork, Encoder


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


class MobileCount(CrowdCounterNetwork):

    def __init__(self, encoder, encoder_params, decoder, decoder_params, upsampling='interpolation'):
        super(MobileCount, self).__init__(encoder, encoder_params, decoder, decoder_params)
        self.layers_sizes = self.encoder.get_layer_sizes()

        self.dropout_clf = nn.Dropout(p=0.5)
        self.clf_conv = nn.Conv2d(self.layers_sizes[0], 1, kernel_size=3, stride=1,
                                  padding=1, bias=True)
        if upsampling == 'convtrans':
            self.upsampling = ConvTransposeUpsampling(1, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif upsampling == 'conv':
            self.upsampling = ConvUpsampling(1)

    def forward(self, x):
        size = x.shape[2:]

        x = self.encoder(x)

        dec = self.decoder(x)
        dec = self.dropout_clf(dec)
        out = self.clf_conv(dec)
        if hasattr(self, 'upsampling'):
            out = self.upsampling(out)
        else:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)

        return out


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


def _make_crp(in_planes, out_planes, stages):
    layers = [CRPBlock(in_planes, out_planes, stages)]
    return nn.Sequential(*layers)


class LWDecoder(nn.Module):
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

    def forward(self, x):
        l1, l2, l3, l4 = x
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

