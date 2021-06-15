"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from block import Bottleneck, CRPBlock, conv1x1, conv3x3, FusionBlock
import torchvision.models as models


class MobileCount(nn.Module):

    def __init__(self, layer_sizes, pretrained=None):
        self.layers_sizes = layer_sizes
        self.inplanes = layer_sizes[0]
        block = Bottleneck
        repetitions = [1, 2, 3, 4]
        expansion = [1, 6, 6, 6]
        strides = [1, 2, 2, 2]

        super(MobileCount, self).__init__()

        self.conv1 = nn.Conv2d(3, layer_sizes[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_sizes[0])
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder = torch.nn.ModuleList(
            [self._make_layer(block, layer_sizes[i], repetitions[i], strides[i], expansion[i]) for i in range(4)])

        self.dropout4 = nn.Dropout(p=0.5)
        self.enlarge1 = conv1x1(layer_sizes[3], layer_sizes[1], bias=False)
        self.crp1 = self._make_crp(layer_sizes[1], layer_sizes[1], 4)

        self.fusion1 = FusionBlock(conv_weight_dim=(layer_sizes[1], layer_sizes[0]),
                                   conv_adapt_dim=(layer_sizes[0], layer_sizes[0]))
        self.dropout3 = nn.Dropout(p=0.5)
        self.enlarge2 = conv1x1(layer_sizes[2], layer_sizes[0], bias=False)
        self.crp2 = self._make_crp(layer_sizes[0], layer_sizes[0], 4)

        self.fusion2 = FusionBlock(conv_weight_dim=(layer_sizes[0], layer_sizes[0]),
                                   conv_adapt_dim=(layer_sizes[0], layer_sizes[0]))

        self.enlarge3 = conv1x1(layer_sizes[1], layer_sizes[0], bias=False)
        self.crp3 = self._make_crp(layer_sizes[0], layer_sizes[0], 4)

        self.fusion3 = FusionBlock(conv_weight_dim=(layer_sizes[0], layer_sizes[0]),
                                   conv_adapt_dim=(layer_sizes[0], layer_sizes[0]))

        self.enlarge4 = conv1x1(layer_sizes[0], layer_sizes[0], bias=False)
        self.crp4 = self._make_crp(layer_sizes[0], layer_sizes[0], 4)

        self.dropout_clf = nn.Dropout(p=0.5)
        self.clf_conv = nn.Conv2d(layer_sizes[0], 1, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            print('load the pre-trained model.')
            resnet = getattr(models, pretrained)(True)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.encoder[0] = resnet.layer1
            self.encoder[1] = resnet.layer2
            self.encoder[2] = resnet.layer3
            self.encoder[3] = resnet.layer4

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        size1 = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.encoder[0](x)
        l2 = self.encoder[1](l1)
        l3 = self.encoder[2](l2)
        l4 = self.encoder[3](l3)

        l4 = self.dropout4(l4)
        x4 = self.enlarge1(l4)
        x4 = self.relu(x4)
        x4 = self.crp1(x4)

        l3 = self.dropout3(l3)
        x3 = self.enlarge2(l3)
        x3 = self.fusion1(x4, x3)
        x3 = self.crp2(x3)

        x2 = self.enlarge3(l2)
        x2 = self.fusion2(x3, x2)
        x2 = self.crp3(x2)

        x1 = self.enlarge4(l1)
        x1 = self.fusion3(x2, x1)
        x1 = self.crp4(x1)

        x1 = self.dropout_clf(x1)
        out = self.clf_conv(x1)

        out = F.interpolate(out, size=size1, mode='bilinear', align_corners=False)

        return out
