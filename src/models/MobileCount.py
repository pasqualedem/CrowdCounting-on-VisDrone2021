"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""

import math

import torch.nn as nn
import torch.nn.functional as F
from block import Bottleneck, CRPBlock, conv1x1, conv3x3

class MobileCount(nn.Module):

    def __init__(self, layer_sizes):
        self.layers_sizes = layer_sizes
        self.inplanes = layer_sizes[0]
        block = Bottleneck
        layers = [1, 2, 3, 4]
        expansion = [1, 6, 6, 6]
        strides = [1, 2, 2, 2]

        super(MobileCount, self).__init__()

        # implement of mobileNetv2
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

        self.conv1 = nn.Conv2d(3, layer_sizes[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_sizes[0])
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layer_sizes[0], layers[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, layer_sizes[1], layers[1], stride=2, expansion=6)
        self.layer3 = self._make_layer(block, layer_sizes[2], layers[2], stride=2, expansion=6)
        self.layer4 = self._make_layer(block, layer_sizes[3], layers[3], stride=2, expansion=6)

        self.dropout4 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl1_dimred = conv1x1(layer_sizes[3], layer_sizes[1], bias=False)
        self.mflow_conv_g1_pool = self._make_crp(layer_sizes[1], layer_sizes[1], 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(layer_sizes[1], layer_sizes[0], bias=False)

        self.dropout3 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl2_dimred = conv1x1(layer_sizes[2], layer_sizes[0], bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(layer_sizes[0], layer_sizes[0], bias=False)
        self.mflow_conv_g2_pool = self._make_crp(layer_sizes[0], layer_sizes[0], 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(layer_sizes[0], layer_sizes[0], bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(layer_sizes[1], layer_sizes[0], bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(layer_sizes[0], layer_sizes[0], bias=False)
        self.mflow_conv_g3_pool = self._make_crp(layer_sizes[0], layer_sizes[0], 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(layer_sizes[0], layer_sizes[0], bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(layer_sizes[0], layer_sizes[0], bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(layer_sizes[0], layer_sizes[0], bias=False)
        self.mflow_conv_g4_pool = self._make_crp(layer_sizes[0], layer_sizes[0], 4)

        self.dropout_clf = nn.Dropout(p=0.5)
        # self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
        #                           padding=1, bias=True)
        self.clf_conv = nn.Conv2d(layer_sizes[0], 1, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.dropout4(l4)
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear')(x4)

        l3 = self.dropout3(l3)
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear')(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear')(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        x1 = self.dropout_clf(x1)
        out = self.clf_conv(x1)

        out = F.interpolate(out, size=size1, mode='bilinear', align_corners=False)

        return out
