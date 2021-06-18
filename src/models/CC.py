import torch
import torch.nn as nn
from torch import nn as nn
from torchvision import models as models


class CrowdCounter(nn.Module):
    """
    Container class for MobileCount networks
    """

    def __init__(self, gpus, ccn):
        super(CrowdCounter, self).__init__()

        self.CCN = ccn
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()

    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse

    def forward(self, img):
        return self.CCN(img)

    def predict(self, img):
        return self(img)

    def load(self, model_path):
        try:
            self.load_state_dict(torch.load(model_path)['model_state_dict'])
        except KeyError:
            self.load_state_dict(torch.load(model_path))  # Retrocompatibility

    def build_loss(self, density_map, gt_data):
        self.loss_mse = self.loss_mse_fn(density_map.squeeze(), gt_data.squeeze())
        return self.loss_mse


class CrowdCounterNetwork(nn.Module):
    def __init__(self, encoder, encoder_params, decoder, decoder_params):
        super().__init__()
        self.encoder = encoder(*encoder_params)
        self.decoder = decoder(self.encoder.get_layer_sizes(), *decoder_params)


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

        return [l1, l2, l3, l4]

    def get_layer_sizes(self):
        return self.layer_sizes


class DoubleEncoder(Encoder):
    def __init__(self, encoder_rgb: Encoder, rgb_args, encoder_tir: Encoder, tir_args):
        super().__init__()
        self.encoder_rgb = encoder_rgb(*rgb_args)
        self.encoder_tir = encoder_tir(*tir_args)
        # if self.encoder_rgb.get_layer_sizes() != self.encoder_tir.get_layer_sizes():
        #     raise Exception('The two encoders must have the same output layer sizes!')
        self.layer_sizes = [out_rgb + out_tir for
                            out_rgb, out_tir in
                            zip(self.encoder_rgb.get_layer_sizes(), self.encoder_rgb.get_layer_sizes())]

    def forward(self, x):
        rgb_out = self.encoder_rgb(x[:, 0:3])
        tir_out = self.encoder_tir(x[:, 3:])
        # return (mid1 + mid2 for mid1, mid2 in zip(rgb_out, tir_out))
        return [torch.hstack((mid1, mid2)) for mid1, mid2 in zip(rgb_out, tir_out)]


class PretrainedEncoder(Encoder):
    def __init__(self, model_name, pretrained, channels=None):
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

        self.channels = channels
        self.layer_sizes = get_layer_sizes(self.layers, model_name)

        if pretrained:
            for param in self.parameters():
                param.requires_grad = False
                self.pretrained = True
            self.train(False)

    def forward(self, x):
        if self.channels != x.shape[1]:
            x = x.repeat(1, self.channels, 1, 1)
        return super().forward(x)


def get_layer_sizes(layers, model_name):
    last_conv = -3
    if model_name == 'resnet34' or model_name == 'resnet18':
        last_conv = -2

    return [list(list(layer.named_children())[-1][1].named_children())[last_conv][1].out_channels
            for layer in layers]
