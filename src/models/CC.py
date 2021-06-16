import torch
import torch.nn as nn
from models.MobileCountVis import MobileCount, PretrainedMobileCount, DoubleEncoderMobileCount
MBVersions = {
    'x0_5': [16, 32, 64, 128],
    'x0_75': [32, 48, 80, 160],
    '': [32, 64, 128, 256],
    'x1_25': [64, 96, 160, 320],
    'x2': [64, 128, 256, 512],
}

# Dict that select which parameter pass to the class for each network
Nets = {
    'Pretrained': (PretrainedMobileCount, ['KNOWN_MODEL', 'PRETRAINED']),
    'MobileCount': (MobileCount, ['VERSION']),
    'DoubleEncoder': (DoubleEncoderMobileCount, ['ENCODER', 'KNOWN_MODEL', 'PRETRAINED', 'VERSION'])
}


def choose_model(model_args):
    """
    Choose which model to use and isntantiate it passing the args
    :param model_args: Dict of args that contain NET key and related arguments
    :return: The instatiated network
    """
    model, args = Nets[model_args.pop('NET')]
    if model == 'DoubleEncoder':
        encoder, args = Nets[model_args.pop('ENCODER')]
        return model(encoder, *[model_args[arg] for arg in args])
    return model(*[model_args[arg] for arg in args])


class CrowdCounter(nn.Module):
    """
    Container class for MobileCount networks
    """
    def __init__(self, gpus, model_args):
        super(CrowdCounter, self).__init__()

        self.CCN = choose_model(model_args)
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

