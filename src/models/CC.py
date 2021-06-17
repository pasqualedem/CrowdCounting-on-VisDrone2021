import torch
from models.MobileCountVis import *
MBVersions = {
    'x0_5': [16, 32, 64, 128],
    'x0_75': [32, 48, 80, 160],
    '': [32, 64, 128, 256],
    'x1_25': [64, 96, 160, 320],
    'x2': [64, 128, 256, 512],
    'x4': [256, 512, 1024, 2048]
}

# Dict that select which parameter pass to the class for each network
Nets = {
    'Pretrained': (PretrainedEncoder, ['ENCODER', 'PRETRAINED']),
    'MobileCount': (LWEncoder, ['CHANNELS'])
}


def choose_encoder(model_args):
    try:
        model, key_args = Nets[model_args['ENCODER']]
    except KeyError:
        model, key_args = Nets['Pretrained']

    args = []
    if model == LWEncoder:
        args = [(MBVersions[model_args.pop('VERSION')])]

    args = args + [model_args[arg] for arg in key_args]

    return model, args


def choose_model(model_args):
    """
    Choose which model to use and instantiate it passing the args
    :param model_args: Dict of args that contain NET key and related arguments
    :return: The instantiated network
    """
    model, args = choose_encoder(model_args)

    if model_args['ENCODER_TIR']:
        tir_encoder_dict = {arg.replace('_TIR', ''): model_args[arg] for arg in model_args if '_TIR' in arg}
        tir_encoder, tir_args = choose_encoder(tir_encoder_dict)
        encoder_rgb = model
        return DoubleEncoderMobileCount([encoder_rgb,
                                         args,
                                         tir_encoder,
                                         tir_args])

    return SingleEncoderMobileCount(model, args)


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

