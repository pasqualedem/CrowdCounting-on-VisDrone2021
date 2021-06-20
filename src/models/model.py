from models.MobileCount import LWEncoder, LWDecoder, MobileCount
from models.ScaleAdaptiveSelection import SASNet, SASDecoder
from models.CC import CrowdCounter, DoubleEncoder, PretrainedEncoder

MBVersions = {
    'x0_5': [16, 32, 64, 128],
    'x0_75': [32, 48, 80, 160],
    '': [32, 64, 128, 256],
    'x1_25': [64, 96, 160, 320],
    'x2': [64, 128, 256, 512],
    'x4': [256, 512, 1024, 2048]
}

# Dict that select which parameter pass to the class for each network
Predictors = {
    'MobileCount': (MobileCount, []),
    'SASNet': (SASNet, ['BLOCK_SIZE'])
             }

Encoders = {
    'Pretrained': (PretrainedEncoder, ['ENCODER', 'PRETRAINED', 'BLOCKS', 'CHANNELS']),
    'LWEncoder': (LWEncoder, ['CHANNELS'])
}

Decoders = {
    'SASDecoder': (SASDecoder, []),
    'LWDecoder': (LWDecoder, [])
}


def choose_encoder(model_args):
    try:
        model, key_args = Encoders[model_args['ENCODER']]
    except KeyError:
        model, key_args = Encoders['Pretrained']

    args = []
    if model == LWEncoder:
        args = [(MBVersions[model_args.pop('VERSION')])]

    args = args + [model_args[arg] for arg in key_args]

    return model, args


def choose_model(gpus, model_args):
    """
    Choose which model to use and instantiate it passing the args
    :param model_args: Dict of args that contain NET key and related arguments
    :return: The instantiated network
    """
    model, args = Predictors[model_args.PREDICTOR]
    encoder, enc_args = choose_encoder(model_args)
    decoder, dec_args = Decoders[model_args.DECODER]

    if model_args['ENCODER_TIR']:
        rgb_encoder = encoder
        encoder = DoubleEncoder
        tir_encoder_dict = {arg.replace('_TIR', ''): model_args[arg] for arg in model_args if '_TIR' in arg}
        tir_encoder, tir_args = choose_encoder(tir_encoder_dict)
        enc_args = [rgb_encoder, enc_args, tir_encoder, tir_args]

    args = [model_args[arg] for arg in args]
    return CrowdCounter(gpus, model(encoder, enc_args, decoder, dec_args, *args))
