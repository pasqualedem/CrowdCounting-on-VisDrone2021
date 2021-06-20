from easydict import EasyDict
import time

__C = EasyDict()
cfg = __C

__C.SEED = 3035  # random seed

# System settings
__C.TRAIN_BATCH_SIZE = 1
__C.VAL_BATCH_SIZE = 6
__C.TEST_BATCH_SIZE = 6
__C.N_WORKERS = 2

# Training settings
__C.RESUME = False

# path settings
__C.EXP_PATH = '../exp'
__C.DATASET = 'VisDrone2021'
__C.DETAILS = ''

# Net settings ########################################
__C.NET = EasyDict()
# PREDICTOR
__C.NET.PREDICTOR = 'MobileCount'
__C.NET.UPSAMPLING = 'interp'
__C.NET.BLOCK_SIZE = 32  # SASNet block size setting

__C.NET.BLOCKS = 4  # Number of blocks of the encoder (and so the decoder)
# ENCODER
__C.NET.COMPOSED = False
__C.NET.ENCODER = 'inception_v3'
# For MobileCount
__C.NET.VERSION = 'x2'
__C.NET.CHANNELS = 3
# For known models
__C.NET.PRETRAINED = True

# Possible second encoder
__C.NET.COMPOSED_TIR = False
__C.NET.ENCODER_TIR = 'inception_v3'
# For MobileCount
__C.NET.VERSION_TIR = ''
__C.NET.CHANNELS_TIR = 3
# For known models
__C.NET.PRETRAINED_TIR = True

# DECODER
__C.NET.DECODER = 'LWDecoder'

# learning optimizer settings ########################################
__C.LR = 1e-4  # learning rate
__C.W_DECAY = 1e-4  # weight decay
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = 0  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 500

__C.OPTIM_ADAM = ('Adam',
                  {
                      'lr': __C.LR,
                      'weight_decay': __C.W_DECAY,
                  })
__C.OPTIM_SGD = ('SGD',
                 {
                     'lr': __C.LR,
                     'weight_decay': __C.W_DECAY,
                     'momentum': 0.95
                 })

__C.OPTIM = __C.OPTIM_ADAM  # Chosen optimizer

__C.PATIENCE = 20
__C.EARLY_STOP_DELTA = 1e-2

# print
__C.PRINT_FREQ = 10

__C.DEVICE = 'cuda'  # cpu or cuda

# ------------------------------VAL------------------------
__C.VAL_SIZE = 0.2
__C.VAL_DENSE_START = 0
__C.VAL_FREQ = 10  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ
