import numpy as np
import os
import time
import torch
import json

from tensorboardX import SummaryWriter


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given number of consecutive epochs"""

    def __init__(self, patience=1, delta=1e-4):
        """
        Instantiate an EarlyStopping object.

        :param patience: The number of consecutive epochs to wait.
        :param delta: The minimum change of the monitored quantity.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.should_stop = False
        self.counter = 0
        self.best_state = None

    def should_stop(self):
        return self.should_stop

    def __call__(self, loss):
        """
        Call the object.

        :param loss: The validation loss measured.
        """
        # Check if an improved of the loss happened
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        # Check if the training should stop
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


class TrainLogger:
    def __init__(self, exp_path, exp_name, cfg):
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        self.writer = SummaryWriter(exp_path + '/' + exp_name)
        self.log = exp_path + '/' + exp_name + '/' + exp_name + '.txt'

        with open(self.log, 'a') as f:
            f.write(''.join(json.dumps(cfg, indent=4, sort_keys=True)) + '\n\n\n\n')

        self.best = False

    def update_model(self, state_dict, epoch, exp_path, exp_name, scores, train_record):
        self.best = False

        self.writer.add_scalar('val_loss', state_dict['val_loss'], epoch)
        for key in scores:
            self.writer.add_scalar(key, scores[key], epoch)

        snapshot_name = 'ep_%d' % epoch
        for key in scores:
            snapshot_name += '_' + key + '_%.1f' % scores[key]

        if state_dict['val_loss'] < train_record['best_val_loss']:
            train_record['best_model_name'] = snapshot_name
            train_record['best_val_loss'] = state_dict['val_loss']
            torch.save(state_dict, os.path.join(exp_path, exp_name, snapshot_name + '.pth'))

        for key in scores:
            if scores[key] < train_record['best_' + key]:
                train_record['best_' + key] = scores[key]

        return train_record

    def summary(self, epoch, scores, timers):
        mae, mse, loss = scores
        out = ('Epoch ' + str(epoch) + ' | ')
        out += ('    [mae %.2f mse %.2f], [val loss %.4f] [forward time %.2f] [train/valid time %.2f / %.2f] --- '
                % (mae, mse, loss,
                   timers['inference time'].average_time * 1000,
                   timers['train time'].diff,
                   timers['val time'].diff))
        if self.best:
            out += "[BEST]"

        print(out)
        with open(self.log, 'a') as f:
            f.write(out + '\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        if hasattr(cur_val, '__iter__'):
            for val in cur_val:
                self._update(val)
        else:
            self._update(cur_val)

    def _update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, cur_val, class_id):
        self.cur_val[class_id] = cur_val
        self.sum[class_id] += cur_val
        self.count[class_id] += 1
        self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def build_exp_name(cfg):
    netname = cfg.NET.PREDICTOR + '_'
    netname += cfg.NET.ENCODER + '_'
    if cfg.NET.ENCODER == 'MobileCount':
        netname += cfg.NET.VERSION

    if cfg.NET.ENCODER == 'MobileCount':
        netname += ('_freeze_' if cfg.NET.PRETRAINED else '')

    if cfg.NET.ENCODER_TIR:
        netname += cfg.NET.ENCODER_TIR + '_'
        if cfg.NET.ENCODER_TIR == 'LWEncoder':
            netname += cfg.NET.VERSION_TIR + '_'

        if cfg.NET.ENCODER_TIR == 'LWEncoder':
            netname += ('_freeze_' if cfg.NET.PRETRAINED_TIR else '')

    netname += cfg.NET.DECODER

    now = time.strftime("%m-%d_%H-%M", time.localtime())

    return now \
           + '_' + cfg.DATASET \
           + '_' + netname \
           + '_' + str(cfg.LR) \
           + '_' + cfg.DETAILS
