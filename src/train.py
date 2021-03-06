import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from config import cfg
from utils import *
from tqdm import tqdm

optimizers = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
}


class Trainer:
    def __init__(self, dataloader, cfg_data, net_fun):
        """
        Initialize the training object with the given parameters and the parameters in the config.py file

        @param dataloader: DataLoader object that iterates the dataset
        @param cfg_data: config data EasyDict object
        @param net_fun: functions the called without parameters, returns the model
        """

        self.cfg_data = cfg_data

        self.exp_name = build_exp_name(cfg)
        self.exp_path = cfg.EXP_PATH

        self.net = net_fun().to(cfg.DEVICE)

        self.optimizer = optimizers[cfg.OPTIM[0]](self.net.parameters(), **cfg.OPTIM[1])
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.epoch = 0
        self.val_loss = np.nan

        if cfg.RESUME:
            checkpoint = torch.load(cfg.RESUME)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.val_loss = checkpoint['val_loss']

        self.train_record = {'best_val_mae': 1e20, 'best_val_rmse': 1e20, 'best_val_loss': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer(), 'inference time': Timer()}
        self.logger = TrainLogger(self.exp_path, self.exp_name, cfg)

        self.i_tb = 0

        self.train_loader, self.val_loader = dataloader()

        with torch.no_grad():
            self.logger.writer.add_graph(self.net, self.train_loader.dataset.__getitem__(0)[0].unsqueeze(0).cuda())

    def train(self):
        """
        Train the model on the dataset using the parameters of the config file.
        """
        print("Experiment: " + self.exp_name)
        early_stop = EarlyStopping(patience=cfg.PATIENCE, delta=cfg.EARLY_STOP_DELTA)
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

            # training
            self.timer['train time'].tic()
            self.forward_dataset()
            self.timer['train time'].toc(average=False)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.validate()

            if early_stop(self.val_loss):
                print('Early stopped! At epoch ' + str(self.epoch))
                break

    def forward_dataset(self):
        """
        Makes a training epoch forwarding the whole dataset. Prints live results using tqdm
        """
        self.net.train()
        out_loss = 0
        time = 0
        norm_gt_count = 0
        norm_pred_count = 0

        tk_train = tqdm(
            enumerate(self.train_loader, 0), total=len(self.train_loader), leave=False,
            bar_format='{l_bar}{bar:32}{r_bar}',
            colour='#ff0de7', desc='Train Epoch %d/%d' % (self.epoch, cfg.MAX_EPOCH)
        )
        postfix = {'loss': out_loss, 'lr': self.optimizer.param_groups[0]['lr'],
                   'time': time, 'gt count': norm_gt_count, 'pred count': norm_pred_count}
        tk_train.set_postfix(postfix, refresh=True)

        for i, data in tk_train:
            self.timer['iter time'].tic()
            img, gt = data
            img = img.to(cfg.DEVICE)
            gt = gt.to(cfg.DEVICE)
            self.optimizer.zero_grad()
            pred_den = self.net.predict(img)
            loss = self.net.build_loss(pred_den, gt)
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.logger.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                out_loss = loss.item()
                time = self.timer['iter time'].diff
                norm_gt_count = torch.mean(torch.sum(gt, dim=(1, 2))).data / self.cfg_data.LOG_PARA
                norm_pred_count = torch.mean(torch.sum(pred_den, dim=(1, 2, 3))).data / self.cfg_data.LOG_PARA
                postfix = {'loss': out_loss, 'lr': self.optimizer.param_groups[0]['lr'], 'time': time,
                           'gt count': norm_gt_count.item(), 'pred count': norm_pred_count.item()}
                tk_train.set_postfix(postfix, refresh=True)

    def validate(self):
        """
        Makes a validation step.
        Validates the model on the validation set, measures the metrics printing it
        and eventually save a checkpoint of the model

        """
        self.timer['val time'].tic()

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        tk_valid = tqdm(
            enumerate(self.val_loader, 0), total=len(self.val_loader),
            leave=False, bar_format='{l_bar}{bar:32}{r_bar}', desc='Validating'
        )

        for vi, data in tk_valid:
            img, gt = data

            with torch.no_grad():
                img = img.to(cfg.DEVICE)
                gt = gt.to(cfg.DEVICE)

                self.timer['inference time'].tic()
                pred_map = self.net.predict(img)
                self.timer['inference time'].toc(average=True)
                self.net.build_loss(pred_map, gt)

                pred_map = pred_map.squeeze(1).data.cpu().numpy()
                gt = gt.data.cpu().numpy()

                pred_cnt = np.sum(pred_map, axis=(1, 2)) / self.cfg_data.LOG_PARA
                gt_count = np.sum(gt, axis=(1, 2)) / self.cfg_data.LOG_PARA

                losses.update(self.net.loss.item())
                maes.update(abs(gt_count - pred_cnt))
                mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        self.timer['val time'].toc(average=False)

        mae = maes.avg
        rmse = np.sqrt(mses.avg)
        self.val_loss = losses.avg
        scores = {'val_mae': mae, 'val_rmse': rmse}

        self.train_record = self.logger.update_model({'model_state_dict': self.net.state_dict(),
                                                      'optimizer_state_dict': self.optimizer.state_dict(),
                                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                                      'epoch': self.epoch,
                                                      'val_loss': self.val_loss
                                                      },
                                                     self.epoch, self.exp_path, self.exp_name,
                                                     scores,
                                                     self.train_record)

        self.logger.summary(self.epoch,
                            [mae, rmse, self.val_loss],
                            self.timer)
