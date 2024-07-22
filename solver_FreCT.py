import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model_FreCT.FreCT import FreCT
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)
warnings.filterwarnings('ignore')

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=2)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + 'Frequency_checkpoint_2.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.build_model()
        self.scale = self.d_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        
    def build_model(self):
        self.model = FreCT(win_size=self.win_size, enc_in=self.input_c, batchsize=self.batch_size, groups=self.groups, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        valid_loss = 0
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            loss_auxi = 0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.scale)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.scale)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.scale)))))
                if self.auxi_mode == "fft":
                    series_auxi = torch.fft.fft(series[u], dim=1) - torch.fft.fft(prior[u], dim=1)
                elif self.auxi_mode == "rfft":
                    if self.auxi_type == 'complex':
                        series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)
                    elif self.auxi_type == 'complex-phase':
                        series_auxi = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).angle()
                    elif self.auxi_type == 'complex-mag-phase':
                        loss_auxi_mag = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).abs()
                        loss_auxi_phase = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    elif self.auxi_type == 'phase':
                        series_auxi = torch.fft.rfft(series[u], dim=-1).angle() - torch.fft.rfft(prior[u], dim=-1).angle()
                    elif self.auxi_type == 'mag':
                        series_auxi = torch.fft.rfft(series[u], dim=-1).abs() - torch.fft.rfft(prior[u], dim=-1).abs()
                    elif self.auxi_type == 'mag-phase':
                        loss_auxi_mag = torch.fft.rfft(series[u], dim=-1).abs() - torch.fft.rfft(prior[u], dim=-1).abs()
                        loss_auxi_phase = torch.fft.rfft(series[u], dim=-1).angle() - torch.fft.rfft(prior[u], dim=-1).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    else:
                        raise NotImplementedError
                elif self.auxi_mode == "rfft-D":
                    series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)

                elif self.auxi_mode == "rfft-2D":
                    series_auxi = torch.fft.rfft2(series[u]) - torch.fft.rfft2(prior[u])
                
                else:
                    raise NotImplementedError
                if self.auxi_loss == "MAE":
                    loss_auxi += series_auxi.abs().mean() if self.module_first else series_auxi.mean().abs()  # check the dim of fft
                else:
                    loss_auxi += (series_auxi.abs()**2).mean() if self.module_first else (series_auxi**2).mean().abs()
                      
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            loss = 0.5 * (prior_loss - series_loss) + 0.5 * loss_auxi
            loss_1.append(loss.item())

        return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            train_loss = 0
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0
                loss_auxi = 0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.scale)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.scale)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)),
                                series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.scale)))))
                    # batch, len, head, d_model
                    if self.auxi_mode == "fft":
                        series_auxi = torch.fft.fft(series[u], dim=1) - torch.fft.fft(prior[u], dim=1)
                    elif self.auxi_mode == "rfft":
                        if self.auxi_type == 'complex':
                            series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)
                        elif self.auxi_type == 'complex-phase':
                            series_auxi = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).angle()
                        elif self.auxi_type == 'complex-mag-phase':
                            loss_auxi_mag = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).abs()
                            loss_auxi_phase = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).angle()
                            series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                        elif self.auxi_type == 'phase':
                            series_auxi = torch.fft.rfft(series[u], dim=-1).angle() - torch.fft.rfft(prior[u], dim=-1).angle()
                        elif self.auxi_type == 'mag':
                            series_auxi = torch.fft.rfft(series[u], dim=-1).abs() - torch.fft.rfft(prior[u], dim=-1).abs()
                        elif self.auxi_type == 'mag-phase':
                            loss_auxi_mag = torch.fft.rfft(series[u], dim=-1).abs() - torch.fft.rfft(prior[u], dim=-1).abs()
                            loss_auxi_phase = torch.fft.rfft(series[u], dim=-1).angle() - torch.fft.rfft(prior[u], dim=-1).angle()
                            series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                        else:
                            raise NotImplementedError
                    elif self.auxi_mode == "rfft-D":
                        series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)

                    elif self.auxi_mode == "rfft-2D":
                        series_auxi = torch.fft.rfft2(series[u]) - torch.fft.rfft2(prior[u])
                    
                    else:
                        raise NotImplementedError
                    # batch, len
                    if self.auxi_loss == "MAE":
                        loss_auxi += series_auxi.abs().mean() if self.module_first else series_auxi.mean().abs()  # check the dim of fft
                    else:
                        loss_auxi += (series_auxi.abs()**2).mean() if self.module_first else (series_auxi**2).mean().abs()
                
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                
                train_loss = prior_loss - series_loss
                loss = 0.5 * train_loss + 0.5 * loss_auxi
                
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            print(
                "Epoch: {0}, Cost time: {1:.3f}s | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, time.time() - epoch_time, loss, vali_loss1
                )
            )
            
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)


            
    def test(self):
        np.set_printoptions(precision=13)
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + 'Frequency_checkpoint_2.pth')))
        self.model.eval()
        temperature=50
        # (1) stastic on the train set
        #len: 527
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            loss_auxi = 0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.scale)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.scale)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)),
                        series[u].detach()) * temperature
                # batch, len, head, d_model
                if self.auxi_mode == "fft":
                        series_auxi = torch.fft.fft(series[u], dim=1) - torch.fft.fft(prior[u], dim=1)
                elif self.auxi_mode == "rfft":
                    if self.auxi_type == 'complex':
                        series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)
                    elif self.auxi_type == 'complex-phase':
                        series_auxi = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).angle()
                    elif self.auxi_type == 'complex-mag-phase':
                        loss_auxi_mag = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).abs()
                        loss_auxi_phase = (torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    elif self.auxi_type == 'phase':
                        series_auxi = torch.fft.rfft(series[u], dim=-1).angle() - torch.fft.rfft(prior[u], dim=-1).angle()
                    elif self.auxi_type == 'mag':
                        series_auxi = torch.fft.rfft(series[u], dim=-1).abs() - torch.fft.rfft(prior[u], dim=-1).abs()
                    elif self.auxi_type == 'mag-phase':
                        loss_auxi_mag = torch.fft.rfft(series[u], dim=-1).abs() - torch.fft.rfft(prior[u], dim=-1).abs()
                        loss_auxi_phase = torch.fft.rfft(series[u], dim=-1).angle() - torch.fft.rfft(prior[u], dim=-1).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    else:
                        raise NotImplementedError
                elif self.auxi_mode == "rfft-D":
                    series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)

                elif self.auxi_mode == "rfft-2D":
                    # [256, 60, 1, 129]
                    series_auxi = torch.fft.rfft2(series[u]) - torch.fft.rfft2(prior[u]) 
                else:
                    raise NotImplementedError
                if self.auxi_loss == "MAE":
                    loss_auxi += series_auxi.abs().mean(dim=3).squeeze(dim=2) if self.module_first else series_auxi.mean(dim=3).squeeze(dim=2).abs()  # check the dim of fft
                else:
                    loss_auxi += (series_auxi.abs()**2).mean(dim=3).squeeze(dim=2) if self.module_first else (series_auxi**2).mean(dim=3).squeeze(dim=2).abs()
                
                # visualization
            
            # 256, 105
            loss = 0.5 * (-series_loss - prior_loss) + 0.5 * loss_auxi
            metric = torch.softmax(loss, dim=-1)
            
            cri = metric.detach().cpu().numpy()
            
            attens_energy.append(cri)
        #909*(64, 90)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
 
        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            loss_auxi = 0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.scale)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.scale)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)),
                        series[u].detach()) * temperature
                if self.auxi_mode == "fft":
                        series_auxi = torch.fft.fft(series[u], dim=1) - torch.fft.fft(prior[u], dim=1)
                elif self.auxi_mode == "rfft":
                    if self.auxi_type == 'complex':
                        series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)
                    elif self.auxi_type == 'complex-phase':
                        series_auxi = (torch.fft.rfft(series[u], dim=1) - torch.fft.rfft(prior[u], dim=1)).angle()
                    elif self.auxi_type == 'complex-mag-phase':
                        loss_auxi_mag = (torch.fft.rfft(series[u], dim=1) - torch.fft.rfft(prior[u], dim=1)).abs()
                        loss_auxi_phase = (torch.fft.rfft(series[u], dim=1) - torch.fft.rfft(prior[u], dim=1)).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    elif self.auxi_type == 'phase':
                        series_auxi = torch.fft.rfft(series[u], dim=1).angle() - torch.fft.rfft(prior[u], dim=1).angle()
                    elif self.auxi_type == 'mag':
                        series_auxi = torch.fft.rfft(series[u], dim=1).abs() - torch.fft.rfft(prior[u], dim=1).abs()
                    elif self.auxi_type == 'mag-phase':
                        loss_auxi_mag = torch.fft.rfft(series[u], dim=1).abs() - torch.fft.rfft(prior[u], dim=1).abs()
                        loss_auxi_phase = torch.fft.rfft(series[u], dim=1).angle() - torch.fft.rfft(prior[u], dim=1).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    else:
                        raise NotImplementedError
                elif self.auxi_mode == "rfft-D":
                    series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)

                elif self.auxi_mode == "rfft-2D":
                    series_auxi = torch.fft.rfft2(series[u]) - torch.fft.rfft2(prior[u])
                    
                else:
                    raise NotImplementedError
                if self.auxi_loss == "MAE":
                    loss_auxi += series_auxi.abs().mean(dim=3).squeeze(dim=2) if self.module_first else series_auxi.mean(dim=3).squeeze(dim=2).abs()  # check the dim of fft
                else:
                    loss_auxi += (series_auxi.abs()**2).mean(dim=3).squeeze(dim=2) if self.module_first else (series_auxi**2).mean(dim=3).squeeze(dim=2).abs()
                
            loss = 0.5 * (-series_loss - prior_loss) + 0.5 * loss_auxi
            metric = torch.softmax(loss, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        # attens_energy :64, 90
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            # 6*[64, 90, 1, 256]
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            loss_auxi = 0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.scale)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.scale)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.scale)),
                        series[u].detach()) * temperature
                if self.auxi_mode == "fft":
                        series_auxi = torch.fft.fft(series[u], dim=1) - torch.fft.fft(prior[u], dim=1)
                elif self.auxi_mode == "rfft":
                    if self.auxi_type == 'complex':
                        series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)
                    elif self.auxi_type == 'complex-phase':
                        series_auxi = (torch.fft.rfft(series[u], dim=1) - torch.fft.rfft(prior[u], dim=1)).angle()
                    elif self.auxi_type == 'complex-mag-phase':
                        loss_auxi_mag = (torch.fft.rfft(series[u], dim=1) - torch.fft.rfft(prior[u], dim=1)).abs()
                        loss_auxi_phase = (torch.fft.rfft(series[u], dim=1) - torch.fft.rfft(prior[u], dim=1)).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    elif self.auxi_type == 'phase':
                        series_auxi = torch.fft.rfft(series[u], dim=1).angle() - torch.fft.rfft(prior[u], dim=1).angle()
                    elif self.auxi_type == 'mag':
                        series_auxi = torch.fft.rfft(series[u], dim=1).abs() - torch.fft.rfft(prior[u], dim=1).abs()
                    elif self.auxi_type == 'mag-phase':
                        loss_auxi_mag = torch.fft.rfft(series[u], dim=1).abs() - torch.fft.rfft(prior[u], dim=1).abs()
                        loss_auxi_phase = torch.fft.rfft(series[u], dim=1).angle() - torch.fft.rfft(prior[u], dim=1).angle()
                        series_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                    else:
                        raise NotImplementedError
                elif self.auxi_mode == "rfft-D":
                    series_auxi = torch.fft.rfft(series[u], dim=-1) - torch.fft.rfft(prior[u], dim=-1)

                elif self.auxi_mode == "rfft-2D":
                    series_auxi = torch.fft.rfft2(series[u]) - torch.fft.rfft2(prior[u])
                

                else:
                    raise NotImplementedError
                if self.auxi_loss == "MAE":
                    loss_auxi += series_auxi.abs().mean(dim=3).squeeze(dim=2) if self.module_first else series_auxi.mean(dim=3).squeeze(dim=2).abs()  # check the dim of fft
                else:
                    loss_auxi += (series_auxi.abs()**2).mean(dim=3).squeeze(dim=2) if self.module_first else (series_auxi**2).mean(dim=3).squeeze(dim=2).abs()
                
            loss = 0.5 * (-series_loss - prior_loss) + 0.5 * loss_auxi
            metric = torch.softmax(loss, dim=-1)
            # batch, length
            cri = metric.detach().cpu().numpy()
            
            attens_energy.append(cri)
            test_labels.append(labels)
              
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        # 69120
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        
        
        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score
    