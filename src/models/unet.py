import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
from time import time

def conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def upsample_layer(size):
    return nn.Upsample(size)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        num_channels = [64, 128, 256, 512, 1024]
        
        # encoder
        self.enc_conv = nn.ModuleList([
            conv_layer(3 if i == 0 else num_channels[i-1], num_channels[i])
            for i in range(len(num_channels))
        ])
        
        self.pool = nn.ModuleList([
            nn.MaxPool2d(2, 2) for _ in range(len(num_channels) - 1)
        ])
        
        # bottleneck
        self.bottleneck_conv = conv_layer(num_channels[-1], num_channels[-1] * 2)
        
        # decoder
        self.dec_conv = nn.ModuleList([
            conv_layer(num_channels[i] * 3 if i > 0 else num_channels[i] * 2, num_channels[i])
            for i in range(len(num_channels) - 1)
        ])
        
        self.upsample = nn.ModuleList([
            upsample_layer(size * 2)
            for size in reversed([16, 32, 64, 128])
        ])
        
        self.final_conv = nn.Conv2d(num_channels[0], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # encoder
        enc_out = [self.enc_conv[0](x)]
        for i in range(1, len(self.enc_conv)):
            enc_out.append(self.enc_conv[i](self.pool[i-1](enc_out[i-1])))
        
        # bottleneck
        bn_out = self.bottleneck_conv(self.pool[-1](enc_out[-1]))
        
        # decoder
        dec_out = [self.dec_conv[0](torch.cat((self.upsample[0](bn_out), enc_out[-1]), dim=1))]
        for i in range(1, len(self.dec_conv)):
            dec_out.append(self.dec_conv[i](
                torch.cat((self.upsample[i](dec_out[i-1]), enc_out[-i-1]), dim=1)
            ))
        
        out = self.final_conv(dec_out[-1])
        
        return out

class UnetModel:
    def __init__(self):
        self.estimator = None
        self.optimizer = None
        self.scheduler = None
        self.train_data = None
        self.val_data = None
        self.losses_history = []
        
    def __set_optimizer(self):
        self.optimizer = optim.Adam(self.estimator.parameters(), lr=0.001)
    
    def __set_scheduler(self):
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        
    def train(self, data: str, imgsz: int, epochs: int, batch: int) -> dict:
        """Train method"""
        self.__load_data(data)
        train_losses = []
        valid_losses = []
        metric_scores = []
        best_metric_score = 0
        
        self.__set_optimizer()
        self.__set_scheduler()
        
        for epoch in range(epochs):
            print('* Epoch %d/%d' % (epoch + 1, epochs))
            self.estimator.train()
            running_train_losses = []
            
            for x_batch, y_batch in self.train_data:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    y_pred = self.estimator(x_batch)
                    loss = self.loss(y_pred, y_batch)
                    running_train_losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                
            epoch_train_loss = np.mean(running_train_losses)
            train_losses.append(epoch_train_loss)
            
            # test phase
            self.estimator.eval()
            running_val_losses = []
            for x_val, y_val in self.val_data:
                with torch.no_grad():
                    y_hat = self.estimator(x_val.to(self.device)).detach().cpu()
                    loss = self.loss(y_hat, y_val)
                    running_val_losses.append(loss.item())
                    
            epoch_val_loss = np.mean(running_val_losses)
            valid_losses.append(epoch_val_loss)
            
            epoch_metric_score = self.__score_model_by_metric(self.estimator, self.__iou, self.train_data)
            metric_scores.append(epoch_metric_score)
            
            if epoch_metric_score > best_metric_score:
                best_metric_score = epoch_metric_score
                
            if self.scheduler:
                self.scheduler.step(epoch_metric_score)
                
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'metric_scores': metric_scores,
            'best_metric_score': best_metric_score
        }


net = UNet()
net.train()