# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import time

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import Parameter, ParameterList
from torchvision.models import vgg16_bn, vgg16
import torchvision

class DropoutLayer(nn.Module):
    def __init__(self, channels, threshes):
        super(DropoutLayer, self).__init__()
        self.channels = channels
        # self.mu = Parameter(torch.ones(self.channels))
        self.mu = ParameterList([])
        # self.log_sigma2 = Parameter(-9 * torch.ones(self.channels))
        self.log_sigma2 = ParameterList([])
        self.noise = torch.distributions.Normal(0,1)
        self.threshes = []   

        self.best_log_alpha = None
        self.dict_mus = {}
        
        self.masks = []
        
    def new_task(self, thresh):
        self.mu.append(Parameter(torch.ones(self.channels).cuda()))
        self.log_sigma2.append(Parameter(-1 * torch.ones(self.channels).cuda()))
        # self.log_sigma2.append(Parameter(((-2-2)*torch.rand(self.channels)+2).cuda()))
        # self.masks.append(torch.ones(self.channels).cuda())
        self.masks.append(Parameter(torch.ones(self.channels).cuda(), requires_grad=False))
        self.threshes.append(thresh)
    
    def log_alpha(self, task_id):
        return self.log_sigma2[task_id] - 2.0*torch.log(torch.abs(self.mu[task_id]) + 1e-8) # mu maybe negative
    
    def forward(self, x, task_id, multi_sample=False):
        if self.training:
            sigma = torch.exp(0.5*self.log_sigma2[task_id])
            epsilon = self.noise.sample(self.mu[task_id].size()).cuda()
            noise = self.mu[task_id] + sigma*epsilon
            noise = noise * self.masks[task_id]
            if (x.size().__len__() == 4):   # CNN
                noise = noise.unsqueeze(1)
                noise = noise.unsqueeze(2)
            return noise * x
        else:
            if multi_sample==False:
                self.update_mask(task_id)
                noise = self.mu[task_id] * self.masks[task_id]
                # noise = self.mu[task_id]
                if (x.size().__len__() == 4):   # CNN
                    noise = noise.unsqueeze(1)
                    noise = noise.unsqueeze(2)
                return noise * x
            else:   # sample multiple times to calculate entropy
                sigma = torch.exp(0.5*self.log_sigma2[task_id])
                epsilon = self.noise.sample(self.mu[task_id].size()).cuda()
                noise = (self.mu[task_id] + sigma * epsilon) * self.masks[task_id]
                if (x.size().__len__() == 4):   # CNN
                    noise = noise.unsqueeze(1)
                    noise = noise.unsqueeze(2)
                
                return noise * x

    def update_mask(self, task_id):  # alpha big ==> not important!
        # mask = (self.log_alpha(task_id) < self.threshes[task_id]).float().clone().cuda()
        # self.masks[task_id] = (self.log_alpha(task_id) < self.threshes[task_id]).float().cuda()
        self.masks[task_id].data = (self.log_alpha(task_id) < self.threshes[task_id]).float().cuda()

        
    def kl_reg(self, task_id):
        kld = 1/2 * torch.log1p(self.mu[task_id] * self.mu[task_id] / (torch.exp(self.log_sigma2[task_id]) + 1e-8))
        return kld.sum()

    
    def sparsity(self, task_id):
        non_zero_num = self.masks[task_id].sum().item()
        return non_zero_num, self.log_alpha(task_id).min().item(), self.log_alpha(task_id).max().item()

    def update_best_log_alpha(self, task_id):
        if self.best_log_alpha == None:
            self.best_log_alpha = self.log_alpha(task_id)
        else:
            self.best_log_alpha = torch.min(self.best_log_alpha, self.log_alpha(task_id))
