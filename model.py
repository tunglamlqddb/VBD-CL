# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import nn
import numpy as np
from layers_VBD import *
from copy import deepcopy

        
class MLPVBD(nn.Module):
    def __init__(self, taskcla, num_classes=10, nf=400, input_size=[1,28,28], threshes=[], split=False):
        super(MLPVBD, self).__init__()

        self.threshes = []
        self.num_classes = num_classes
        self.split = split
        self.taskcla = taskcla
        
        self.input_size = np.prod(input_size)
        self.drop1 = DropoutLayer(self.input_size, self.threshes)
        self.fc1 = nn.Linear(self.input_size, nf)
        self.drop2 = DropoutLayer(nf, self.threshes)
        self.fc2 = nn.Linear(nf, nf)
        self.drop3 = DropoutLayer(nf, self.threshes)
        if self.split:
            self.last = nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(nn.Linear(nf,n))
        else:
            self.fc3 = nn.Linear(nf,num_classes)
        
        self.last_fc_biases = {}

        self.relu = nn.ReLU()

        self.normal_drop1 = nn.Dropout(0.2)
        self.normal_drop2 = nn.Dropout(0.5)

        nn.init.uniform_(self.fc1.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc2.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        

    def update_mask(self, task_id):    
        for module in self.children():
            if hasattr(module, 'mu'):
                module.update_mask(task_id)
        
    def new_task(self, thresh):
        self.threshes.append(thresh)
        for module in self.children():
            if hasattr(module, 'mu'):
                module.new_task(thresh)
        
    def forward(self, x, task_id, multi_sample=False):
        x = x.view(x.size(0), -1)
        # x = self.normal_drop2(x)
        x = self.drop1(x, task_id, multi_sample)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.normal_drop2(x)

        x = self.drop2(x, task_id, multi_sample)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.normal_drop2(x)

        x = self.drop3(x, task_id, multi_sample)
        
        if self.split:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](x))
        else:
            y = self.fc3(x)

        return y
    
    def kl_reg(self,task_id):
        kld = 0
        kld += self.drop1.kl_reg(task_id)
        kld += self.drop2.kl_reg(task_id)
        kld += self.drop3.kl_reg(task_id)

        return kld

    def l1(self, task_id):
        l1_reg = 0.
        for name, module in self.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l1_reg += module.weight.abs().sum()
                l1_reg += module.bias.abs().sum()
            elif isinstance(module, nn.ModuleList):
                for n, p in module.named_children():
                    if str(task_id) in n:
                        l1_reg += p.weight.abs().sum()
                        l1_reg += p.bias.sum()
                        break
        return l1_reg


    def l2(self, task_id):
        l2_reg = 0.
        for name, module in self.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l2_reg += module.weight.pow(2.0).sum()
                l2_reg += module.bias.pow(2.0).sum()
            elif isinstance(module, nn.ModuleList):
                for n, p in module.named_children():
                    if str(task_id) in n:
                        l2_reg += p.weight.pow(2.0).sum()
                        l2_reg += p.bias.pow(2.0).sum()
        return l2_reg

    def group_Lasso(self):
        lasso_out = 0.
        lasso_in = 0.
        for name, module in self.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                lasso_in += module.weight.pow(2.0).sum(dim=0).add(1e-8).pow(1/2.).sum()
                # bias?
                lasso_out += module.weight.pow(2.0).sum(dim=1).add(1e-8).pow(1/2.).sum()
        return lasso_out + lasso_in

    def get_model(self):
        return deepcopy(self.state_dict())
    
    def set_model(self, new_state_dict):
        self.load_state_dict(deepcopy(new_state_dict))
        return
        
    def update_best_log_alphas(self, task_id):
        self.best_log_alphas = {}
        
        for name, module in self.named_children():
            if hasattr(module, 'mu'):
                module.update_best_log_alpha(task_id)
                self.best_log_alphas[name] = module.best_log_alpha

    def update_mask_grad(self, task_id, training_thresh=None, fix=True, drop_rate=0.5, s=25):
        
        def gate(log_alpha):
            alpha = torch.exp(log_alpha)
            p = alpha / (alpha + 1)
            return 1 / (1 + torch.exp(-s*(p-drop_rate)))
        
        if training_thresh == None:
          thr = self.threshes[task_id]
        else:
          thr = training_thresh

        self.mask_weight = {}    # pre-calculated mask of each DNN weight (Linear)
        self.mask_bias = {}
        
        # hard code
        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                if name=='fc1':   # first Linear layer
                    mask_w_prev  = self.best_log_alphas['drop1'].view(1,-1).expand_as(self.fc1.weight)
                    mask_w_aft   = self.best_log_alphas['drop2'].view(-1,1).expand_as(self.fc1.weight)
                    mask_b_layer = self.best_log_alphas['drop2'].view(-1)
                    if fix==True:
                        self.mask_weight['fc1'] = ~((mask_w_prev < thr) * (mask_w_aft < thr))
                        self.mask_bias['fc1']   = ~(mask_b_layer < thr)
                    else:
                        self.mask_weight['fc1'] = ((mask_w_prev<thr)*(mask_w_aft<thr)) * torch.min(gate(mask_w_prev), gate(mask_w_aft)) + ~((mask_w_prev < thr)*(mask_w_aft<thr))
                        self.mask_bias['fc1']   = (mask_b_layer < thr) * gate(mask_b_layer) + ~(mask_b_layer < thr)
                        
                elif name=='fc3' and self.split==False:   # last Linear layer
                    mask_w_layer = self.best_log_alphas['drop3'].view(1,-1).expand_as(self.fc3.weight)
                    # mask_b_layer = torch.ones_like(self.fc3.bias)
                    if fix==True:
                        self.mask_weight['fc3'] = ~(mask_w_layer < thr)
                        # self.mask_bias['fc3']   = mask_b_layer
                    else:
                        self.mask_weight['fc3'] = (mask_w_layer < thr) * gate(mask_w_layer) + ~(mask_w_layer < thr) 
                        # self.mask_bias['fc3'] = mask_b_layer
                        
                else:               # intermediate layers ==> fc2
                    mask_w_prev  = self.best_log_alphas['drop2'].view(1,-1).expand_as(self.fc2.weight)
                    mask_w_aft   = self.best_log_alphas['drop3'].view(-1,1).expand_as(self.fc2.weight)
                    mask_b_layer = self.best_log_alphas['drop3'].view(-1)
                    if fix==True:
                        self.mask_weight['fc2'] = ~((mask_w_prev < thr) * (mask_w_aft < thr))
                        self.mask_bias['fc2'] = ~(mask_b_layer < thr)
                    else:
                        self.mask_weight['fc2'] = ((mask_w_prev<thr)*(mask_w_aft<thr)) * torch.min(gate(mask_w_prev), gate(mask_w_aft)) + ~((mask_w_prev<thr)*(mask_w_aft<thr))
                        self.mask_bias['fc2']   = (mask_b_layer<thr) * gate(mask_b_layer) + ~(mask_b_layer<thr)

        # store bias of Last Linear layer fc3
        if not self.split:
            self.last_fc_biases[task_id] = self.fc3.bias.clone().detach().cpu().numpy()

    def update_grad(self):
        # update grad of Linear layers
        for name,module in self.named_children():
            if isinstance(module, nn.Linear):   # current layer is Linear
                if name != 'fc3':
                    module.weight.grad.data *= self.mask_weight[name].cuda()
                    module.bias.grad.data *= self.mask_bias[name].cuda()
                else:
                    module.weight.grad.data *= self.mask_weight[name].cuda()
                
