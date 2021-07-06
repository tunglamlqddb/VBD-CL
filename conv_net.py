import torch, math
import torch.nn as nn
import numpy as np
from layers_VBD import DropoutLayer
from copy import deepcopy


def compute_conv_output_size(size_in, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor(size_in + 2*padding - dilation*(kernel_size-1)-1 / float(stride) + 1))

class ConvNet(nn.Module):
    def __init__(self, taskcla, input_size=[3,32,32], threshes=[]):
        super().__init__()
        self.taskcla = taskcla
        self.threshes = []
        ncha, size = input_size[0], input_size[1]

        # self.drop_conv1 = DropoutLayer(ncha, self.threshes)
        self.c1 = nn.Conv2d(ncha, 64, kernel_size=3, padding=1)
        s = compute_conv_output_size(size, 3, padding=1)   # 32
        s = s//2   # 16
        self.drop_conv2 = DropoutLayer(64, self.threshes)
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)      # 16
        s = s//2   # 8
        self.drop_conv3 = DropoutLayer(128, self.threshes)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)      # 8
        s = s//2   # 4
        self.drop_conv4 = DropoutLayer(256, self.threshes)
        self.last_conv_size = s
        self.drop1 = DropoutLayer(s*s*256, self.threshes)
        self.fc1 = nn.Linear(s*s*256, 2048)
        self.drop2 = DropoutLayer(2048, self.threshes)
        self.fc2 = nn.Linear(2048, 2048)
    
        self.drop3 = DropoutLayer(2048, self.threshes)
        self.last = nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(nn.Linear(2048, n))
    
        self.MaxPool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.split = True

        self.normal_drop1 = nn.Dropout(0.2)
        self.normal_drop2 = nn.Dropout(0.5)

        # nn.init.kaiming_uniform_(self.c1.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.c2.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.c3.weight, a=math.sqrt(5))


    def forward(self, x, task_id, multi_sample=False):
        bs = x.shape[0]
        # x = self.drop_conv1(x, task_id,multi_sample)
        x = self.c1(x)
        
        x = self.relu(x)
        x = self.drop_conv2(x, task_id, multi_sample)
        x = self.MaxPool(x)
        # x = self.normal_drop1(x)
        
        x = self.c2(x)  
        
        x = self.relu(x)
        x = self.drop_conv3(x, task_id, multi_sample)
        x = self.MaxPool(x)
        # x = self.normal_drop1(x)

        x = self.c3(x)
        
        x = self.relu(x)
        x = self.drop_conv4(x, task_id, multi_sample)    # todo: check position
        x = self.MaxPool(x)
        # x = self.normal_drop1(x)
        
        x = x.view(bs,-1)
        x = self.drop1(x, task_id, multi_sample)
        x = self.relu(self.fc1(x))
        # x = self.normal_drop2(x)
        x = self.drop2(x, task_id, multi_sample)
        x = self.relu(self.fc2(x))
        # x = self.normal_drop2(x)
        x = self.drop3(x, task_id, multi_sample)
        y = []
        for t,n in self.taskcla:
            y.append(self.last[t](x))
        
        return y
    
    def update_mask(self, task_id):
        for module in self.children():
            if hasattr(module, 'mu'):
                module.update_mask(task_id)
    
    def new_task(self, thresh):
        self.threshes.append(thresh)
        for module in self.children():
            if hasattr(module, 'mu'):
                module.new_task(thresh)
    
    def kl_reg(self, task_id):
        kld = 0
        for module in self.children():
            if hasattr(module, 'kl_reg'):
                kld += module.kl_reg(task_id)
        return kld
    
    def l1(self, task_id):
        l1_reg = 0.
        # for name, param in self.named_parameters():
        #     if ('weight' in name or 'bias' in name) and 'last' not in name:
        #         l1_reg += param.abs().sum()
        #     if 'last' in name:
        #         l1_reg += 
        
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
        # for name, param in self.named_parameters():
        #     if 'weight' in name or 'bias' in name:
        #         l2_regularization += param.pow(2.0).sum()
        # return l2_regularization

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

    def update_mask_grad(self, task_id, training_thresh=None, fix=True, drop_rate=0.5, s=50):
        
        def gate(log_alpha, name):
            alpha = torch.exp(log_alpha)
            p = alpha / (alpha + 1)
            # if name == 'c1':
            #     c1_log_alpha = -1.0
            #     c1_rate = math.exp(c1_log_alpha) / (math.exp(c1_log_alpha) + 1)
            #     return 1 / (1 + torch.exp(-s*(p-c1_rate)))
            # else:    
            return 1 / (1 + torch.exp(-s*(p-drop_rate)))

        if training_thresh == None:
            thr = self.threshes[task_id]
        else:
            thr = training_thresh
        
        self.mask_weight = {}
        self.mask_bias = {}

        for name,module in self.named_children():
            if isinstance(module, nn.Conv2d):
                if name == 'c1':
                    mask_w_aft = self.best_log_alphas['drop_conv2'].view(-1,1,1,1).expand_as(module.weight)
                    mask_b_aft = self.best_log_alphas['drop_conv2'].view(-1)
                    # self.mask_weight['c1'] = (mask_w_aft < thr)*torch.exp(torch.abs(1/mask_w_aft)) + ~(mask_w_aft < thr)
                    # self.mask_bias['c1'] =  (mask_b_aft < thr)*torch.exp(torch.abs(1/mask_b_aft)) + ~(mask_b_aft < thr)
                    if fix:
                        self.mask_weight['c1'] = ~(mask_w_aft < thr)
                        self.mask_bias['c1']   =  ~(mask_b_aft < thr)
                    else:
                        self.mask_weight['c1'] = (mask_w_aft < thr) * gate(mask_w_aft, name) + ~(mask_w_aft < thr)
                        self.mask_bias['c1']   = (mask_b_aft < thr) * gate(mask_b_aft, name) + ~(mask_b_aft < thr)
                if name == 'c2':
                    mask_w_aft = self.best_log_alphas['drop_conv3'].view(-1,1,1,1).expand_as(module.weight)
                    mask_w_pre = self.best_log_alphas['drop_conv2'].view(1,-1,1,1).expand_as(module.weight)
                    mask_b_aft = self.best_log_alphas['drop_conv3'].view(-1)
                    if fix:
                        self.mask_weight['c2'] = ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        # self.mask_weight['c2'] = ~(mask_w_aft < thr)
                        self.mask_bias['c2']   = ~(mask_b_aft < thr)
                    else:
                        self.mask_weight['c2'] = ((mask_w_aft < thr) * (mask_w_pre < thr)) * torch.min(gate(mask_w_aft,name),gate(mask_w_pre,name)) + ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        self.mask_bias['c2']   = (mask_b_aft < thr) * gate(mask_b_aft,name) + ~(mask_b_aft < thr)
                if name == 'c3':
                    # drop_conv3 & drop1
                    mask_w_pre = self.best_log_alphas['drop_conv3'].view(1,-1,1,1).expand_as(module.weight)
                    # mask_aft = self.best_log_alphas['drop1'].view(module.weight.shape[0], self.last_conv_size, self.last_conv_size)
                    # mask_aft = torch.amin(mask_aft, dim=(1,2))
                    # mask_w_aft = mask_aft.view(-1,1,1,1)
                    # mask_w_aft = mask_w_aft.expand_as(module.weight)
                    # self.mask_weight['c3'] = ~((mask_w_aft < thr) * (mask_w_pre < thr))
                    # mask_b_aft = mask_aft.view(-1)
                    # self.mask_bias['c3'] = ~(mask_b_aft < thr)
                    
                    # drop_conv4 & drop1 & no drop_conv3
                    mask_conv_aft = self.best_log_alphas['drop_conv4']   # 256
                    mask_fc_aft = self.best_log_alphas['drop1'].view(module.weight.shape[0], self.last_conv_size, self.last_conv_size) # 4096=256*4*4
                    mask_fc_aft = torch.amin(mask_fc_aft, dim=(1,2)) # 256
                    mask_aft = torch.max(mask_conv_aft, mask_fc_aft)  #256
                    # mask_aft = mask_conv_aft
                    mask_w_aft = mask_aft.view(-1,1,1,1).expand_as(module.weight)
                    mask_b_aft = mask_aft.view(-1)   # 256
                    if fix:
                        self.mask_weight['c3'] = ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        self.mask_bias['c3']   = ~(mask_b_aft < thr)
                    else: 
                        self.mask_weight['c3'] = ((mask_w_aft < thr) * (mask_w_pre < thr)) * torch.min(gate(mask_w_aft,name), gate(mask_w_pre,name)) + ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        self.mask_bias['c3']   = (mask_b_aft < thr) * gate(mask_b_aft,name) + ~(mask_b_aft < thr)

            if isinstance(module, nn.Linear):
                if name == 'fc1':
                    num_chan_pre_conv = self.c3.weight.shape[0]   # 256
                    mask_conv_pre = self.best_log_alphas['drop_conv4'].view(-1,1,1).expand(num_chan_pre_conv, self.last_conv_size, self.last_conv_size)  # 256*4*4
                    mask_conv_pre = mask_conv_pre.reshape(-1)   # 4096
                    mask_fc_pre = self.best_log_alphas['drop1'] # 4096
                    mask_w_pre = torch.max(mask_conv_pre, mask_fc_pre)   
                    # mask_w_pre = mask_conv_pre
                    mask_w_pre = mask_w_pre.view(1,-1).expand_as(module.weight)
                    # mask_w_pre = self.best_log_alphas['drop1'].view(1,-1).expand_as(module.weight)   # if not conv c3 before
                    mask_w_aft = self.best_log_alphas['drop2'].view(-1,1).expand_as(module.weight)
                    mask_b_aft = self.best_log_alphas['drop2'].view(-1)
                    if fix:
                        self.mask_weight['fc1'] = ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        self.mask_bias['fc1']   = ~(mask_b_aft < thr)
                    else:
                        self.mask_weight['fc1'] = ((mask_w_aft < thr) * (mask_w_pre < thr)) * torch.min(gate(mask_w_aft,name), gate(mask_w_pre,name)) + ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        self.mask_bias['fc1']   = (mask_b_aft < thr) * gate(mask_b_aft,name) + ~(mask_b_aft < thr)

                if name == 'fc2':
                    mask_w_aft = self.best_log_alphas['drop3'].view(-1,1).expand_as(module.weight)
                    mask_w_pre = self.best_log_alphas['drop2'].view(1,-1).expand_as(module.weight)
                    mask_b_aft = self.best_log_alphas['drop3'].view(-1)
                    if fix:
                        self.mask_weight['fc2'] = ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        self.mask_bias['fc2']   = ~(mask_b_aft < thr)
                    else:
                        self.mask_weight['fc2'] = ((mask_w_aft < thr) * (mask_w_pre < thr)) * torch.min(gate(mask_w_aft,name), gate(mask_w_pre,name)) + ~((mask_w_aft < thr) * (mask_w_pre < thr))
                        self.mask_bias['fc2']   = (mask_b_aft < thr) * gate(mask_b_aft,name) + ~(mask_b_aft < thr)
        
    def update_grad(self):
        for name,module in self.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):  
                # if name != 'c1' : 
                module.weight.grad.data *= self.mask_weight[name].cuda()
                module.bias.grad.data *= self.mask_bias[name].cuda()
                
