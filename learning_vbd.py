import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import time, math

# torch.autograd.set_detect_anomaly(True)


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from layers_VBD import * 

class mywork():
    def __init__(self, model, train_x,train_y, valid_x,valid_y, test_x,test_y, task_id, sbatch=256, lr_patience=5, lr_factor=3, lr_min=1e-5, s=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100000000], gamma=1)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.model = model
        self.kl_weight = 0
        self.l1_weight = 0
        self.l2_weight = 0
        self.lasso_weight = 0
        self.task_id = task_id
        self.ce = torch.nn.CrossEntropyLoss()
        self.sbatch = sbatch
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.s = s
        self.used_neurons = {}

    def _get_optimizer(self,lr=None, optimizer='Adam'):
        if lr is None: lr=self.lr
        # old_state = self.optim.state_dict()
        # if optimizer == 'SGD':
        #     self.optim = torch.optim.SGD(self.model.parameters(),lr=lr)
        #     self.optim.load_state_dict(old_state)
        # if optimizer == 'Adam':
        #     self.optim = torch.optim.Adam(self.model.parameters(),lr=lr)
        #     self.optim.load_state_dict(old_state)
        for g in self.optim.param_groups:
            g['lr'] = lr

    def train(self, epoch=300, kl = None, l1_weight=None, l2_weight=None, lasso_weight=None, scaling = 0, lr = 1e-4, training_thresh=None, kl_lr=1, scheduler='lambda', fix=True, optimizer='Adam'):
        self.model = self.model.to(self.device)
        best_model = self.model.get_model()    # state dict
        best_loss = 1e8
        best_epoch = 0
        if optimizer == 'Adam':
            # self.optim = optim.Adam(
            #         [{'params':layer, 'lr':kl_lr * lr} if (name.split('.')[-1]=='mu' or name.split('.')[-1]=='log_sigma2') else \
            #         {'params':layer} for name, layer in self.model.named_parameters() if name.split('.')[-1] != 'bias'], lr=lr
            #         , betas=(0.9,0.999)
            # )
            self.optim = optim.Adam(self.model.parameters(),lr=lr, betas=(0.9,0.999))
        if optimizer == 'SGD':
            self.optim = optim.SGD(
                    [{'params':layer, 'lr':kl_lr * lr} if (name.split('.')[-1]=='mu' or name.split('.')[-1]=='log_sigma2') else \
                    {'params':layer} for name, layer in self.model.named_parameters() if name.split('.')[-1] != 'bias'], lr=lr
            )

        # self.scheduler = ReduceLROnPlateau(self.optim,'min',factor=1/3 ,patience=5, verbose =False, min_lr=1e-5)
        # lamb = lambda e: 1 if epoch < 20 else 1-(epoch-20)/30
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lamb)

        if kl!=None:
            self.kl_weight = kl
        if l1_weight != None:
            self.l1_weight = l1_weight
        if l2_weight != None:
            self.l2_weight = l2_weight
        if lasso_weight != None:
            self.lasso_weight = lasso_weight   

        print("Staring training")
        for e in (range(0,epoch)):
            self.model.train()
            self.kl_weight = min(self.kl_weight+ scaling, 1)
            running_loss = 0.0
            ce_loss = 0.0
            train_acc = 0
            i = 0 
            start = time.time()
            
            self.used_neurons[e] = []
            if (e+1)%5==0:
                print('Task %s Epoch %s: ' % (self.task_id, e+1), end='')
            for name, module in self.model.named_children():
                if hasattr(module, 'mu'):
                    if module.best_log_alpha != None:
                        # used = ((torch.min(module.log_alpha(self.task_id), module.best_log_alpha) < module.threshes[self.task_id]).float() != 0).nonzero()
                        used = (torch.min(module.log_alpha(self.task_id), module.best_log_alpha) < module.threshes[self.task_id]).float().sum().item()
                    else:
                        # used = ((module.log_alpha(self.task_id) < module.threshes[self.task_id]).float() != 0).nonzero()
                        used = (module.log_alpha(self.task_id) < module.threshes[self.task_id]).float().sum().item()
                    # print("\n", name, ':', module.log_alpha(self.task_id).min().item(), '/', module.log_alpha(self.task_id).max().item(), end='; ')
                    if (e+1)%5==0:
                        print(name,':',used, end='----')
                    # self.used_neurons[e].append((name,used))
            if (e+1)%5==0:
                print()

            r=np.arange(self.train_x.size(0))
            np.random.shuffle(r)
            r=torch.LongTensor(r)
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                data=self.train_x[b].to(self.device)
                target=self.train_y[b].to(self.device)

                self.optim.zero_grad()
                if self.model.split:
                    output = self.model(data, self.task_id)[self.task_id]
                    # target %= self.model.taskcla[self.task_id][1]
                else:
                    output = self.model(data, self.task_id)
                # pred = output.data.max(1)[1] 
                # train_acc += np.sum(pred.cpu().numpy() == target.cpu().numpy())

                loss = self.ce(output, target) + self.model.kl_reg(self.task_id) * self.kl_weight / len(target)
                if l1_weight != None:
                    loss += self.l1_weight*self.model.l1(self.task_id) / len(target)
                if l2_weight != None:
                    loss += self.l2_weight*self.model.l2(self.task_id) / len(target)
                if lasso_weight != None:
                    loss += self.lasso_weight*self.model.group_Lasso() / len(target)
                
                loss.backward()

                if self.task_id > 0:
                    self.model.update_grad()
                    
                self.optim.step()
                
            self.model.update_mask(self.task_id)
                
            finish = time.time()
            train_acc, train_loss, train_ce = self.test(self.train_x,self.train_y, self.task_id)
            eval_time = time.time()
            # print('\n[%d] Time: %.4f Train Acc: %.8f Loss: %.8f CE loss: %.8f ' %
            #   (e + 1, time.time()-start, 100.0*train_acc/len(self.trainset.dataset), running_loss / len(self.trainset.dataset), ce_loss/len(self.trainset.dataset)), end=' ')    
            
            if (e+1)%5==0:
                print('|Epoch{:3d}, time={:5.1f}ms/time={:5.1f}ms | Train: loss={:.5f}, acc={:5.4f}% |'.format(
                    e+1, 1000*(finish-start), 1000*(eval_time-finish), train_loss, 100*train_acc), end='')

            valid_acc, valid_loss, valid_ce = self.test(self.valid_x,self.valid_y, self.task_id)
            if (e+1)%5==0:
                print(' Valid: loss={:.5f}, ce={:.5f}, acc={:5.4f}% |'.format(valid_loss, valid_ce, 100*valid_acc), end='')
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.get_model()
                best_epoch = e+1
                patience = self.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    if (e+1)%5==0:
                        print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self._get_optimizer(lr)                        
            if (e+1)%5==0:
                print(" at epoch", best_epoch)
                # print("\tReal test:", end=' ')
            # test_acc, test_loss, test_ce = self.test(self.test_x,self.test_y, self.task_id)
            # if (e+1)%5==0:
                # print('loss={:.5f}, ce={:.5f}, acc={:5.4f}% |'.format(test_loss, test_ce, 100*test_acc), end='')
                print()

            # if optimizer == 'SGD':
            #       torch.nn.utils.clip_grad_norm(self.model.parameters(), 100)
                  
            # self.scheduler.step(valid_loss)
            # self.scheduler.step()

            # self.used_neurons[e] = []
            # for name, module in self.model.named_children():
            #     if hasattr(module, 'mu'):
            #         if module.best_log_alpha != None:
            #             used = ((torch.min(module.log_alpha(self.task_id), module.best_log_alpha) < module.threshes[self.task_id]).float() != 0).nonzero()
            #         else:
            #             used = ((module.log_alpha(self.task_id) < module.threshes[self.task_id]).float() != 0).nonzero()
            #         # print("\n", name, ':', module.log_alpha(self.task_id).min().item(), '/', module.log_alpha(self.task_id).max().item(), end='; ')
            #         # print('\n', name, ':', used)
            #         self.used_neurons[e].append((name,used))

        # if self.task_id > 0:
        self.model.set_model(best_model)
        self.model.update_mask(self.task_id)
        self.model.update_best_log_alphas(self.task_id)
        self.sparsity()
        # if self.task_id == 0:
        #     PATH = '/content/drive/MyDrive/SBP_pytorch/Struct-Sparse-Pruning/save_models'
        #     path = PATH + '/model_cifar10_' + str(self.task_id) + '_' + str(self.s) + '_' + str(self.model.threshes[self.task_id]) + '.pt'
        #     torch.save(self.model, path)
        drop_rate = math.exp(self.model.threshes[self.task_id]) / (math.exp(self.model.threshes[self.task_id]) + 1)
        self.model.update_mask_grad(self.task_id, training_thresh, fix=fix, drop_rate=drop_rate, s=self.s)  
        
        # for i, value in self.used_neurons.items():
        #     print('epochs', i)
        #     for name, used in value:
        #         print(name, ',', used.tolist())
        
    def test_2(self, test_loader, task_id, during_train=True, thresh=None):
        if thresh!=None:
           remember = self.change_thresh(thresh)
        self.model.eval()
        test_loss = 0 
        test_acc = 0
        ce_loss = 0
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            if self.model.split:
                output = self.model(data, task_id)[task_id]
                # target %= self.model.taskcla[task_id][1]
            else:
                output = self.model(data, task_id)
            pred = output.data.max(1)[1] 
            test_acc += np.sum(pred.cpu().numpy() == target.cpu().numpy())
            ce_tmp = self.ce(output, target)
            loss = ce_tmp + self.model.kl_reg(self.task_id) * self.kl_weight / len(target)
            if self.l1_weight > 0:
                loss += self.l1_weight*self.model.l1(self.task_id) / len(target)
            if self.l2_weight > 0:
                loss += self.l2_weight*self.model.l2(self.task_id) / len(target)
            if self.lasso_weight > 0:
                loss += self.lasso_weight*self.model.group_Lasso() / len(target)

            ce_loss += ce_tmp
            test_loss += loss.item() 

        if thresh!=None:
            self.change_thresh(remember)
            
        return test_acc/len(test_loader.dataset), test_loss/len(test_loader), ce_loss/len(test_loader)

    def test(self, x,y, task_id):
        total_loss=0
        ce_loss = 0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),256):
            if i+256<=len(r): b=r[i:i+256]
            else: b=r[i:]
            images=x[b].to(self.device)
            target=y[b].to(self.device)
            
            # Forward
            if self.model.split:
                outputs = self.model.forward(images, task_id)[task_id]
            else:
                outputs = self.model.forward(images, task_id)
            ce_tmp = self.ce(outputs,target)
            loss = ce_tmp + self.model.kl_reg(self.task_id) * self.kl_weight / len(target)
            if self.l1_weight > 0:
                loss += self.l1_weight*self.model.l1(self.task_id) / len(target)
            if self.l2_weight > 0:
                loss += self.l2_weight*self.model.l2(self.task_id) / len(target)
            if self.lasso_weight > 0:
                loss += self.lasso_weight*self.model.group_Lasso() / len(target)
                
            values,indices=outputs.max(1)
            hits=(indices==target).float()

            total_loss+=loss.data.cpu().numpy()*len(b)
            ce_loss+=ce_tmp.data.cpu().numpy()*len(b) 
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_acc/total_num, total_loss/total_num, ce_loss/total_num

    def cal_entropy(self, current_task_id, other_task_loader, num_samples, taskcla): # after training current task calculate entropy of output prediction from 1st to current task          
        if not self.model.split:
            total_predictions = np.zeros([len(other_task_loader.dataset), self.model.num_classes])
        else:
            total_predictions = np.zeros([len(other_task_loader.dataset), taskcla[current_task_id][1]])
        total_accs = 0
        for num in range(num_samples):
            all_output = list()
            test_acc = 0
            for data, target in other_task_loader:
                data, target = data.to(self.device), target.to(self.device)
                if self.model.split:
                    batch_output = self.model(data, current_task_id, multi_sample=True)[current_task_id]
                    target %= self.model.taskcla[current_task_id][1]
                else:
                    batch_output = self.model(data, current_task_id, multi_sample=True)
                all_output.append(F.softmax(batch_output, dim=1).detach().cpu().numpy())
                pred = batch_output.data.max(1)[1]
                test_acc += np.sum(pred.cpu().numpy() == target.cpu().numpy())
            test_acc /= len(other_task_loader.dataset)
            total_accs += test_acc
            all_output = np.vstack(all_output)
            total_predictions += all_output
            
        total_predictions /= num_samples
        entropy = -np.sum(np.log(total_predictions + 1e-12) * total_predictions, axis=1)
        total_accs /= num_samples

        return entropy, total_accs

    def change_thresh(self, newt):
        if newt==None:
            pass
        # for module in self.model.features.children():
        #     if hasattr(module, 'kl_reg'):
        #         k = module.thresh
        #         module.thresh = newt
        for module in self.model.children():
            if hasattr(module, 'mu'):
                k = module.thresh
                module.thresh = newt
        self.model.thresh = newt
        return k

    def sparsity(self): 
        path = './result/neuron_omniglot_50_0.txt'
        print('-----Sparsity-----')
        with open(path, 'a') as f:
            f.write("Task %s\n" % self.task_id)
            for name, module in self.model.named_children():
                if hasattr(module, 'mu'):
                    mask = (module.best_log_alpha < self.model.threshes[self.task_id]).float()
                    print(name, ':', mask.sum().item(), '/', module.channels, end='; ')
                    f.write(name + ':' + str(mask.sum().item()) + '/' + str(module.channels) + '; ')
            f.write('\n')
        print()
        # for name, module in self.model.named_children():
        #     if hasattr(module, 'mu'):
        #         min_alpha = module.best_log_alpha.min().item()
        #         max_alpha = module.best_log_alpha.max().item()
        #         print(name, ':', min_alpha, '/', max_alpha, end='; ')
        #         print(module.best_log_alpha)
        # print()      
        # with open(path, 'a') as f:
        #     f.write("Task %s\n" % self.task_id)
        #     for name, module in self.model.named_children():
        #         if hasattr(module, 'mu'):
        #             f.write(name + ':' + str(module.best_log_alpha.clone().detach().cpu().numpy()) + '\n')
        # with open(path, 'a') as f:
        #     f.write("Task %s\n" % self.task_id)
        #     for name, module in self.model.named_children():
        #         if hasattr(module, 'mu'):
        #             f.write("%s\n" % name) 
        #             f.write('MU:' + str((module.mu[self.task_id]*(1-module.masks[self.task_id])).clone().detach().cpu().numpy()) + '\n')
        #             f.write('Log_sigma2:' + str(module.log_sigma2[self.task_id].clone().detach().cpu().numpy()) + '\n')

    def save_model(self, PATH):
        path = PATH + '/model_' + str(self.task_id) + '.pt'
        torch.save(self.model, path)
