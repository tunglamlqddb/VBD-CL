from pickle import TRUE
from pydoc import locate
from model import MLPVBD
from learning_vbd import *
import torch
import numpy as np
from conv_net import ConvNet


lr = 0.001
kl = 0.1
fix = False
split_mnist = False
permuted_mnist = True
cifar100 = False
cifar10_100 = False
omniglot = False


if permuted_mnist:
    n_tasks = 10
    from data import get_permuted_mnist
    data, taskcla, size = get_permuted_mnist(tasknum=n_tasks)
    n_epochs = 100
    use_conv = False
    split = False
    
if split_mnist:   
    n_epochs = 100
    use_conv = False
    n_tasks = 5
    from data import get_split_mnist
    data, taskcla, size = get_split_mnist()
    split = True
    
if cifar100==True:
    n_tasks = 10
    n_epochs = 150
    use_conv =True
    from data import get_split_cifar100
    data, taskcla, size = get_split_cifar100()
    split = True

if cifar10_100==True:
    n_tasks = 11
    n_epochs = 150
    use_conv =True
    from data import get_split_cifar10_100
    data, taskcla, size = get_split_cifar10_100()
    split = True

if omniglot==True:
    n_tasks = 50
    n_epochs = 100
    use_conv =True
    from data import get_omniglot
    data, taskcla, size = get_omniglot()
    split = True


# train_loader = {}
# test_loader = {}
# valid_loader = {}
# for task_id in range(n_tasks):
#     train_dataset = torch.utils.data.TensorDataset(data[task_id]['train']['x'], data[task_id]['train']['y'])
#     train_loader[task_id] = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)
#     valid_dataset = torch.utils.data.TensorDataset(data[task_id]['valid']['x'], data[task_id]['valid']['y'])
#     valid_loader[task_id] = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
#     test_dataset = torch.utils.data.TensorDataset(data[task_id]['test']['x'], data[task_id]['test']['y'])
#     test_loader[task_id] = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)


threshes = n_tasks*[2.0]
s=27
if use_conv == True:
    model = ConvNet(taskcla, size, [])
else:
    model = MLPVBD(taskcla, num_classes=data['ncla'], nf=400, input_size=[1,28,28], threshes=[], split=split)
# print("Model architeture: ", model)
accs = np.zeros([n_tasks, n_tasks])
for task_id in range(n_tasks):
    print("Task ", task_id)
    model.new_task(threshes[task_id])
    work = mywork(model, data[task_id]['train']['x'],data[task_id]['train']['y'], data[task_id]['valid']['x'],data[task_id]['valid']['y'], data[task_id]['test']['x'],data[task_id]['test']['y'], task_id, 256, s=s)
    if task_id == 0:
        work.train(epoch=n_epochs, lr=lr, kl=kl, kl_lr=1, fix=False, l1_weight=0.01, l2_weight=None, lasso_weight=None, training_thresh=None, optimizer='Adam')
    #   torch.save(model, "./save_models/cifar100_soft_task0_s"+str(s)+"_thr"+str(threshes[0])+"_kl0.1_l10.01"+".pt")
    #   model = torch.load("./save_models/cifar100_soft_task0_s"+str(s)+"_thr"+str(threshes[0])+"_kl0.1_l10.01"+".pt")
    #   work.model = model
    #   drop_rate = math.exp(work.model.threshes[task_id]) / (math.exp(work.model.threshes[task_id]) + 1)
    #   work.model.update_mask_grad(task_id, training_thresh=None, fix=True, drop_rate=drop_rate, s=s) 
    else:
        work.train(epoch=n_epochs, lr=lr, kl=kl, kl_lr=1, fix=False, l1_weight=0.01, l2_weight=None, lasso_weight=None, training_thresh=None, optimizer='Adam')

    print("----Testing----")
    for task_before in range(0, task_id+1):
        if permuted_mnist:
            work.model.fc3.bias = torch.nn.Parameter(torch.from_numpy(work.model.last_fc_biases[task_before]).cuda())
        test_acc, test_loss, test_ce = work.test(data[task_before]['test']['x'],data[task_before]['test']['y'], task_before)
        print("Task %s Acc %.4f CE %.5f" %(task_before, 100*test_acc, test_ce))
        accs[task_id][task_before] = test_acc
    # if task_id == 0:
    #   break
print(100*"*", end=' ')
for i in range(n_tasks):
    print('\nTask', i, end=': ')
    for j in range(n_tasks-1,-1,-1):
        if accs[j][i] > 0:
            print('%.2f' % (100*accs[j][i]), end=' ')
