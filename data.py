from sklearn.utils import shuffle
import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms

# SPLIT
def get_split_mnist(seed=0, fixed_order=False, pc_valid=0, tasknum = 5):
    if tasknum>5:
        tasknum = 5
    data = {}
    taskcla = []
    size = [1, 28, 28]
    
    # Pre-load
    # MNIST
    mean = (0.1307,)
    std = (0.3081,)
    if not os.path.isdir('./dat/binary_split_mnist'):
        os.makedirs('./dat/binary_split_mnist')
        dat = {}
        dat['train'] = datasets.MNIST('./dat/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST('./dat/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i in range(5):
            data[i] = {}
            data[i]['name'] = 'split_mnist-{:d}'.format(i)
            data[i]['ncla'] = 2
            data[i]['train'] = {'x': [], 'y': []}
            data[i]['test'] = {'x': [], 'y': []}
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                task_idx = target.numpy()[0] // 2
                data[task_idx][s]['x'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0]%2)

        for i in range(5):
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x'])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('./dat/binary_split_mnist'), 'data'+ str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('./dat/binary_split_mnist'), 'data'+ str(i) + s + 'y.bin'))
    else:
        # Load binary files
        for i in range(5):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 2
            data[i]['name'] = 'split_mnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('./dat/binary_split_mnist'), 'data'+ str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('./dat/binary_split_mnist'), 'data'+ str(i) + s + 'y.bin'))
        
    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data, taskcla, size

#PERMUTED
def get_permuted_mnist(seed=0, fixed_order=False, pc_valid=0, tasknum = 10):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # Pre-load
    # MNIST
    mean = torch.Tensor([0.1307])
    std = torch.Tensor([0.3081])
    dat = {}
    dat['train'] = datasets.MNIST('./dat/', train=True, download=True)
    dat['test'] = datasets.MNIST('./dat/', train=False, download=True)
    
    for i in range(tasknum):
        print(i, end=',')
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = 'pmnist-{:d}'.format(i)
        data[i]['ncla'] = 10
        if i == 0: permutation = np.arange(28*28)
        else: permutation = np.random.RandomState(seed=i).permutation(28*28)
        # permutation = np.random.RandomState(seed=seed+i).permutation(28*28)

        for s in ['train', 'test']:
            if s == 'train':
                arr = dat[s].train_data.view(dat[s].train_data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].train_labels)
            else:
                arr = dat[s].test_data.view(dat[s].test_data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].test_labels)
                
            arr = (arr/255 - mean) / std
            data[i][s]={}
            data[i][s]['x'] = arr[:,permutation].view(-1, size[0], size[1], size[2])
            data[i][s]['y'] = label
            
    # Validation
    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size

# CIFAR100
def get_split_cifar100(seed = 0, pc_valid=0.1):
    # data_path = '/content/drive/MyDrive/Colab_Notebooks/LAB/Some Models/SBP_pytorch/Struct-Sparse-Pruning/dat/binary_split_cifar100/'

    data={}
    taskcla = []
    size=[3,32,32]
    ids=list(shuffle(np.arange(10),random_state=seed)+1)
    # ids = [9, 3, 5, 10, 2, 7, 8, 4, 1, 6]

    print('Task order =',ids)
    data[0] = dict.fromkeys(['name','ncla','train','test'])
    for i in range(10):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./dat/binary_split_cifar100'), 'data'+ str(ids[i]) + s + 'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./dat/binary_split_cifar100'), 'data'+ str(ids[i]) + s + 'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='cifar100-'+str(ids[i-1])
            
    # Validation
    for t in range(10):
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in range(10):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    return data,taskcla,size

# CIFAR10-100 
def get_split_cifar10_100(seed=0,pc_valid=0.10, tasknum = 10):
    data={}
    taskcla=[]
    size=[3,32,32]

    if not os.path.isdir('./dat/binary_split_cifar100'):
        os.makedirs('./dat/binary_split_cifar100')
        os.makedirs('./dat/binary_cifar10')

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        
        # CIFAR10
        dat={}
        dat['train']=datasets.CIFAR10('./dat/',train=True,download=True,
                                      transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./dat/',train=False,download=True,
                                     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        data[0]={}
        data[0]['name']='cifar10'
        data[0]['ncla']=10
        data[0]['train']={'x': [],'y': []}
        data[0]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(target.numpy()[0])
        
        
        # CIFAR100
        dat={}
        
        dat['train']=datasets.CIFAR100('./dat/',train=True,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./dat/',train=False,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for n in range(1,11):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=10
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                task_idx = target.numpy()[0] // 10 + 1
                data[task_idx][s]['x'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0]%10)

        # "Unify" and save
        for s in ['train','test']:
            data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
            data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
            torch.save(data[0][s]['x'], os.path.join(os.path.expanduser('./dat/binary_cifar10'),'data'+s+'x.bin'))
            torch.save(data[0][s]['y'], os.path.join(os.path.expanduser('./dat/binary_cifar10'),'data'+s+'y.bin'))
        for t in range(1,11):
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('./dat/binary_split_cifar100'),
                                                         'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('./dat/binary_split_cifar100'),
                                                         'data'+str(t)+s+'y.bin'))
    
    # Load binary files
    data={}
    data[0] = dict.fromkeys(['name','ncla','train','test'])
    for s in ['train','test']:
        data[0][s]={'x':[],'y':[]}
        data[0][s]['x']=torch.load(os.path.join(os.path.expanduser('./dat/binary_cifar10'),'data'+s+'x.bin'))
        data[0][s]['y']=torch.load(os.path.join(os.path.expanduser('./dat/binary_cifar10'),'data'+s+'y.bin'))
    data[0]['ncla']=len(np.unique(data[0]['train']['y'].numpy()))
    data[0]['name']='cifar10'
    
    ids=list(shuffle(np.arange(10),random_state=seed) + 1)
#     ids=list(range(1,11))
    print('Task order =',ids)
    for i in range(1,11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./dat/binary_split_cifar100'),
                                                    'data'+str(ids[i-1])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./dat/binary_split_cifar100'),
                                                    'data'+str(ids[i-1])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='cifar100-'+str(ids[i-1])
            
    # Validation
    for t in range(11):
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in range(11):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

#OMNIGLOT
def get_omniglot(seed=0, fixed_order=False, pc_valid=0, tasknum = 50):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    tasknum = 50

    if not os.path.isdir('./dat/binary_omniglot'):
        os.makedirs('./dat/binary_omniglot')
        
        filename = 'Permuted_Omniglot_task50.pt'
        filepath = os.path.join(os.getcwd(), 'dat/')
#         filepath = os.path.join(os.getcwd(), '')
        f = torch.load(os.path.join(filepath,filename))
        ncla_dict = {}
        for i in range(tasknum):
            data[i] = {}
            data[i]['name'] = 'omniglot-{:d}'.format(i)
            data[i]['ncla'] = (torch.max(f['Y']['train'][i]) + 1).int().item()
            ncla_dict[i] = data[i]['ncla']
                
#                 loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[i]['train'] = {'x': [], 'y': []}
            data[i]['test'] = {'x': [], 'y': []}
            data[i]['valid'] = {'x': [], 'y': []}

            image = f['X']['train'][i]
            target = f['Y']['train'][i]

            index_arr = np.arange(len(image))
            np.random.shuffle(index_arr)
            train_ratio = (len(image)//10)*8
            valid_ratio = (len(image)//10)*1
            test_ratio = (len(image)//10)*1
            
            train_idx = index_arr[:train_ratio]
            valid_idx = index_arr[train_ratio:train_ratio+valid_ratio]
            test_idx = index_arr[train_ratio+valid_ratio:]

            data[i]['train']['x'] = image[train_idx]
            data[i]['train']['y'] = target[train_idx]
            data[i]['valid']['x'] = image[valid_idx]
            data[i]['valid']['y'] = target[valid_idx]
            data[i]['test']['x'] = image[test_idx]
            data[i]['test']['y'] = target[test_idx]
            

            # "Unify" and save
            for s in ['train', 'test', 'valid']:
#                 data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('./dat/binary_omniglot'), 'data' + str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('./dat/binary_omniglot'), 'data' + str(i) + s + 'y.bin'))
        torch.save(ncla_dict, os.path.join(os.path.expanduser('./dat/binary_omniglot'), 'ncla_dict.pt'))

    else:
        ncla_dict = torch.load(os.path.join(os.path.expanduser('./dat/binary_omniglot'), 'ncla_dict.pt'))
        # Load binary files
#         ids=list(shuffle(np.arange(tasknum),random_state=seed))
        ids=list(np.arange(tasknum))
        print('Task order =',ids)
        for i in range(tasknum):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test','valid'])
            data[i]['ncla'] = ncla_dict[ids[i]]
            data[i]['name'] = 'omniglot-{:d}'.format(i)
            # Load
            for s in ['train', 'test', 'valid']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('./dat/binary_omniglot'), 
                                                          'data' + str(ids[i]) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('./dat/binary_omniglot'), 
                                                          'data' + str(ids[i]) + s + 'y.bin'))


    # Others
    n = 0
    data_num = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
        # print('Task %d: %d classes'%(t+1,data[t]['ncla']))
#         print(data[t]['train']['x'].shape[0])
        data_num += data[t]['train']['x'].shape[0]
    # print(data_num)
        
    data['ncla'] = n
    # print(n)
    return data, taskcla, size
