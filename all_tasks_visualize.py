import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

experiment = 'cifar10-100'
n_tasks = 11
main = True

if experiment != 'cifar10-100':
    font = {'size' : 20}
else:
    font = {'size' : 25}
matplotlib.rc('font', **font)


accs = []

accs_fix = np.loadtxt('./result/' + experiment + '/hard_dropout.txt')
accs_soft = np.loadtxt('./result/'+ experiment + '/soft_dropout.txt')
accs_ucl = np.loadtxt('./result/' + experiment +'/ucl.txt')
accs_agscl = np.loadtxt('./result/' + experiment + '/ags-cl.txt')
accs_ewc = np.loadtxt('./result/' + experiment + '/ewc.txt')
accs_hat = np.loadtxt('./result/' + experiment + '/hat.txt')

if main:
    accs.append(accs_soft)
    accs.append(accs_ucl)
    accs.append(accs_agscl)
    accs.append(accs_ewc)
    accs.append(accs_hat)
else:
    accs.append(accs_fix)
    accs.append(accs_soft)

step = np.linspace(1,n_tasks,n_tasks)
if main:
    names = ['VBD-CL', 'UCL', 'AGS-CL', 'EWC', 'HAT']
else:
    names = ['Hard-VBD-CL', 'VBD-CL']

if experiment == 'cifar10-100':
    fig, axs = plt.subplots(3,5, figsize=(20,15))
else:
    fig, axs = plt.subplots(2,5, figsize=(17,9))
for task in range(n_tasks):
    for s in range(len(names)):    
        task_acc = accs[s][task:n_tasks,task]
        step_i = step[task:n_tasks]
        axs[int(task>=5)+int(task>=10),task%5].plot(step_i, task_acc, label=names[s], marker='o')
        plt.setp(axs[int(task>=5)+int(task>=10),task%5], xlabel='Tác vụ ' + str(task+1))
        axs[int(task>=5)+int(task>=10),task%5].xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.setp(axs[:,0], ylabel='Độ chính xác trung bình')
if experiment == 'cifar10-100':    
    axs[2,1].set_visible(False)
    axs[2,2].set_visible(False)
    axs[2,3].set_visible(False)
    axs[2,4].set_visible(False)

fig.tight_layout(pad=1)
if experiment == 'cifar10-100':
    axs[2,0].legend(loc='lower center', bbox_to_anchor=(1.8, 0))
else:
    plt.legend(loc='lower center', bbox_to_anchor=(-2.5, -0.39),fancybox=True, shadow=True, ncol=8, fontsize='medium')

path = '/home/lam/Desktop/Thesis_final/VBD/result/' + experiment
if main:
    plt.savefig(path + '/main_all_tasks.pdf')
else:
    plt.savefig(path + '/ablation_all_tasks.pdf', bbox_inches='tight')

plt.show()
    

# tmp = np.round(np.loadtxt('./result/fix.txt'), 4)
# print(tmp)
# res = np.zeros([50,50])
# for i in range(50):
#     for j in range(i+1):
#         res[i][j] = tmp[j]
# print(res)
# np.savetxt('./result/acc_fix.txt', res, fmt="%.4f")
