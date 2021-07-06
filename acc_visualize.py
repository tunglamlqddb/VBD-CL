import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

font = {'size' : 18}
matplotlib.rc('font', **font)

experiment = 'cifar10-100'
n_tasks = 11
main = False
accs = []
task20 = False

accs_fix = np.loadtxt('./result/' + experiment + '/hard_dropout.txt')
accs_soft = np.loadtxt('./result/'+ experiment + '/soft_dropout.txt')
accs_ucl = np.loadtxt('./result/' + experiment +'/ucl.txt')
accs_agscl = np.loadtxt('./result/' + experiment + '/ags-cl.txt')
accs_ewc = np.loadtxt('./result/' + experiment + '/ewc.txt')
accs_hat = np.loadtxt('./result/' + experiment + '/hat.txt')

if task20:
    accs_hat_800_01 = np.loadtxt('./result/' + experiment + '/hat_20_800_0.1.txt')
    accs_hat_800_05 = np.loadtxt('./result/' + experiment + '/hat_20_800_0.5.txt')
    accs_hat_1000_05 = np.loadtxt('./result/' + experiment + '/hat_20_1000_0.5.txt')
    accs_soft_20 = np.loadtxt('./result/' + experiment + '/soft_dropout_20.txt')

if task20:
    accs.append(accs_hat_800_01)
    accs.append(accs_hat_800_05)
    accs.append(accs_hat_1000_05)
    accs.append(accs_soft_20)
else:
    if main:
        accs.append(accs_soft)
        accs.append(accs_ucl)
        accs.append(accs_agscl)
        accs.append(accs_ewc)
        accs.append(accs_hat)
    else:
        accs.append(accs_fix)
        accs.append(accs_soft)


if task20:
    step = np.arange(n_tasks*2)
else:
    step = np.arange(n_tasks)

if task20:
    names = ['HAT_800_0.1', 'HAT_800_0.5', 'HAT_1000_0.5', 'VBD-CL']
else:
    if main:
        names = ['VBD-CL', 'UCL', 'AGS-CL', 'EWC', 'HAT']
    else:
        names = ['Hard-VBD-CL', 'VBD-CL']

for s in range(len(names)):    
    ave = np.sum(accs[s], axis=1)
    ave /= (step+1)
    plt.plot(step, ave, label=names[s], marker='.')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))


plt.xlabel("Tác vụ")
plt.ylabel("Độ chính xác trung bình")
plt.tight_layout()
if experiment == 'omniglot' or task20== True:
    plt.legend(fontsize='small')
else:
    plt.legend(bbox_to_anchor=(0.45, -0.4), loc='lower center', ncol=3, fontsize='medium')
if main==False:
    plt.legend()
path = '/home/lam/Desktop/Thesis_final/VBD/result/' + experiment  
if task20:
    plt.savefig(path + '/ablation_longer_tasks.pdf')
else:
    if main:
        plt.savefig(path + '/main.pdf', bbox_inches='tight')
    else:
        plt.savefig(path + '/ablation.pdf', bbox_inches='tight')

plt.show()
    
# tmp = np.round(np.loadtxt('./result/fix.txt'), 4)
# print(tmp)
# res = np.zeros([50,50])
# for i in range(50):
#     for j in range(i+1):
#         res[i][j] = tmp[j]
# print(res)
# np.savetxt('./result/acc_fix.txt', res, fmt="%.4f")
