import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import append


experiment = 'omniglot'
n_tasks = 50
main = False
accs = []

path = './result/' + experiment + '/used_neurons.txt'

used_neurons = []

with open(path, 'r') as f:
    rows = f.readlines()
    for row in rows[0:-1]:
        layers = row.split(';')
        num = []
        denom = []
        print(layers)
        for layer in layers:
            if layer != ' \n' and layer != '\n' and layer != '  \n':
                num.append(layer.split(':')[1].split('/')[0])
                denom.append(layer.split(':')[1].split('/')[1])
        num = [float(item) for item in num]
        denom = [float(item) for item in denom]
        used_neurons.append(round(100.0*(sum(num) / sum(denom)),1))
print(used_neurons)