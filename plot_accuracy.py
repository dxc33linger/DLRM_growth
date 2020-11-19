import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio
import collections
import pandas as pd

file_name ='savemat_loss0.5966_accu0.7515'
content = scio.loadmat('./log/' + file_name + '.mat')


gL_log = content['gL_log'][0]
gA_log = content['gA_log'][0]
num_iteration = len(gL_log)
assert len(gL_log) == len(gA_log)
title_font = {'size': '8', 'color': 'black', 'weight': 'normal'}  # Bottom vertical alignment for more space
axis_font = {'size': '10'}
plt.figure()
x = np.linspace(0, num_iteration, num=num_iteration)
print(x.shape)
print(gL_log.shape)
plt.xlim(0, num_iteration)
plt.xlabel('Iteration')
plt.ylabel('BCELoss')
plt.plot(x,gL_log, 'b-o', alpha=1.0, label='FC growth')
plt.yticks(np.arange(0.4, 0.6, step=0.1))
plt.xticks(np.arange(0, num_iteration + 1, step=20))
plt.legend(loc='best')
plt.savefig('./log/learning_curve_loss{:.4f}.png'.format(gL_log[-1]))
plt.title('Kaggle Display Advertising Challenge Dataset')
plt.show()