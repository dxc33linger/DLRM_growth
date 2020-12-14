import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio
import collections
import pandas as pd
import matplotlib
matplotlib.use('AGG')


label1 = 'Growth'
label2 = 'Pruning'

filename ='savemat_bot13-64-32-16-16-loss0.44864_accu0.79139_flop27.52G'
filename2 = 'savemat_bot13-64-32-16-16_loss0.44907_accu0.79121_flop27.06G'


content = scio.loadmat('./log/' + filename + '.mat')
gL_log = content['gL_log'][0]
gA_log = content['gA_log'][0]
ln_bot = content['ln_bot'][0]
ln_top = content['ln_top'][0]


assert filename != filename2, 'Check the provided .mat file. It should not the the same as filename2!'
filename2 = scio.loadmat('./log/' + filename2 + '.mat')
gL_log_base = filename2['gL_log'][0]
gA_log_base = filename2['gA_log'][0]
ln_bot_filename2 = filename2['ln_bot'][0]
ln_top_filename2 = filename2['ln_top'][0]
print('ln_bot', ln_bot,  ln_bot_filename2, '\n')
print('ln_top', ln_top, ln_top_filename2, '\n' )
assert len(gL_log) == len(gA_log)
assert len(gL_log_base) == len(gL_log)
assert ln_top.any() == ln_top_filename2.any(), 'ln_top != ln_top_filename2'
assert ln_bot.any() == ln_bot_filename2.any(), 'ln_bot != ln_bot_filename2'

param_list = content['param_FC_log'][0]
param_list_filename2 = filename2['param_FC_log'][0]

print(param_list)
print(param_list_filename2)
title_font = {'size': '8', 'color': 'black', 'weight': 'normal'}  # Bottom vertical alignment for more space
axis_font = {'size': '10'}
# Plot 1:

plt.figure(figsize=(16, 6.5))
plt.subplot(131)
# plt.figure()
x = np.linspace(0, len(gL_log), num=len(gL_log))
plt.xlim(0, len(gL_log))
plt.ylim(0.43, 0.5)
plt.xlabel("Num of Iterations")
plt.ylabel('BCELoss')
plt.plot(x, gL_log, 'r-', alpha=1.0, label='Loss - ' +  label1)
plt.plot(x, gL_log_base, 'k--', alpha=1.0, label='Loss - ' +  label2)
plt.yticks(np.arange(0.40, 0.5, step=0.01))
plt.xticks(np.arange(0, len(gL_log) + 1, step=20), rotation=45)
plt.legend(loc='lower right')
plt.title('Kaggle Display Advertising Challenge Dataset')

plt.subplot(132)
x = np.linspace(0, len(gA_log), num=len(gA_log))
plt.xlim(0, len(gA_log))
plt.ylim(0.765, 0.795)
plt.xlabel("Num of Iterations")
plt.ylabel('Accuracy %')
plt.plot(x, gA_log, 'g-', alpha=1.0, label='Accuracy - ' + label1)
plt.plot(x, gA_log_base, 'k--', alpha=1.0, label='Accuracy - ' +  label2)
plt.yticks(np.arange(0.765, 0.795, step=0.005))
plt.xticks(np.arange(0, len(gA_log) + 1, step=20), rotation=45)
plt.legend(loc='lower left')
plt.title('Kaggle Display Advertising Challenge Dataset')


plt.subplot(133)
x = np.linspace(0, len(param_list), num=len(param_list))
plt.xlim(0, len(param_list))
plt.xlabel("Num of Iterations")
plt.ylabel('Num of parameters (K)')
plt.plot(x, param_list, 'b-', alpha=1.0, label='FC param - ' + label1)
plt.plot(x, param_list_filename2, 'k--', alpha=1.0, label='FC param - ' + label2)
plt.xticks(np.arange(0, len(param_list) + 1, step=20*2048), rotation=45)
plt.legend(loc='lower right')
plt.title('Kaggle Display Advertising Challenge Dataset')
plt.savefig('./log/learning_curve_loss{:.5f}_Accu{:.5f}.png'.format(gL_log[-1], gA_log[-1]))



flops = sum(param_list) * 2
flops_filename2 = sum(param_list_filename2) * 2
print('#Param (k)', np.unique(param_list), np.unique(param_list_filename2))
print('Growth #FLOPs: {:.2f}G\nfilename2 #FLOPS: {:.2f}G\nsaving percentage {:.1f}%'.format(flops/ 1000000, flops_filename2/ 1000000, (flops_filename2 - flops) / flops_filename2 * 100))
