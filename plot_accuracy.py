import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio
import collections
import pandas as pd
import matplotlib
matplotlib.use('AGG')

filename ='savemat_bot13-512-256-64-16_step2_loss0.44641_accu0.79155'
content = scio.loadmat('./log/' + filename + '.mat')
gL_log = content['gL_log'][0]
gA_log = content['gA_log'][0]
ln_bot = content['ln_bot'][0]
ln_top = content['ln_top'][0]


baseline = 'savemat_bot13-512-256-64-16_step0_loss0.4463_accu0.7919'
assert filename != baseline, 'Check the provided .mat file. It should not the the same as baseline!'
baseline = scio.loadmat('./log/' + baseline + '.mat')
gL_log_base = baseline['gL_log'][0]
gA_log_base = baseline['gA_log'][0]
ln_bot_baseline = baseline['ln_bot'][0]
ln_top_baseline = baseline['ln_top'][0]
print('ln_bot', ln_bot,  ln_bot_baseline, '\n')
print('ln_top', ln_top, ln_top_baseline, '\n' )
assert len(gL_log) == len(gA_log)
assert len(gL_log_base) == len(gL_log)
assert ln_top.any() == ln_top_baseline.any(), 'ln_top != ln_top_baseline'
assert ln_bot.any() == ln_bot_baseline.any(), 'ln_bot != ln_bot_baseline'

param_list = content['param_FC_log'][0]
param_list_baseline = baseline['param_FC_log'][0]

print(param_list)
print(param_list_baseline)
title_font = {'size': '8', 'color': 'black', 'weight': 'normal'}  # Bottom vertical alignment for more space
axis_font = {'size': '10'}
# Plot 1:

plt.figure(figsize=(16, 6.5))
plt.subplot(131)
# plt.figure()
x = np.linspace(0, len(gL_log), num=len(gL_log))
plt.xlim(0, len(gL_log))
plt.xlabel("Num of Iterations")
plt.ylabel('BCELoss')
plt.plot(x, gL_log, 'r-', alpha=1.0, label='Loss - FC growth')
plt.plot(x, gL_log_base, 'k--', alpha=1.0, label='Loss - Baseline')
plt.yticks(np.arange(0.4, 0.6, step=0.1))
plt.xticks(np.arange(0, len(gL_log) + 1, step=20), rotation=45)
plt.legend(loc='lower right')
plt.title('Kaggle Display Advertising Challenge Dataset')

plt.subplot(132)
x = np.linspace(0, len(gA_log), num=len(gA_log))
plt.xlim(0, len(gA_log))
plt.xlabel("Num of Iterations")
plt.ylabel('Accuracy %')
plt.plot(x, gA_log, 'g-', alpha=1.0, label='Accuracy - FC growth')
plt.plot(x, gA_log_base, 'k--', alpha=1.0, label='Accuracy - Baseline')
plt.yticks(np.arange(0.70, 0.80, step=1))
plt.xticks(np.arange(0, len(gA_log) + 1, step=20), rotation=45)
plt.legend(loc='lower left')
plt.title('Kaggle Display Advertising Challenge Dataset')


plt.subplot(133)
x = np.linspace(0, len(param_list), num=len(param_list))
plt.xlim(0, len(param_list))
plt.xlabel("Num of Iterations")
plt.ylabel('Num of parameters (K)')
plt.plot(x, param_list, 'b-', alpha=1.0, label='FC param - FC growth')
plt.plot(x, param_list_baseline, 'k--', alpha=1.0, label='FC param - Baseline')
plt.xticks(np.arange(0, len(param_list) + 1, step=20*2048), rotation=45)
plt.legend(loc='lower right')
plt.title('Kaggle Display Advertising Challenge Dataset')
plt.savefig('./log/learning_curve_loss{:.5f}_Accu{:.5f}.png'.format(gL_log[-1], gA_log[-1]))



flops = sum(param_list) * 2
flops_baseline = sum(param_list_baseline) * 2
print('#Param (k)', np.unique(param_list), np.unique(param_list_baseline))
print('Growth #FLOPs: {:.2f}G\nBaseline #FLOPS: {:.2f}G\nsaving percentage {:.1f}%'.format(flops/ 1000000, flops_baseline/ 1000000, (flops_baseline - flops) / flops_baseline * 100))
