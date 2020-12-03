import os
import shutil
import scipy.io


if os.path.exists('./saved_model'):
    shutil.rmtree('./saved_model')
os.mkdir('./saved_model')

i = 1
for growth_step in [2, 3]:
    for initial_cap in [0.333, 0.5, 0.7]:
        growth_ratio = (1.0 - initial_cap) / (growth_step-1)
        scipy.io.savemat('./log/start_tuning_{}.mat'.format(i), {'i': i})

        command_tmp = 'python dlrm_s_pytorch.py --gpu 1 --initialization zero --growth-stop-horizon 1.0' + \
                      ' --initial-capacity ' + str(initial_cap) + ' --growth-step ' + str(growth_step) + ' --growth-ratio ' +str(growth_ratio)
        print('command:\n', command_tmp)
        os.system(command_tmp)

        scipy.io.savemat('./log/end_tuning_{}.mat'.format(i), {'i': i})
        i += 1
