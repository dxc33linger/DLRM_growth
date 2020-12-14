import os
import shutil
import scipy.io

if not os.path.exists('./log'):
    os.mkdir('./log')
if os.path.exists('./saved_model'):
    shutil.rmtree('./saved_model')
os.mkdir('./saved_model')

# i = 1
# for initial_cap in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     for initialization in ['zero']:
#         growth_step = 2
#         debuglog = 'testCode'
#
#         # initial_cap  = 1.0 / growth_step
#         growth_ratio = (1.0 - initial_cap) / (growth_step-1)
#         scipy.io.savemat('./log/start_tuning_{}.mat'.format(i), {'i': i})
#
#         command_tmp = 'python dlrm_s_pytorch.py --gpu 1 --growth-stop-horizon 1.0' + ' --initialization ' + initialization + \
#                       ' --initial-capacity ' + str(initial_cap) + ' --growth-step ' + str(growth_step) + ' --growth-ratio ' +str(growth_ratio) + ' --debuglog ' + debuglog
#         print('command:\n', command_tmp)
#         os.system(command_tmp)
#
#
#         scipy.io.savemat('./log/end_tuning_{}.mat'.format(i), {'i': i})
#         i += 1
#'0.0-0.9', '0.0-0.8','0.0-0.7','0.0-0.6','0.0-0.5','0.0-0.4','0.0-0.3','0.0-0.2','0.0-0.1', '0.0-0.0
#'0.0-0.7','0.0-0.6','0.0-0.5'
i = 1


#,'0.7-0.0','0.6-0.0', '0.5-0.0'
for mask_delay in ['0.0-0.3', '0.0-0.5']:  # , '0.8-0.0'
    for mask_ratio in ['0.9-0.0']:
        debuglog = 'scanGrowthSmallSize'

        scipy.io.savemat('./log/start_Grow_tuning_{}.mat'.format(i), {'i': i})

        command_tmp = 'python dsd_dlrm_s_pytorch.py --gpu 0' + \
                      ' --masking-delay ' + mask_delay + ' --masking-ratio ' + mask_ratio + ' --debuglog ' + debuglog
        print('command:\n', command_tmp)
        os.system(command_tmp)


        scipy.io.savemat('./log/end_Grow_tuning_{}.mat'.format(i), {'i': i})
        i += 1


# '0.0-0.9-0.0', '0.0-0.8-0.0','0.0-0.7-0.0','0.0-0.6-0.0','0.0-0.5-0.0','0.0-0.4-0.0','0.0-0.3-0.0','0.0-0.2-0.0','0.0-0.1-0.0', '0.0-0.0-0.0'
for mask_delay in ['0.0-0.1-0.4', '0.0-0.1-0.6']:  # '0.0-0.25-0.75', , '0.0-0.8-0.0' , '0.0-0.2-0.5',
    for mask_ratio in ['0.0-0.9-0.0']:
        debuglog = 'scanDSDSmallSize'

        scipy.io.savemat('./log/start_DSD_tuning_{}.mat'.format(i), {'i': i})

        command_tmp = 'python dsd_dlrm_s_pytorch.py --gpu 1' + \
                      ' --masking-delay ' + mask_delay + ' --masking-ratio ' + mask_ratio + ' --debuglog ' + debuglog
        print('command:\n', command_tmp)
        os.system(command_tmp)


        scipy.io.savemat('./log/end_DSD_tuning_{}.mat'.format(i), {'i': i})
        i += 1




for mask_delay in ['0.0-0.7', '0.0-0.5']:
    for mask_ratio in ['0.0-0.9']: # ,'0.0-0.8'
        debuglog = 'scanPruningSmallSize'

        scipy.io.savemat('./log/start_pruning_tuning_{}.mat'.format(i), {'i': i})

        command_tmp = 'python dsd_dlrm_s_pytorch.py --gpu 0' + \
                      ' --masking-delay ' + mask_delay + ' --masking-ratio ' + mask_ratio + ' --debuglog ' + debuglog
        print('command:\n', command_tmp)
        os.system(command_tmp)


        scipy.io.savemat('./log/end_pruning_tuning_{}.mat'.format(i), {'i': i})
        i += 1
