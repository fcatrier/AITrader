#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

python

# -----------------------------------------------------------------------------
# Exploitation / visualisation des résultats de l'apprentissage
#
#   => analyse de l'overfitting
#
# -----------------------------------------------------------------------------

import os
import sys


cur_dir = os.getcwd()
if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
    sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\pythonProject\\test-master')
    py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
elif cur_dir == 'C:\\Users\\Frédéri\\PycharmProjects\\pythonProject':
    py_dir = 'C:\\Users\\Frédéri\\Py'
else:
    sys.path.append('E:\\Py\\pythonProject')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v7.6.5 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v8.0.3.33 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')
    py_dir = 'E:\\Py'


import learn_history

#
_dataset_name = 'work'
_dir_npy='\\npy_current'

df = learn_history.npy_results(_dataset_name, _dir_npy)



cols = ['step3_samples_by_class','step3_time_depth','config_LSTM','config_Dense','acc','val_acc','loss','val_loss','result','step3_columns']
df[cols]

cols = ['acc','val_acc','loss','val_loss','result','step3_columns']
df[cols]

cols = ['conv1D_block1_filters', 'conv1D_block1_kernel_size', 'conv1D_block1_MaxPooling1D_pool_size','acc','val_acc','loss','val_loss','result','res_eval_result_atr' ]
df[cols]


df[(df['result']>0.0)&(df['res_eval_result_atr']>0.0)][cols]

df[(df['result']>0.0)]

df[(df['res_eval_result_atr']>0.0)][cols]

df = df.sort_values(by=['val_loss'], ascending=True)

df = df.sort_values(by=['res_eval_result_atr'], ascending=False) 


cols = ['step3_time_depth', \
        'acc','val_acc','test1_acc','test2_acc', \
        'res_eval_result_atr_val','res_eval_result_atr_test1','res_eval_result_atr_test2']

df = df.sort_values(by=['val_acc'], ascending=False)
df[cols]

