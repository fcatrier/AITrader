#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import os
import sys


cur_dir = os.getcwd()
if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
    sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\AITrader')
    py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
elif cur_dir == 'C:\\Users\\Frédéri\\PycharmProjects\\AITrader':
    py_dir = 'C:\\Users\\Frédéri\\Py'
else:
    sys.path.append('E:\\Py\\AITrader')
    py_dir = 'E:\\Py'

import learn_script


_dataset_name = 'work'
_dir_npy = '\\npy_current'

if __name__ == '__main__':
    learn_script.execute(_dataset_name, _dir_npy)


# import arbo
# import learn_history

# if __name__ == '__main__':
    # npy_path = arbo.get_study_dir(py_dir, _dataset_name) + _dir_npy
    # df = learn_history.npy_results(npy_path,669)
    # print(df.tail())
