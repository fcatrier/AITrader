#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

# import os
# import sys
# import keras

# cur_dir = os.getcwd()
# if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
#     sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\pythonProject\\test-master')
#     py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
# elif cur_dir == 'C:\\Users\\Frédéri\\PycharmProjects\\pythonProject':
#     py_dir = 'C:\\Users\\Frédéri\\Py'
# else:
#     sys.path.append('E:\\Py\\pythonProject')
#     sys.path.append('E:\\Py\\pythonProject\\__pycache__')
#     py_dir = 'E:\\Py'
#     sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v7.6.5 for CUDA 10.1\\bin')
#     sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v8.0.3.33 for CUDA 10.1\\bin')
#     sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')


#
# imports from this project
#
import arbo
import learn_evaluate_results
import learn_history
import step2_dataset_prepare_target_data as step2
import step3_dataset_prepare_learning_input_data as step3
import utils
from model_manager import ModelManager


def learn(dataset_name, dir_npy, model_manager, learning_data, loops_count=1):
    #
    npy_path = arbo.npy_path(dataset_name, dir_npy)
    idx_run_loop = learn_history.new_npy_idx(npy_path)
    #
    # Setting of last model manager properties that need information about data generation
    #
    input_features = learning_data['train']['np_X'].shape[2]
    input_timesteps = learning_data['train']['np_X'].shape[1]
    output_shape = learning_data['train']['df_y_Nd'].shape[1]
    train_params = learning_data['train']['np_X'].shape[0]
    print("input_features=", input_features)
    print("input_timesteps=", input_timesteps)
    print("output_shape=", output_shape)
    print("train_params=", train_params)
    #
    _mm_dict = model_manager.get_properties()
    #
    _mm_dict['input_features'] = input_features
    _mm_dict['input_timesteps'] = input_timesteps
    _mm_dict['output_shape'] = output_shape
    _mm_dict['train_params'] = train_params
    #
    model_manager.update_properties(_mm_dict)
    #
    for i in range(0, loops_count):
        print("")
        print("-----------------------------------------------------------")
        print("Repeat #", i + 1, " /", loops_count)
        print("-----------------------------------------------------------")
        print("")
        #
        model = model_manager.create_compile_model()
        #
        if i == 0:  # affiché uniquement au premier passage pour désaturer l'affichage
            print(model.summary())
            print("model.count_params()=", model.count_params())
            print("train_params : ", train_params)
        #
        model_manager.fit(learning_data)
        #
        post_learning_metrics_val   = learn_evaluate_results.post_learning_metrics(model, learning_data, 'val')
        post_learning_metrics_test1 = learn_evaluate_results.post_learning_metrics(model, learning_data, 'test1')
        post_learning_metrics_test2 = learn_evaluate_results.post_learning_metrics(model, learning_data, 'test2')
        #
        path = arbo.npy_path_with_prefix(dataset_name, dir_npy, idx_run_loop)
        #
        utils.dictionary_save(path, model_manager.get_properties())
        utils.dictionary_save(path, step2.step2_params)
        utils.dictionary_save(path, step3.step3_params)
        utils.dictionary_save(path, post_learning_metrics_val,   'val')
        utils.dictionary_save(path, post_learning_metrics_test1, 'train1')
        utils.dictionary_save(path, post_learning_metrics_test2, 'train2')
        #
        idx_run_loop += 1
        #
        del model
    #


#
# Ici point d'entrée pour ajuster les paramètres ou coder les boucles de variations
#
def execute(dataset_name, dir_npy):
    #
    from train_data_generator import FCTrainDataGenerator
    fcg = FCTrainDataGenerator()
    fcg.load_compute_raw_data_additional_params(dataset_name)
    fcg.load_compute_raw_data()
    #
    step2.step2_params['step2_target_class_col_name'] = 'target_class'
    step2.step2_params['step2_profondeur_analyse'] = 3
    step2.step2_params['step2_target_period'] = 'M15'
    # paramètres spécifiques à 'generate_big_define_target'
    step2.step2_params['step2_symbol_for_target'] = 'UsaInd'
    step2.step2_params['step2_targets_classes_count'] = 3
    step2.step2_params['step2_symbol_spread'] = 2.5
    # step2_params['step2_targetLongShort'] = 20.0
    # step2_params['step2_ratio_coupure'] = 1.3
    # step2_params['step2_use_ATR'] = False
    step2.step2_params['step2_targetLongShort'] = 0.95
    step2.step2_params['step2_ratio_coupure'] = 1.1
    step2.step2_params['step2_use_ATR'] = True
    #
    fcg.compute_target_additional_params(step2.step2_params)
    fcg.compute_target()
    #
    model_manager = ModelManager()
    #
    # step3 parameters : unchanged during loop
    #
    step3.step3_params['step3_column_names_to_scale'] = []
    step3.step3_params['step3_column_names_not_to_scale'] = [
        'UsaInd_M15_time_slot',
        'UsaInd_M15_pRSI_3', 'UsaInd_M15_pRSI_5', 'UsaInd_M15_pRSI_8', 'UsaInd_M15_pRSI_13', 'UsaInd_M15_pRSI_21']
    step3.step3_params['step3_tests_by_class'] = 66
    step3.step3_params['step3_idx_start'] = 0  # step3_idx_start = int(random.random()*1000)
    #
    for step3_recouvrement in (8, 13, 21, 34, 55, 89, 144, 233):   # (2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233):
        for step3_samples_by_class in (330, 660):    # (330, 660, 990, 1320, 1650, 1980):
            #
            # step3 parameters : modified by this loop
            #
            step3.step3_params['step3_recouvrement'] = step3_recouvrement  # 1 / proportion recouvrement
            step3.step3_params['step3_time_depth'] = step3_recouvrement
            step3.step3_params['step3_samples_by_class'] = step3_samples_by_class
            #
            try:
                fcg.compute_learning_data_GRU_LSTM_Conv1D_additional_params(step3.step3_params)
                learning_data = fcg.compute_learning_data_GRU_LSTM_Conv1D()
            except :
                print("fcg.create_step3_data failed. STOP")
                return
            #
            # Model and learning parameters : unchanged during loop
            #
            _mm_dict = model_manager.get_properties()
            #
            _mm_dict['model_architecture'] = 'Conv1D_Dense'
            _mm_dict['conv1D_block1_MaxPooling1D_pool_size'] = 2
            _mm_dict['config_GRU_LSTM_units'] = 128
            _mm_dict['config_Dense_units'] = 96
            _mm_dict['dropout_rate'] = 0.5
            _mm_dict['optimizer_name'] = 'adam'
            _mm_dict['optimizer_modif_learning_rate'] = 0.75
            #
            _mm_dict['fit_batch_size'] = 32
            _mm_dict['fit_epochs_max'] = 500
            _mm_dict['fit_earlystopping_patience'] = 100
            #
            model_manager.update_properties(_mm_dict)
            #
            for conv1D_block1_filters in (55, 89, 144, 233, 377, 610, 987):
                for conv1D_block1_kernel_size in (2, 3, 5):
                    #
                    # Model and learning parameters : modified by this loop
                    #
                    _mm_dict = model_manager.get_properties()
                    #
                    _mm_dict['conv1D_block1_filters'] = conv1D_block1_filters
                    _mm_dict['conv1D_block1_kernel_size'] = conv1D_block1_kernel_size
                    #
                    model_manager.update_properties(_mm_dict)
                    #
                    learn(dataset_name, dir_npy, model_manager, learning_data)
