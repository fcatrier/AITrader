#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import arbo
import learn_evaluate_results
import learn_history
import step2_dataset_prepare_target_data as step2
import step3_dataset_prepare_learning_input_data as step3
import utils


#
# entry point
#
def execute(dataset_name, dir_npy):
    #
    from train_data_generator import FCTrainDataGenerator
    data_generator = FCTrainDataGenerator()
    #
    from model_manager import ModelManager
    model_manager = ModelManager()
    #
    loop_step_raw_data(dataset_name, dir_npy, data_generator, model_manager)


def loop_step_raw_data(dataset_name, dir_npy, data_generator, model_manager):
    #
    data_generator.load_compute_raw_data_additional_params(dataset_name)
    data_generator.load_compute_raw_data()
    #
    # (iterate on) next step
    #
    loop_step_target(dataset_name, dir_npy, data_generator, model_manager):


def loop_step_target(dataset_name, dir_npy, data_generator, model_manager):
    #
    # step2 parameters : unchanged during loop
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
    data_generator.compute_target_additional_params(step2.step2_params)
    data_generator.compute_target()
    #
    # (iterate on) next step
    #
    loop_learning_data(dataset_name, dir_npy, data_generator, model_manager):


def loop_learning_data(dataset_name, dir_npy, data_generator, model_manager):
    #
    # step3 parameters : unchanged during loop
    #
    step3.step3_params['step3_column_names_to_scale'] = []
    step3.step3_params['step3_column_names_not_to_scale'] = [
        'UsaInd_M15_time_slot',
        'UsaInd_M15_pRSI_3', 'UsaInd_M15_pRSI_5', 'UsaInd_M15_pRSI_8', 'UsaInd_M15_pRSI_13', 'UsaInd_M15_pRSI_21']
    step3.step3_params['step3_tests_by_class'] = 66
    step3.step3_params['step3_idx_start'] = 0  # may be step3_idx_start = int(random.random()*1000)
    #
    # step3 parameters : modified during loop + (iterate on) next step
    #
    #
    for step3_time_depth in (2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233):
        for step3_samples_by_class in (330, 660):    # (330, 660, 990, 1320, 1650, 1980):
            #
            # step3 parameters : modified by this loop
            #
            step3.step3_params['step3_time_depth'] = step3_time_depth
            step3.step3_params['step3_recouvrement'] = step3_time_depth
            step3.step3_params['step3_samples_by_class'] = step3_samples_by_class
            #
            try:
                data_generator.compute_learning_data_GRU_LSTM_Conv1D_additional_params(step3.step3_params)
                learning_data = data_generator.compute_learning_data_GRU_LSTM_Conv1D()
                #
                # Setting of last model manager properties provided by data generation
                model_manager.update_properties_from_learning_data(self, learning_data)
                #
            except:
                print("data_generator.create_step3_data failed. STOP")
                return
            #
            loop_model(dataset_name, dir_npy, data_generator, model_manager)


def loop_model(dataset_name, dir_npy, data_generator, model_manager):
    #
    # Model and learning parameters : unchanged during loop
    #
    mm_dict = model_manager.get_properties()
    #
    mm_dict['model_architecture'] = 'Conv1D_Dense'
    mm_dict['conv1D_block1_MaxPooling1D_pool_size'] = 0
    mm_dict['config_GRU_LSTM_units'] = 128
    mm_dict['config_Dense_units'] = 96
    mm_dict['dropout_rate'] = 0.5
    mm_dict['optimizer_name'] = 'adam'
    mm_dict['optimizer_modif_learning_rate'] = 0.75
    #
    mm_dict['fit_batch_size'] = 32
    mm_dict['fit_epochs_max'] = 500
    mm_dict['fit_earlystopping_patience'] = 100
    #
    model_manager.update_properties(mm_dict)
    #
    #
    # Model and learning parameters : unchanged during loop
    #
    for conv1D_block1_filters in (21, 55, 89, 144, 233, 377, 610, 987):
        for conv1D_block1_kernel_size in (2, 3, 5, 8):
            #
            if conv1D_block1_kernel_size >= step3_time_depth:
                continue
            #
            for config_Dense_units in (13, 21, 34, 55, 89):
                #
                # Model and learning parameters : modified by this loop
                #
                mm_dict = model_manager.get_properties()
                #
                mm_dict['conv1D_block1_filters'] = conv1D_block1_filters
                mm_dict['conv1D_block1_kernel_size'] = conv1D_block1_kernel_size
                mm_dict['config_Dense_units'] = config_Dense_units
                #
                model_manager.update_properties(mm_dict)
                #
                learn(dataset_name, dir_npy, model_manager, learning_data)


def learn(dataset_name, dir_npy, model_manager, learning_data, loops_count=1):
    #
    npy_path = arbo.npy_path(dataset_name, dir_npy)
    idx_run_loop = learn_history.new_npy_idx(npy_path)
    npy_path_with_prefix = arbo.npy_path_with_prefix(dataset_name, dir_npy, idx_run_loop)
    #
    for i in range(0, loops_count):
        #
        print("")
        print("-----------------------------------------------------------")
        print("Repeat #", i + 1, " /", loops_count)
        print("-----------------------------------------------------------")
        print("")
        #
        model = model_manager.create_compile_model()
        #
        if i == 0:  # affiché uniquement au premier passage pour désaturer l'affichage
            model_manager.display_model_info(model)
        #
        learning_metrics_this_fit = model_manager.fit(learning_data)
        #
        utils.dictionary_save(npy_path_with_prefix, step2.step2_params)
        utils.dictionary_save(npy_path_with_prefix, step3.step3_params)
        utils.dictionary_save(npy_path_with_prefix, model_manager.get_properties())
        utils.dictionary_save(npy_path_with_prefix, learning_metrics_this_fit)
        #
        # calcul et sauvagarde des métriques post apprentissage uniquement si
        #  l'apprentissage a été un minimum concluant
        #
        if learning_metrics_this_fit['val_accuracy'] >= 0.5:
            for part_of_dataset in ('val', 'test1', 'test2'):
                post_learning_metrics = learn_evaluate_results.post_learning_metrics(model, learning_data, part_of_dataset)
                utils.dictionary_save(npy_path_with_prefix, post_learning_metrics, part_of_dataset)
        #
        idx_run_loop += 1
        #
        del model