#
# Copyright (c) 2020 by Frederi CATRIER - All rights reserved.
#

# -----------------------------------------------------------------------------
# Méthodes pour l'analyse des résultats post learning
# -----------------------------------------------------------------------------

import os
import sys

import numpy
import pandas
from sklearn.metrics import confusion_matrix

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

import step2_dataset_prepare_target_data as step2


learning_metrics_template = {'train_loss': None, 'val_loss': None, 'train_accuracy': None, 'val_accuracy': None}

post_learning_metrics_template = {'acc': None, 'res_eval_result_atr': None, 'cm': None, 'pc_resultat': None}

obsolete_metrics_for_backward_compatibility = {
    'model_dropout_rate': None,
    'model_fit_batch_size': None,
    'model_fit_earlystopping_patience': None,
    'model_fit_epochs_max': None,
    'optimizer_modif_learning_rate': None,
    'test1_acc': None,
    'test2_acc': None,
    'val_acc': None,
    'create_model_algo': None,
    'batch_size': None,
    'dropout': None,
    'layer_size': None,
    'layers_count': None,
    'factor': None,
    'config_Conv1D': None,
    'conv1D_filters': None,
    'conv1D_kernel_size': None,
    'conv1D_block2_filters': None,
    'conv1D_block2_kernel_size': None,
    'conv1D_block2_GlobalAveragePooling1D': None,
    'config_LSTM': None,
    'config_Dense': None,
    'acc': None,
    'loss': None,
    'epochs_sum': None,
    'cm': None,
    'result': None,
    'res_eval_result_atr': None,
    'step3_columns': None}


def post_learning_metrics(model, learning_data, train_val_test):
    #
    np_X = learning_data[train_val_test]['np_X']
    df_y_1d = learning_data[train_val_test]['df_y_1d']
    df_atr = learning_data[train_val_test]['df_atr']
    #
    y_pred_raw = model.predict(np_X)  # avec les poids sortie modèle
    df_y_pred = pandas.DataFrame(numpy.argmax(y_pred_raw, axis=-1))  # avec les poids forcés à 0/1
    df_y_pred.index = df_y_1d.index
    #
    acc, res_eval_result_atr, cm, pc_resultat = evaluate_atr(
        df_y_pred, df_y_1d, df_atr,
        step2.step2_params['step2_symbol_spread'], step2.step2_params['step2_ratio_coupure'])
    print("--- results analysis for ", train_val_test)
    print('res_eval_result_atr = ', res_eval_result_atr, '\tpc_resultat =', pc_resultat)
    print('acc = ', acc)
    print(cm)
    cm_metrics(cm)
    print("---")
    #
    result = post_learning_metrics_template.copy()
    result['acc'] = acc
    result['res_eval_result_atr'] = res_eval_result_atr
    result['cm'] = cm
    result['pc_resultat'] = pc_resultat
    return result


def evaluate_not_atr(df_y_pred, df_y_1d_test, df_atr_test, target_long_short, step2_symbol_spread, step2_ratio_coupure):
    cm, test_acc, dfMerged = compute_cm(df_y_pred, df_y_1d_test, df_atr_test)
    res = 0.0
    try:
        gain = target_long_short
        #
        true_1_predicted_1 = cm[0, 0]
        true_1_predicted_2 = cm[0, 1]
        true_1_predicted_3 = cm[0, 2]
        #
        true_2_predicted_1 = cm[1, 0]
        true_2_predicted_2 = cm[1, 1]
        true_2_predicted_3 = cm[1, 2]
        #
        true_3_predicted_1 = cm[2, 0]
        true_3_predicted_2 = cm[2, 1]
        true_3_predicted_3 = cm[2, 2]
        #
        res += true_1_predicted_1 * (+gain - step2_symbol_spread)
        res += true_2_predicted_1 * (-gain * step2_ratio_coupure - step2_symbol_spread)
        res += true_3_predicted_1 * (-step2_symbol_spread)
        #
        res += true_1_predicted_2 * (-gain * step2_ratio_coupure - step2_symbol_spread)
        res += true_2_predicted_2 * (+gain - step2_symbol_spread)
        res += true_3_predicted_2 * (-step2_symbol_spread)
        #
        res += true_1_predicted_3 * 0.0  # predicted 3 = no action
        res += true_2_predicted_3 * 0.0  # predicted 3 = no action
        res += true_3_predicted_3 * 0.0  # predicted 3 = no action
        #
    except:
        pass
    return test_acc, res, cm


def evaluate_atr(df_y_pred, df_y_1d_test, df_atr_test, step2_symbol_spread, step2_ratio_coupure):
    cm, test_acc, dfMerged = compute_cm(df_y_pred, df_y_1d_test, df_atr_test)
    #
    dfMerged_y_pred_0 = dfMerged[(dfMerged['y_pred'] == 0)]
    dfMerged_y_pred_1 = dfMerged[(dfMerged['y_pred'] == 1)]
    dfMerged_y_pred_2 = dfMerged[(dfMerged['y_pred'] == 2)]
    #
    dfMerged_y_pred_0_true_0 = dfMerged_y_pred_0[(dfMerged_y_pred_0['y_true'] == 0)]
    dfMerged_y_pred_0_true_1 = dfMerged_y_pred_0[(dfMerged_y_pred_0['y_true'] == 1)]
    dfMerged_y_pred_0_true_2 = dfMerged_y_pred_0[(dfMerged_y_pred_0['y_true'] == 2)]
    #
    dfMerged_y_pred_1_true_0 = dfMerged_y_pred_1[(dfMerged_y_pred_1['y_true'] == 0)]
    dfMerged_y_pred_1_true_1 = dfMerged_y_pred_1[(dfMerged_y_pred_1['y_true'] == 1)]
    dfMerged_y_pred_1_true_2 = dfMerged_y_pred_1[(dfMerged_y_pred_1['y_true'] == 2)]
    #
    # gains = atr
    #
    gains = dfMerged_y_pred_0_true_0['atr_test'].sum() + dfMerged_y_pred_1_true_1['atr_test'].sum()
    #
    # perte = atr * step2_ratio_coupure
    #     Nota : les prédictions 0 ou 1 et réel 2 sont comptabilisées neutres
    #            (coût = spread, inclus dans coût total spread)
    #
    pertes = (dfMerged_y_pred_0_true_1['atr_test'].sum() + dfMerged_y_pred_1_true_0[
        'atr_test'].sum()) * step2_ratio_coupure
    #
    # cout_total_spread = step2_symbol_spread * nb_operations
    nb_operations = dfMerged_y_pred_0.shape[0] + dfMerged_y_pred_1.shape[0]
    cout_total_spread = step2_symbol_spread * nb_operations
    #
    # resultat
    #
    resultat = gains - pertes - cout_total_spread
    max_possible_resultat = dfMerged['atr_test'].sum() - step2_symbol_spread * cm.sum()
    min_possible_resultat = - dfMerged['atr_test'].sum() * step2_ratio_coupure - step2_symbol_spread * cm.sum()
    pc_resultat = 100.0 * (resultat - min_possible_resultat) / (max_possible_resultat - min_possible_resultat)
    #
    return test_acc, int(resultat), cm, int(pc_resultat)


def compute_cm(df_y_pred, df_y_1d_test, df_atr_test):
    df_y_1d_test_translated_0_2 = df_y_1d_test[0] - 1
    dfMerged = pandas.concat([df_y_pred, df_y_1d_test_translated_0_2, df_atr_test], join='inner', axis=1)
    dfMerged.columns = ['y_pred', 'y_true', 'atr_test']
    #
    y_true = numpy.array(df_y_1d_test_translated_0_2.values)
    y_pred = numpy.array(df_y_pred.values)
    cm = confusion_matrix(y_true, y_pred)
    #
    # numpy.trace(cm) = somme sur la diagonale (prédictions correctes)
    test_acc = round(numpy.trace(cm) / numpy.sum(cm), 2)
    #
    return cm, test_acc, dfMerged


def cm_metrics(cm):
    #
    # sommes par colonnes : analyse des prédictions
    #
    pred_1_count = cm[:, 0].sum()
    pred_2_count = cm[:, 1].sum()
    pred_3_count = cm[:, 2].sum()
    arr_for_std = []
    arr_for_std.append(100.0 * pred_1_count / cm[:].sum())
    arr_for_std.append(100.0 * pred_3_count / cm[:].sum())
    arr_for_std.append(100.0 * pred_3_count / cm[:].sum())
    print("pred_1 =", round(pred_1_count / (pred_1_count + pred_2_count + pred_3_count), 2),
          "\tpred_2 =", round(pred_2_count / (pred_1_count + pred_2_count + pred_3_count), 2),
          "\tpred_3 =", round(pred_3_count / (pred_1_count + pred_2_count + pred_3_count), 2),
          "\tecart-type pred 1/2/3 =", int(numpy.std(arr_for_std)))
    # #
    # # sommes par lignes : analyse des valeurs réelles (ici doit faire step3_tests_by_class)
    # #
    # true_1_count = cm[0].sum()
    # true_2_count = cm[1].sum()
    # true_3_count = cm[2].sum()
    # #
    # # analyse détaillée de chaque cas
    # #
    # true_1_predicted_1 = cm[0,0]
    # true_1_predicted_2 = cm[0,1]
    # true_1_predicted_3 = cm[0,2]
    # #
    # true_2_predicted_1 = cm[1,0]
    # true_2_predicted_2 = cm[1,1]
    # true_2_predicted_3 = cm[1,2]
    # #
    # true_3_predicted_1 = cm[2,0]
    # true_3_predicted_2 = cm[2,1]
    # true_3_predicted_3 = cm[2,2]
    # #
    # pc_predicted_true_class_1 = 0.0
    # pc_predicted_true_class_2 = 0.0
    # pc_predicted_true_class_3 = 0.0
    # ratio_OK_vs_KO_true_1_predicted_2 = 0.0
    # ratio_OK_vs_KO_true_2_predicted_1 = 0.0
    # try:
    # pc_predicted_true_class_1 = round(pred_1_count/true_1_count,2)
    # pc_predicted_true_class_2 = round(pred_2_count/true_2_count,2)
    # pc_predicted_true_class_3 = round(pred_3_count/true_3_count,2)
    # #
    # ratio_OK_vs_KO_true_1_predicted_2 = round(true_1_predicted_1/true_1_predicted_2,2)
    # ratio_OK_vs_KO_true_2_predicted_1 = round(true_2_predicted_2/true_2_predicted_1,2)
    # except:
    # pass
    # #
    # print("   pc_predicted_true_class_1 =",pc_predicted_true_class_1)
    # print("   pc_predicted_true_class_2 =",pc_predicted_true_class_2)
    # print("   pc_predicted_true_class_3 =",pc_predicted_true_class_3)
    # print("   ratio_OK_vs_KO_true_1_predicted_2 =",ratio_OK_vs_KO_true_1_predicted_2, \
    # "(erreur true_1_predicted_2=",true_1_predicted_2, \
    # " vs ok true_1_predicted_1=", true_1_predicted_1,")")
    # print("   ratio_OK_vs_KO_true_2_predicted_1 =",ratio_OK_vs_KO_true_2_predicted_1, \
    # "(erreur true_2_predicted_1=",true_2_predicted_1, \
    # " vs ok true_2_predicted_2=", true_2_predicted_2,")")
    # #
