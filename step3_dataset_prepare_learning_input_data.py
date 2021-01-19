#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

#
# TODO
#
#  6 - ajouter informations (voir données dans XL Cockpit)
#        par exemple : - pics et creux
#                      - seuils symboliques
#                      - xMA
#


"""




interface à minima

train_data, target_data


conventions (usuelles dans les samples et les codes sur le Web)
    X = données d'apprentissage (X majuscule)
    y = données cible (y minuscule)

conventions Frédéri
    préfixe np = numpy.array
    préfixe df = pandas.dataframe


np_learn_X, df_y_Nd, df_y_1d


supposons que l'on ait défini un jeu d'apprentissage simple :

date    close       target


alors à minima
    X = close
    y = target

mais pour les LSTM/GRU et Conv1D on a besoin de données d'apprentissage sous la forme 
d'un historique de X pour chaque y

         |-   (X0,X1,X2)      y0
n        |
samples  |    (X1,X2,X3)      y1
         |    ...
         |-   (Xn,Xn+1,Xn+2)  Yn

illustration pour un exemple un peu plus complexe :

données = date, O, H, L, C, rsi14, target

alors

X = O, H, L, C, rsi14
y = target

une fois formaté pour LSTM/GRU ou Conv1D :

( (O_0, H_0, L_0, C_0, rsi14_0),
  (O_1, H_1, L_1, C_1, rsi14_1),
  (O_2, H_2, L_2, C_2, rsi14_2) )       y0

( (O_1, H_1, L_1, C_1, rsi14_1),
  (O_2, H_2, L_2, C_2, rsi14_2),
  (O_3, H_3, L_3, C_3, rsi14_3) )       y1

...

=> c'est pour cela que j'utilise un tableau numpy pour les X.

sa dimension est :

profondeur            ( (nombre de colonnes),
d'historique            (nombre de colonnes),
injecté pour 1 sample   (nombre de colonnes),
4 ici par exp.          (nombre de colonnes) )       y0






"""

# python

import os
import sys

import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

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


import arbo
import step2_dataset_prepare_target_data as step2


step3_params = {
    'step3_column_names_to_scale' : [],
    'step3_column_names_not_to_scale' : [],
    'step3_idx_start' : 0,
    'step3_recouvrement' : 0,
    'step3_samples_by_class' : 0,
    'step3_tests_by_class' : 0,
    'step3_time_depth' : 0 }


def get_param_list():
    param_list = []
    for key, key_value in step3_params.items():
        param_list.append(key)
    return param_list


# -----------------------------------------------------------------------------
# Définition des fonctions
# -----------------------------------------------------------------------------



def generate_learning_data_dynamic_scale_column(df,column):
   toReshape = numpy.array(df[column]).reshape(-1, 1)
   scaler = MinMaxScaler()
   scaler.fit(toReshape)
   reshaped=scaler.transform(toReshape)
   df[column] = reshaped
   return df



def generate_learning_data_dynamic3(big_for_target, idx_start, recouvrement, samples_by_class, time_depth, column_names_to_scale, column_names_not_to_scale):
   #
   np_X_arr_test            = []
   target_arr_test          = []
   date_index_arr_test      = []
   atr_arr_test             = []
   #
   target1_count = 0
   target2_count = 0
   target3_count = 0
   #
   target1_test_count = 0
   target2_test_count = 0
   target3_test_count = 0
   #
   column_names = []
   for col in column_names_to_scale:
      column_names.append(col)
   #
   for col in column_names_not_to_scale:
      column_names.append(col)
   #
   column_names.append('target_class')
   column_names.append('UsaInd_M15_ATR_13')
   #
   dfTarget = big_for_target.copy()
   dfTarget = dfTarget.sort_index(ascending=False)
   #
   idx = time_depth + idx_start
   idx_max = len(dfTarget)-1
   step = max(1, time_depth // recouvrement)  # max ici pour éviter step = 0
   print("idx_step=",step)
   max_records = (idx_max-idx) // step
   approximate_max_samples_by_class = max_records // 3
   if(approximate_max_samples_by_class<=samples_by_class):
      print("approximate_max_samples_by_class<=samples_by_class, approximate_max_samples_by_class=",approximate_max_samples_by_class)
      return False, 0,0,0
   #
   # !!! : à partir d'ici on est avec indice 0 = le plus récent et idx_max = le plus ancien
   #
   while( (idx <= idx_max) &
          ( (target1_count<samples_by_class) or
            (target2_count<samples_by_class) or
            (target3_count<samples_by_class) ) ):
      #
      dfExtract = dfTarget[column_names][idx-time_depth:idx]
      target = dfExtract['target_class'][0]
      #
      if( ((target==1) & (target1_count<samples_by_class)) or
          ((target==2) & (target2_count<samples_by_class)) or
          ((target==3) & (target3_count<samples_by_class)) ):
         #
         # rescaling sur l'intervalle des colonnes le nécessitant
         #
         for col in column_names_to_scale:
            dfExtract = generate_learning_data_dynamic_scale_column(dfExtract,col)
         #
         # mémorisation avant suppression de la colonne 'target_class' (pour debug)
         #input_df_arr.append(dfExtract)
         #
         target = dfExtract['target_class'     ][0]
         dt     = dfExtract.index[0]
         atr    = dfExtract['UsaInd_M15_ATR_13'][0]
         dfExtract = dfExtract.drop(['target_class'],axis=1)
         #
         if( (target==1) & (target1_test_count<samples_by_class) ):
            np_X_arr_test.append(numpy.array(dfExtract))
            target_arr_test.append(target)
            date_index_arr_test.append(dt)
            atr_arr_test.append(atr)
            #
            target1_test_count += 1
            target1_count += 1
         elif( (target==2) & (target2_test_count<samples_by_class) ):
            np_X_arr_test.append(numpy.array(dfExtract))
            target_arr_test.append(target)
            date_index_arr_test.append(dt)
            atr_arr_test.append(atr)
            #
            target2_test_count += 1
            target2_count += 1
         elif( (target==3) & (target3_test_count<samples_by_class) ):
            np_X_arr_test.append(numpy.array(dfExtract))
            target_arr_test.append(target)
            date_index_arr_test.append(dt)
            atr_arr_test.append(atr)
            #
            target3_test_count += 1
            target3_count += 1
      #
      idx += step
   #
   print("target1 : ",target1_count,"(",round(target1_count/(target1_count+target2_count+target3_count),2),"%)",
         "target2 : ",target2_count,"(",round(target2_count/(target1_count+target2_count+target3_count),2),"%)",
         "target3 : ",target3_count,"(",round(target3_count/(target1_count+target2_count+target3_count),2),"%)")
   #
   np_X_test      = numpy.array(np_X_arr_test)
   #
   df_y_1d_test       = pandas.DataFrame(target_arr_test)
   #
   df_date_index_test      = pandas.DataFrame(date_index_arr_test)
   df_atr_test             = pandas.DataFrame(atr_arr_test)
   #
   df_y_Nd_test = pandas.DataFrame()
   for col_value in range(df_y_1d_test[0].min(),df_y_1d_test[0].max()+1):
      df_y_Nd_test[col_value]=1*(df_y_1d_test[0]==col_value)
   #
   # remise des dates en tant qu'index (pour faciliter le debug et les échanges/tests avec MT5)
   #
   df_y_1d_test['DateTime_tmp_idx']=df_date_index_test
   df_y_1d_test.index=df_y_1d_test['DateTime_tmp_idx']
   df_y_1d_test = df_y_1d_test.drop(['DateTime_tmp_idx' ],axis=1)
   #
   df_atr_test['DateTime_tmp_idx']=df_date_index_test
   df_atr_test.index=df_atr_test['DateTime_tmp_idx']
   df_atr_test = df_atr_test.drop(['DateTime_tmp_idx' ],axis=1)
   #
   df_y_Nd_test['DateTime_tmp_idx']=df_date_index_test
   df_y_Nd_test.index=df_y_Nd_test['DateTime_tmp_idx']
   df_y_Nd_test = df_y_Nd_test.drop(['DateTime_tmp_idx' ],axis=1)
   #
   resOK = ( (target1_count+target2_count+target3_count) == (3*samples_by_class) )
   #
   return resOK, idx, np_X_test, df_y_Nd_test, df_y_1d_test, df_atr_test




"""

Préparation des données d'entrée de l'apprentissage sur la base d'un dataset
contenant les target_class (LONG / SHORT / OUT et éventuellement LONGSHORT)

Attendu en entrée : fichier bix_XX avec une colonne target_xxx (nom de cette
   colonne à mettre dans la variable target_class_col_name) présente et définie

   Paramètres à prendre en compte

      paramètres injectés dans l'apprentissage (OHLC, OHLC en HA, d'autres à voir
      par exemple RSI ou autre indicateur sur la base de centiles)
         => voir la relation entre les paramètres injectés et les hyperparamètres du réseau)
         
      profondeur de l'historique injecté (remis à plat pour chaque date pour
      fournir une donnée complète)
         => voir la relation entre l'ajout de profondeur et les hyperparamètres du réseau)
      
      scaling

         regroupement de paramètres qui doivent être "rescalé" ensembles (par défaut
         le scaling fonctionne sur une seule colonne)
         
         algo de scaling


"""

# def get_df(symbol,period):
   # if(symbol=='Usa500'):
      # if(period=='H1'):
         # return Usa500_dfH1
      # elif(period=='M15'):
         # return Usa500_dfM15
      # elif(period=='M5'):
         # return Usa500_dfM5
      # elif(period=='M1'):
         # return Usa500_dfM1
   # elif(symbol=='UsaInd'):
      # if(period=='H1'):
         # return UsaInd_dfH1
      # elif(period=='M15'):
         # return UsaInd_dfM15
      # elif(period=='M5'):
         # return UsaInd_dfM5
      # elif(period=='M1'):
         # return UsaInd_dfM1
   # elif(symbol=='UsaTec'):
      # if(period=='H1'):
         # return UsaTec_dfH1
      # elif(period=='M15'):
         # return UsaTec_dfM15
      # elif(period=='M5'):
         # return UsaTec_dfM5
      # elif(period=='M1'):
         # return UsaTec_dfM1
   # elif(symbol=='Ger30'):
      # if(period=='H1'):
         # return Ger30_dfH1
      # elif(period=='M15'):
         # return Ger30_dfM15
      # elif(period=='M5'):
         # return Ger30_dfM5
      # elif(period=='M1'):
         # return Ger30_dfM1


def extract_columns(all_df_bigs,symbol,periods,dateJoin,time_depth,column_set):
   #
   #print('IN extract_columns(',symbol,periods,dateJoin,time_depth,column_set)
   #
   dfExtracts = []
   columns_to_drop = []
   needRescaling = True
   #
   for period in periods:
      big_Pn=step2.get_big2(all_df_bigs,period)
      dfJoinPn = big_Pn[(big_Pn['DateTime']<=dateJoin)]
      dfExtractJoinPn=dfJoinPn[len(dfJoinPn)-time_depth:len(dfJoinPn)]
      #print("DEBUG extract_columns",dfExtractJoinPn)
      mycolumnsPn, needRescaling, columns_to_drop = get_columns_to_extract(column_set,symbol,period)
      dfExtractJoinPn = dfExtractJoinPn[mycolumnsPn]
      if(len(dfExtractJoinPn)<time_depth):
         print('Pn',period, 'manque données')
         return False,0,0,0
      dfExtracts.append(dfExtractJoinPn)
   #
   return True, dfExtracts, needRescaling, columns_to_drop


def get_columns_to_extract(column_set,symbol,period):
   #print('IN get_columns_to_extract',column_set,symbol,period)
   needRescaling=True
   base_columns = []
   base_columns_to_drop = []
   # column_set 1 : hour, minute to check that M5, M15 and H1 are consistent
   if(column_set==1):     
      base_columns.append('hour')
      base_columns.append('minute')
      needRescaling=False
   # column_set 1 : OHLC et OHLC en HA pour normalisation données OHLC
   elif(column_set==2):
      base_columns.append('time_slot')
      needRescaling=False
   elif(column_set==3):
      base_columns.append('Open_'    +period)
      base_columns.append('High_'    +period)
      base_columns.append('Low_'     +period)
      base_columns.append('Close_'   +period)
      base_columns.append('HA_Open_' +period)
      base_columns.append('HA_High_' +period)
      base_columns.append('HA_Low_'  +period)
      base_columns.append('HA_Close_'+period)
   # column_set 3 : OHLC pour normalisation données OHLC
   elif(column_set==4):
      base_columns.append('Open_'    +period)
      base_columns.append('High_'    +period)
      base_columns.append('Low_'     +period)
      base_columns.append('Close_'   +period)
   # column_set 4 : OHLC en HA pour normalisation données OHLC
   elif(column_set==5):
      base_columns.append('HA_Open_' +period)
      base_columns.append('HA_High_' +period)
      base_columns.append('HA_Low_'  +period)
      base_columns.append('HA_Close_'+period)
   #
   elif(column_set==6):
      base_columns.append('class_vs_pivot_H4')
      base_columns.append('class_vs_pivot_D1')
      needRescaling=False
   #
   elif(column_set==7):
      base_columns.append('pRSI')
      needRescaling=False
   #
   elif(column_set==8):
      base_columns.append('pBOLL')
      needRescaling=False
   #
   elif(column_set==9):
      base_columns.append('pADX')
      needRescaling=False
   #
   elif(column_set==10):
      base_columns.append('pCCI')
      needRescaling=False
   #
   elif(column_set==11):
      base_columns.append('pMassIndex')
      needRescaling=False
   #
   elif(column_set==12):
      base_columns.append('pATR')
      needRescaling=False
   #
   elif(column_set==13):
      base_columns.append('pUltimateOscillator')
      needRescaling=False
   #
   elif(column_set==14):
      base_columns.append('pWilliamsRIndicator')
      needRescaling=False
   #
   elif(column_set==20):
      base_columns.append('plogClose_' +period+'_1')
      needRescaling=False
   #
   elif(column_set==21):
      base_columns.append('plogClose_' +period+'_2')
      needRescaling=False
   #
   elif(column_set==22):
      base_columns.append('plogClose_' +period+'_3')
      needRescaling=False
   #
   elif(column_set==23):
      base_columns.append('plogClose_' +period+'_5')
      needRescaling=False
   #
   elif(column_set==24):
      base_columns.append('plogClose_' +period+'_8')
      needRescaling=False
   #
   elif(column_set==25):
      base_columns.append('plogClose_' +period+'_13')
      needRescaling=False
   #
   elif(column_set==26):
      base_columns.append('plogClose_' +period+'_21')
      needRescaling=False
   #
   elif(column_set==100):   #debug OHLC non rescalé
      base_columns.append('Open_'    +period)
      base_columns.append('High_'    +period)
      base_columns.append('Low_'     +period)
      base_columns.append('Close_'   +period)
      needRescaling=False
   #
   elif(column_set==101):   #debug Close non rescalé
      base_columns.append('Close_'   +period)
      needRescaling=False
   #
   columns = []
   columns_to_drop = []
   #
   for column in base_columns:
      columns.append(symbol+'_'+period+'_'+column)
   #
   for column_to_drop in base_columns_to_drop:
      columns_to_drop.append(symbol+'_'+period+'_'+column_to_drop)
   #
   #print('OUT get_columns_to_extract',columns, needRescaling, columns_to_drop)
   return columns, needRescaling, columns_to_drop


def clean_idx_generate_scaled_ohlc(dfLearn,symbol):
   columns_to_drop = []
   columns_to_drop.append('idx2')
   columns_to_drop.append(symbol+'_idx')
   columns_to_drop.append(symbol+'_idx_x')
   columns_to_drop.append(symbol+'_idx_y')
   #
   dfLearn = step2.secure_drop_columns(dfLearn,columns_to_drop)
   #
   return dfLearn


def merge(parts_to_merge):
   #
   #print("IN merge",len(parts_to_merge))
   #for i in range(0,len(parts_to_merge)):
   #   print(parts_to_merge[i])
   #print("IN merge -- fin")
   #
   parts_to_merge_count = len(parts_to_merge)
   #
   if(parts_to_merge_count==1):
      merge_result = parts_to_merge[0]
   #
   elif(parts_to_merge_count==2):
      merge_result = merge_parts(parts_to_merge[0], parts_to_merge[1])
   #
   elif(parts_to_merge_count>2):
      merge_result = merge_parts(parts_to_merge[0], parts_to_merge[1])
      for i in range(2,len(parts_to_merge)):
         merge_result = merge_parts(merge_result, parts_to_merge[i])
   #
   else:
      merge_result = []
   #
   return merge_result


def merge_parts(part1,part2):
   #
   copy_part1 = part1.copy()
   copy_part1.insert(1, 'idx_merge_part', range(0, len(copy_part1)))
   copy_part1
   #
   copy_part2 = part2.copy()
   copy_part2.insert(1, 'idx_merge_part', range(0, len(copy_part2)))
   copy_part2
   # 4/12 : ajout de left_index=True pour conserver l'index DateTime et pouvoir trier à la fin en
   #        ordre inverse pour les entrées TimeSeries
   #merged = pandas.merge(left=copy_part1, right=copy_part2, left_on='idx_merge_part', right_on='idx_merge_part')
   merged = pandas.merge(left=copy_part1, right=copy_part2, left_on='idx_merge_part', right_on='idx_merge_part',left_index=True)
   merged = merged.drop('idx_merge_part',axis=1)
   merged
   #
   return merged


def flatten_one_raw_scaled_ohlc_KO(target_class_value,date_join,to_flatten,time_depth,dfLearn_columns,target_class_col_name):
   flatten_raw = numpy.array(to_flatten)
   flatten_raw = flatten_raw.reshape(1,-1)
   df_flatten_raw = pandas.DataFrame(flatten_raw)
   # ajout colonne target
   df_flatten_raw.insert(len(df_flatten_raw.columns),target_class_col_name,target_class_value)
   df_flatten_raw.insert(len(df_flatten_raw.columns),'date_join',date_join)
   #
   df_flatten_raw.columns = dfLearn_columns
   return df_flatten_raw

#ICI en cours, pas utile pour LSTM mais toujours KO pour dfLearn
def flatten_one_raw_scaled_ohlc(target_class_value,date_join,to_flatten,time_depth,dfLearn_columns,target_class_col_name):
   #
   flatten_raws = []
   flatten_raws_1D = []
   for idx in range(0,len(to_flatten)):
      flatten_raw = []
      for col in to_flatten.columns:
         partial_to_flatten = to_flatten[col][idx]
         flatten_raw.append(partial_to_flatten)
         flatten_raws_1D.append(partial_to_flatten)
      flatten_raws.append(flatten_raw)
   #
   #np_flatten_raws = numpy.array(flatten_raws)
   df_flatten_raws    = pandas.DataFrame(flatten_raws)
   df_flatten_raws_1D = pandas.DataFrame(flatten_raws_1D)
   #
   # ajout colonne target
   df_flatten_raws_1D.insert(len(df_flatten_raws_1D.columns),target_class_col_name,target_class_value)
   df_flatten_raws_1D.insert(len(df_flatten_raws_1D.columns),'date_join',date_join)
   #
   return df_flatten_raws_1D


def check_hour_minute_3_periods_v2(symbol,periods,dfExtracts):
   #
   periods_count=len(periods)
   idx=0
   check_hour   = True
   check_minute = True
   hour_for_check = 0
   minute_for_check = 0
   #
   hour_M1  = -1
   hour_M5  = -1
   hour_M15 = -1
   hour_M30 = -1
   hour_H1  = -1
   #
   minute_M1  = -1
   minute_M5  = -1
   minute_M15 = -1
   minute_M30 = -1
   #
   for period in periods:
      if(period=='M1'):
         dfExtractJoinM1 = dfExtracts[idx]
         hour_M1   = dfExtractJoinM1 [symbol+'_'+'M1'+'_hour'  ].tail(1)[0]
         minute_M1 = dfExtractJoinM1 [symbol+'_'+'M1'+'_minute'].tail(1)[0]
         if(idx==0):
            hour_for_check = hour_M1
         else:
            check_hour &= (hour_M1 == hour_for_check)
         idx += 1
      elif(period=='M5'):
         dfExtractJoinM5 = dfExtracts[idx]
         hour_M5   = dfExtractJoinM5 [symbol+'_'+'M5'+'_hour'  ].tail(1)[0]
         minute_M5 = dfExtractJoinM5 [symbol+'_'+'M5'+'_minute'].tail(1)[0]
         if(idx==0):
            hour_for_check = hour_M5
         else:
            check_hour &= (hour_M5 == hour_for_check)
         idx += 1
      elif(period=='M15'):
         dfExtractJoinM15 = dfExtracts[idx]
         hour_M15   = dfExtractJoinM15[symbol+'_'+'M15'+'_hour'  ].tail(1)[0]
         minute_M15 = dfExtractJoinM15[symbol+'_'+'M15'+'_minute'].tail(1)[0]
         if(idx==0):
            hour_for_check = hour_M15
         else:
            check_hour &= (hour_M15 == hour_for_check)
         idx += 1
      elif(period=='M30'):
         dfExtractJoinM30 = dfExtracts[idx]
         hour_M30   = dfExtractJoinM30[symbol+'_'+'M30'+'_hour'  ].tail(1)[0]
         minute_M30 = dfExtractJoinM30[symbol+'_'+'M30'+'_minute'].tail(1)[0]
         if(idx==0):
            hour_for_check = hour_M30
         else:
            check_hour &= (hour_M30 == hour_for_check)
         idx += 1
      elif(period=='H1'):
         dfExtractJoinH1 = dfExtracts[idx]
         hour_H1   = dfExtractJoinH1 [symbol+'_'+ 'H1'+'_hour'  ].tail(1)[0]
         if(idx==0):
            hour_for_check = hour_H1
         else:
            check_hour &= (hour_H1 == hour_for_check)
         idx += 1
   #
   if(periods_count==1):
      return True
   #
   if(minute_M30!=-1):
      if(minute_M15!=-1):
         check_minute &= ((minute_M15//30)*30)==minute_M30
      if(minute_M5!=-1):
         check_minute &= ((minute_M5 //30)*30)==minute_M30
      if(minute_M1!=-1):
         check_minute &= ((minute_M1 //30)*30)==minute_M30
   if(minute_M15!=-1):
      if(minute_M5!=-1):
         check_minute &= ((minute_M5//15)*15)==minute_M15
      if(minute_M1!=-1):
         check_minute &= ((minute_M1//15)*15)==minute_M15
   #
   if(check_hour==False):
      print("check_hour==False",hour_for_check,hour_H1,hour_M30,hour_M15,hour_M5,hour_M1)
   if(check_minute==False):
      print("check_minute==False",minute_M1,minute_M5,minute_M15,minute_M30)
   #
   return check_hour & check_minute


def prepare_dfLearn(all_df_bigs,big_for_target,target_class_col_name,symbols_set,periods,time_depth,samples_by_class,shuffle_in_data,column_sets,old_to_recent,fix_0412):
   #
   dfLearn, dfLearn_columns = prepare_dfLearn_columns(all_df_bigs,big_for_target,symbols_set,periods,time_depth,column_sets,target_class_col_name,fix_0412)
   #
   #shuffle ici sinon les samples seront extraits de la date la plus ancienne à la plus récente
   big_for_target_shuffled = big_for_target.copy()
   if(shuffle_in_data==True):
      big_for_target_shuffled = shuffle(big_for_target_shuffled)
   #
   min_target_class = big_for_target_shuffled[target_class_col_name].min()
   max_target_class = big_for_target_shuffled[target_class_col_name].max()
   for class_id in range(min_target_class,max_target_class+1):
      count_for_class=0
      if(old_to_recent):
         idx=0
      else:
         idx=len(big_for_target_shuffled)-1
      #
      while((count_for_class<samples_by_class)&(idx<len(big_for_target_shuffled))):
         if(big_for_target_shuffled[target_class_col_name][idx]==class_id):
            dateJoin=big_for_target_shuffled['DateTime'][idx]
            #dateJoin
            resOK, merged_M5_M15_H1_rescaled, merged_M5_M15_H1 = \
               generate_scaled_ohlc_v3(all_df_bigs,symbols_set,periods,dateJoin,time_depth,column_sets,fix_0412)
            if resOK:
               target_class_value=big_for_target_shuffled[target_class_col_name][idx]
               flattened_merged_M5_M15_H1_rescaled = \
                     flatten_one_raw_scaled_ohlc(target_class_value,
                                                dateJoin,
                                                merged_M5_M15_H1_rescaled,
                                                time_depth,
                                                dfLearn_columns,target_class_col_name)
               dfLearn = dfLearn.append(flattened_merged_M5_M15_H1_rescaled)
               count_for_class += 1
               print(symbols_set,'\t','class:',class_id,'\t','count_for_class: (',samples_by_class,')',count_for_class)
            else:
               print(symbols_set,'\t','erreur : données manquantes')
         #
         if(old_to_recent):
            idx +=1
         else:
            idx -=1
   #
   dfLearn.insert(1, 'idx',range( 0, len(dfLearn)))
   dfLearn.index = dfLearn['idx']
   dfLearn = dfLearn.drop(['idx'],axis=1)
   dfLearn
   #
   # 04/12 : ajout d'un sort sur le résultat final pour ne pas être regroupé par classes
   #         mais par dates descendantes
   dfLearn = dfLearn.sort_values(by=['date_join'],ascending=False)
   #
   print(pandas.value_counts(dfLearn[target_class_col_name]))
   #
   return dfLearn


def prepare_XYlearn_LSTM2(all_df_bigs,big_for_target,target_class_col_name,symbols_set,periods,time_depth,samples_by_class,column_sets,old_to_recent,fix_0412):
   #
   arr_learn_X   = []
   arr_learn_Y   = []
   arr_date_join = []
   #
   big_for_target = big_for_target.copy()
   #
   min_target_class = big_for_target[target_class_col_name].min()
   max_target_class = big_for_target[target_class_col_name].max()
   classes_count = max_target_class - min_target_class + 1
   #
   # arr_classes_count = []
   # for class_id in range(min_target_class,max_target_class+1):
      # arr_classes_count.append(0)
   #
   #
   # parcours recent -> ancien pour extraire les données les plus récentes
   #
   for class_id in range(min_target_class,max_target_class+1):
      count_for_class=0
      if(old_to_recent):
         idx=0
      else:
         idx=len(big_for_target)-1
      #
      while((count_for_class<samples_by_class)&(idx<len(big_for_target))&(idx>=0)):
         if(big_for_target[target_class_col_name][idx]==class_id):
            dateJoin=big_for_target['DateTime'][idx]
            #
            resOK, merged_M5_M15_H1_rescaled, merged_M5_M15_H1 = \
               generate_scaled_ohlc_v3(all_df_bigs,symbols_set,periods,dateJoin,time_depth,column_sets,fix_0412)
            if resOK:
               # ajout colonne target
               arr_learn_Y.append(class_id)
               arr_learn_X.append(merged_M5_M15_H1_rescaled)
               arr_date_join.append(dateJoin)
               #
               count_for_class += 1
               print(symbols_set,'\t','class:',class_id,'\t','count_for_class: (',samples_by_class,')',count_for_class)
            else:
               print(symbols_set,'\t','erreur : données manquantes')
         #
         if(old_to_recent):
            idx +=1
         else:
            idx -=1
   #
   # remise des dates old -> recent (LSTM)
   #
   arr_learn_X_old_to_recent   = []
   arr_learn_Y_old_to_recent   = []
   arr_date_join_old_to_recent = []
   idx_max = len(arr_learn_Y)-1
   for i in range(idx_max,-1,-1):
      arr_learn_X_old_to_recent.append  (arr_learn_X  [i])
      arr_learn_Y_old_to_recent.append  (arr_learn_Y  [i])
      arr_date_join_old_to_recent.append(arr_date_join[i])
   #
   # formatage des sorties
   #
   np_learn_X = numpy.array(arr_learn_X_old_to_recent)     # en numpy
   #
   df_y_1d = pandas.DataFrame(arr_learn_Y_old_to_recent)   # en df
   cols = []
   cols.append(target_class_col_name)
   df_y_1d.columns = cols
   df_y_1d
   #
   df_y_Nd = pandas.DataFrame()              #en df
   for col_value in range(min(arr_learn_Y),max(arr_learn_Y)+1):
      df_y_Nd[target_class_col_name+'_'+str(col_value)]=1*(df_y_1d[target_class_col_name]==col_value)
   #
   df_date_join = pandas.DataFrame(arr_date_join_old_to_recent)
   return np_learn_X, df_y_1d, df_y_Nd, df_date_join


def prepare_XYlearn_LSTM (all_df_bigs,big_for_target,target_class_col_name,symbols_set,periods,time_depth,samples_by_class,column_sets,fix_0412):
   #
   arr_learn_X   = []
   arr_learn_Y   = []
   arr_date_join = []
   #
   big_for_target = big_for_target.copy()
   #
   min_target_class = big_for_target[target_class_col_name].min()
   max_target_class = big_for_target[target_class_col_name].max()
   classes_count = max_target_class - min_target_class + 1
   #
   arr_classes_count = []
   for class_id in range(min_target_class,max_target_class+1):
      arr_classes_count.append(0)
   #
   idx=len(big_for_target)-1
   #
   # parcours recent -> ancien pour extraire les données les plus récentes
   #
   while(min(arr_classes_count)<samples_by_class):
      #
      cur_class_id = big_for_target[target_class_col_name][idx]
      dateJoin     = big_for_target['DateTime'][idx]
      #
      # on génère exactement le nombre données par classes demandées
      if(arr_classes_count[cur_class_id-min_target_class]<samples_by_class):
         #
         resOK, merged_M5_M15_H1_rescaled, merged_M5_M15_H1 = \
            generate_scaled_ohlc_v3(all_df_bigs,symbols_set,periods,dateJoin,time_depth,column_sets,fix_0412)
         if resOK:
            # ajout colonne target
            arr_learn_Y.append(cur_class_id)
            arr_learn_X.append(merged_M5_M15_H1_rescaled)
            arr_date_join.append(dateJoin)
            #
            arr_classes_count[cur_class_id-min_target_class] += 1
            #
            print(symbols_set,'\t',min(arr_classes_count),'/',samples_by_class)
         else:
            print(symbols_set,'\t','erreur : données manquantes')
         #
      idx -= 1
   #
   # remise des dates old -> recent (LSTM)
   #
   # arr_learn_X_old_to_recent   = []
   # arr_learn_Y_old_to_recent   = []
   # arr_date_join_old_to_recent = []
   # idx_max = len(arr_learn_Y)-1
   # for i in range(idx_max,-1,-1):
      # arr_learn_X_old_to_recent.append  (arr_learn_X  [i])
      # arr_learn_Y_old_to_recent.append  (arr_learn_Y  [i])
      # arr_date_join_old_to_recent.append(arr_date_join[i])
   #
   # formatage des sorties
   #
   #np_learn_X = numpy.array(arr_learn_X_old_to_recent)     # en numpy
   np_learn_X = numpy.array(arr_learn_X)     # en numpy
   #
   #df_y_1d = pandas.DataFrame(arr_learn_Y_old_to_recent)   # en df
   df_y_1d = pandas.DataFrame(arr_learn_Y)   # en df
   cols = []
   cols.append(target_class_col_name)
   df_y_1d.columns = cols
   df_y_1d
   #
   df_y_Nd = pandas.DataFrame()              #en df
   for col_value in range(min(arr_learn_Y),max(arr_learn_Y)+1):
      df_y_Nd[target_class_col_name+'_'+str(col_value)]=1*(df_y_1d[target_class_col_name]==col_value)
   #
   #df_date_join = pandas.DataFrame(arr_date_join_old_to_recent)
   df_date_join = pandas.DataFrame(arr_date_join)
   return np_learn_X, df_y_1d, df_y_Nd, df_date_join


def prepare_dfLearn_columns(all_df_bigs,big_for_target,symbols_set,periods,time_depth,column_sets,target_class_col_name,fix_0412):
   # préparation de la structure : on génère une donnée avec la dernière date pour récupérer les colonnes
   # et créer le dataframe d'apprentissage (dfLearn) qui sera ensuite complété ligne par ligne
   lastDate=big_for_target['DateTime'][len(big_for_target)-1]
   dateJoin=lastDate
   #
   isOK, partial_merge_rescaled, sample_to_be_flatten = generate_scaled_ohlc_v3(all_df_bigs,symbols_set,periods,dateJoin,time_depth,column_sets,fix_0412)
   #sample_to_be_flatten = step2.secure_drop_columns(sample_to_be_flatten,columns_to_drop)
   #
   dfLearn = create_empty_dataframe_learn(sample_to_be_flatten,time_depth,target_class_col_name)
   #dfLearn_columns = dfLearn.columns
   return dfLearn, dfLearn.columns

def generate_scaled_ohlc_v3(all_df_bigs,symbols_set,periods,dateJoin,time_depth,column_sets,fix_0412):
   #print(">>> 1")
   check_all_symbols = True
   for symbol in symbols_set:
      # 1 - extrait 3 périodes avec column_set = 1 (idx,hour,minute) pour vérifier cohérence sur les 3 périodes 
      resOK, dfExtracts, needRescaling, columns_to_drop = extract_columns(all_df_bigs,symbol,periods,dateJoin,time_depth,1)
      #print(">>> 1.1")
      if(columns_to_drop!=0):
         dfExtracts = step2.secure_drop_columns(dfExtracts,columns_to_drop)
      #print(">>> 1.2")
      # 2 - vérification cohérence sur les 3 périodes
      check_all_symbols &= resOK
      if(check_all_symbols==True):
         check_all_symbols &= check_hour_minute_3_periods_v2(symbol,periods,dfExtracts)
      #print(">>> 1.3")
   #
   if(check_all_symbols==False):
      print('check_all_symbols==False')
      return False,0,0
   #
   #print(">>> 2")
   parts_to_merge = []
   parts_to_merge_rescaled = []
   for symbol in symbols_set:
      for column_set in column_sets:
         resOK, dfExtracts, needRescaling, columns_to_drop = extract_columns(all_df_bigs,symbol,periods,dateJoin,time_depth,column_set)
         #print(">>> 3")
         merged_P1_P2_P3 = merge(dfExtracts)
         #print(">>> 4")
         merged_P1_P2_P3 = clean_idx_generate_scaled_ohlc(merged_P1_P2_P3,symbol)
         merged_P1_P2_P3 = step2.secure_drop_columns(merged_P1_P2_P3,columns_to_drop)
         #
         if(needRescaling==True):
            merged_P1_P2_P3_rescaled = scale_dataset(merged_P1_P2_P3,time_depth)
            merged_P1_P2_P3_rescaled
         else:
            merged_P1_P2_P3_rescaled = merged_P1_P2_P3
         #
         parts_to_merge.append(merged_P1_P2_P3)
         parts_to_merge_rescaled.append(merged_P1_P2_P3_rescaled)
   #
   #print(">>> 5")
   partial_merge = merge(parts_to_merge)
   #print(">>> 6")
   partial_merge_rescaled = merge(parts_to_merge_rescaled)
   # 4/12 : inversion de l'index pour avoir le plus récent en premier pour les TimeSeries
   if(fix_0412==True):
      partial_merge = partial_merge.sort_index(ascending=False)
      partial_merge_rescaled = partial_merge_rescaled.sort_index(ascending=False)
   #
   return True, partial_merge_rescaled, partial_merge


def scale_dataset(dataset_to_scale,time_depth):
   #
   dataset_to_scale_forScaler = []
   for i in range(0,time_depth):
      for col in dataset_to_scale.columns:
         dataset_to_scale_forScaler.append(dataset_to_scale[col][i])
   #
   dataset_to_scale_forScaler = numpy.array(dataset_to_scale_forScaler)
   dataset_to_scale_forScaler.shape
   dataset_to_scale_forScaler = dataset_to_scale_forScaler.reshape(-1,1)
   dataset_to_scale_forScaler
   dataset_to_scale_forScaler.shape
   theMin = dataset_to_scale_forScaler.min()
   theMax = dataset_to_scale_forScaler.max()
   #
   scaler = MinMaxScaler()
   dataset_to_scale_scaled = scaler.fit_transform(dataset_to_scale_forScaler)
   dataset_to_scale_scaled
   dataset_to_scale_scaled.shape
   #
   dataset_to_scale_forScaler = dataset_to_scale_forScaler.reshape(-1,len(dataset_to_scale.columns))
   dataset_to_scale_forScaler
   dataset_to_scale_forScaler.shape
   #
   dataset_to_scale_scaled = dataset_to_scale_scaled.reshape(-1,len(dataset_to_scale.columns))
   dataset_to_scale_scaled
   dataset_to_scale_scaled.shape
   # remise sous forme de pandas.Dataframe
   colNames = dataset_to_scale.columns
   dataset_to_scale = pandas.DataFrame(dataset_to_scale_forScaler)
   dataset_to_scale.columns = colNames
   dataset_to_scale
   #
   dataset_to_scale_rescaled = pandas.DataFrame(dataset_to_scale_scaled)
   dataset_to_scale_rescaled.columns = colNames
   dataset_to_scale_rescaled
   #
   return dataset_to_scale_rescaled


#
# Création d'un dataframe qui sera injecté en entrée du réseau
#   -> mise à plat du dataframe multi-ut et de profondeur time_step
#
# def create_empty_dataframe_learn(merged_M5_M15_H1,time_depth,target_class_col_name):
   # empty_dfLearn = pandas.DataFrame()
   # colNames = merged_M5_M15_H1.columns
   # for time_step in range(0,time_depth):
      # if(time_step==0):
         # postfix=''
      # else:
         # postfix='_'+str(time_step)
      # #
      # for i in range(0,len(colNames)):
         # empty_dfLearn.insert(len(empty_dfLearn.columns),colNames[i]+postfix,0)
   # #
   # empty_dfLearn.insert(len(empty_dfLearn.columns),target_class_col_name,0)
   # empty_dfLearn.insert(len(empty_dfLearn.columns),'date_join',0)
   # return empty_dfLearn

def create_empty_dataframe_learn(merged_M5_M15_H1,time_depth,target_class_col_name):
   empty_dfLearn = pandas.DataFrame()
   colNames = merged_M5_M15_H1.columns
   for i in range(0,len(colNames)):
      for time_step in range(0,time_depth):
         if(time_step==0):
            postfix=''
         else:
            postfix='_'+str(time_step)
         #
         empty_dfLearn.insert(len(empty_dfLearn.columns),colNames[i]+postfix,0)
   #
   empty_dfLearn.insert(len(empty_dfLearn.columns),target_class_col_name,0)
   empty_dfLearn.insert(len(empty_dfLearn.columns),'date_join',0)
   return empty_dfLearn


def generate_all_symbols_sets(all_symbols,symbols_count):
   if(symbols_count==1):
      all_symbols_set = []
      for symbol in all_symbols:
         symbols=[]
         symbols.append(symbol)
         all_symbols_set.append(symbols)
      #
      return all_symbols_set
   elif(symbols_count==2):
      return generate_all_symbols_set_pair(all_symbols)
   elif(symbols_count==3):
      return generate_all_symbols_set_trio(all_symbols)


def generate_all_symbols_set_pair(all_symbols):
   all_symbols_pair = []
   for i in range(0,len(all_symbols)):
      for j in range(i+1,len(all_symbols)):
         # print(all_symbols[i],'\t',all_symbols[j])
         cur_symbol_pair = []
         cur_symbol_pair.append(all_symbols[i])
         cur_symbol_pair.append(all_symbols[j])
         all_symbols_pair.append(cur_symbol_pair)
   #
   return all_symbols_pair


def generate_all_symbols_set_trio(all_symbols):
   all_symbols_trio = []
   for i in range(0,len(all_symbols)):
      for j in range(i+1,len(all_symbols)):
         for k in range(j+1,len(all_symbols)):
            # print(all_symbols[i],'\t',all_symbols[j],'\t',all_symbols[k])
            cur_symbol_trio = []
            cur_symbol_trio.append(all_symbols[i])
            cur_symbol_trio.append(all_symbols[j])
            cur_symbol_trio.append(all_symbols[k])
            all_symbols_trio.append(cur_symbol_trio)
   #
   return all_symbols_trio


# TU symbols = ['Usa500','UsaInd','UsaTec','Ger30']
# TU for i in (1,2,3):
# TU    print('--',i)
# TU    symbols_sets = generate_all_symbols_sets(symbols,i)
# TU    for symbols_set in symbols_sets:
# TU       print(symbols_set)
# TU       for symbol in symbols_set:
# TU        print(symbol)


def dfLearn_from_file(dataset_name,target_period,profondeur_analyse,spread_x,ratio_potentiels,symbol_for_target,targetLongShort,ratio_coupure,symbols_set,time_depth,samples_by_class,shuffle_in_data,periods,column_sets,old_to_recent):
   # sauvegarde
   filename = get_dfLearn_filename(target_period,profondeur_analyse,spread_x,ratio_potentiels,
               #symbol_for_target,targetLongShort,ratio_coupure,
               symbols_set,time_depth,samples_by_class,shuffle_in_data,periods,column_sets,old_to_recent)
   #
   dfLearn = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename, index_col=0)
   #
   return dfLearn


############################




def check_file_exists(full_filename,ext):
   here = False
   if(ext=='npy'):
      try:
         numpy.load(full_filename)
         here = True
      except FileNotFoundError:
         here = False
   elif(ext=='csv'):
      try:
         pandas.read_csv(full_filename)
         here = True
      except FileNotFoundError:
         here = False
   #
   return here

#
# Génération des fichiers (non LSTM)
#



def get_dfLearn_filename_old(target_period,profondeur_analyse,spread_x,ratio_potentiels,symbols_set,time_depth,samples_by_class,shuffle_in_data,periods,column_sets,old_to_recent):
   filename ='dfLearn_'
   filename+='target_period='
   filename+=target_period
   filename+='_profondeur_analyse='
   filename+=str(profondeur_analyse)
   filename+='_spread_x='
   filename+=str(spread_x)
   filename+='_ratio_potentiels='
   filename+=str(ratio_potentiels)
   filename+='_(B)_symbols_set='
   first_symbol=True
   for symbol in symbols_set:
      if(first_symbol==True):
         first_symbol=False
      else:
         filename+='_'
      #
      filename+=symbol
   #
   filename+='_time_depth='
   filename+=str(time_depth)
   filename+='_samples_by_class='
   filename+=str(samples_by_class)
   filename+='_shuffle_in_data='
   filename+=str(shuffle_in_data)
   #
   filename+='_periods='
   first_period=True
   for period in periods:
      if(first_period==True):
         first_period=False
      else:
         filename+='_'
      filename+=period
   #
   filename+='_column_sets='
   first_column_set = True
   for column_set in column_sets:
      if(first_column_set==True):
         first_column_set=False
      else:
         filename+='_'
      filename+=str(column_set)
   #
   filename+='_old_to_recent='
   filename+=str(old_to_recent)
   filename+='.csv'
   return filename


def get_dfLearn_filename(
      target_period,profondeur_analyse,
      spread_x,ratio_potentiels,
      #symbol_for_target, targetLongShort, ratio_coupure,
      symbols_set,time_depth,samples_by_class,shuffle_in_data,periods,column_sets,old_to_recent):
   # entête
   filename ='learn_type=not_LSTM.'
   # step 1 : pas de paraètre
   # step 2 : paramètres communs
   filename+='target_period='
   filename+=target_period
   filename+='.profondeur_analyse='
   filename+=str(profondeur_analyse)
   # step 2 : paramètres génération par potentiel
   filename+='.spread_x='
   filename+='{:.2f}'.format(spread_x)
   filename+='.ratio_potentiels='
   filename+='{:.2f}'.format(ratio_potentiels)
   # step 2 : paramètres génération par target
   # filename+='.symbol_for_target='
   # filename+=symbol_for_target
   # filename+='.targetLongShort='
   # filename+='{:.2f}'.format(targetLongShort)
   # filename+='.ratio_coupure='
   # filename+='{:.2f}'.format(ratio_coupure)
   #step 3
   filename+='.symbols_set='
   first_symbol=True
   for symbol in symbols_set:
      if(first_symbol==True):
         first_symbol=False
      else:
         filename+='_'
      #
      filename+=symbol
   #
   filename+='.time_depth='
   filename+=str(time_depth)
   filename+='.samples_by_class='
   filename+=str(samples_by_class)
   filename+='.shuffle_in_data='
   filename+=str(shuffle_in_data)
   #
   filename+='.periods='
   first_period=True
   for period in periods:
      if(first_period==True):
         first_period=False
      else:
         filename+='_'
      filename+=period
   #
   filename+='.column_sets='
   first_column_set = True
   for column_set in column_sets:
      if(first_column_set==True):
         first_column_set=False
      else:
         filename+='-'
      filename+=str(column_set)
   #
   filename+='.old_to_recent='
   filename+=str(old_to_recent)
   filename+='.csv'
   return filename


import shutil

def rename_not_LSTM(dataset_name,target_period,profondeur_analyse,spread_x,ratio_potentiels,symbol_for_target,targetLongShort,ratio_coupure,symbols_set,time_depth,samples_by_class,periods,column_sets,shuffle_in_data,old_to_recent):
   filename_renameFrom = get_dfLearn_filename_old(
      target_period,
      profondeur_analyse,
      spread_x,
      ratio_potentiels,
      #symbol_for_target,targetLongShort,ratio_coupure,
      symbols_set,
      time_depth,
      samples_by_class,
      shuffle_in_data,
      periods,
      column_sets,
      old_to_recent)
   print(filename_renameFrom)
   check_exists_filename = check_file_exists(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_renameFrom,'csv')
   if(check_exists_filename==True):
      print("Fichier existant OK, renommage")
      filename_renameTo = get_dfLearn_filename(
         target_period,
         profondeur_analyse,
         spread_x,
         ratio_potentiels,
         #symbol_for_target,targetLongShort,ratio_coupure,
         symbols_set,
         time_depth,
         samples_by_class,
         shuffle_in_data,
         periods,
         column_sets,
         old_to_recent)
      shutil.copyfile(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_renameFrom, arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_renameTo)
   else:
      print("Fichier NON exitant, pas de renommage")


def generate_not_LSTM(filenames_list_not_LSTM,dataset_name,all_df_bigs,
      big_for_target,target_class_col_name,target_period,profondeur_analyse,
      spread_x,ratio_potentiels,symbols_set,time_depth,samples_by_class,periods,column_sets,shuffle_in_data,old_to_recent):
   #
   fileset = []
   #
   filename = get_dfLearn_filename(
      target_period,
      profondeur_analyse,
      spread_x,
      ratio_potentiels,
      #symbol_for_target,targetLongShort,ratio_coupure,
      symbols_set,
      time_depth,
      samples_by_class,
      shuffle_in_data,
      periods,
      column_sets,
      old_to_recent)
   #
   check_exists_filename = check_file_exists(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename,'csv')
   if(check_exists_filename==True):
      print("Fichier déjà existant")
   else:
      print("Génération du fichier")
      #
      # génération
      #
      dfLearn = prepare_dfLearn(
         all_df_bigs,
         big_for_target,
         target_class_col_name,
         symbols_set,
         periods,
         time_depth,
         samples_by_class,
         shuffle_in_data,
         column_sets,
         old_to_recent)
      #
      # sauvegarde
      #
      dfLearn.to_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename)
   #
   filenames_list_not_LSTM.append(filename)
   return filenames_list_not_LSTM




############################


def get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,contents):
   if(contents=='np_learn_X'):
      filename='np_learn_X'
   elif(contents=='df_y_1d'):
      filename='df_y_1d'
   elif(contents=='df_y_Nd'):
      filename='df_y_Nd'
   elif(contents=='df_date_join'):
      filename='df_date_join'
   filename+='_'
   filename+='target_period='
   filename+=target_period
   filename+='_profondeur_analyse='
   filename+=str(profondeur_analyse)
   filename+='_(B)_symbols_set='
   first_symbol=True
   for symbol in symbols_set:
      if(first_symbol==True):
         first_symbol=False
      else:
         filename+='_'
      #
      filename+=symbol
   #
   filename+='_time_depth='
   filename+=str(time_depth)
   filename+='_samples_by_class='
   filename+=str(samples_by_class)
   #
   filename+='_periods='
   first_period=True
   for period in periods:
      if(first_period==True):
         first_period=False
      else:
         filename+='_'
      filename+=period
   #
   filename+='_column_sets='
   first_column_set = True
   for column_set in column_sets:
      if(first_column_set==True):
         first_column_set=False
      else:
         filename+='_'
      filename+=str(column_set)
   #
   if(contents=='np_learn_X'):
      filename+='.npy'
   else:
      filename+='.csv'
   #
   return filename


def save_learn_files_LSTM(dataset_name,np_learn_X,df_y_1d,df_y_Nd,df_date_join,
   target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets):
   #
   filename_np_learn_X = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'np_learn_X')
   numpy.save(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_np_learn_X, np_learn_X)
   #
   filename_df_y_1d = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_y_1d')
   df_y_1d.to_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_y_1d)
   #
   filename_df_y_Nd = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_y_Nd')
   df_y_Nd.to_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_y_Nd)
   #
   filename_df_date_join = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_date_join')
   df_date_join.to_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_date_join)
   #
   return filename_np_learn_X, filename_df_y_1d, filename_df_y_Nd, filename_df_date_join


def get_filenames_LSTM(dataset_name,target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets):
   #
   filename_np_learn_X = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'np_learn_X')
   #
   filename_df_y_1d = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_y_1d')
   #
   filename_df_y_Nd = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_y_Nd')
   #
   filename_df_date_join = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_date_join')
   #
   return filename_np_learn_X, filename_df_y_1d, filename_df_y_Nd, filename_df_date_join


def load_learn_from_files_LSTM(dataset_name,target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets):
   #
   filename_np_learn_X = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'np_learn_X')
   np_learn_X = numpy.load(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_np_learn_X)
   #
   filename_df_y_1d = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_y_1d')
   df_y_1d = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_y_1d, index_col=0)
   #
   filename_df_y_Nd = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_y_Nd')
   df_y_Nd = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_y_Nd, index_col=0)
   #
   filename_df_date_join = get_dfLearn_filename_LSTM(target_period,profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,'df_date_join')
   df_date_join = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_date_join, index_col=0)
   #
   return np_learn_X, df_y_1d, df_y_Nd, df_date_join


def load_learn_from_fileset_LSTM(dataset_name,fileset):
   #
   filename_np_learn_X = fileset[0]
   np_learn_X = numpy.load(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_np_learn_X)
   #
   filename_df_y_1d = fileset[1]
   df_y_1d = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_y_1d, index_col=0)
   #
   filename_df_y_Nd = fileset[2]
   df_y_Nd = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_y_Nd, index_col=0)
   #
   filename_df_date_join = fileset[3]
   df_date_join = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_df_date_join, index_col=0)
   #
   return np_learn_X, df_y_1d, df_y_Nd, df_date_join


#
# Génération des fichiers LSTM
#

def generate_LSTM(filenames_list_LSTM,dataset_name,target_period,all_df_bigs,big_for_target,target_class_col_name,
                  profondeur_analyse,symbols_set,time_depth,samples_by_class,periods,column_sets,fix_0412):
   #
   fileset = []
   #
   filename_to_check = get_dfLearn_filename_LSTM(
         target_period,
         profondeur_analyse,
         symbols_set,
         time_depth,
         samples_by_class,
         periods,
         column_sets,'np_learn_X')
   check_exists_filename_np_learn_X = check_file_exists(arbo.get_study_dir(py_dir,dataset_name)+'\\'+filename_to_check,'npy')
   if(check_exists_filename_np_learn_X==True):
      print("Fichier déjà existant")
      #
      filename_np_learn_X, filename_df_y_1d, filename_df_y_Nd, filename_df_date_join = \
         get_filenames_LSTM(dataset_name,
                                  target_period,   
                                  profondeur_analyse,
                                  symbols_set,
                                  time_depth,
                                  samples_by_class,
                                  periods,
                                  column_sets)
      #
      fileset.append(filename_np_learn_X)
      fileset.append(filename_df_y_1d)
      fileset.append(filename_df_y_Nd)
      fileset.append(filename_df_date_join)
   else:
      print("Génération du fichier")
      np_learn_X, df_y_1d, df_y_Nd, df_date_join = prepare_XYlearn_LSTM(
         all_df_bigs,
         big_for_target,
         target_class_col_name,
         symbols_set,
         periods,
         time_depth,
         samples_by_class,
         column_sets,
         fix_0412)
      #
      # sauvegarde
      #
      fileset = []
      #
      filename_np_learn_X, filename_df_y_1d, filename_df_y_Nd, filename_df_date_join = \
         save_learn_files_LSTM(
                  dataset_name,
                  np_learn_X,df_y_1d,df_y_Nd,df_date_join,
                  target_period,
                  profondeur_analyse,
                  symbols_set,
                  time_depth,
                  samples_by_class,
                  periods,
                  column_sets)
      #
      fileset.append(filename_np_learn_X)
      fileset.append(filename_df_y_1d)
      fileset.append(filename_df_y_Nd)
      fileset.append(filename_df_date_join)
   #
   filenames_list_LSTM.append(fileset)
   return filenames_list_LSTM




"""
# processus de génération : big_XX est utilisé pour fournir des dates pour lesquelles
#           on a la certitude d'avoir des valeurs pour tous les symboles, garanti par le 
#           big = pandas.concat([copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30],join='inner',axis=1)
"""


# def generate_learning_data_dynamic(big_for_target, recouvrement, samples_by_class, time_depth, column_names_to_scale, column_names_not_to_scale):
   # #
   # np_learn_X_arr = []
   # target_arr     = []
   # date_index_arr = []
   # #
   # target1_count = 0
   # target2_count = 0
   # target3_count = 0
   # #
   # column_names = []
   # for col in column_names_to_scale:
      # column_names.append(col)
   # #
   # for col in column_names_not_to_scale:
      # column_names.append(col)
   # #
   # column_names.append('target_class')
   # #
   # dfTarget = big_for_target.copy()
   # dfTarget = dfTarget.sort_index(ascending=False)
   # #
   # idx = time_depth
   # idx_max = len(dfTarget)-1
   # step = time_depth // recouvrement
   # max_records = (idx_max-idx) // step
   # approximate_max_samples_by_class = max_records // 3
   # if(approximate_max_samples_by_class<=samples_by_class):
      # print("approximate_max_samples_by_class<=samples_by_class, approximate_max_samples_by_class=",approximate_max_samples_by_class)
      # return False, 0,0,0
   # #
   # # !!! : à partir d'ici on est avec indice 0 = le plus récent et idx_max = le plus ancien
   # #
   # while( (idx <= idx_max) &
          # ( (target1_count<samples_by_class) or
            # (target2_count<samples_by_class) or
            # (target3_count<samples_by_class) ) ):
      # #
      # dfExtract = dfTarget[column_names][idx-time_depth:idx]
      # target = dfExtract['target_class'][0]
      # date_index_arr.append(dfExtract.index[0])
      # #
      # if( ((target==1) & (target1_count<samples_by_class)) or
          # ((target==2) & (target2_count<samples_by_class)) or
          # ((target==3) & (target3_count<samples_by_class)) ):
         # #
         # # rescaling sur l'intervalle des colonnes le nécessitant
         # #
         # for col in column_names_to_scale:
            # dfExtract = generate_learning_data_dynamic_scale_column(dfExtract,col)
         # #
         # # mémorisation avant suppression de la colonne 'target_class' (pour debug)
         # #input_df_arr.append(dfExtract)
         # #
         # target = dfExtract['target_class'][0]
         # dfExtract = dfExtract.drop(['target_class'],axis=1)
         # np_learn_X_arr.append(numpy.array(dfExtract))
         # #
         # target_arr.append(target)
         # if(target==1):
            # target1_count += 1
         # elif(target==2):
            # target2_count += 1
         # elif(target==3):
            # target3_count += 1
         # else:
            # print("!!!")
      # #
      # idx += ( time_depth // recouvrement )
   # #
   # print("target1 : ",target1_count,"(",round(target1_count/(target1_count+target2_count+target3_count),2),"%)",
         # "target2 : ",target2_count,"(",round(target2_count/(target1_count+target2_count+target3_count),2),"%)",
         # "target3 : ",target3_count,"(",round(target3_count/(target1_count+target2_count+target3_count),2),"%)")
   # #
   # np_learn_X    = numpy.array(np_learn_X_arr)
   # df_y_1d       = pandas.DataFrame(target_arr)
   # df_date_index = pandas.DataFrame(date_index_arr)
   # #
   # df_y_Nd = pandas.DataFrame()
   # for col_value in range(df_y_1d[0].min(),df_y_1d[0].max()+1):
      # df_y_Nd[col_value]=1*(df_y_1d[0]==col_value)
   # #
   # # remise des dates en tant qu'index (pour faciliter le debug et les échanges/tests avec MT5)
   # #
   # df_y_1d['DateTime_tmp_idx']=df_date_index
   # df_y_1d.index=df_y_1d['DateTime_tmp_idx']
   # df_y_1d = df_y_1d.drop(['DateTime_tmp_idx' ],axis=1)
   # #
   # df_y_Nd['DateTime_tmp_idx']=df_date_index
   # df_y_Nd.index=df_y_Nd['DateTime_tmp_idx']
   # df_y_Nd = df_y_Nd.drop(['DateTime_tmp_idx' ],axis=1)
   # #
   # return True, np_learn_X, df_y_Nd, df_y_1d



# def generate_learning_data_dynamic2(big_for_target, idx_start, recouvrement, samples_by_class, tests_by_class, time_depth, column_names_to_scale, column_names_not_to_scale):
   # #
   # np_X_arr_test            = []
   # np_X_arr_train_val       = []
   # target_arr_test          = []
   # target_arr_train_val     = []
   # date_index_arr_test      = []
   # date_index_arr_train_val = []
   # atr_arr_test             = []
   # #
   # target1_count = 0
   # target2_count = 0
   # target3_count = 0
   # #
   # target1_test_count = 0
   # target2_test_count = 0
   # target3_test_count = 0
   # #
   # column_names = []
   # for col in column_names_to_scale:
      # column_names.append(col)
   # #
   # for col in column_names_not_to_scale:
      # column_names.append(col)
   # #
   # column_names.append('target_class')
   # column_names.append('UsaInd_M15_ATR_13')
   # #
   # dfTarget = big_for_target.copy()
   # dfTarget = dfTarget.sort_index(ascending=False)
   # #
   # idx = time_depth + idx_start
   # idx_max = len(dfTarget)-1
   # step = time_depth // recouvrement
   # print("idx_step=",step)
   # max_records = (idx_max-idx) // step
   # approximate_max_samples_by_class = max_records // 3
   # if(approximate_max_samples_by_class<=samples_by_class):
      # print("approximate_max_samples_by_class<=samples_by_class, approximate_max_samples_by_class=",approximate_max_samples_by_class)
      # return False, 0,0,0
   # #
   # # !!! : à partir d'ici on est avec indice 0 = le plus récent et idx_max = le plus ancien
   # #
   # while( (idx <= idx_max) &
          # ( (target1_count<samples_by_class) or
            # (target2_count<samples_by_class) or
            # (target3_count<samples_by_class) ) ):
      # #
      # dfExtract = dfTarget[column_names][idx-time_depth:idx]
      # target = dfExtract['target_class'][0]
      # #
      # if( ((target==1) & (target1_count<samples_by_class)) or
          # ((target==2) & (target2_count<samples_by_class)) or
          # ((target==3) & (target3_count<samples_by_class)) ):
         # #
         # # rescaling sur l'intervalle des colonnes le nécessitant
         # #
         # for col in column_names_to_scale:
            # dfExtract = generate_learning_data_dynamic_scale_column(dfExtract,col)
         # #
         # # mémorisation avant suppression de la colonne 'target_class' (pour debug)
         # #input_df_arr.append(dfExtract)
         # #
         # target = dfExtract['target_class'     ][0]
         # dt     = dfExtract.index[0]
         # atr    = dfExtract['UsaInd_M15_ATR_13'][0]
         # dfExtract = dfExtract.drop(['target_class'],axis=1)
         # #
         # if( (target==1) & (target1_test_count<tests_by_class) ):
            # np_X_arr_test.append(numpy.array(dfExtract))
            # target_arr_test.append(target)
            # date_index_arr_test.append(dt)
            # atr_arr_test.append(atr)
            # #
            # target1_test_count += 1
            # target1_count += 1
         # elif( (target==2) & (target2_test_count<tests_by_class) ):
            # np_X_arr_test.append(numpy.array(dfExtract))
            # target_arr_test.append(target)
            # date_index_arr_test.append(dt)
            # atr_arr_test.append(atr)
            # #
            # target2_test_count += 1
            # target2_count += 1
         # elif( (target==3) & (target3_test_count<tests_by_class) ):
            # np_X_arr_test.append(numpy.array(dfExtract))
            # target_arr_test.append(target)
            # date_index_arr_test.append(dt)
            # atr_arr_test.append(atr)
            # #
            # target3_test_count += 1
            # target3_count += 1
         # else:
            # np_X_arr_train_val.append(numpy.array(dfExtract))
            # target_arr_train_val.append(target)
            # date_index_arr_train_val.append(dt)
            # #
            # if(target==1):
               # target1_count += 1
            # elif(target==2):
               # target2_count += 1
            # elif(target==3):
               # target3_count += 1
            # else:
               # print("!!!")
      # #
      # idx += ( time_depth // recouvrement )
   # #
   # print("target1 : ",target1_count,"(",round(target1_count/(target1_count+target2_count+target3_count),2),"%)",
         # "target2 : ",target2_count,"(",round(target2_count/(target1_count+target2_count+target3_count),2),"%)",
         # "target3 : ",target3_count,"(",round(target3_count/(target1_count+target2_count+target3_count),2),"%)")
   # #
   # np_X_test      = numpy.array(np_X_arr_test)
   # np_X_train_val = numpy.array(np_X_arr_train_val)
   # #
   # df_y_1d_test       = pandas.DataFrame(target_arr_test)
   # df_y_1d_train_val  = pandas.DataFrame(target_arr_train_val)
   # #
   # df_date_index_test      = pandas.DataFrame(date_index_arr_test)
   # df_atr_test             = pandas.DataFrame(atr_arr_test)
   # df_date_index_train_val = pandas.DataFrame(date_index_arr_train_val)
   # #
   # df_y_Nd_test = pandas.DataFrame()
   # for col_value in range(df_y_1d_test[0].min(),df_y_1d_test[0].max()+1):
      # df_y_Nd_test[col_value]=1*(df_y_1d_test[0]==col_value)
   # #
   # df_y_Nd_train_val = pandas.DataFrame()
   # for col_value in range(df_y_1d_train_val[0].min(),df_y_1d_train_val[0].max()+1):
      # df_y_Nd_train_val[col_value]=1*(df_y_1d_train_val[0]==col_value)
   # #
   # # remise des dates en tant qu'index (pour faciliter le debug et les échanges/tests avec MT5)
   # #
   # df_y_1d_test['DateTime_tmp_idx']=df_date_index_test
   # df_y_1d_test.index=df_y_1d_test['DateTime_tmp_idx']
   # df_y_1d_test = df_y_1d_test.drop(['DateTime_tmp_idx' ],axis=1)
   # #
   # df_atr_test['DateTime_tmp_idx']=df_date_index_test
   # df_atr_test.index=df_atr_test['DateTime_tmp_idx']
   # df_atr_test = df_atr_test.drop(['DateTime_tmp_idx' ],axis=1)
   # #
   # df_y_1d_train_val['DateTime_tmp_idx']=df_date_index_train_val
   # df_y_1d_train_val.index=df_y_1d_train_val['DateTime_tmp_idx']
   # df_y_1d_train_val = df_y_1d_train_val.drop(['DateTime_tmp_idx' ],axis=1)
   # #
   # df_y_Nd_test['DateTime_tmp_idx']=df_date_index_test
   # df_y_Nd_test.index=df_y_Nd_test['DateTime_tmp_idx']
   # df_y_Nd_test = df_y_Nd_test.drop(['DateTime_tmp_idx' ],axis=1)
   # #
   # df_y_Nd_train_val['DateTime_tmp_idx']=df_date_index_train_val
   # df_y_Nd_train_val.index=df_y_Nd_train_val['DateTime_tmp_idx']
   # df_y_Nd_train_val = df_y_Nd_train_val.drop(['DateTime_tmp_idx' ],axis=1)
   # #
   # resOK = ( (target1_count+target2_count+target3_count) == (3*samples_by_class) )
   # #
   # return resOK, np_X_test,    np_X_train_val,    \
                 # df_y_Nd_test, df_y_Nd_train_val, \
                 # df_y_1d_test, df_y_1d_train_val, \
                 # df_atr_test


# # -----------------------------------------------------------------------------
# # Scripts de génération du dfLearn
# # -----------------------------------------------------------------------------

# #
# # Batch production -- start
# #
# os.system('cls')

# # variables à reprendre de "2_dataset_prepare_target_data"
# target_class_col_name = 'target_class'

# # variables
# symbols = ['UsaInd','UsaTec','Ger30','Usa500']
# periods = ['M15','H1']

# column_sets = [2]

# shuffle_in_data  = False
# samples_by_class = 1000
# symbols_sets_symbols_count = 1

# old_to_recent = True

# all_generated_dfLearn = []

# symbols_sets = generate_all_symbols_sets(symbols,symbols_sets_symbols_count)

# for symbols_set in symbols_sets:
   # print("MAIN : ",symbols_sets)
   # for time_depth in (2,3,5,8):
      # #time_depth=2
      # dfLearn = prepare_dfLearn(big_for_target,target_class_col_name,symbols_set,periods,time_depth,samples_by_class,shuffle_in_data,column_sets,old_to_recent)
      # dfLearn
      # all_generated_dfLearn.append(dfLearn)
      # # sauvegarde
      # filename ='\\dfLearn_M15_profondeur_analyse=3_spread_x=4_ratio=2_(B)_'
      # filename+='symbols_set='
      # first_symbol=True
      # for symbol in symbols_set:
         # if(first_symbol==True):
            # first_symbol=False
         # else:
            # filename+='_'
         # filename+=symbol
      # filename+='_time_depth='+str(time_depth)
      # filename+='_samples_by_class='+str(samples_by_class)
      # filename+='_shuffle_in_data='+str(shuffle_in_data)
      # filename+='_M15_H1.csv'
      # filename
      # dfLearn.to_csv(arbo.get_study_dir(py_dir,dataset_name)+filename)


# for i in range(0,len(all_generated_dfLearn)):
   # print(all_generated_dfLearn[i].columns)
   # print(all_generated_dfLearn[i])


#
# Batch production -- end
#


#################

# ### dev ici
# big_for_target['UsaInd_RSI'].tail(10)

# column_sets = [2,5]
# symbols = ['Usa500'] #,'UsaInd','UsaTec','Ger30']
# #for i in (1,2,3):
# i=1
# print('--',i)
# all_symbols_set = generate_all_symbols_sets(symbols,i)
# print("all_symbols_set:",all_symbols_set)
# for symbols_set in all_symbols_set:
   # print("avant appel:",symbols_set)
   # dfLearn, dfLearn.columns = prepare_dfLearn_columns(all_df_bigs,big_M5,symbols_set,periods,time_depth,column_sets,target_class_col_name)
   # print(dfLearn.columns)

# ###fin dev ici

# #
# # Single -- start
# #

# # variables à reprednre de "2_dataset_prepare_target_data"
# target_class_col_name = 'target_class'


# # variables
# symbols          = ['Ger30']   #,'Usa500']  #, 'UsaInd','UsaTec']   #, 'Ger30']
# time_depth       = 8
# samples_by_class = 10
# periods          = ['M5','M15','H1']


# dfLearn = prepare_dfLearn(big_for_target,target_class_col_name,symbols_set,periods,time_depth,samples_by_class,shuffle_in_data,column_sets,old_to_recent)
# dfLearn

# # sauvegarde
# dfLearn.to_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\dfLearn_M5_profondeur_analyse=3_spread_x=4_ratio=2_(B)_Ger30_time_depth=8_samples_by_class=1000_M5_M15_H1.csv')

# #
# # Single -- end
# #
# # -- début reprise

# python

# import os
# cur_dir=os.getcwd()
# import sys
# if(cur_dir=='C:\\Users\\T0042310\\MyApp\\miniconda3'):
   # sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\pythonProject\\test-master')
   # py_dir='C:\\Users\\T0042310\\Documents\\Perso\\Py'
# else:
   # sys.path.append('E:\\Py\\TF')
   # py_dir='E:\\Py'

# import arbo

# import pandas
# import numpy
# import ta
# import os

# import sklearn
# from sklearn.utils import shuffle
# from numpy import asarray
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# dataset_name = "work"

# del dfLearn
# dfLearn = pandas.read_csv(arbo.get_study_dir(py_dir,dataset_name)+'\\dfLearn_15000_3_equi.csv', index_col=0)


# # variables à reprednre de "2_dataset_prepare_target_data"
# target_class_col_name = 'target_class'

# # -- fin reprise

# def finalize_learning_data(dfLearn):
   # dfLearn2 = dfLearn.copy()
   # dfLearn2 = dfLearn2.drop(['date_join'],axis=1)
   # #
   # dfLearn2.dtypes
   # dfLearn2.shape
   # dfLearn2.columns
   # #
   # # split AND shuffle
   # #
   # # df_train : 60%
   # # df_val   : 30%
   # # df_test  : 10%
   # #
   # #
   # df_train, df_val_test = \
      # train_test_split(dfLearn2, test_size=0.40, random_state=137)
   # df_val, df_test = \
      # train_test_split(df_val_test, test_size=0.25, random_state=137)
   # #
   # df_train
   # df_val
   # df_test
   # #
   # # Y
   # #
   # df_y_1d_train, df_y_Nd_train = single_to_classes_target_cols(df_train,target_class_col_name)
   # df_y_1d_val,   df_y_Nd_val   = single_to_classes_target_cols(df_val,  target_class_col_name)
   # df_y_1d_test,  df_y_Nd_test  = single_to_classes_target_cols(df_test, target_class_col_name)
   # #
   # df_y_1d_train
   # df_y_Nd_train
   # #
   # df_y_1d_val
   # df_y_Nd_val
   # #
   # df_y_1d_test
   # df_y_Nd_test
   # #
   # # X
   # #
   # df_x_train = df_train.drop([target_class_col_name],axis=1)
   # df_x_val   = df_val.drop  ([target_class_col_name],axis=1)
   # df_x_test  = df_test.drop ([target_class_col_name],axis=1)
   # #
   # return df_x_train, df_x_val, df_x_test, df_y_1d_train, df_y_Nd_train, df_y_1d_val, df_y_Nd_val, df_y_1d_test, df_y_Nd_test



# df_x_train, df_x_val, df_x_test, df_y_1d_train, df_y_Nd_train, df_y_1d_val, df_y_Nd_val, df_y_1d_test, df_y_Nd_test = finalize_learning_data(dfLearn)


# # -- fin ici, passer à "dataset_learn.py"







# # TEST : génération d'un jeu d'apprentissage aléatoire

# import random

# count_learning=10000
# features = 200

# x_full = []
# for i in range (0,count_learning):
   # new_line = []
   # for i in range(0,features):
      # new_line.append(random.random())
   # x_full.append(new_line)

# df_x_full = pd.DataFrame(x_full)
# df_x_full.astype('float32')

# df_x_full


# y_full = []
# for i in range (0,count_learning):
   # new_line = []
   # rnd=random.random()
   # if(rnd<.33):
      # new_line.append(1)
      # new_line.append(0)
      # new_line.append(0)
   # elif(rnd>.33)&(rnd<.66):
      # new_line.append(0)
      # new_line.append(1)
      # new_line.append(0)
   # else:
      # new_line.append(0)
      # new_line.append(0)
      # new_line.append(1)
   # y_full.append(new_line)

# df_y_full = pd.DataFrame(y_full)
# df_y_full

# np_x_full = df_x_full.values
# np_y_full = df_y_full.values


# # fin TEST : génération d'un jeu d'apprentissage aléatoire

