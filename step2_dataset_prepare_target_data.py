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

# python
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

import os


step2_params = {
    'step2_target_class_col_name' : [],
    'step2_profondeur_analyse' : 0,
    'step2_target_period' : 0,
    'step2_symbol_for_target' : '',
    'step2_symbol_spread' : 0.0,
    'step2_targets_classes_count' : 0,
    'step2_targetLongShort' : 0.0,
    'step2_ratio_coupure' : 0.0,
    'step2_use_ATR' : True}

def get_param_list():
    param_list = []
    for key, key_value in step2_params.items():
        param_list.append(key)
    return param_list

"""

Préparation des classes de données (LONG / SHORT / OUT et éventuellement LONGSHORT)

   Paramètres à prendre en compte

      répartition de chaque classe pour faciliter l'apprentissatge
   
      ratio coupure (soit le ratio gain/perte) qui est directement lié au taux
      de réussite à atteindre à la reconnaissance pour disposer d'un système rentable
         voir feuille Excel mais en résumé <2.5 voire 2 encore mieux

Identification des configurations pertinentes :

M1 : define_potential
         extrait d'un df suite à sélection unitaire ratio pot_long / pot_short 

   exemple :
      dfSearch_Long = big_M15 \
         [(big_M15['Ger30_Pot_Long_3']>2)&\
          (big_M15['Ger30_Pot_Long_3']>4*abs(big_M15['Ger30_Pot_Short_3']))&\
          (big_M15['Usa500_Pot_Long_3']>1)&\
          (big_M15['Usa500_Pot_Long_3']>4*abs(big_M15['Usa500_Pot_Short_3']))&\
          (big_M15['UsaInd_Pot_Long_3']>2)&\
          (big_M15['UsaInd_Pot_Long_3']>4*abs(big_M15['Usa500_Pot_Short_3']))&\
          (big_M15['UsaTec_Pot_Long_3']>1)&\
          (big_M15['UsaTec_Pot_Long_3']>4*abs(big_M15['Usa500_Pot_Short_3']))
         ]
      dfSearch_Long

   (+)   approche rapide
   (+)   part du résultat voulu au lieu d'une hypothèse qui est ensuite évaluée
   (+)   nécessite de fixer peu de paramètres
               - profondeur_analyse
   (-)   potentiel != réalité (ignore les coupures potentielles)

M2 : define_target
         évaluation d'une pseudo stratégie sur un df déjà filtré

   (+)   approche la plus précise
   (-)   nécessite de fixer plus de paramètres
               - profondeur_analyse
               - niveaux long/short (soit par quantiles, soit en absolu)
               - ratio_coupure vs target


"""

# -----------------------------------------------------------------------------
# Définition des fonctions
# -----------------------------------------------------------------------------

# ENTETE A REPRENDRE, CORPS OK
#              1             2             3             4             5            6             7
#        Long2 =N/A    Long2 =N/A    Long2 =N/A    Long2 =N/A    Long2 =N/A   Long2 =False  Long2 =True
#        Long1 =False  Long1 =True   Long1 =False  Long1 =True   Long1 =True  Long1 =True   Long1 =True
#        Short1=False  Short1=False  Short1=True   Short1=True   Short1=True  Short1=True   Short1=True
#        Short2=N/A    Short2=N/A    Short2=N/A    Short2=False  Short2=True  Short2=N/A    Short2=N/A
#                        long OK                      long OK      long KO      short OK     short KO
#                                      short OK      non coupé      coupé       non coupé      coupé
# +2 qtl     (N/A)         (N/A)         (N/A)         (N/A)         (N/A)          F            T
#                                                                                             
# +1 qtl       F             T             F             T             T            T            T
#                                                                                             
# -1 qtl       F             F             T             T             T            T            T
#                                                                                             
# -2 qtl     (N/A)         (N/A)         (N/A)           F             T          (N/A)        (N/A)     
#

def secure_drop_columns(df,columns_to_drop):
   for column in columns_to_drop:
      if column in df.columns:
         df = df.drop([column],axis=1)
   return df


def define_target_quantile(df,symbol,profondeur_analyse,quantile_value,ratio_coupure=2.0,use_ATR=False):
   qtlLong  = df['Pot_Long' +'_'+str(profondeur_analyse)].quantile(quantile_value)
   qtlShort = df['Pot_Short'+'_'+str(profondeur_analyse)].quantile(quantile_value)
   return define_target_core(df,symbol,period,profondeur_analyse,qtlLong,qtlShort,ratio_coupure,use_ATR)


def define_target_core(df,symbol,period,profondeur_analyse,targetLong,targetShort,ratio_coupure=2.0,use_ATR=False):
   # clean des targets précédentes
   columns_to_drop = []
   columns_to_drop.append(symbol+'_'+'target_LONG'     )
   columns_to_drop.append(symbol+'_'+'target_SHORT'    )
   columns_to_drop.append(symbol+'_'+'target_LONGSHORT')
   columns_to_drop.append(symbol+'_'+'target_OUT'      )
   df = secure_drop_columns(df,columns_to_drop)
   #
   if(use_ATR==True):
      atr=df['UsaInd_M15_ATR_13']
      df[symbol+'_'+'target_long_1x' ]=(df[symbol+'_'+period+'_'+'Pot_Long' +'_'+str(profondeur_analyse)]>=atr*targetLong)
      df[symbol+'_'+'target_long_2x' ]=(df[symbol+'_'+period+'_'+'Pot_Long' +'_'+str(profondeur_analyse)]>=atr*ratio_coupure*targetLong)
      df[symbol+'_'+'target_short_1x']=(df[symbol+'_'+period+'_'+'Pot_Short'+'_'+str(profondeur_analyse)]>=atr*targetShort)
      df[symbol+'_'+'target_short_2x']=(df[symbol+'_'+period+'_'+'Pot_Short'+'_'+str(profondeur_analyse)]>=atr*ratio_coupure*targetShort)
   else:
      df[symbol+'_'+'target_long_1x' ]=(df[symbol+'_'+period+'_'+'Pot_Long' +'_'+str(profondeur_analyse)]>=targetLong)
      df[symbol+'_'+'target_long_2x' ]=(df[symbol+'_'+period+'_'+'Pot_Long' +'_'+str(profondeur_analyse)]>=ratio_coupure*targetLong)
      df[symbol+'_'+'target_short_1x']=(df[symbol+'_'+period+'_'+'Pot_Short'+'_'+str(profondeur_analyse)]>=targetShort)
      df[symbol+'_'+'target_short_2x']=(df[symbol+'_'+period+'_'+'Pot_Short'+'_'+str(profondeur_analyse)]>=ratio_coupure*targetShort)   
   #
   # LONG  : objectif KO, coupure KO
   # SHORT : objectif KO, coupure KO
   df[symbol+'_'+'target_1']=(df[symbol+'_'+'target_long_1x']==False)&(df[symbol+'_'+'target_short_1x']==False)
   #
   # LONG  : objectif KO, coupure ADU
   # SHORT : objectif OK, coupure KO
   df[symbol+'_'+'target_2']=(df[symbol+'_'+'target_long_1x']==False)&(df[symbol+'_'+'target_short_1x']==True)
   #
   # LONG  : objectif KO, coupure OK
   df[symbol+'_'+'target_2_1']=df[symbol+'_'+'target_2']&(df[symbol+'_'+'target_short_2x']==True)
   # LONG  : objectif KO, coupure KO
   df[symbol+'_'+'target_2_2']=df[symbol+'_'+'target_2']&(df[symbol+'_'+'target_short_2x']==False)
   #
   # LONG  : objectif OK, coupure KO
   # SHORT : objectif KO, coupure ADU
   df[symbol+'_'+'target_3']=(df[symbol+'_'+'target_long_1x']==True)&(df[symbol+'_'+'target_short_1x']==False)
   #
   # SHORT : objectif KO, coupure OK
   df[symbol+'_'+'target_3_1']=df[symbol+'_'+'target_3']&(df[symbol+'_'+'target_long_2x']==True)
   # SHORT : objectif KO, coupure KO
   df[symbol+'_'+'target_3_2']=df[symbol+'_'+'target_3']&(df[symbol+'_'+'target_long_2x']==False)
   #
   # LONG  : objectif OK, coupure ADU
   # SHORT : objectif OK, coupure ADU
   df[symbol+'_'+'target_4']=(df[symbol+'_'+'target_long_1x']==True)&(df[symbol+'_'+'target_short_1x']==True )
   #
   # LONG  : objectif OK, coupure POTENTIELLE (? si 1) objectif ou 1) coupure)
   # SHORT : objectif OK, coupure POTENTIELLE (? si 1) objectif ou 1) coupure)
   df[symbol+'_'+'target_4_1']=df[symbol+'_'+'target_4']&(df[symbol+'_'+'target_long_2x']==True)&(df[symbol+'_'+'target_short_2x']==True)
   # LONG  : objectif OK, coupure KO
   # SHORT : objectif OK, coupure POTENTIELLE (? si 1) objectif ou 1) coupure)
   df[symbol+'_'+'target_4_2']=df[symbol+'_'+'target_4']&(df[symbol+'_'+'target_long_2x']==True)&(df[symbol+'_'+'target_short_2x']==False)
   # LONG  : objectif OK, coupure POTENTIELLE (? si 1) objectif ou 1) coupure)
   # SHORT : objectif OK, coupure KO
   df[symbol+'_'+'target_4_3']=df[symbol+'_'+'target_4']&(df[symbol+'_'+'target_long_2x']==False)&(df[symbol+'_'+'target_short_2x']==True)
   # LONG  : objectif OK, coupure KO
   # SHORT : objectif OK, coupure KO
   df[symbol+'_'+'target_4_4']=df[symbol+'_'+'target_4']&(df[symbol+'_'+'target_long_2x']==False)&(df[symbol+'_'+'target_short_2x']==False)
   # TEST le 30/10, correction le 12/11
   df[symbol+'_'+'target_LONG'     ]=((df[symbol+'_'+'target_3']==True)|(df[symbol+'_'+'target_4_2']==True))&(df[symbol+'_'+'target_4_4']==False)
   df[symbol+'_'+'target_SHORT'    ]=((df[symbol+'_'+'target_2']==True)|(df[symbol+'_'+'target_4_3']==True))&(df[symbol+'_'+'target_4_4']==False)
   df[symbol+'_'+'target_LONGSHORT']=(df[symbol+'_'+'target_4_4']==True)
   df[symbol+'_'+'target_OUT'      ]=(df[symbol+'_'+'target_LONG']==False)&(df[symbol+'_'+'target_SHORT']==False)&(df[symbol+'_'+'target_4_4']==False)
#
# LONG
#     Objectif OK       Coupure  KO             => gain = targetLong
#           df['target_3']
#           df['target_4_2']
#     Objectif OK       Coupure  POTENTIELLE    => gain = %G x qltLong - %P x 2 x targetShort
#           df['target_4_1']
#           df['target_4_3']
#     Objectif KO       Coupure  KO             => gain = 0
#           df['target_1']
#           df['target_2_2']
#     Objectif KO       Coupure  OK             => gain = - 2 x targetShort
#           df['target_2_1']
#
#
# SHORT
#     Objectif OK       Coupure  KO             => gain = targetShort
#           df['target_2']                    
#           df['target_4_3']                  
#     Objectif OK       Coupure  POTENTIELLE    => gain = %G x targetShort - %P x 2 x targetLong
#           df['target_4_1']                  
#           df['target_4_2']                  
#     Objectif KO       Coupure  KO             => gain = 0
#           df['target_1']                    
#           df['target_3_2']                  
#     Objectif KO       Coupure  OK             => gain = - 2 x targetLong
#           df['target_3_1']
#
   return df


def define_target_clean(df,symbol):
   columns_to_drop = []
   #
   columns_to_drop.append(symbol+'_'+'target_long_1x'  )
   columns_to_drop.append(symbol+'_'+'target_long_2x'  )
   columns_to_drop.append(symbol+'_'+'target_short_1x' )
   columns_to_drop.append(symbol+'_'+'target_short_2x' )
   columns_to_drop.append(symbol+'_'+'target_1'        )
   columns_to_drop.append(symbol+'_'+'target_2'        )
   columns_to_drop.append(symbol+'_'+'target_2_1'      )
   columns_to_drop.append(symbol+'_'+'target_2_2'      )
   columns_to_drop.append(symbol+'_'+'target_3'        )
   columns_to_drop.append(symbol+'_'+'target_3_1'      )
   columns_to_drop.append(symbol+'_'+'target_3_2'      )
   columns_to_drop.append(symbol+'_'+'target_4'        )
   columns_to_drop.append(symbol+'_'+'target_4_1'      )
   columns_to_drop.append(symbol+'_'+'target_4_2'      )
   columns_to_drop.append(symbol+'_'+'target_4_3'      )
   columns_to_drop.append(symbol+'_'+'target_4_4'      )
   columns_to_drop.append(symbol+'_'+'target_LONG'     )
   columns_to_drop.append(symbol+'_'+'target_SHORT'    )
   columns_to_drop.append(symbol+'_'+'target_LONGSHORT')
   columns_to_drop.append(symbol+'_'+'target_OUT'      )
   #
   df = secure_drop_columns(df,columns_to_drop)
   #
   return df


def eval_by_quantile(df,symbol,profondeur_analyse,qtl,spread,ratio_coupure=2.0):
   df, target_long, target_short = define_target_quantile(df,profondeur_analyse,qtl,ratio_coupure)
   #print('\t\t','qtlLong=',round(qtlLong,1),'qtlShort=',round(qtlShort,1))
   return eval_by_targets(df,symbol,profondeur_analyse,target_long,target_short,spread,ratio_coupure=2.0)


def eval_by_targets(df,symbol,profondeur_analyse,target_long,target_short,spread,ratio_coupure=2.0):
   count_target_1   =(df[symbol+'_'+'target_1'  ]==True).sum()
   count_target_2   =(df[symbol+'_'+'target_2'  ]==True).sum()
   count_target_2_1 =(df[symbol+'_'+'target_2_1']==True).sum()
   count_target_2_2 =(df[symbol+'_'+'target_2_2']==True).sum()
   count_target_3   =(df[symbol+'_'+'target_3'  ]==True).sum()
   count_target_3_1 =(df[symbol+'_'+'target_3_1']==True).sum()
   count_target_3_2 =(df[symbol+'_'+'target_3_2']==True).sum()
   count_target_4   =(df[symbol+'_'+'target_4'  ]==True).sum()
   count_target_4_1 =(df[symbol+'_'+'target_4_1']==True).sum()
   count_target_4_2 =(df[symbol+'_'+'target_4_2']==True).sum()
   count_target_4_3 =(df[symbol+'_'+'target_4_3']==True).sum()
   count_target_4_4 =(df[symbol+'_'+'target_4_4']==True).sum()
   #
   # LONG
   #     Objectif OK       Coupure  KO             => gain = qtlLong
   #           df['target_3']
   #           df['target_4_2']
   #     Objectif OK       Coupure  POTENTIELLE    => gain = %G x qltLong - %P x 2 x target_short
   #           df['target_4_1']
   #           df['target_4_3']
   #     Objectif KO       Coupure  KO             => gain = 0
   #           df['target_1']
   #           df['target_2_2']
   #     Objectif KO       Coupure  OK             => gain = - 2 x target_short
   #           df['target_2_1']
   #
   pcGain=0.5
   pcPerte=1.0-pcGain
   neutreLong=spread
   gainLong =(spread+target_long)
   perteLong=(spread-ratio_coupure*target_short)
   longWin  = 0.0
   longWin += (count_target_3+count_target_4_2+count_target_4_4)  * gainLong
   longWin += (count_target_4_1+count_target_4_3)* (pcGain*gainLong+pcPerte*perteLong)
   longWin += (count_target_1+count_target_2_2)  * neutreLong
   longWin += count_target_2_1                   * perteLong
   #
   # SHORT
   #     Objectif OK       Coupure  KO             => gain = target_short
   #           df['target_2']                    
   #           df['target_4_3']                  
   #     Objectif OK       Coupure  POTENTIELLE    => gain = %G x target_short - %P x 2 x target_long
   #           df['target_4_1']                  
   #           df['target_4_2']                  
   #     Objectif KO       Coupure  KO             => gain = 0
   #           df['target_1']                    
   #           df['target_3_2']                  
   #     Objectif KO       Coupure  OK             => gain = - 2 x target_long
   #           df['target_3_1']
   #
   neutreShort=spread
   gainShort  =(spread+target_short)
   perteShort =(spread-ratio_coupure*target_long)
   shortWin  = 0.0
   shortWin += (count_target_2+count_target_4_3+count_target_4_4)  * gainShort
   shortWin += (count_target_4_1+count_target_4_2)* (pcGain*gainShort+pcPerte*perteShort)
   shortWin += (count_target_1+count_target_3_2)  * neutreShort
   shortWin += count_target_3_1                   * perteShort
   #
   #define_target_clean(df)
   return round(longWin,0), round(shortWin,0), round(target_long,1), round(target_short,1)


def get_spreads():
   spread_Ger30  = 1.5
   spread_Usa500 = 0.35
   spread_UsaInd = 2.5
   spread_UsaTec = 1.0
   #
   return spread_Ger30, spread_Usa500, spread_UsaInd, spread_UsaTec


def define_potential(df_big,profondeur_analyse,ratio_potentiels,spread_x):
   dfSearch_Long =df_big.copy()
   dfSearch_Short=df_big.copy()
   #
   spread_Ger30, spread_Usa500, spread_UsaInd, spread_UsaTec = get_spreads()
   #
   dfSearch_Long = dfSearch_Long \
      [(dfSearch_Long[ 'Ger30_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_Ger30)&\
       (dfSearch_Long[ 'Ger30_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(dfSearch_Long['Ger30_Pot_Short_'+str(profondeur_analyse)]))&\
       (dfSearch_Long['Usa500_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_Usa500)&\
       (dfSearch_Long['Usa500_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(dfSearch_Long['Usa500_Pot_Short_'+str(profondeur_analyse)]))&\
       (dfSearch_Long['UsaInd_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_UsaInd)&\
       (dfSearch_Long['UsaInd_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(dfSearch_Long['Usa500_Pot_Short_'+str(profondeur_analyse)]))&\
       (dfSearch_Long['UsaTec_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_UsaTec)&\
       (dfSearch_Long['UsaTec_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(dfSearch_Long['Usa500_Pot_Short_'+str(profondeur_analyse)]))
      ]
   #
   dfSearch_Short = dfSearch_Short \
      [(dfSearch_Short[ 'Ger30_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_Ger30)&\
       (dfSearch_Short[ 'Ger30_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(dfSearch_Short[ 'Ger30_Pot_Long_'+str(profondeur_analyse)]))&\
       (dfSearch_Short['Usa500_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_Usa500)&\
       (dfSearch_Short['Usa500_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(dfSearch_Short['Usa500_Pot_Long_'+str(profondeur_analyse)]))&\
       (dfSearch_Short['UsaInd_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_UsaInd)&\
       (dfSearch_Short['UsaInd_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(dfSearch_Short['Usa500_Pot_Long_'+str(profondeur_analyse)]))&\
       (dfSearch_Short['UsaTec_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_UsaTec)&\
       (dfSearch_Short['UsaTec_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(dfSearch_Short['Usa500_Pot_Long_'+str(profondeur_analyse)]))
      ]
   #
   return dfSearch_Long, dfSearch_Short


def define_potentialbis(df_big,period,profondeur_analyse,ratio_potentiels,spread_x):
   #
   spread_Ger30, spread_Usa500, spread_UsaInd, spread_UsaTec = get_spreads()
   #
   df_big['m1_long'] = \
      (df_big[ 'Ger30_'+period+'_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_Ger30)&\
      (df_big[ 'Ger30_'+period+'_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(df_big['Ger30_'+period+'_Pot_Short_'+str(profondeur_analyse)]))&\
      (df_big['Usa500_'+period+'_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_Usa500)&\
      (df_big['Usa500_'+period+'_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(df_big['Usa500_'+period+'_Pot_Short_'+str(profondeur_analyse)]))&\
      (df_big['UsaInd_'+period+'_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_UsaInd)&\
      (df_big['UsaInd_'+period+'_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(df_big['Usa500_'+period+'_Pot_Short_'+str(profondeur_analyse)]))&\
      (df_big['UsaTec_'+period+'_Pot_Long_'+str(profondeur_analyse)]>spread_x*spread_UsaTec)&\
      (df_big['UsaTec_'+period+'_Pot_Long_'+str(profondeur_analyse)]>=ratio_potentiels*abs(df_big['Usa500_'+period+'_Pot_Short_'+str(profondeur_analyse)]))
   #
   df_big['m1_short'] = \
      (df_big[ 'Ger30_'+period+'_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_Ger30)&\
      (df_big[ 'Ger30_'+period+'_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(df_big[ 'Ger30_'+period+'_Pot_Long_'+str(profondeur_analyse)]))&\
      (df_big['Usa500_'+period+'_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_Usa500)&\
      (df_big['Usa500_'+period+'_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(df_big['Usa500_'+period+'_Pot_Long_'+str(profondeur_analyse)]))&\
      (df_big['UsaInd_'+period+'_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_UsaInd)&\
      (df_big['UsaInd_'+period+'_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(df_big['Usa500_'+period+'_Pot_Long_'+str(profondeur_analyse)]))&\
      (df_big['UsaTec_'+period+'_Pot_Short_'+str(profondeur_analyse)]>spread_x*spread_UsaTec)&\
      (df_big['UsaTec_'+period+'_Pot_Short_'+str(profondeur_analyse)]>ratio_potentiels*abs(df_big['Usa500_'+period+'_Pot_Long_'+str(profondeur_analyse)]))
   #
   return df_big


def target_core_to_synthetic_col4(dfCore,symbol,target_class_col_name):
   dfCore[target_class_col_name]= 0
   dfCore[target_class_col_name]= \
        1 * ( dfCore[symbol+'_target_LONG'     ]==True) \
      + 2 * ( dfCore[symbol+'_target_SHORT'    ]==True) \
      + 3 * ( dfCore[symbol+'_target_OUT'      ]==True) \
      + 4 * ( dfCore[symbol+'_target_LONGSHORT']==True)     # en dernier car très rare => facilite étapes suivantes
   #print(pandas.value_counts(dfCore[target_class_col_name]))
   return dfCore


def target_core_to_synthetic_col3(dfCore,symbol,target_class_col_name):
   ## patch -- début
   dfCore = dfCore.copy()
   #dfCore[symbol+'_target_OUT'] &= dfCore[symbol+'_target_LONGSHORT']
   #dfCore[symbol+'_target_LONGSHORT'] = False
   #dfCore = dfCore.drop([symbol+'_target_LONGSHORT'],axis=1)
   ## patch -- fin
   dfCore[target_class_col_name]= 0
   dfCore[target_class_col_name]= \
        1 * ( dfCore[symbol+'_target_LONG'     ]==True) \
      + 2 * ( dfCore[symbol+'_target_SHORT'    ]==True) \
      + 3 * ( dfCore[symbol+'_target_OUT'      ]==True) \
      + 3 * ( dfCore[symbol+'_target_LONGSHORT']==True)
   #print(pandas.value_counts(dfCore[target_class_col_name]))
   return dfCore


# -----------------------------------------------------------------------------
# Méthodes de génération
# -----------------------------------------------------------------------------

"""
# Paramètres "define_target"
#
#  -> période de référence (big_Periode)
#  -> symbol
#  -> profondeur
#  -> niveau target (long/short)
#  -> ratio_coupure
#
# SORTIE : 3 (long/short/out) ou 4 targets (long/short/out/longshort)
"""

def generate_big_define_target(symbol_for_target,period,big_Period,profondeur_analyse,targetLongShort,ratio_coupure,targets_classes_count,target_class_col_name,use_ATR=False):
   #
   big_for_target = big_Period.copy()
   big_for_target = define_target_core(big_for_target,symbol_for_target,period,profondeur_analyse,targetLongShort,targetLongShort,ratio_coupure,use_ATR)
   #
   pcLong      =round((big_for_target[symbol_for_target+'_target_LONG'     ]==True).sum()/len(big_for_target),2)
   pcShort     =round((big_for_target[symbol_for_target+'_target_SHORT'    ]==True).sum()/len(big_for_target),2)
   pcLongShort =round((big_for_target[symbol_for_target+'_target_LONGSHORT']==True).sum()/len(big_for_target),2)
   pcOut       =round((big_for_target[symbol_for_target+'_target_OUT'      ]==True).sum()/len(big_for_target),2)
   #print('profondeur_analyse:',profondeur_analyse,'\t', \
   #      'targetLong:',targetLongShort,'\t','targetShort:',targetLongShort,'\t','ratio_coupure:',ratio_coupure,'\t', \
   #      'pcLong:',pcLong,'\t','pcShort:',pcShort,'\t','pcLongShort:',pcLongShort,'\t','pcOut:',pcOut)
   #
   if(targets_classes_count==4):
      big_for_target = target_core_to_synthetic_col4(big_for_target,symbol_for_target,target_class_col_name)
   elif(targets_classes_count==3):
      big_for_target = target_core_to_synthetic_col3(big_for_target,symbol_for_target,target_class_col_name)
   else:
      print('Erreur: valeur invalide pour variable targets_classes_count:',targets_classes_count,' (doit être dans intervalle[3-4]')
   #
   #big_for_target = define_target_clean(big_for_target,symbol_for_target)
   #
   return big_for_target

"""
# Paramètres "define_potential"
#
#  -> période de référence (big_Periode)
#  (NON -> symbol)
#  -> profondeur
#  -> niveau target (long/short)
#  (NON -> ratio_coupure)
#  -> ratio potentiels long/short vs short/long
#  -> spread_x
#
# SORTIE : 3 targets (long/short/out)
"""

def generate_big_define_potential(big_Period,period,profondeur_analyse,ratio_potentiels,spread_x,target_class_col_name):
   #
   big_for_target = big_Period.copy()
   big_for_target=define_potentialbis(big_for_target,period,profondeur_analyse,ratio_potentiels,spread_x)
   #
   toUseLong      = big_for_target[(big_for_target['m1_long' ]==True)]
   toUseShort     = big_for_target[(big_for_target['m1_short']==True)]
   toUseOut       = big_for_target[(big_for_target['m1_long' ]==False)&(big_for_target['m1_short']==False)]
   toUseLongShort = big_for_target[(big_for_target['m1_long' ]==True )&(big_for_target['m1_short']==True )]
   #
   big_for_target[target_class_col_name]= 0
   big_for_target[target_class_col_name]= \
        1 * ( big_for_target['m1_long'  ]==True) \
      + 2 * ( big_for_target['m1_short' ]==True) \
      + 3 * ((big_for_target['m1_long'  ]==False)& (big_for_target['m1_short' ]==False))
   #
   #print(pandas.value_counts(big_for_target[target_class_col_name]))
   #
   # columns_to_drop = []
   # columns_to_drop.append('m1_long' )
   # columns_to_drop.append('m1_short')
   # big_for_target = secure_drop_columns(big_for_target,columns_to_drop)
   #
   return big_for_target, toUseLong, toUseShort, toUseOut

def prepare_target_data_with_define_potential(all_df_bigs,profondeur_analyse,period,spread_x,ratio_potentiels,target_class_col_name):
   big_Period = get_big2(all_df_bigs,period)
   big_for_target, toUseLong, toUseShort, toUseOut = generate_big_define_potential(big_Period,period,profondeur_analyse,ratio_potentiels,spread_x,target_class_col_name)
   return big_for_target

def prepare_target_data_with_define_target(all_df_bigs,profondeur_analyse,period,symbol_for_target,targetLongShort,ratio_coupure,targets_classes_count,target_class_col_name,use_ATR):
   big_Period = get_big2(all_df_bigs,period)
   big_for_target = generate_big_define_target(symbol_for_target,period,big_Period,profondeur_analyse,targetLongShort,ratio_coupure,targets_classes_count,target_class_col_name,use_ATR)
   return big_for_target

def get_big2(all_df_bigs,period):
   if(period=='H1'):
      return all_df_bigs['H1'][0]
   elif(period=='M30'):
      return all_df_bigs['M30'][0]
   elif(period=='M15'):
      return all_df_bigs['M15'][0]
   elif(period=='M5'):
      return all_df_bigs['M5'][0]

def analyze_parameters_prepare_target_data_with_define_potential(
      all_df_bigs, profondeur_analyse, target_period, target_class_col_name,
      range_spread_x,range_ratio_potentiels_x10):
   os.system('cls')
   print('spread_x','\t','ratio_potentiels','\tclass 1 (long)','\tclass 2 (short)','\tclass 3 (out)')
   for spread_x in range_spread_x:
      for ratio_potentiels_x10 in range_ratio_potentiels_x10:
         ratio_potentiels = ratio_potentiels_x10 / 10.0
         big_for_target = prepare_target_data_with_define_potential(
               all_df_bigs,profondeur_analyse,target_period,spread_x,ratio_potentiels,target_class_col_name)
         pc1 = (big_for_target[target_class_col_name]==1).sum()/len(big_for_target)
         pc2 = (big_for_target[target_class_col_name]==2).sum()/len(big_for_target)
         pc3 = (big_for_target[target_class_col_name]==3).sum()/len(big_for_target)
         highlight = (pc1>=0.3)&(pc2>=0.3)&(pc3>=0.3)
         if(highlight==True):
            print(spread_x,'\t',ratio_potentiels,
               '\t','{:.2f}'.format(pc1),
               '\t','{:.2f}'.format(pc2),
               '\t','{:.2f}'.format(pc3),
               '\t <---')
         else:
            print(spread_x,'\t',ratio_potentiels,
               '\t','{:.2f}'.format(pc1),
               '\t','{:.2f}'.format(pc2),
               '\t','{:.2f}'.format(pc3) )

def analyze_parameters_prepare_target_data_with_define_target(
      all_df_bigs, profondeur_analyse, target_period, target_class_col_name,
      symbol_for_target, targets_classes_count, range_targetLongShort_x100, range_ratio_coupure_x100, use_ATR):
   #
   os.system('cls')
   print(symbol_for_target)
   if(targets_classes_count==3):
      print('targetLongShort','\t','ratio_coupure','\tclass 1 (long)','\tclass 2 (short)','\tclass 3 (out)')
   if(targets_classes_count==4):
      print('targetLongShort','\t','ratio_coupure','\tclass 1 (long)','\tclass 2 (short)','\tclass 3 (out)','\tclass 4 (longshort)')
   #
   for targetLongShort_x100 in range_targetLongShort_x100:
      targetLongShort = targetLongShort_x100 / 100.0
      for ratio_coupure_x100 in range_ratio_coupure_x100:
         ratio_coupure = ratio_coupure_x100 / 100.0
         pc_GP_equilibre = ratio_coupure / (1.0+ratio_coupure)
         big_for_target = prepare_target_data_with_define_target(
                           all_df_bigs,
                           profondeur_analyse,
                           target_period,
                           symbol_for_target,
                           targetLongShort,
                           ratio_coupure,
                           targets_classes_count,
                           target_class_col_name, use_ATR)
         pc1 = (big_for_target[target_class_col_name]==1).sum()/len(big_for_target)
         pc2 = (big_for_target[target_class_col_name]==2).sum()/len(big_for_target)
         pc3 = (big_for_target[target_class_col_name]==3).sum()/len(big_for_target)
         pc4 = (big_for_target[target_class_col_name]==4).sum()/len(big_for_target)
         if(targets_classes_count==3):
            highlight = (pc1>=0.3)&(pc2>=0.3)&(pc3>=0.3)
            if(highlight==True):
               print(targetLongShort,'\t',ratio_coupure,
                  '\t','{:.2f}'.format(pc1),
                  '\t','{:.2f}'.format(pc2),
                  '\t','{:.2f}'.format(pc3),
                  '\t <---',
                  '(reco > '+'{:.2f}'.format(pc_GP_equilibre)+')')
            else:
               print(targetLongShort,'\t',ratio_coupure,
                  '\t','{:.2f}'.format(pc1),
                  '\t','{:.2f}'.format(pc2),
                  '\t','{:.2f}'.format(pc3) )
         if(targets_classes_count==4):
            highlight = (pc1>=0.2)&(pc2>=0.2)&(pc3>=0.2)
            if(highlight==True):
               print(targetLongShort,'\t',ratio_coupure,
                  '\t','{:.2f}'.format(pc1),
                  '\t','{:.2f}'.format(pc2),
                  '\t','{:.2f}'.format(pc3),
                  '\t','{:.2f}'.format(pc4),
                  '\t <---',
                  '(reco > '+'{:.2f}'.format(pc_GP_equilibre)+')')
            else:
               print(targetLongShort,'\t',ratio_coupure,
                  '\t','{:.2f}'.format(pc1),
                  '\t','{:.2f}'.format(pc2),
                  '\t','{:.2f}'.format(pc3),
                  '\t','{:.2f}'.format(pc4) )


# #
# # Points d'entrée et variables globales à partir d'ici
# #

# os.system('cls')

# # variables de structure
# target_class_col_name = 'target_class'

# # paramètres pour la définiton de la 'generate_big_define_target' et 'generate_big_define_potential'
# profondeur_analyse = 3
# period             = 'M15'

# # paramètres spécifiques à 'generate_big_define_potential'
# spread_x           = 4
# ratio              = 2

# big_for_target = prepare_target_data_with_define_potential(profondeur_analyse,period,spread_x,ratio)


# # paramètres spécifiques à 'generate_big_define_target'
# # symbol_for_target = 'Ger30'
# # targetLongShort    = 4.0
# # ratio_coupure      = 1.5
# # targets_classes_count      = 3

# # big_for_target = prepare_target_data_with_define_target( profondeur_analyse,  \
                                                         # # period,              \
                                                         # # symbol_for_target,   \
                                                         # # targetLongShort,     \
                                                         # # ratio_coupure,       \
                                                         # # targets_classes_count,       \
                                                         # # target_class_col_name)

## === >> ici big_for_target est prêt pour début étape 3


# -----------------------------------------------------------------------------
# Analyse croisée : choix avec generate_big_define_potential et analyse dataset
# créé avec generate_big_define_target
# -----------------------------------------------------------------------------

# os.system('cls')

# # paramètres pour la définiton de la 'generate_big_define_target' et 'generate_big_define_potential'
# profondeur_analyse = 2
# big_Period         = big_M5

# # paramètres spécifiques à 'generate_big_define_potential'
# spread_x           = 4
# ratio              = 2

# big_for_target, toUseLong, toUseShort, toUseOut = generate_big_define_potential(big_Period,period,profondeur_analyse,ratio,spread_x,target_class_col_name)
# big_for_target

# # paramètres spécifiques à 'generate_big_define_target'
# # voir boucle symbol_for_target = 'Ger30'
# targetLongShort    = 4.0
# ratio_coupure      = 1.5
# targets_classes_count      = 3

# print('>>> LONG')
# big_for_target_inLong = toUseLong.copy()
# for symbol_for_target in ('Usa500','UsaInd','UsaTec','Ger30'):
   # print(symbol_for_target)
   # big_for_target = generate_big_define_target(symbol_for_target,period,big_for_target_inLong,profondeur_analyse,targetLongShort,ratio_coupure,targets_classes_count,target_class_col_name)

# print('>>> SHORT')
# big_for_target_inShort = toUseShort.copy()
# for symbol_for_target in ('Usa500','UsaInd','UsaTec','Ger30'):
   # print(symbol_for_target)
   # big_for_target = generate_big_define_target(symbol_for_target,period,big_for_target_inShort,profondeur_analyse,targetLongShort,ratio_coupure,targets_classes_count,target_class_col_name)

# print('>>> OUT')
# big_for_target_inOut = toUseOut.copy()
# for symbol_for_target in ('Usa500','UsaInd','UsaTec','Ger30'):
   # print(symbol_for_target)
   # big_for_target = generate_big_define_target(symbol_for_target,period,big_for_target_inOut,profondeur_analyse,targetLongShort,ratio_coupure,targets_classes_count,target_class_col_name)














# ##########################################################

# ## POUBELLE ou à déplacer


# # paramètres pour la définiton de la 'target'
# symbol_for_target = 'Ger30'
# big_for_target    = big_M5.copy()

# profondeur_analyse = 3
# targetLong         = 5.0
# targetShort        = 5.0
# ratio_coupure      = 1.5


# #big_for_target = define_target_core(big_for_target,symbol_for_target,period,profondeur_analyse,targetLong,targetShort,ratio_coupure,use_ATR)
# big_for_target = define_target_core(toUseLong,symbol_for_target,period,profondeur_analyse,targetLong,targetShort,ratio_coupure,use_ATR)

# pcLong      =round((big_for_target[symbol_for_target+'_target_LONG'     ]==True).sum()/len(big_for_target),2)
# pcShort     =round((big_for_target[symbol_for_target+'_target_SHORT'    ]==True).sum()/len(big_for_target),2)
# pcLongShort =round((big_for_target[symbol_for_target+'_target_LONGSHORT']==True).sum()/len(big_for_target),2)
# pcOut       =round((big_for_target[symbol_for_target+'_target_OUT'      ]==True).sum()/len(big_for_target),2)
# print('profondeur_analyse:',profondeur_analyse,'\t', \
      # 'targetLong:',targetLong,'\t','targetShort:',targetShort,'\t','ratio_coupure:',ratio_coupure,'\t', \
      # 'pcLong:',pcLong,'\t','pcShort:',pcShort,'\t','pcLongShort:',pcLongShort,'\t','pcOut:',pcOut)


# #big_for_target = define_target_core(big_for_target,symbol_for_target,period,profondeur_analyse,targetLong,targetShort,ratio_coupure,use_ATR)
# big_for_target = define_target_core(toUseShort,symbol_for_target,period,profondeur_analyse,targetLong,targetShort,ratio_coupure,use_ATR)

# pcLong      =round((big_for_target[symbol_for_target+'_target_LONG'     ]==True).sum()/len(big_for_target),2)
# pcShort     =round((big_for_target[symbol_for_target+'_target_SHORT'    ]==True).sum()/len(big_for_target),2)
# pcLongShort =round((big_for_target[symbol_for_target+'_target_LONGSHORT']==True).sum()/len(big_for_target),2)
# pcOut       =round((big_for_target[symbol_for_target+'_target_OUT'      ]==True).sum()/len(big_for_target),2)
# print('profondeur_analyse:',profondeur_analyse,'\t', \
      # 'targetLong:',targetLong,'\t','targetShort:',targetShort,'\t','ratio_coupure:',ratio_coupure,'\t', \
      # 'pcLong:',pcLong,'\t','pcShort:',pcShort,'\t','pcLongShort:',pcLongShort,'\t','pcOut:',pcOut)



# # -----------------------------------------------------------------------------
# # Script pour analyse meilleur ratio entre classes suivant les paramètres
# # de définition de la 'target'
# # -----------------------------------------------------------------------------

# os.system('cls')

# big_for_target = big_M5
# for symbol in ('Usa500','UsaInd','UsaTec','Ger30'):
   # print('>>',symbol)
   # profondeur_analyse=3
   # for target_x10 in range(10,100,10):
      # targetLong = target_x10 / 10.0
      # targetShort = target_x10 / 10.0
      # for ratio_coupure_x2 in range(3,10):
         # ratio_coupure=ratio_coupure_x2/2.0
         # dfTest = define_target_core(big_for_target,symbol,period,profondeur_analyse,targetLong,targetShort,ratio_coupure,use_ATR)
         # pcLong      =round((dfTest[symbol+'_target_LONG'     ]==True).sum()/len(dfTest),2)
         # pcShort     =round((dfTest[symbol+'_target_SHORT'    ]==True).sum()/len(dfTest),2)
         # pcLongShort =round((dfTest[symbol+'_target_LONGSHORT']==True).sum()/len(dfTest),2)
         # pcOut       =round((dfTest[symbol+'_target_OUT'      ]==True).sum()/len(dfTest),2)
         # if(pcLong>0.15)&(pcShort>0.15):
            # print('target:',round(target_x10/10.0,0),'\t','ratio_coupure:',ratio_coupure,'\t','pcLong:',pcLong,'\t','pcShort:',pcShort,'\t','pcLongShort:',pcLongShort,'\t','pcOut:',pcOut)
      # print('--')




# ##=== plus bas : à ranger


# os.system('cls')

# big_XX = big_M5
# for profondeur_analyse in (2,3,5):
   # for ratio in (1.5,2,3,4):
      # for spread_x in (2,3,4):
         # dfTest_Long, dfTest_Short = define_potential(big_XX,profondeur_analyse,ratio,spread_x)
         # print('profondeur_analyse:',profondeur_analyse,'\t','spread_x:',spread_x,'\t','ratio:', ratio,'\t',\
               # len(dfTest_Long ),'(',round(len(dfTest_Long )/len(big_XX),2),')','\t',\
               # len(dfTest_Short),'(',round(len(dfTest_Short)/len(big_XX),2),')','\t',\
               # len(big_XX))




# Génération :

# M1 images
   # - définir la composition horizontale (périodes)
   # - définir la composition verticale (symboles)
   # - définir le contenu de chaque graphe
   # - concaténer le tout pour générer images
   # - RQ : génération possible directement dans chaque répertoire, par classe
          # sans passer par target si on utilise la méthode 1


# Identification des configurations pertinentes :

# M1 : extrait d'un df suite à sélection unitaire ratio pot_long / pot_short 

   # exemple :
      # dfSearch_Long = big_M15 \
         # [(big_M15['Ger30_Pot_Long_3']>2)&\
          # (big_M15['Ger30_Pot_Long_3']>4*abs(big_M15['Ger30_Pot_Short_3']))&\
          # (big_M15['Usa500_Pot_Long_3']>1)&\
          # (big_M15['Usa500_Pot_Long_3']>4*abs(big_M15['Usa500_Pot_Short_3']))&\
          # (big_M15['UsaInd_Pot_Long_3']>2)&\
          # (big_M15['UsaInd_Pot_Long_3']>4*abs(big_M15['Usa500_Pot_Short_3']))&\
          # (big_M15['UsaTec_Pot_Long_3']>1)&\
          # (big_M15['UsaTec_Pot_Long_3']>4*abs(big_M15['Usa500_Pot_Short_3']))
         # ]
      # dfSearch_Long

   # (+)   approche rapide
   # (+)   part du résultat voulu au lieu d'une hypothèse qui est ensuite évaluée
   # (-)   potentiel != réalité (ignore les coupures potentielles)


# M2 : évaluation d'un ratio de potentiel long/Short sur un df préalablement filtré
      
   # def eval_ratio_potentiel(dfEval,profondeur_analyse):
      # sumPotLong =round(dfEval['Pot_Long'  +'_'+str(profondeur_analyse)].sum(),0)
      # sumPotShort=round(dfEval['Pot_Short' +'_'+str(profondeur_analyse)].sum(),0)
      # ratioPotLongOnPotShort=round(sumPotLong/sumPotShort,2)
      # return ratioPotLongOnPotShort, sumPotLong, sumPotShort

   # (+)   approche rapide
   # (+)   nécessite de fixer peu de paramètres
               # - profondeur_analyse
   # (-)   potentiel != réalité (ignore les coupures potentielles)


# M3 : évaluation d'une pseudo stratégie sur un df déjà filtré

   # eval_by_quantile ou eval_by_target

   # nécessite de définir une target (define_target) avec les paramètres
      # - profondeur_analyse
      # - niveaux long/short (soit par quantiles, soit en absolu)
      # - ratio_coupure vs target

   # def eval_by_targets(df,symbol,profondeur_analyse,target_long,target_short,spread,ratio_coupure=2.0):
      
   # (+)   approche la plus précise
   # (-)   nécessite de fixer plus de paramètres
               # - profondeur_analyse
               # - niveaux long/short (soit par quantiles, soit en absolu)
               # - ratio_coupure vs target


# def filtre_delta_close(df,delta_close_long):
   # dfFiltered=df.copy()
   # if(delta_close_long==True):
      # dfFiltered=dfFiltered[(dfFiltered['delta_Close']>=0)]
   # else:
      # dfFiltered=dfFiltered[(dfFiltered['delta_Close']<=0)]
   # return dfFiltered




# def eval_ratio_potentiel(dfEval,profondeur_analyse):
   # sumPotLong =round(dfEval['Pot_Long'  +'_'+str(profondeur_analyse)].sum(),0)
   # sumPotShort=round(dfEval['Pot_Short' +'_'+str(profondeur_analyse)].sum(),0)
   # ratioPotLongOnPotShort=round(sumPotLong/sumPotShort,2)
   # return ratioPotLongOnPotShort, sumPotLong, sumPotShort


# def filter_by_hour_minute(dfToFilter,hour,minute):
   # dfFiltered=dfToFilter.copy()
   # dfFiltered=dfFiltered[((dfFiltered['hour']==hour)&(dfFiltered['minute']==minute))]
   # return dfFiltered


# def etude_position_vs_pivot(symbol,period_pivot,hour,minute):
   # #os.system('cls')
   # print('symbol:',symbol,'period_pivot:',period_pivot,'hour:',hour,'minute:',minute)
   # dfToAnalyze = get_df(symbol,period)
   # dfXX        = filter_by_hour_minute(dfToAnalyze,hour,minute)
   # tab_values_counts=pandas.value_counts(dfXX['class_vs_pivot_'+period_pivot])
   # for i in (-2,-1,1,2):   #(-4,-3,-2,-1,1,2,3,4):
      # print('pivot:',i)
      # for profondeur_analyse in (2,3,5):  #,8,13,21):
         # dfEvalRatioPotentiel=dfXX[(dfXX['class_vs_pivot_'+period_pivot]==i)]
         # ratioPotLongOnPotShort, sumPotLong, sumPotShort = eval_ratio_potentiel(dfEvalRatioPotentiel,profondeur_analyse)
         # print('\t','profondeur_analyse:',str(profondeur_analyse).zfill(2),'\t','pivot class:',i,\
            # '\t','(',len(dfEvalRatioPotentiel),')', \
            # '\t','ratioPotLongOnPotShort:',round(ratioPotLongOnPotShort,2))



# def etude_potentiel_indicateur(dfBase,indicateur,profondeur,spread_not_used):
   # step=20
   # for indicateur_qtl in range(0,100,step):
      # qtl=indicateur_qtl/100.0
      # qtlValMin=dfBase[indicateur].quantile(qtl)
      # qtlValMax=dfBase[indicateur].quantile(qtl+step/100.0)
      # dfXX=dfBase[(dfBase[indicateur]>=qtlValMin)&(dfBase[indicateur]<qtlValMax)]
      # sumPotLong =round(dfXX['Pot_Long'  +'_'+str(profondeur_analyse)].sum(),0)
      # sumPotShort=round(dfXX['Pot_Short' +'_'+str(profondeur_analyse)].sum(),0)
      # valForQtl=round(sumPotLong/sumPotShort,2)
      # if(valForQtl>1.1):
         # print('>>> ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t',round(qtlValMin,1),'\t',round(qtlValMax,1),'\t',sumPotLong,'\t',sumPotShort)
      # elif((valForQtl<0.9)&(valForQtl>0)):
         # print('<<< ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t',round(qtlValMin,1),'\t',round(qtlValMax,1),'\t',sumPotLong,'\t',sumPotShort)
      # else:
         # print('    ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t',round(qtlValMin,1),'\t',round(qtlValMax,1),'\t',sumPotLong,'\t',sumPotShort)


# def get_dfXX_etude_potentiel_indicateur(dfBase,indicateur,profondeur,spread_not_used,indicateur_qtl):
   # step=5
   # qtl=indicateur_qtl/100.0
   # qtlValMin=dfBase[indicateur].quantile(qtl)
   # qtlValMax=dfBase[indicateur].quantile(qtl+step/100.0)
   # dfXX=dfBase[(dfBase[indicateur]>=qtlValMin)&(dfBase[indicateur]<qtlValMax)]
   # return dfXX


# def zoom_year_month(dfBase,indicateur,profondeur,spread_not_used,indicateur_qtl):
   # dfXX=get_dfXX_etude_potentiel_indicateur(dfBase,indicateur,profondeur,spread_not_used,indicateur_qtl)
   # for year in range(2018,2021):
      # for month in range(1,10):
         # dfYearMonth=dfXX[(dfXX['year']==year)&(dfXX['month']==month)]
         # sumPotLong =round(dfYearMonth['Pot_Long'  +'_'+str(profondeur_analyse)].sum(),0)
         # sumPotShort=round(dfYearMonth['Pot_Short' +'_'+str(profondeur_analyse)].sum(),0)
         # valForQtl=round(sumPotLong/sumPotShort,2)
         # if(valForQtl>1.1):
            # print('year:',year,'month:',month,'\t','>>> ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)
         # elif((valForQtl<0.9)&(valForQtl>0)):
            # print('year:',year,'month:',month,'\t','<<< ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)
         # else:
            # print('year:',year,'month:',month,'\t','    ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)

# def zoom_dayofweek(dfBase,indicateur,profondeur,spread_not_used,indicateur_qtl):
   # dfXX=get_dfXX_etude_potentiel_indicateur(dfBase,indicateur,profondeur,spread_not_used,indicateur_qtl)
   # for dayofweek in range(0,5):
      # dfDayOfWeek=dfXX[(dfXX['dayofweek']==dayofweek)]
      # sumPotLong =round(dfDayOfWeek['Pot_Long'  +'_'+str(profondeur_analyse)].sum(),0)
      # sumPotShort=round(dfDayOfWeek['Pot_Short' +'_'+str(profondeur_analyse)].sum(),0)
      # valForQtl=round(sumPotLong/sumPotShort,2)
      # if(valForQtl>1.1):
         # print('dayofweek:',dayofweek,'\t','>>> ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)
      # elif((valForQtl<0.9)&(valForQtl>0)):
         # print('dayofweek:',dayofweek,'\t','<<< ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)
      # else:
         # print('dayofweek:',dayofweek,'\t','    ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)

# def zoom_hour(dfBase,indicateur,profondeur,spread_not_used,indicateur_qtl):
   # dfXX=get_dfXX_etude_potentiel_indicateur(dfBase,indicateur,profondeur,spread_not_used,indicateur_qtl)
   # for hour in range(0,24):
      # dfHour=dfXX[(dfXX['hour']==hour)]
      # sumPotLong =round(dfHour['Pot_Long'  +'_'+str(profondeur_analyse)].sum(),0)
      # sumPotShort=round(dfHour['Pot_Short' +'_'+str(profondeur_analyse)].sum(),0)
      # valForQtl=round(sumPotLong/sumPotShort,2)
      # if(valForQtl>1.1):
         # print('hour:',hour,'\t','>>> ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)
      # elif((valForQtl<0.9)&(valForQtl>0)):
         # print('hour:',hour,'\t','<<< ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)
      # else:
         # print('hour:',hour,'\t','    ',valForQtl,'\t',indicateur,'\t',profondeur,'\t',indicateur_qtl,'\t','\t',sumPotLong,'\t',sumPotShort)

# def etude_potentiel_indicateur2(dfBase,indicateur,profondeur,spread):
   # for indicateur_qtl in range(0,10):
      # qtl=indicateur_qtl/10.0
      # qtlValMin=dfBase[indicateur].quantile(qtl)
      # qtlValMax=dfBase[indicateur].quantile(qtl+0.1)
      # dfXX=dfBase[(dfBase[indicateur]>=qtlValMin)&(dfBase[indicateur]<qtlValMax)]
      # for qtlx10 in range(1,10):
         # qtl=qtlx10/10.0
         # longWin, shortWin, qtlLong, qtlShort, error = eval_by_quantile(dfXX,profondeur,qtl,spread)
         # if(longWin>0):
            # print(indicateur,'\t',profondeur,'\t',qtlx10,'\t',indicateur_qtl,'\t',round(qtlValMin,1),'\t',round(qtlValMax,1),'\t',round(qtlLong,1),'\t',round(qtlShort,1),'\t >>> longWin:',round(longWin,0))
         # #else:
         # #   print(indicateur,'\t',profondeur,'\t',qtlx10,'\t',indicateur_qtl,'\t',round(qtlValMin,1),'\t',round(qtlValMax,1),'\t',round(qtlLong,1),'\t',round(qtlShort,1),'\t     longWin:',round(longWin,0))
         # if(shortWin>0):
            # print(indicateur,'\t',profondeur,'\t',qtlx10,'\t',indicateur_qtl,'\t',round(qtlValMin,1),'\t',round(qtlValMax,1),'\t',round(qtlLong,1),'\t',round(qtlShort,1),'\t >>> shortWin:',round(shortWin,0))
         # #else:
         # #   print(indicateur,'\t',profondeur,'\t',qtlx10,'\t',indicateur_qtl,'\t',round(qtlValMin,1),'\t',round(qtlValMax,1),'\t',round(qtlLong,1),'\t',round(qtlShort,1),'\t     shortWin:',round(shortWin,0))



# dfToAnalyze = get_df(symbol,period)
# dfXX        = dfToAnalyze  #filter_by_hour_minute(dfToAnalyze,hour,minute)


# spread=0.0
# for qtl_x10 in range(0,10):
   # qtl=qtl_x10/10.0
   # for ratio_coupure in (1.0,1.5,2.0):
      # longWin,shortWin,qtlLong,qtlShort=eval_by_quantile(dfXX,profondeur_analyse,qtl,spread,ratio_coupure)
      # print(qtl_x10,ratio_coupure,'\t',longWin,'\t',shortWin,'\t',qtlLong,'\t',qtlShort,'\t')


# spread=-2.0
# for target in range(2,20,2):
   # for ratio_coupure in (1.0,1.5,2.0):
      # longWin,shortWin,qtlLong,qtlShort=eval_by_targets(dfXX,symbol,profondeur_analyse,target,target,spread,ratio_coupure)
      # print(target,'\t',ratio_coupure,'\t',longWin,'\t',shortWin)
