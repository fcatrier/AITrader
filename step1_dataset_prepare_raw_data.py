#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

"""

Tri plus ancien (head) vers plus récent (tail): (df = df.sort_index(ascending=True))
  - shift( 1) = valeur hier
  - shift(-1) = valeur demain
  - calculs avec librairie ta : OK

Mais attention ce tri n'est pas adapté à l'extaction : ne pas oublier de l'inverser avec la commande
   df = df.sort_index(ascending=False)


# # remise dans l'ordre chronologique du plus récent en 1er au plus ancien en dernier
# # attention à l'application des indicateurs techniques : appliqués de haut en bas dans le df
# # de qui n'est pas le sens 
# g_big_for_target = g_big_for_target.sort_index(ascending=True)


# # le 10/12 : Potentiels vérifiés, OK
# # TODO : indicateurs techniques et plogClose_period_xx
# cols = ['Close_M15','High_M15','Pot_Long_2']
# UsaInd_dfM15[cols]

# indicator_EMA_3 = ta.trend.EMAIndicator(g_big_for_target['Usa500_M15_idx'],3)
# g_big_for_target['Usa500_M15_idx.EMAIndicator_3']=indicator_EMA_3.ema_indicator()
# g_big_for_target['Usa500_M15_idx.ema_indicator_3']=ta.trend.ema_indicator(g_big_for_target['Usa500_M15_idx'],3)

# g_big_for_target = g_big_for_target.sort_index(ascending=False)


"""
#python

import os
import sys

# cur_dir = os.getcwd()
# if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
#     sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\pythonProject\\test-master')
#     py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
# elif cur_dir == 'C:\\Users\\Frédéri\\PycharmProjects\\pythonProject':
#     py_dir = 'C:\\Users\\Frédéri\\Py'
# else:
#     sys.path.append('E:\\Py\\pythonProject')
#     sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v7.6.5 for CUDA 10.1\\bin')
#     sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v8.0.3.33 for CUDA 10.1\\bin')
#     sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')
#     py_dir = 'E:\\Py'

import arbo
import pandas
import numpy
import ta
from scipy import stats

# -----------------------------------------------------------------------------
# Définition des fonctions
# -----------------------------------------------------------------------------

def get_prepared_dataset_full_filename(dataset_name,symbol,period):
   return arbo.get_source_data_dir(arbo.get_py_dir(),dataset_name)+'\\'+symbol+'_2018-2020-10-18_'+period+'.csv'

def read_data(data_dir):
   df=pandas.read_csv(data_dir, index_col=0, parse_dates=True)
   df.insert(1, 'idx', range(0, len(df)))
   df['DateTime'] = df.index
   df['year']      = [(df['DateTime'][i].year      ) for i in range(0, len(df))]
   df['month']     = [(df['DateTime'][i].month     ) for i in range(0, len(df))]
   df['day']       = [(df['DateTime'][i].day       ) for i in range(0, len(df))]
   df['dayofweek'] = [(df['DateTime'][i].dayofweek ) for i in range(0, len(df))]
   df['hour']      = [(df['DateTime'][i].hour      ) for i in range(0, len(df))]
   df['minute']    = [(df['DateTime'][i].minute    ) for i in range(0, len(df))]
   df['time_slot'] = ( df['hour'] * 4 + df['minute'] // 15 ) / 100.0
   df.head()
   df.dtypes
   return df

def step_from_period(period):
   step=0
   if(period=="M1"):
      step=1
   elif(period=="M5"):
      step=5
   elif(period=="M15"):
      step=15
   elif(period=="M30"):
      step=30
   elif(period=="H1"):
      step=60
   elif(period=="H4"):
      step=4*60
   return step

def dfMn_allData(df,period):
   filter=[]
   step=step_from_period(period)
   for i in range(0,60,step):
      filter.append(i)
   dfMn = df.copy()
   dfMn = dfMn[dfMn['minute'].isin(filter)]
   return dfMn

def add_potentiel_long_short(df,n):
   df['Pot_Long' +'_'+str(n)] =   df['High'].rolling(n).max().shift(-n) - df['Close']
   df['Pot_Short'+'_'+str(n)] = - df['Low' ].rolling(n).min().shift(-n) + df['Close']
   return df

def add_log_close(df,period):
   for depth in (1,2,3,5,8,13,21):
      df['logClose_'+period+'_'+str(depth)] = numpy.log(df['Close_'+period]/df['Close_'+period].shift(depth))
      df['plogClose_'+period+'_'+str(depth)]=[(stats.percentileofscore(df['logClose_'+period+'_'+str(depth)],df['logClose_'+period+'_'+str(depth)][i])/100.0) for i in range(0, len(df))]
   for depth in (1,2,3,5,8,13,21):
      df = df.drop(['logClose_'+period+'_'+str(depth)],axis=1)
   return df

# pré-requis pour calculer les Heikin-Ashi
# crée deux colonnes prev_Open_period et prev_Close_period
def add_prev_OpenClose(df,period):
   df['prev_Open'  +'_'+period]=df['Open' +'_'+period].shift(1)
   df['prev_Close' +'_'+period]=df['Close'+'_'+period].shift(1)
   df['delta_Close'+'_'+period]=df['Close'+'_'+period]-df['prev_Close'+'_'+period]
   return df

def add_HA(df,period):
   df = df.copy()
   df['HA_Close'+'_'+period]=(df['Open'+'_'+period]+df['High'+'_'+period]+df['Low'+'_'+period]+df['Close'+'_'+period])/4
   df['HA_Open' +'_'+period]=(df['prev_Open'+'_'+period]+df['prev_Close'+'_'+period])/2
   df['HA_Open' +'_'+period]=round(df['HA_Open' +'_'+period],2)
   df['HA_Close'+'_'+period]=round(df['HA_Close'+'_'+period],2)
   df['HA_High' +'_'+period]=df[['HA_Open'+'_'+period,'HA_Close'+'_'+period,'High'+'_'+period]].max(axis=1)
   df['HA_Low'  +'_'+period]=df[['HA_Open'+'_'+period,'HA_Close'+'_'+period,'Low' +'_'+period]].min(axis=1)
   df['HA_High' +'_'+period]=round(df['HA_High'+'_'+period],2)
   df['HA_Low'  +'_'+period]=round(df['HA_Low' +'_'+period],2)
   return df

def add_pivot(df,period_pivot):
   """
   # Pivot = (H + B + C) / 3
   # S1 = (2 x Pivot) - H
   # S2 = Pivot - (H - B)
   # S3 = B - 2x (H - Pivot)
   # R1 = (2 x Pivot) - B
   # R2 = Pivot + (H - B)
   # R3 = H + 2x (Pivot - B)
   """
   df['Pivot_'+period_pivot]=(df['High_'+period_pivot]+df['Low_'+period_pivot]+df['Close_'+period_pivot])/3.0
   df['S1_'+period_pivot]=2.0*df['Pivot_'+period_pivot]-df['High_'+period_pivot]
   df['S2_'+period_pivot]=df['Pivot_'+period_pivot]-(df['High_'+period_pivot]-df['Low_'+period_pivot])
   df['S3_'+period_pivot]=df['Low_'+period_pivot]-2.0*(df['High_'+period_pivot]-df['Pivot_'+period_pivot])
   df['R1_'+period_pivot]=2.0*df['Pivot_'+period_pivot]-df['Low_'+period_pivot]
   df['R2_'+period_pivot]=df['Pivot_'+period_pivot]+(df['High_'+period_pivot]-df['Low_'+period_pivot])
   df['R3_'+period_pivot]=df['High_'+period_pivot]+2.0*(df['Pivot_'+period_pivot]-df['Low_'+period_pivot])
   df['Pivot_'+period_pivot]=round(df['Pivot_'+period_pivot],2)
   df['S1_'+period_pivot]=round(df['S1_'+period_pivot],2)
   df['S2_'+period_pivot]=round(df['S2_'+period_pivot],2)
   df['S3_'+period_pivot]=round(df['S3_'+period_pivot],2)
   df['R1_'+period_pivot]=round(df['R1_'+period_pivot],2)
   df['R2_'+period_pivot]=round(df['R2_'+period_pivot],2)
   df['R3_'+period_pivot]=round(df['R3_'+period_pivot],2)
   df=class_vs_pivot(df,period_pivot)
   df=clean_pivot(df,period_pivot)
   return df

def class_vs_pivot(df,period_pivot):
   df['class_vs_pivot_'+period_pivot]= \
      ( \
         ( \
            (+4) * (                                           (df['Close']>df['R3_'   +period_pivot]) ) + \
            (+3) * ( (df['Close']<df['R3_'   +period_pivot]) & (df['Close']>df['R2_'   +period_pivot]) ) + \
            (+2) * ( (df['Close']<df['R2_'   +period_pivot]) & (df['Close']>df['R1_'   +period_pivot]) ) + \
            (+1) * ( (df['Close']<df['R1_'   +period_pivot]) & (df['Close']>df['Pivot_'+period_pivot]) ) + \
            (-1) * ( (df['Close']<df['Pivot_'+period_pivot]) & (df['Close']>df['S1_'   +period_pivot]) ) + \
            (-2) * ( (df['Close']<df['S1_'   +period_pivot]) & (df['Close']>df['S2_'   +period_pivot]) ) + \
            (-3) * ( (df['Close']<df['S2_'   +period_pivot]) & (df['Close']>df['S3_'   +period_pivot]) ) + \
            (-4) * ( (df['Close']<df['S3_'   +period_pivot])                                         ) \
         ) \
      + 4 ) / 8.0
   return df

def clean_pivot(df,period_pivot):
   df = df.drop(['Pivot_'+period_pivot],axis=1)
   df = df.drop(['S1_'   +period_pivot],axis=1)
   df = df.drop(['S2_'   +period_pivot],axis=1)
   df = df.drop(['S3_'   +period_pivot],axis=1)
   df = df.drop(['R1_'   +period_pivot],axis=1)
   df = df.drop(['R2_'   +period_pivot],axis=1)
   df = df.drop(['R3_'   +period_pivot],axis=1)
   return df

def add_ta_tmp_test(dfMn):
   # RSI
   indicator_rsi = ta.momentum.RSIIndicator(close=dfMn['Close'])
   dfMn['RSI']=indicator_rsi.rsi()
   dfMn['pRSI2']=[(stats.percentileofscore(dfMn['RSI'],dfMn['RSI'][i])/100.0) for i in range(0, len(dfMn))]
   return dfMn

def add_ta(dfMn):
   # RSI
   indicator_rsi = ta.momentum.RSIIndicator(dfMn['Close'])
   dfMn['RSI']=indicator_rsi.rsi()
   dfMn['pRSI']=[(stats.percentileofscore(dfMn['RSI'],dfMn['RSI'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['RSI'],axis=1)
   #
   for period in (3,5,8,13,21):
      indicator_rsi = ta.momentum.RSIIndicator(dfMn['Close'],period)
      dfMn['RSI_'+str(period)]=indicator_rsi.rsi()
      dfMn['pRSI_'+str(period)]=[(stats.percentileofscore(dfMn['RSI_'+str(period)],dfMn['RSI_'+str(period)][i])/100.0) for i in range(0, len(dfMn))]
      dfMn = dfMn.drop(['RSI_'+str(period)],axis=1)
   #
   # %B
   indicator_boll = ta.volatility.BollingerBands(close=dfMn['Close'])
   dfMn['BOLL']=(dfMn['Close']-indicator_boll.bollinger_lband())/(indicator_boll.bollinger_hband()-indicator_boll.bollinger_lband())
   dfMn['pBOLL']=[(stats.percentileofscore(dfMn['BOLL'],dfMn['BOLL'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['BOLL'],axis=1)
   # ADX   
   indicator_ADX = ta.trend.ADXIndicator(high=dfMn['High'], low=dfMn['Low'], close=dfMn['Close'])
   dfMn['ADX']=indicator_ADX.adx()
   dfMn['pADX']=[(stats.percentileofscore(dfMn['ADX'],dfMn['ADX'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['ADX'],axis=1)
   # CCI
   indicator_CCI = ta.trend.CCIIndicator(high=dfMn['High'], low=dfMn['Low'], close=dfMn['Close'])
   dfMn['CCI']=indicator_CCI.cci()
   dfMn['pCCI']=[(stats.percentileofscore(dfMn['CCI'],dfMn['CCI'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['CCI'],axis=1)
   # MassIndex
   indicator_MassIndex = ta.trend.MassIndex(high=dfMn['High'], low=dfMn['Low'])
   dfMn['MassIndex']=indicator_MassIndex.mass_index()
   dfMn['pMassIndex']=[(stats.percentileofscore(dfMn['MassIndex'],dfMn['MassIndex'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['MassIndex'],axis=1)
   # ATR
   indicator_ATR = ta.volatility.AverageTrueRange(high=dfMn['High'], low=dfMn['Low'], close=dfMn['Close'])
   dfMn['ATR']=indicator_ATR.average_true_range()
   dfMn['pATR']=[(stats.percentileofscore(dfMn['ATR'],dfMn['ATR'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['ATR'],axis=1)
   #
   for period in (3,5,8,13,21):
      indicator_ATR = ta.volatility.AverageTrueRange(dfMn['High'],dfMn['Low'],dfMn['Close'],period)
      dfMn['ATR_'+str(period)]=indicator_ATR.average_true_range()
   #
   # Ultimate Oscillator
   indicator_UltimateOscillator = ta.momentum.UltimateOscillator(high=dfMn['High'], low=dfMn['Low'], close=dfMn['Close'])
   dfMn['UltimateOscillator']=indicator_UltimateOscillator.uo()
   dfMn['pUltimateOscillator']=[(stats.percentileofscore(dfMn['UltimateOscillator'],dfMn['UltimateOscillator'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['UltimateOscillator'],axis=1)
   #Williams %R
   indicator_WilliamsRIndicator = ta.momentum.WilliamsRIndicator(high=dfMn['High'], low=dfMn['Low'], close=dfMn['Close'])
   dfMn['WilliamsRIndicator']=indicator_WilliamsRIndicator.wr()
   dfMn['pWilliamsRIndicator']=[(stats.percentileofscore(dfMn['WilliamsRIndicator'],dfMn['WilliamsRIndicator'][i])/100.0) for i in range(0, len(dfMn))]
   dfMn = dfMn.drop(['WilliamsRIndicator'],axis=1)
   return dfMn

def prepare_raw_dataset_M1(df):
   # M1
   dfM1 = df.copy()
   dfM1['Open' ] = dfM1['Open_M1' ]
   dfM1['High' ] = dfM1['High_M1' ]
   dfM1['Low'  ] = dfM1['Low_M1'  ]
   dfM1['Close'] = dfM1['Close_M1']
   # Suppression de toutes les périodes "Mxx"
   # dfM1 = dfM1.drop(['Open_M1' ],axis=1)
   # dfM1 = dfM1.drop(['High_M1' ],axis=1)
   # dfM1 = dfM1.drop(['Low_M1'  ],axis=1)
   # dfM1 = dfM1.drop(['Close_M1'],axis=1)
   #
   dfM1 = dfM1.drop(['Open_M5'  ],axis=1)
   dfM1 = dfM1.drop(['High_M5'  ],axis=1)
   dfM1 = dfM1.drop(['Low_M5'   ],axis=1)
   dfM1 = dfM1.drop(['Close_M5' ],axis=1)
   dfM1 = dfM1.drop(['Open_M15' ],axis=1)
   dfM1 = dfM1.drop(['High_M15' ],axis=1)
   dfM1 = dfM1.drop(['Low_M15'  ],axis=1)
   dfM1 = dfM1.drop(['Close_M15'],axis=1)
   dfM1 = dfM1.drop(['Open_M30' ],axis=1)
   dfM1 = dfM1.drop(['High_M30' ],axis=1)
   dfM1 = dfM1.drop(['Low_M30'  ],axis=1)
   dfM1 = dfM1.drop(['Close_M30'],axis=1)
   #
   dfM1 = add_prev_OpenClose(dfM1,'M1')
   # dfM1 = add_prev_OpenClose(dfM1,'M5')
   # dfM1 = add_prev_OpenClose(dfM1,'M15')
   # dfM1 = add_prev_OpenClose(dfM1,'M30')
   # dfM1 = add_prev_OpenClose(dfM1,'H1')
   dfM1 = add_HA(dfM1,'M1')
   # dfM1 = add_HA(dfM1,'M5')
   # dfM1 = add_HA(dfM1,'M15')
   # dfM1 = add_HA(dfM1,'M30')
   # dfM1 = add_HA(dfM1,'H1')
   dfM1 = add_log_close(dfM1,'M1')
   # dfM1 = add_log_close(dfM1,'M5')
   # dfM1 = add_log_close(dfM1,'M15')
   # dfM1 = add_log_close(dfM1,'M30')
   # dfM1 = add_log_close(dfM1,'H1')
   dfM1  = add_pivot(dfM1, 'H1')
   dfM1  = add_pivot(dfM1, 'H4')
   dfM1  = add_pivot(dfM1, 'D1')
   # suppressions à faire après avoir ajouté les pivots
   dfM1 = dfM1.drop(['Open_H1' ],axis=1)
   dfM1 = dfM1.drop(['High_H1' ],axis=1)
   dfM1 = dfM1.drop(['Low_H1'  ],axis=1)
   dfM1 = dfM1.drop(['Close_H1'],axis=1)
   dfM1 = dfM1.drop(['Open_H4' ],axis=1)
   dfM1 = dfM1.drop(['High_H4' ],axis=1)
   dfM1 = dfM1.drop(['Low_H4'  ],axis=1)
   dfM1 = dfM1.drop(['Close_H4'],axis=1)
   dfM1 = dfM1.drop(['Open_D1' ],axis=1)
   dfM1 = dfM1.drop(['High_D1' ],axis=1)
   dfM1 = dfM1.drop(['Low_D1'  ],axis=1)
   dfM1 = dfM1.drop(['Close_D1'],axis=1)
   #
   dfM1  = add_ta(dfM1)
   dfM1=add_potentiel_long_short(dfM1,2)
   dfM1=add_potentiel_long_short(dfM1,3)
   dfM1=add_potentiel_long_short(dfM1,5)
   dfM1=add_potentiel_long_short(dfM1,8)
   dfM1=add_potentiel_long_short(dfM1,13)
   dfM1=add_potentiel_long_short(dfM1,21)
   #Clean
   dfM1 = dfM1.replace([numpy.inf, -numpy.inf], numpy.nan)
   dfM1 = dfM1.dropna()
   return dfM1

def prepare_raw_dataset_M5(df):
   # M5
   dfM5 = dfMn_allData(df,'M5')
   dfM5['Open' ] = dfM5['Open_M5' ]
   dfM5['High' ] = dfM5['High_M5' ]
   dfM5['Low'  ] = dfM5['Low_M5'  ]
   dfM5['Close'] = dfM5['Close_M5']
   # Suppression de toutes les périodes "Mxx"
   dfM5 = dfM5.drop(['Open_M1' ],axis=1)
   dfM5 = dfM5.drop(['High_M1' ],axis=1)
   dfM5 = dfM5.drop(['Low_M1'  ],axis=1)
   dfM5 = dfM5.drop(['Close_M1'],axis=1)
   #
   dfM5 = dfM5.drop(['Open_M15' ],axis=1)
   dfM5 = dfM5.drop(['High_M15' ],axis=1)
   dfM5 = dfM5.drop(['Low_M15'  ],axis=1)
   dfM5 = dfM5.drop(['Close_M15'],axis=1)
   dfM5 = dfM5.drop(['Open_M30' ],axis=1)
   dfM5 = dfM5.drop(['High_M30' ],axis=1)
   dfM5 = dfM5.drop(['Low_M30'  ],axis=1)
   dfM5 = dfM5.drop(['Close_M30'],axis=1)
   #
   dfM5 = add_prev_OpenClose(dfM5,'M5')
   # dfM5 = add_prev_OpenClose(dfM5,'M15')
   # dfM5 = add_prev_OpenClose(dfM5,'M30')
   # dfM5 = add_prev_OpenClose(dfM5,'H1')
   dfM5 = add_HA(dfM5,'M5')
   # dfM5 = add_HA(dfM5,'M15')
   # dfM5 = add_HA(dfM5,'M30')
   # dfM5 = add_HA(dfM5,'H1')
   dfM5 = add_log_close(dfM5,'M5')
   # dfM5 = add_log_close(dfM5,'M15')
   # dfM5 = add_log_close(dfM5,'M30')
   # dfM5 = add_log_close(dfM5,'H1')
   dfM5  = add_pivot(dfM5, 'H1')
   dfM5  = add_pivot(dfM5, 'H4')
   dfM5  = add_pivot(dfM5, 'D1')
   # suppressions à faire après avoir ajouté les pivots
   dfM5 = dfM5.drop(['Open_H1' ],axis=1)
   dfM5 = dfM5.drop(['High_H1' ],axis=1)
   dfM5 = dfM5.drop(['Low_H1'  ],axis=1)
   dfM5 = dfM5.drop(['Close_H1'],axis=1)
   dfM5 = dfM5.drop(['Open_H4' ],axis=1)
   dfM5 = dfM5.drop(['High_H4' ],axis=1)
   dfM5 = dfM5.drop(['Low_H4'  ],axis=1)
   dfM5 = dfM5.drop(['Close_H4'],axis=1)
   dfM5 = dfM5.drop(['Open_D1' ],axis=1)
   dfM5 = dfM5.drop(['High_D1' ],axis=1)
   dfM5 = dfM5.drop(['Low_D1'  ],axis=1)
   dfM5 = dfM5.drop(['Close_D1'],axis=1)
   #
   dfM5  = add_ta(dfM5)
   dfM5=add_potentiel_long_short(dfM5,2)
   dfM5=add_potentiel_long_short(dfM5,3)
   dfM5=add_potentiel_long_short(dfM5,5)
   dfM5=add_potentiel_long_short(dfM5,8)
   dfM5=add_potentiel_long_short(dfM5,13)
   dfM5=add_potentiel_long_short(dfM5,21)
   #Clean
   dfM5 = dfM5.replace([numpy.inf, -numpy.inf], numpy.nan)
   dfM5 = dfM5.dropna()
   return dfM5

def prepare_raw_dataset_M15(df):
   # M15
   dfM15 = dfMn_allData(df,'M15')
   dfM15['Open' ] = dfM15['Open_M15' ]
   dfM15['High' ] = dfM15['High_M15' ]
   dfM15['Low'  ] = dfM15['Low_M15'  ]
   dfM15['Close'] = dfM15['Close_M15']
   # Suppression de toutes les périodes "Mxx"
   dfM15 = dfM15.drop(['Open_M1' ],axis=1)
   dfM15 = dfM15.drop(['High_M1' ],axis=1)
   dfM15 = dfM15.drop(['Low_M1'  ],axis=1)
   dfM15 = dfM15.drop(['Close_M1'],axis=1)
   dfM15 = dfM15.drop(['Open_M5' ],axis=1)
   dfM15 = dfM15.drop(['High_M5' ],axis=1)
   dfM15 = dfM15.drop(['Low_M5'  ],axis=1)
   dfM15 = dfM15.drop(['Close_M5'],axis=1)
   #
   dfM15 = dfM15.drop(['Open_M30' ],axis=1)
   dfM15 = dfM15.drop(['High_M30' ],axis=1)
   dfM15 = dfM15.drop(['Low_M30'  ],axis=1)
   dfM15 = dfM15.drop(['Close_M30'],axis=1)
   dfM15 = add_prev_OpenClose(dfM15,'M15')
   # dfM15 = add_prev_OpenClose(dfM15,'M30')
   # dfM15 = add_prev_OpenClose(dfM15,'H1')
   dfM15 = add_HA(dfM15,'M15')
   # dfM15 = add_HA(dfM15,'M30')
   # dfM15 = add_HA(dfM15,'H1')
   dfM15 = add_log_close(dfM15,'M15')
   # dfM15 = add_log_close(dfM15,'M30')
   # dfM15 = add_log_close(dfM15,'H1')
   dfM15 = add_pivot(dfM15, 'H1')
   dfM15 = add_pivot(dfM15, 'H4')
   dfM15 = add_pivot(dfM15, 'D1')
   # suppressions à faire après avoir ajouté les pivots
   dfM15 = dfM15.drop(['Open_H1' ],axis=1)
   dfM15 = dfM15.drop(['High_H1' ],axis=1)
   dfM15 = dfM15.drop(['Low_H1'  ],axis=1)
   dfM15 = dfM15.drop(['Close_H1'],axis=1)
   dfM15 = dfM15.drop(['Open_H4' ],axis=1)
   dfM15 = dfM15.drop(['High_H4' ],axis=1)
   dfM15 = dfM15.drop(['Low_H4'  ],axis=1)
   dfM15 = dfM15.drop(['Close_H4'],axis=1)
   dfM15 = dfM15.drop(['Open_D1' ],axis=1)
   dfM15 = dfM15.drop(['High_D1' ],axis=1)
   dfM15 = dfM15.drop(['Low_D1'  ],axis=1)
   dfM15 = dfM15.drop(['Close_D1'],axis=1)
   #
   dfM15 = add_ta(dfM15)
   dfM15=add_potentiel_long_short(dfM15,2)
   dfM15=add_potentiel_long_short(dfM15,3)
   dfM15=add_potentiel_long_short(dfM15,5)
   dfM15=add_potentiel_long_short(dfM15,8)
   dfM15=add_potentiel_long_short(dfM15,13)
   dfM15=add_potentiel_long_short(dfM15,21)
   #Clean
   dfM15 = dfM15.replace([numpy.inf, -numpy.inf], numpy.nan)
   dfM15 = dfM15.dropna()
   return dfM15

def prepare_raw_dataset_M30(df):
   # M30
   dfM30 = dfMn_allData(df,'M30')
   dfM30['Open' ] = dfM30['Open_M30' ]
   dfM30['High' ] = dfM30['High_M30' ]
   dfM30['Low'  ] = dfM30['Low_M30'  ]
   dfM30['Close'] = dfM30['Close_M30']
   # Suppression de toutes les périodes "Mxx"
   dfM30 = dfM30.drop(['Open_M1' ],axis=1)
   dfM30 = dfM30.drop(['High_M1' ],axis=1)
   dfM30 = dfM30.drop(['Low_M1'  ],axis=1)
   dfM30 = dfM30.drop(['Close_M1'],axis=1)
   dfM30 = dfM30.drop(['Open_M5' ],axis=1)
   dfM30 = dfM30.drop(['High_M5' ],axis=1)
   dfM30 = dfM30.drop(['Low_M5'  ],axis=1)
   dfM30 = dfM30.drop(['Close_M5'],axis=1)
   dfM30 = dfM30.drop(['Open_M15' ],axis=1)
   dfM30 = dfM30.drop(['High_M15' ],axis=1)
   dfM30 = dfM30.drop(['Low_M15'  ],axis=1)
   dfM30 = dfM30.drop(['Close_M15'],axis=1)
   # dfM30 = dfM30.drop(['Open_M30' ],axis=1)
   # dfM30 = dfM30.drop(['High_M30' ],axis=1)
   # dfM30 = dfM30.drop(['Low_M30'  ],axis=1)
   # dfM30 = dfM30.drop(['Close_M30'],axis=1)
   dfM30 = add_prev_OpenClose(dfM30,'M30')
   # dfM30 = add_prev_OpenClose(dfM30,'H1')
   dfM30 = add_HA(dfM30,'M30')
   # dfM30 = add_HA(dfM30,'H1')
   dfM30 = add_log_close(dfM30,'M30')
   # dfM30 = add_log_close(dfM30,'H1')
   dfM30 = add_pivot(dfM30, 'H1')
   dfM30 = add_pivot(dfM30, 'H4')
   dfM30 = add_pivot(dfM30, 'D1')
   # suppressions à faire après avoir ajouté les pivots
   dfM30 = dfM30.drop(['Open_H1' ],axis=1)
   dfM30 = dfM30.drop(['High_H1' ],axis=1)
   dfM30 = dfM30.drop(['Low_H1'  ],axis=1)
   dfM30 = dfM30.drop(['Close_H1'],axis=1)
   dfM30 = dfM30.drop(['Open_H4' ],axis=1)
   dfM30 = dfM30.drop(['High_H4' ],axis=1)
   dfM30 = dfM30.drop(['Low_H4'  ],axis=1)
   dfM30 = dfM30.drop(['Close_H4'],axis=1)
   dfM30 = dfM30.drop(['Open_D1' ],axis=1)
   dfM30 = dfM30.drop(['High_D1' ],axis=1)
   dfM30 = dfM30.drop(['Low_D1'  ],axis=1)
   dfM30 = dfM30.drop(['Close_D1'],axis=1)
   #
   dfM30 = add_ta(dfM30)
   dfM30=add_potentiel_long_short(dfM30,2)
   dfM30=add_potentiel_long_short(dfM30,3)
   dfM30=add_potentiel_long_short(dfM30,5)
   dfM30=add_potentiel_long_short(dfM30,8)
   dfM30=add_potentiel_long_short(dfM30,13)
   dfM30=add_potentiel_long_short(dfM30,21)
   #Clean
   dfM30 = dfM30.replace([numpy.inf, -numpy.inf], numpy.nan)
   dfM30 = dfM30.dropna()
   return dfM30

def prepare_raw_dataset_H1(df):
   # H1
   dfH1 = dfMn_allData(df,'H1')
   dfH1['Open' ] = dfH1['Open_H1' ]
   dfH1['High' ] = dfH1['High_H1' ]
   dfH1['Low'  ] = dfH1['Low_H1'  ]
   dfH1['Close'] = dfH1['Close_H1']
   # Suppression de toutes les périodes "Mxx"
   dfH1 = dfH1.drop(['Open_M1'  ],axis=1)
   dfH1 = dfH1.drop(['High_M1'  ],axis=1)
   dfH1 = dfH1.drop(['Low_M1'   ],axis=1)
   dfH1 = dfH1.drop(['Close_M1' ],axis=1)
   dfH1 = dfH1.drop(['Open_M5'  ],axis=1)
   dfH1 = dfH1.drop(['High_M5'  ],axis=1)
   dfH1 = dfH1.drop(['Low_M5'   ],axis=1)
   dfH1 = dfH1.drop(['Close_M5' ],axis=1)
   dfH1 = dfH1.drop(['Open_M15' ],axis=1)
   dfH1 = dfH1.drop(['High_M15' ],axis=1)
   dfH1 = dfH1.drop(['Low_M15'  ],axis=1)
   dfH1 = dfH1.drop(['Close_M15'],axis=1)
   dfH1 = dfH1.drop(['Open_M30' ],axis=1)
   dfH1 = dfH1.drop(['High_M30' ],axis=1)
   dfH1 = dfH1.drop(['Low_M30'  ],axis=1)
   dfH1 = dfH1.drop(['Close_M30'],axis=1)
   # dfH1 = dfH1.drop(['Open_H1'  ],axis=1)
   # dfH1 = dfH1.drop(['High_H1'  ],axis=1)
   # dfH1 = dfH1.drop(['Low_H1'   ],axis=1)
   # dfH1 = dfH1.drop(['Close_H1' ],axis=1)
   dfH1 = add_prev_OpenClose(dfH1,'H1')
   dfH1 = add_HA(dfH1,'H1')
   dfH1 = add_log_close(dfH1,'H1')
   dfH1 = add_pivot(dfH1, 'H4')
   dfH1 = add_pivot(dfH1, 'D1')
   # suppressions à faire après avoir ajouté les pivots
   dfH1 = dfH1.drop(['Open_H4' ],axis=1)
   dfH1 = dfH1.drop(['High_H4' ],axis=1)
   dfH1 = dfH1.drop(['Low_H4'  ],axis=1)
   dfH1 = dfH1.drop(['Close_H4'],axis=1)
   dfH1 = dfH1.drop(['Open_D1' ],axis=1)
   dfH1 = dfH1.drop(['High_D1' ],axis=1)
   dfH1 = dfH1.drop(['Low_D1'  ],axis=1)
   dfH1 = dfH1.drop(['Close_D1'],axis=1)
   #
   dfH1 = add_ta(dfH1)
   dfH1=add_potentiel_long_short(dfH1,2)
   dfH1=add_potentiel_long_short(dfH1,3)
   dfH1=add_potentiel_long_short(dfH1,5)
   dfH1=add_potentiel_long_short(dfH1,8)
   dfH1=add_potentiel_long_short(dfH1,13)
   dfH1=add_potentiel_long_short(dfH1,21)
   #Clean
   dfH1 = dfH1.replace([numpy.inf, -numpy.inf], numpy.nan)
   dfH1 = dfH1.dropna()
   return dfH1

def get_raw_dataset_full_filename(dataset_name,symbol):
   return arbo.get_source_data_dir(arbo.get_py_dir(),dataset_name)+'\\DownloaderNN_'+symbol+'_M1_NNDataset_2018-2020-10-18.csv'

# -----------------------------------------------------------------------------
# Script de génération sur la base de données brutes (raw MT5)
# -----------------------------------------------------------------------------

def generate_raw_data(dataset_name):
   symbol='Usa500'
   #
   Usa500_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   #
   Usa500_dfH1  = prepare_raw_dataset_H1 (Usa500_dfRaw)
   Usa500_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   #
   Usa500_dfM30 = prepare_raw_dataset_M30(Usa500_dfRaw)
   Usa500_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   #
   Usa500_dfM15 = prepare_raw_dataset_M15(Usa500_dfRaw)
   Usa500_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   #
   Usa500_dfM5  = prepare_raw_dataset_M5 (Usa500_dfRaw)
   Usa500_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # Usa500_dfM1  = prepare_raw_dataset_M1 (Usa500_dfRaw)
   # Usa500_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   # symbol='UsaInd'
   # #
   # UsaInd_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   # #
   # UsaInd_dfH1  = prepare_raw_dataset_H1 (UsaInd_dfRaw)
   # UsaInd_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   # #
   # UsaInd_dfM30 = prepare_raw_dataset_M30(UsaInd_dfRaw)
   # UsaInd_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   # #
   # UsaInd_dfM15 = prepare_raw_dataset_M15(UsaInd_dfRaw)
   # UsaInd_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   # #
   # UsaInd_dfM5  = prepare_raw_dataset_M5 (UsaInd_dfRaw)
   # UsaInd_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # UsaInd_dfM1  = prepare_raw_dataset_M1 (UsaInd_dfRaw)
   # UsaInd_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   symbol='UsaTec'
   #
   UsaTec_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   #
   UsaTec_dfH1  = prepare_raw_dataset_H1 (UsaTec_dfRaw)
   UsaTec_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   #
   UsaTec_dfM30 = prepare_raw_dataset_M30(UsaTec_dfRaw)
   UsaTec_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   #
   UsaTec_dfM15 = prepare_raw_dataset_M15(UsaTec_dfRaw)
   UsaTec_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   #
   UsaTec_dfM5  = prepare_raw_dataset_M5 (UsaTec_dfRaw)
   UsaTec_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # # UsaTec_dfM1  = prepare_raw_dataset_M1 (UsaTec_dfRaw)
   # # UsaTec_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   symbol='Ger30'
   #
   Ger30_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   #
   Ger30_dfH1  = prepare_raw_dataset_H1 (Ger30_dfRaw)
   Ger30_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   #
   Ger30_dfM30 = prepare_raw_dataset_M30(Ger30_dfRaw)
   Ger30_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   #
   Ger30_dfM15 = prepare_raw_dataset_M15(Ger30_dfRaw)
   Ger30_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   #
   Ger30_dfM5  = prepare_raw_dataset_M5 (Ger30_dfRaw)
   Ger30_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # Ger30_dfM1  = prepare_raw_dataset_M1 (Ger30_dfRaw)
   # Ger30_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   symbol='EURUSD'
   #
   EURUSD_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   #
   EURUSD_dfH1  = prepare_raw_dataset_H1 (EURUSD_dfRaw)
   EURUSD_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   #
   EURUSD_dfM30 = prepare_raw_dataset_M30(EURUSD_dfRaw)
   EURUSD_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   #
   EURUSD_dfM15 = prepare_raw_dataset_M15(EURUSD_dfRaw)
   EURUSD_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   #
   EURUSD_dfM5  = prepare_raw_dataset_M5 (EURUSD_dfRaw)
   EURUSD_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # # EURUSD_dfM1  = prepare_raw_dataset_M1 (EURUSD_dfRaw)
   # # EURUSD_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   # symbol='USDJPY'
   #
   USDJPY_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   #
   USDJPY_dfH1  = prepare_raw_dataset_H1 (USDJPY_dfRaw)
   USDJPY_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   #
   USDJPY_dfM30 = prepare_raw_dataset_M30(USDJPY_dfRaw)
   USDJPY_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   #
   USDJPY_dfM15 = prepare_raw_dataset_M15(USDJPY_dfRaw)
   USDJPY_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   #
   USDJPY_dfM5  = prepare_raw_dataset_M5 (USDJPY_dfRaw)
   USDJPY_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # # USDJPY_dfM1  = prepare_raw_dataset_M1 (USDJPY_dfRaw)
   # # USDJPY_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   symbol='GOLD'
   #
   GOLD_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   #
   GOLD_dfH1  = prepare_raw_dataset_H1 (GOLD_dfRaw)
   GOLD_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   #
   GOLD_dfM30 = prepare_raw_dataset_M30(GOLD_dfRaw)
   GOLD_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   #
   GOLD_dfM15 = prepare_raw_dataset_M15(GOLD_dfRaw)
   GOLD_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   #
   GOLD_dfM5  = prepare_raw_dataset_M5 (GOLD_dfRaw)
   GOLD_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # GOLD_dfM1  = prepare_raw_dataset_M1 (GOLD_dfRaw)
   # GOLD_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   # symbol='LCrude'
   # #
   # LCrude_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   # #
   # LCrude_dfH1  = prepare_raw_dataset_H1 (LCrude_dfRaw)
   # LCrude_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   # #
   # LCrude_dfM30 = prepare_raw_dataset_M30(LCrude_dfRaw)
   # LCrude_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   # #
   # LCrude_dfM15 = prepare_raw_dataset_M15(LCrude_dfRaw)
   # LCrude_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   # #
   # LCrude_dfM5  = prepare_raw_dataset_M5 (LCrude_dfRaw)
   # LCrude_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   #
   # LCrude_dfM1  = prepare_raw_dataset_M1 (LCrude_dfRaw)
   # LCrude_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   #
   # symbol='Brent'
   # #
   # Brent_dfRaw = read_data(get_raw_dataset_full_filename(dataset_name,symbol))
   # #
   # Brent_dfH1  = prepare_raw_dataset_H1 (Brent_dfRaw)
   # Brent_dfH1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   # #
   # Brent_dfM30 = prepare_raw_dataset_M30(Brent_dfRaw)
   # Brent_dfM30.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   # #
   # Brent_dfM15 = prepare_raw_dataset_M15(Brent_dfRaw)
   # Brent_dfM15.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   # #
   # Brent_dfM5  = prepare_raw_dataset_M5 (Brent_dfRaw)
   # Brent_dfM5.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   # #
   # # Brent_dfM1  = prepare_raw_dataset_M1 (Brent_dfRaw)
   # # Brent_dfM1.to_csv(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))




   
# -----------------------------------------------------------------------------
# Script de chargement de données préparées
# -----------------------------------------------------------------------------

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


def get_prepared_dataset_full_filename(dataset_name, symbol, period):
   return arbo.get_source_data_dir(arbo.get_py_dir(),dataset_name)+'\\'+symbol+'_2018-2020-10-18_'+period+'.csv'

def read_prepared_dataset(full_filename):
   df=pandas.read_csv(full_filename, index_col=0, parse_dates=True)
   return df

def prefix_columns(df,symbol,period):
   oldCols = df.columns
   newCols = []
   for i in range(0,len(oldCols)):
      newCols.append(symbol+'_'+period+'_'+oldCols[i])
   return newCols

def create_big(period, \
               copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30,\
               copy_EURUSD, copy_USDJPY, \
               copy_GOLD, copy_LCrude):   #, copy_Brent):
   #
   newCols = prefix_columns(copy_Usa500,'Usa500',period)
   copy_Usa500.columns = newCols
   newCols = prefix_columns(copy_UsaInd,'UsaInd',period)
   copy_UsaInd.columns = newCols
   newCols = prefix_columns(copy_UsaTec,'UsaTec',period)
   copy_UsaTec.columns = newCols
   newCols = prefix_columns(copy_Ger30,'Ger30',period)
   copy_Ger30.columns = newCols
   newCols = prefix_columns(copy_EURUSD,'EURUSD',period)
   copy_EURUSD.columns = newCols
   newCols = prefix_columns(copy_USDJPY,'USDJPY',period)
   copy_USDJPY.columns = newCols
   newCols = prefix_columns(copy_GOLD,'GOLD',period)
   copy_GOLD.columns = newCols
   newCols = prefix_columns(copy_LCrude,'LCrude',period)
   copy_LCrude.columns = newCols
   # newCols = prefix_columns(copy_Brent,'Brent',period)
   # copy_Brent.columns = newCols
   #
   big = pandas.concat([copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30,\
                        copy_EURUSD, copy_USDJPY, \
                        copy_GOLD, copy_LCrude],join='inner',axis=1)
   # ajout de la colonne DateTime (qui est masquée mais présente en tant qu'index)
   big['DateTime'] = big.index
   return big

def load_prepared_data(dataset_name):
   symbol='Usa500'
   #
   Usa500_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   Usa500_dfH1
   Usa500_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   Usa500_dfM30
   Usa500_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   Usa500_dfM15
   Usa500_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   Usa500_dfM5
   # Usa500_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # Usa500_dfM1
   #
   symbol='UsaInd'
   #
   UsaInd_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   UsaInd_dfH1
   UsaInd_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   UsaInd_dfM30
   UsaInd_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   UsaInd_dfM15
   UsaInd_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   UsaInd_dfM5
   # UsaInd_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # UsaInd_dfM1
   #
   symbol='UsaTec'
   #
   UsaTec_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   UsaTec_dfH1
   UsaTec_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   UsaTec_dfM30
   UsaTec_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   UsaTec_dfM15
   UsaTec_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   UsaTec_dfM5
   # UsaTec_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # UsaTec_dfM1
   #
   symbol='Ger30'
   #
   Ger30_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   Ger30_dfH1
   Ger30_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   Ger30_dfM30
   Ger30_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   Ger30_dfM15
   Ger30_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   Ger30_dfM5
   # Ger30_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # Ger30_dfM1
   #
   symbol='EURUSD'
   #
   EURUSD_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   EURUSD_dfH1
   EURUSD_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   EURUSD_dfM30
   EURUSD_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   EURUSD_dfM15
   EURUSD_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   EURUSD_dfM5
   # EURUSD_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # EURUSD_dfM1
   #
   symbol='USDJPY'
   #
   USDJPY_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   USDJPY_dfH1
   USDJPY_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   USDJPY_dfM30
   USDJPY_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   USDJPY_dfM15
   USDJPY_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   USDJPY_dfM5
   # USDJPY_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # USDJPY_dfM1
   #
   symbol='GOLD'
   #
   GOLD_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   GOLD_dfH1
   GOLD_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   GOLD_dfM30
   GOLD_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   GOLD_dfM15
   GOLD_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   GOLD_dfM5
   # GOLD_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # GOLD_dfM1
   #
   symbol='LCrude'
   #
   LCrude_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   LCrude_dfH1
   LCrude_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   LCrude_dfM30
   LCrude_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   LCrude_dfM15
   LCrude_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   LCrude_dfM5
   # LCrude_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # LCrude_dfM1
   #
   # symbol='Brent'
   # #
   # Brent_dfH1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'H1'))
   # Brent_dfH1
   # Brent_dfM30 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M30'))
   # Brent_dfM30
   # Brent_dfM15 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M15'))
   # Brent_dfM15
   # Brent_dfM5 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M5'))
   # Brent_dfM5
   # # Brent_dfM1 = read_prepared_dataset(get_prepared_dataset_full_filename(dataset_name,symbol,'M1'))
   # # Brent_dfM1
   #
   #
   period='H1'
   copy_Usa500 = Usa500_dfH1.copy()
   copy_UsaInd = UsaInd_dfH1.copy()
   copy_UsaTec = UsaTec_dfH1.copy()
   copy_Ger30  = Ger30_dfH1.copy()
   copy_EURUSD = EURUSD_dfH1.copy()
   copy_USDJPY = USDJPY_dfH1.copy()
   copy_GOLD   = GOLD_dfH1.copy()
   copy_LCrude = LCrude_dfH1.copy()
   # copy_Brent  =  Brent_dfH1.copy()
   big_H1 = create_big(period, \
                       copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30,\
                       copy_EURUSD, copy_USDJPY, \
                       copy_GOLD, copy_LCrude)#, copy_Brent)
   #
   period='M30'
   copy_Usa500 = Usa500_dfM30.copy()
   copy_UsaInd = UsaInd_dfM30.copy()
   copy_UsaTec = UsaTec_dfM30.copy()
   copy_Ger30  =  Ger30_dfM30.copy()
   copy_EURUSD = EURUSD_dfM30.copy()
   copy_USDJPY = USDJPY_dfM30.copy()
   copy_GOLD   = GOLD_dfM30.copy()
   copy_LCrude = LCrude_dfM30.copy()
   # copy_Brent  =  Brent_dfM30.copy()
   big_M30 = create_big(period, \
                        copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30,\
                        copy_EURUSD, copy_USDJPY, \
                        copy_GOLD, copy_LCrude)#, copy_Brent)
   #
   period='M15'
   copy_Usa500 = Usa500_dfM15.copy()
   copy_UsaInd = UsaInd_dfM15.copy()
   copy_UsaTec = UsaTec_dfM15.copy()
   copy_Ger30  =  Ger30_dfM15.copy()
   copy_EURUSD = EURUSD_dfM15.copy()
   copy_USDJPY = USDJPY_dfM15.copy()
   copy_GOLD   = GOLD_dfM15.copy()
   copy_LCrude = LCrude_dfM15.copy()
   # copy_Brent  =  Brent_dfM15.copy()
   big_M15 = create_big(period, \
                        copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30,\
                        copy_EURUSD, copy_USDJPY, \
                        copy_GOLD, copy_LCrude)#, copy_Brent)
   #
   period='M5'
   copy_Usa500 = Usa500_dfM5.copy()
   copy_UsaInd = UsaInd_dfM5.copy()
   copy_UsaTec = UsaTec_dfM5.copy()
   copy_Ger30  =  Ger30_dfM5.copy()
   copy_EURUSD = EURUSD_dfM5.copy()
   copy_USDJPY = USDJPY_dfM5.copy()
   copy_GOLD   = GOLD_dfM5.copy()
   copy_LCrude = LCrude_dfM5.copy()
   # copy_Brent  =  Brent_dfM5.copy()
   big_M5 = create_big(period, \
                       copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30,\
                       copy_EURUSD, copy_USDJPY, \
                       copy_GOLD, copy_LCrude)#, copy_Brent)
   #
   # period='M1'
   # copy_Usa500 = Usa500_dfM1.copy()
   # copy_UsaInd = UsaInd_dfM1.copy()
   # copy_UsaTec = UsaTec_dfM1.copy()
   # copy_Ger30  =  Ger30_dfM1.copy()
   # copy_EURUSD = EURUSD_dfM1.copy()
   # copy_USDJPY = USDJPY_dfM1.copy()
   # copy_GOLD   = GOLD_dfM1.copy()
   # copy_LCrude = LCrude_dfM1.copy()
   # copy_Brent  =  Brent_dfM1.copy()
   # big_M1 = create_big(period, \
   #                     copy_Usa500,copy_UsaInd,copy_UsaTec,copy_Ger30,\
   #                     copy_EURUSD, copy_USDJPY, \
   #                     copy_GOLD, copy_LCrude, copy_Brent)
   #
   # dates finales manquantes sur M15 => nettoyage
   big_H1  = big_H1 [(big_H1 ['DateTime']<="2020-10-29 00:00:00")]
   big_M30 = big_M30[(big_M30['DateTime']<="2020-10-29 00:00:00")]
   big_M15 = big_M15[(big_M15['DateTime']<="2020-10-29 00:00:00")]
   big_M5  = big_M5 [(big_M5 ['DateTime']<="2020-10-29 00:00:00")]
   #big_M1 = big_M1 [(big_M1 ['DateTime']<="2020-10-29 00:00:00")]
   #
   return   Usa500_dfH1, Usa500_dfM30, Usa500_dfM15, Usa500_dfM5, \
            UsaInd_dfH1, UsaInd_dfM30, UsaInd_dfM15, UsaInd_dfM5, \
            UsaTec_dfH1, UsaTec_dfM30, UsaTec_dfM15, UsaTec_dfM5, \
            Ger30_dfH1,  Ger30_dfM30,  Ger30_dfM15,  Ger30_dfM5, \
            EURUSD_dfH1, EURUSD_dfM30, EURUSD_dfM15, EURUSD_dfM5, \
            USDJPY_dfH1, USDJPY_dfM30, USDJPY_dfM15, USDJPY_dfM5, \
            GOLD_dfH1,   GOLD_dfM30,   GOLD_dfM15,   GOLD_dfM5, \
            LCrude_dfH1, LCrude_dfM30, LCrude_dfM15, LCrude_dfM5, \
            big_H1,      big_M30,      big_M15,      big_M5


#
# Points d'entrée et variables globales à partir d'ici
#

# dataset_name = 'work'

# Usa500_dfH1, Usa500_dfM30, Usa500_dfM15, Usa500_dfM5, \
# UsaInd_dfH1, UsaInd_dfM30, UsaInd_dfM15, UsaInd_dfM5, \
# UsaTec_dfH1, UsaTec_dfM30, UsaTec_dfM15, UsaTec_dfM5, \
# Ger30_dfH1,  Ger30_dfM30,  Ger30_dfM15,  Ger30_dfM5,  \
# big_H1,      big_M30,      big_M15,      big_M5 = load_prepared_data(dataset_name)


# Usa500_dfH1, Usa500_dfM30, Usa500_dfM15, Usa500_dfM5, Usa500_dfM1, \
# UsaInd_dfH1, UsaInd_dfM30, UsaInd_dfM15, UsaInd_dfM5, UsaInd_dfM1, \
# UsaTec_dfH1, UsaTec_dfM30, UsaTec_dfM15, UsaTec_dfM5, UsaTec_dfM1, \
# Ger30_dfH1,  Ger30_dfM30,  Ger30_dfM15,  Ger30_dfM5,  Ger30_dfM1,  \
# big_H1,      big_M30,      big_M15,      big_M5,      big_M1 = load_prepared_data(dataset_name)


# pCols = ['pRSI','pBOLL','pADX','pCCI','pMassIndex','pATR','pUltimateOscillator','pWilliamsRIndicator']
# for i in range (0,len(pCols)):
   # for j in range(i+1,len(pCols)):
      # numpy.corrcoef(Ger30_dfM5[pCols[i]],Ger30_dfM5[pCols[j]])

# future use
# def steps_from_period(period):
   # stepMin=0
   # stepHour=0
   # if(period=="M1"):
      # stepMin=1
      # stepHour=0
   # elif(period=="M5"):
      # stepMin=5
      # stepHour=0
   # elif(period=="M15"):
      # stepMin=15
      # stepHour=0
   # elif(period=="M30"):
      # stepMin=30
      # stepHour=0
   # elif(period=="H1"):
      # stepMin=60
      # stepHour=1
   # elif(period=="H4"):
      # stepMin=60
      # stepHour=4
   # return stepMin, stepHour
