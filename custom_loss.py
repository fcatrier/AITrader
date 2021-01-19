#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import keras
import numpy
import pandas
import tensorflow


def my_loss_fn1(df_y_1d_test, df_y_pred):
   cce = keras.losses.CategoricalCrossentropy()
   return cce(df_y_1d_test, df_y_pred)

def my_loss_fn2(y_pred, y_true):
   np_y_true = numpy.array(y_true.shape)
   np_y_pred = numpy.array(y_pred.shape)
   numpy.save('e:\\my_loss_fn2.npy', np_y_true)
   cce = keras.losses.CategoricalCrossentropy()
   return cce(y_pred, y_true)


def my_loss_fn3(df_y_1d_test, df_y_pred):
   df_y_1d_test[0] = df_y_1d_test[0]-1
   #  
   df_result = pandas.concat([df_y_pred,df_y_1d_test],join='inner',axis=1)
   df_result['res'] = 0
   df_result.columns = ['df_y_pred','df_y_1d_test','res']
   #
   df_result['res'][((df_result['df_y_pred']==0)&( df_result['df_y_1d_test']==1))] = 1
   df_result['res'][((df_result['df_y_pred']==0)&( df_result['df_y_1d_test']==2))] = 0.33
   #
   df_result['res'][((df_result['df_y_pred']==1)&( df_result['df_y_1d_test']==0))] = 1
   df_result['res'][((df_result['df_y_pred']==1)&( df_result['df_y_1d_test']==2))] = 0.33
   #
   df_result['res'][((df_result['df_y_pred']==3)&( df_result['df_y_1d_test']==0))] = 0.25
   df_result['res'][((df_result['df_y_pred']==3)&( df_result['df_y_1d_test']==2))] = 0.25
   #
   cce = keras.losses.CategoricalCrossentropy()
   return cce(df_y_1d_test, df_y_pred) * df_result['res'].sum()


def my_loss_fn4(df_y_1d_test, df_y_pred):
   #df_y_1d_test[0] = df_y_1d_test[0]-1
   #  
   df_result = pandas.concat([df_y_pred,df_y_1d_test],join='inner',axis=1)
   df_result['res'] = 0
   df_result.columns = ['df_y_pred','df_y_1d_test','res']
   #
   df_result['res'][((df_result['df_y_pred']==0)&( df_result['df_y_1d_test']==1))] = 1
   df_result['res'][((df_result['df_y_pred']==0)&( df_result['df_y_1d_test']==2))] = 0.33
   #
   df_result['res'][((df_result['df_y_pred']==1)&( df_result['df_y_1d_test']==0))] = 1
   df_result['res'][((df_result['df_y_pred']==1)&( df_result['df_y_1d_test']==2))] = 0.33
   #
   df_result['res'][((df_result['df_y_pred']==3)&( df_result['df_y_1d_test']==0))] = 0.25
   df_result['res'][((df_result['df_y_pred']==3)&( df_result['df_y_1d_test']==2))] = 0.25
   #
   cce = keras.losses.CategoricalCrossentropy()
   return cce(df_y_1d_test, df_y_pred) * df_result['res'].sum()


def my_loss_fn5(y_true, y_pred):
   try:
      #
      #y_true = numpy.array(y_true)
      #y_pred = numpy.array(y_pred)
      #
      arr_malus = []
      for i in range(0,y_true.shape[0]):
         malus = 0.0
         # pred = 0, true = 0
         if((y_pred[i][0]>0.35)&(y_true[i][0]==1)):
            malus += 0.0
         # pred = 0, true = 1
         if((y_pred[i][0]>0.35)&(y_true[i][1]==1)):
            malus += 1.0
         # pred = 0, true = 2
         if((y_pred[i][0]>0.35)&(y_true[i][2]==1)):
            malus += 0.33
         # pred = 1, true = 0
         if((y_pred[i][1]>0.35)&(y_true[i][0]==1)):
            malus += 1.0
         # pred = 1, true = 1
         if((y_pred[i][1]>0.35)&(y_true[i][1]==1)):
            malus += 0.0
         # pred = 1, true = 2
         if((y_pred[i][1]>0.35)&(y_true[i][2]==1)):
            malus += 0.33
         # pred = 2, true = 0
         if((y_pred[i][2]>0.35)&(y_true[i][0]==1)):
            malus += 0.25
         # pred = 2, true = 1
         if((y_pred[i][2]>0.35)&(y_true[i][1]==1)):
            malus += 0.25
         # pred = 2, true = 2
         if((y_pred[i][2]>0.35)&(y_true[i][2]==1)):
            malus += 0.0
         #
         arr_malus.append(malus)
      #
      np_malus = numpy.array(arr_malus)
      tf_malus = tensorflow.convert_to_tensor(np_malus, dtype=tensorflow.float32)
      return tf_malus
   except:
      return 10.0
   #
   #return tf_malus

# https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras

# t = [0.1,0.2,0.3,0.4]
# t = numpy.array(t)
# t01 = numpy.where(t >= 0.35, 1, 0)

# numpy.where
# numpy.where(np_y_pred > 0.35, 1, 0)
# numpy.where(t >= 0.3, 1, 0)
# # array([0, 0, 0, 1, 1, 1])


# # tf.make_ndarray(
    # # tensor
# # )


# # arr = [0.1,0.2,0.3]
# # np_arr = numpy.array(arr)

# # np_arr_back = numpy.array(tf_arr)


# y_true = tensorflow.constant([[1, 0, 0]      , [1, 0, 0]      , [0, 1, 0]      , [1, 0, 0]      , [0, 1, 0]]     )
# y_pred = tensorflow.constant([[0.95, 0.05, 0], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]  ])

# np_y_true = numpy.array(y_true)
# np_y_pred = numpy.array(y_pred)
# np_y_pred = numpy.where(np_y_pred > 0.35, 1, 0)

# np_y_true
# np_y_pred

# np_delta = np_y_true - np_y_pred
# np_delta

# array([[ 0,  0,  0],
       # [ 1, -1,  0],
       # [-1,  1,  0],
       # [ 1,  0, -1],
       # [ 0,  1, -1]])

# f = [1,-1,0]   # error pred 1 true 2 => malus
# f = [-1,1,0]   # error pred 2 true 1 => malus
# f = [1,0,-1]   # error pred 1 true 3 => malus
# f = [0,1,-1]   # error pred 2 true 3 => malus

# np_f = numpy.array(f)

# for row in np_delta:
   # print((row==np_f).all())



# r1 = np_delta[0]
# r1

# r1 == np_f


# f = [1,-1,0]
# np_f = numpy.array(f)
# np_f

# f1 = [1,-1,0]
# np_f1 = numpy.array(f)
# np_f1

# (np_f==np_f1).all()
# numpy.array_equal(np_f,np_f1)

# np_delta * np_f


# # loss = tensorflow.keras.losses.categorical_crossentropy(y_true, y_pred)
# # loss

# # arr_malus = []
# # for i in range(0,y_true.shape[0]):
   # # malus = 0.0
   # # for j in range(0,y_true.shape[1]):
      
      # # if((y_pred[i][j]==0)&(y_true[i][j]==1)):
         # # malus += 1.0
      # # if((y_pred[i][j]==0)&(y_true[i][j]==2)):
         # # malus += 0.33
      
      # # if((y_pred[i][j]==1)&(y_true[i][j]==0)):
         # # malus += 1
      # # if((y_pred[i][j]==1)&(y_true[i][j]==2)):
         # # malus += 0.33
      
      # # if((y_pred[i][j]==3)&(y_true[i][j]==0)):
         # # malus += 0.25
      # # if((y_pred[i][j]==3)&(y_true[i][j]==2)):
         # # malus += 0.25
      
   # # arr_malus.append(malus)

# # np_malus = numpy.array(arr_malus)



# # arr1 = [ 1,   2 ]
# # arr2 = [ 0.5, 2 ]

# # np_arr1 = numpy.array(arr1)
# # np_arr2 = numpy.array(arr2)

# # np_arr3 = np_arr1 * np_arr2


# en développement
# https://github.com/keras-team/keras/issues/2115
# https://stackoverflow.com/questions/56696069/keras-apply-different-weight-to-different-misclassification
# https://www.gitmemory.com/issue/keras-team/keras/2115/569543576

# from keras import backend as K

# def my_loss_fn(df_y_1d_test, df_y_pred):
   # df_y_1d_test[0] = df_y_1d_test[0]-1
   # #  
   # df_result = pandas.concat([df_y_pred,df_y_1d_test],join='inner',axis=1)
   # df_result['res'] = 0
   # df_result.columns = ['df_y_pred','df_y_1d_test','res']
   # #
   # df_result['res'][((df_result['df_y_pred']==0)&( df_result['df_y_1d_test']==1))] = 1
   # df_result['res'][((df_result['df_y_pred']==0)&( df_result['df_y_1d_test']==2))] = 0.33
   # #
   # df_result['res'][((df_result['df_y_pred']==1)&( df_result['df_y_1d_test']==0))] = 1
   # df_result['res'][((df_result['df_y_pred']==1)&( df_result['df_y_1d_test']==2))] = 0.33
   # #
   # df_result['res'][((df_result['df_y_pred']==3)&( df_result['df_y_1d_test']==0))] = 0.25
   # df_result['res'][((df_result['df_y_pred']==3)&( df_result['df_y_1d_test']==2))] = 0.25
   # #
   # #return tensorflow.Tensor(numpy.array(df_result['res'].values),value_index=-1,dtype=float)
   # return numpy.array(df_result['res'].values)

# from keras import backend as K

# def my_loss_fn2(y_pred, y_true):
   # #loss = K.categorical_crossentropy(y_true,y_pred)
   # cce = keras.losses.CategoricalCrossentropy()
   # print(cce(y_pred, y_true))
   # return cce(y_pred, y_true)

