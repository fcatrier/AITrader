#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

class ModelsCatalog:

    def __init__(self):
        pass

    def create_model(self, model_dict):
        #
        model = None
        if model_dict['model_architecture'] == 'Conv1D_Dense':
            model = self.__create_model_Conv1D_Dense(model_dict)
        elif model_dict['model_architecture'] == 'Dense_Dense':
            model = self.__create_model_Dense_Dense(model_dict)
        elif model_dict['model_architecture'] == 'LSTM_Dense':
            model = self.__create_model_LSTM_Dense(model_dict)
        else:
            raise ValueError('Unknown model name')
        #
        return model

    @staticmethod
    def __create_model_Dense_Dense(model_dict):
        #
        import keras
        #
        model = keras.Sequential()
        #
        # entrée du modèle
        model.add(keras.Input(shape=(model_dict['input_timesteps'], model_dict['input_features'])))
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(model_dict['config_Dense_units'], activation='relu'))
        model.add(keras.layers.Dropout(model_dict['dropout_rate']))
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(model_dict['config_Dense_units2'], activation='relu'))
        model.add(keras.layers.Dropout(model_dict['dropout_rate']))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(model_dict['output_shape'], activation='softmax'))
        return model

    @staticmethod
    def __create_model_LSTM_Dense(model_dict):
        #
        import keras
        #
        model = keras.Sequential()
        #
        # entrée du LSTM
        #
        model.add(keras.layers.LSTM(model_dict['config_GRU_LSTM_units'], return_sequences=True,
                                    input_shape=(model_dict['input_timesteps'],
                                                 model_dict['input_features'])))
        model.add(keras.layers.Dropout(model_dict['dropout_rate']))
        #
        # ajout d'une couche Flatten intermédiaire pour ne pas avoir à gérer des soucis de
        # taille de données (=> à partir d'ici on est en 1D)
        #
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(model_dict['config_Dense_units'], activation='relu'))
        model.add(keras.layers.Dropout(model_dict['dropout_rate']))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(model_dict['output_shape'], activation='softmax'))
        return model

    @staticmethod
    def __create_model_Conv1D_Dense(model_dict):
        #
        import keras
        from keras.layers import Dropout
        from keras.layers.convolutional import Conv1D, MaxPooling1D
        #        #
        model = keras.Sequential()
        #
        model.add(Conv1D(filters=model_dict['conv1D_block1_filters'],
                         kernel_size=model_dict['conv1D_block1_kernel_size'],
                         activation='relu',
                         input_shape=(model_dict['input_timesteps'],
                                      model_dict['input_features'])))
        model.add(Dropout(model_dict['dropout_rate']))
        #
        if model_dict['conv1D_block1_MaxPooling1D_pool_size'] != 0:
            model.add(MaxPooling1D(model_dict['conv1D_block1_MaxPooling1D_pool_size']))
            model.add(Dropout(model_dict['dropout_rate']))
        #
        # ajout d'une couche Flatten intermédiaire pour ne pas avoir à gérer des soucis de
        # taille de données (=> à partir d'ici on est en 1D)
        #
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(model_dict['config_Dense_units'], activation='relu'))
        model.add(keras.layers.Dropout(model_dict['dropout_rate']))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(model_dict['output_shape'], activation='softmax'))
        return model
