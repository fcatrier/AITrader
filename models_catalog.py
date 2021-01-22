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
        elif model_dict['model_architecture'] == 'Conv1D_x2_Dense':
            model = self.__create_model_Conv1D_x2_Dense(model_dict)
        elif model_dict['model_architecture'] == 'Dense_Dense':
            model = self.__create_model_Dense_Dense(model_dict)
        elif model_dict['model_architecture'] == 'LSTM_Dense':
            model = self.__create_model_LSTM_Dense(model_dict)
        elif model_dict['model_architecture'] == 'ResNet1D_dev':
            model = self.__create_model_resnet1D_dev(model_dict)
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

    @staticmethod
    def __create_model_Conv1D_x2_Dense(model_dict):
        #
        import keras
        from keras.layers import Dropout
        from keras.layers.convolutional import Conv1D, MaxPooling1D
        #
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
        model.add(Conv1D(filters=model_dict['conv1D_block1_filters'],
                         kernel_size=model_dict['conv1D_block1_kernel_size'],
                         activation='relu'))
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

    def __create_model_resnet1D_dev(self, model_dict):
        #
        return self.__resnet1D_dev(model_dict)

    def __resnet1D_dev(self, model_dict):
        #
        #  https://pylessons.com/Keras-ResNet-tutorial/
        #  https://missinglink.ai/guides/keras/keras-resnet-building-training-scaling-residual-nets-keras/
        #
        import keras
        #
        x, x_input = self.__resnet_stage_entry(model_dict)
        #
        x = self.res_conv(x, 64, model_dict)
        # x = self.res_conv(x, 128, model_dict)
        # x = self.res_conv(x, 256, model_dict)
        # ...
        #
        model = self.__resnet_stage_output(x, x_input, model_dict)
        #
        return model

    def __resnet_stage_entry(self, model_dict):
        #
        import keras
        #

        # # Define the input as a tensor with shape input_shape
        # X_input = Input(input_shape)
        # # Zero-Padding
        # X = ZeroPadding2D((3, 3))(X_input)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x_input = keras.Input(shape = (model_dict['input_timesteps'],model_dict['input_features']))

        x = keras.layers.Conv1D(64, kernel_size=1)(x_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling1D(2)(x)

        return x, x_input

    @staticmethod
    def res_conv(x, filters, model_dict):
        #
        import keras
        #
        x_skip = x

        # first block
        x = keras.layers.Conv1D(filters, kernel_size=1)(x)
        # when s = 2 then it is like downsizing the feature map
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)

        # second block
        x = keras.layers.Conv1D(filters, kernel_size=3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)

        #third block
        x = keras.layers.Conv1D(filters, kernel_size=1)(x)
        x = keras.layers.BatchNormalization()(x)

        # shortcut
        x_skip = keras.layers.Conv1D(filters, kernel_size=1)(x_skip)
        x_skip = keras.layers.BatchNormalization()(x_skip)

        # add
        x = keras.layers.Add()([x, x_skip])
        x = keras.layers.Activation(keras.activations.relu)(x)

        return x


    def __resnet_stage_output(self, x, x_input, model_dict):

        x = keras.layers.AveragePooling1D(2)(x)
        x = keras.layers.Flatten()(x)
        #
        # sortie des classes
        #
        x = keras.layers.Dense(model_dict['output_shape'], activation='softmax', kernel_initializer='he_normal')(x) #multi-class

        # define the model
        model = keras.Model(inputs = x_input, outputs = x, name='ResNet1D_dev')

        # # Create model
        # model = Model(inputs = X_input, outputs = X, name='ResNet50')

        return model


