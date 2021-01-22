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
        elif model_dict['model_architecture'] == 'ResNet50':
            model = self.__create_model_ResNet50(model_dict)
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

    def __create_model_ResNet50(self, model_dict):
        #
        return self.__resnet50(model_dict)

    @staticmethod
    def res_conv(x, s, filters):
        #
        import keras
        #
        x_skip = x

        # first block
        x = keras.layers.Conv1D(filters, kernel_size=1, strides=s, padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        # when s = 2 then it is like downsizing the feature map
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)

        # second block
        x = keras.layers.Conv1D(filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)

        #third block
        x = keras.layers.Conv1D(filters, kernel_size=1, strides=1, padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.BatchNormalization()(x)

        # shortcut
        x_skip = keras.layers.Conv1D(filters, kernel_size=1, strides=s, padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x_skip)
        x_skip = keras.layers.BatchNormalization()(x_skip)

        # add
        x = keras.layers.Add()([x, x_skip])
        x = keras.layers.Activation(keras.activations.relu)(x)

        return x

    # def conv_block(input_tensor,
    #                kernel_size,
    #                filters,
    #                stage,
    #                block,
    #                strides=2):
    #     """A block that has a conv layer at shortcut.
    #     # Arguments
    #         input_tensor: input tensor
    #         kernel_size: default 3, the kernel size of
    #             middle conv layer at main path
    #         filters: list of integers, the filters of 3 conv layer at main path
    #         stage: integer, current stage label, used for generating layer names
    #         block: 'a','b'..., current block label, used for generating layer names
    #         strides: Strides for the first conv layer in the block.
    #     # Returns
    #         Output tensor for the block.
    #     Note that from stage 3,
    #     the first conv layer at main path is with strides=(2, 2)
    #     And the shortcut should have strides=(2, 2) as well
    #     """
    #     filters1, filters2, filters3 = filters
    #     if backend.image_data_format() == 'channels_last':
    #         bn_axis = 3
    #     else:
    #         bn_axis = 1
    #     conv_name_base = 'res' + str(stage) + block + '_branch'
    #     bn_name_base = 'bn' + str(stage) + block + '_branch'
    #
    #     x = layers.Conv1D(filters1, 1, strides=strides,
    #                       kernel_initializer='he_normal',
    #                       name=conv_name_base + '2a')(input_tensor)
    #     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    #     x = layers.Activation('relu')(x)
    #
    #     x = layers.Conv1D(filters2, kernel_size, padding='same',
    #                       kernel_initializer='he_normal',
    #                       name=conv_name_base + '2b')(x)
    #     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    #     x = layers.Activation('relu')(x)
    #
    #     x = layers.Conv2D(filters3, (1, 1),
    #                       kernel_initializer='he_normal',
    #                       name=conv_name_base + '2c')(x)
    #     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    #
    #     shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
    #                              kernel_initializer='he_normal',
    #                              name=conv_name_base + '1')(input_tensor)
    #     shortcut = layers.BatchNormalization(
    #         axis=bn_axis, name=bn_name_base + '1')(shortcut)
    #
    #     x = layers.add([x, shortcut])
    #     x = layers.Activation('relu')(x)
    #     return x
    #
    # def identity_block(input_tensor, kernel_size, filters, stage, block):
    #     """The identity block is the block that has no conv layer at shortcut.
    #     # Arguments
    #         input_tensor: input tensor
    #         kernel_size: default 3, the kernel size of
    #             middle conv layer at main path
    #         filters: list of integers, the filters of 3 conv layer at main path
    #         stage: integer, current stage label, used for generating layer names
    #         block: 'a','b'..., current block label, used for generating layer names
    #     # Returns
    #         Output tensor for the block.
    #     """
    #     filters1, filters2, filters3 = filters
    #     if backend.image_data_format() == 'channels_last':
    #         bn_axis = 3
    #     else:
    #         bn_axis = 1
    #     conv_name_base = 'res' + str(stage) + block + '_branch'
    #     bn_name_base = 'bn' + str(stage) + block + '_branch'
    #
    #     x = layers.Conv2D(filters1, (1, 1),
    #                       kernel_initializer='he_normal',
    #                       name=conv_name_base + '2a')(input_tensor)
    #     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    #     x = layers.Activation('relu')(x)
    #
    #     x = layers.Conv2D(filters2, kernel_size,
    #                       padding='same',
    #                       kernel_initializer='he_normal',
    #                       name=conv_name_base + '2b')(x)
    #     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    #     x = layers.Activation('relu')(x)
    #
    #     x = layers.Conv2D(filters3, (1, 1),
    #                       kernel_initializer='he_normal',
    #                       name=conv_name_base + '2c')(x)
    #     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    #
    #     x = layers.add([x, input_tensor])
    #     x = layers.Activation('relu')(x)
    #     return x

    def __resnet50(self, model_dict):
        #
        import keras
        #

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = keras.Input(shape = (model_dict['input_timesteps'],model_dict['input_features']))
        x = keras.layers.Conv1D(64, kernel_size=1, strides=2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling1D(2, strides=2)(x)

        #2nd stage
        # frm here on only conv block and identity block, no pooling

        x = self.res_conv(x, s=1, filters=64)
        # x = res_identity(x, filters=(64, 256))
        # x = res_identity(x, filters=(64, 256))
        #
        # # 3rd stage
        #
        # x = res_conv(x, s=2, filters=(128, 512))
        # x = res_identity(x, filters=(128, 512))
        # x = res_identity(x, filters=(128, 512))
        # x = res_identity(x, filters=(128, 512))
        #
        # # 4th stage
        #
        # x = res_conv(x, s=2, filters=(256, 1024))
        # x = res_identity(x, filters=(256, 1024))
        # x = res_identity(x, filters=(256, 1024))
        # x = res_identity(x, filters=(256, 1024))
        # x = res_identity(x, filters=(256, 1024))
        # x = res_identity(x, filters=(256, 1024))
        #
        # # 5th stage
        #
        # x = res_conv(x, s=2, filters=(512, 2048))
        # x = res_identity(x, filters=(512, 2048))
        # x = res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection

        x = keras.layers.AveragePooling1D(2, padding='same')(x)

        x = keras.layers.Flatten()(x)
        #
        # sortie des classes
        #
        x = keras.layers.Dense(model_dict['output_shape'], activation='softmax', kernel_initializer='he_normal')(x) #multi-class

        # define the model

        model = keras.Model(outputs=x, name='Resnet50')

        return model

