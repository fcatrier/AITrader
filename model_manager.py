#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

from models_catalog import ModelsCatalog
import learn_evaluate_results


class ModelManager:
    #
    __model_dict = dict([
        #
        ('model_architecture', 'None'),
        #
        ('conv1D_block1_filters', -1),
        ('conv1D_block1_kernel_size', -1),
        ('conv1D_block1_MaxPooling1D_pool_size', -1),
        #
        ('config_GRU_LSTM_units', -1),
        #
        ('config_Dense_units', -1),
        ('config_Dense_units2', -1),
        #
        ('dropout_rate', 0.0),
        ('optimizer_name', 'None'),
        ('optimizer_modif_learning_rate', 0.0),
        #
        ('input_features', -1),
        ('input_timesteps', -1),
        ('output_shape', -1),
        ('model_count_params', -1),
        ('X_train_params', -1),
        #
        ('fit_batch_size', 32),
        ('fit_epochs_max', 500),
        ('fit_earlystopping_patience', 100)
    ])

    def __init__(self):
        pass

    def get_properties(self):
        return self.__model_dict

    def update_properties_from_learning_data(self, learning_data):
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
        self.__model_dict['input_features'] = input_features
        self.__model_dict['input_timesteps'] = input_timesteps
        self.__model_dict['output_shape'] = output_shape
        self.__model_dict['train_params'] = train_params

    def update_properties(self, model_dict):
        self.__model_dict = model_dict

    def __create_optimizer(self):
        #
        import keras
        #
        if self.__model_dict['optimizer_name'] == 'sgd':
            learning_rate = 0.01 * self.__model_dict['optimizer_modif_learning_rate']
            return keras.optimizers.SGD(learning_rate)
        elif self.__model_dict['optimizer_name'] == 'adam':
            learning_rate = 0.001 * self.__model_dict['optimizer_modif_learning_rate']
            return keras.optimizers.Adam(learning_rate)
        else:
            raise ValueError('Unknown optimizer_choice')

    def create_compile_model(self):
        #
        models_catalog = ModelsCatalog()
        model = models_catalog.create_model(model_dict)
        #
        model.compile(optimizer=self.__create_optimizer(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        #
        self.__model_dict['model_count_params'] = model.count_params()
        #
        return model

    def display_model_info(self, model):
        print(model.summary())
        print("model.count_params()=", model.count_params())
        print("train_params : ", self.__model_dict['train_params'])

    def fit(self, learning_data):
        #
        import keras
        #
        model = self.create_compile_model()
        #
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 patience=self.__model_dict['fit_earlystopping_patience'],
                                                 restore_best_weights=True)
        history = model.fit(learning_data['train']['np_X'],
                            learning_data['train']['df_y_Nd'],
                            self.__model_dict['fit_batch_size'],
                            shuffle=False,
                            epochs=self.__model_dict['fit_epochs_max'],
                            callbacks=[callback],
                            verbose=0,
                            validation_data=(learning_data['val']['np_X'],
                                             learning_data['val']['df_y_Nd']))
        #
        train_loss = round(min(history.history['loss']), 3)
        val_loss = round(min(history.history['val_loss']), 3)
        train_accuracy = round(max(history.history['accuracy']), 2)
        val_accuracy = round(max(history.history['val_accuracy']), 2)
        print("train_loss     = ", train_loss)
        print("val_loss       = ", val_loss)
        print("train_accuracy = ", train_accuracy)
        print("val_accuracy   = ", val_accuracy)
        print("---")
        #
        learning_metrics_dict_this_fit = learn_evaluate_results.learning_metrics_dict.copy()
        learning_metrics_dict_this_fit['train_loss'] = train_loss
        learning_metrics_dict_this_fit['val_loss'] = val_loss
        learning_metrics_dict_this_fit['train_accuracy'] = train_accuracy
        learning_metrics_dict_this_fit['val_accuracy'] = val_accuracy
        return learning_metrics_dict_this_fit