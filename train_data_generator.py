#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

learning_data_base_template = {
    'np_X': None,
    'df_y_1d': None,
    'df_y_Nd': None,
    'df_atr': None}


learning_data_template = {
    'train': None,
    'val': None,
    'test1': None,
    'test2': None}


class AtomicLearningData:
    X = None  # shall be set with a numpy.array
    y_1d = None  # shall be set with a pandas.Dataframe with a datetime index
    y_Nd = None  # shall be set with a pandas.Dataframe with a datetime index
    atr = None  # shall be set with a pandas.Dataframe with a datetime index


class LearningData:
    train = None  # shall be set with a AtomicLearningData object
    val = None  # shall be set with a AtomicLearningData object
    test = None  # shall be set with a AtomicLearningData object
    test1 = None  # shall be set with a AtomicLearningData object
    test2 = None  # shall be set with a AtomicLearningData object


class TrainDataGeneratorBase:
    raw_data = None  # = output of loader
    raw_data_with_target = None  # = output of trigger
    learning_data = None  # = output of norm data ? Shall be set with a AtomicLearningData object

    # -------------------------------------------------------------------------
    # Members functions
    # -------------------------------------------------------------------------

    #
    # Set __df_raw_data from outside this class
    #
    def load_compute_raw_data(self):
        raise NotImplementedError("Pure method to be implemented in child class")

    #
    # Add if not present and set field 'target' to  __df_raw_data => __df_raw_data_with_target
    #
    def compute_target(self):
        raise NotImplementedError("Pure method to be implemented in child class")

    #
    # Generate __df_learning_data for networks with Flat entry
    #
    def compute_learning_data_flat(self):
        raise NotImplementedError("Pure method to be implemented in child class")

    #
    # Generate __df_learning_data for networks with LSTM/GRU or Conv1D entry 
    #
    def compute_learning_data_GRU_LSTM_Conv1D(self):
        raise NotImplementedError("Pure method to be implemented in child class")


class FCTrainDataGenerator(TrainDataGeneratorBase):
    #
    # Set __df_raw_data from outside this class
    #
    __dataset_name = None
    __step2_params = None
    __step3_params = None

    def load_compute_raw_data_additional_params(self, dataset_name):
        self.__dataset_name = dataset_name

    def load_compute_raw_data(self):
        #
        import pandas
        import step1_dataset_prepare_raw_data as step1
        #
        Usa500_dfH1, Usa500_dfM30, Usa500_dfM15, Usa500_dfM5, \
        UsaInd_dfH1, UsaInd_dfM30, UsaInd_dfM15, UsaInd_dfM5, \
        UsaTec_dfH1, UsaTec_dfM30, UsaTec_dfM15, UsaTec_dfM5, \
        Ger30_dfH1, Ger30_dfM30, Ger30_dfM15, Ger30_dfM5, \
        EURUSD_dfH1, EURUSD_dfM30, EURUSD_dfM15, EURUSD_dfM5, \
        USDJPY_dfH1, USDJPY_dfM30, USDJPY_dfM15, USDJPY_dfM5, \
        GOLD_dfH1, GOLD_dfM30, GOLD_dfM15, GOLD_dfM5, \
        LCrude_dfH1, LCrude_dfM30, LCrude_dfM15, LCrude_dfM5, \
        big_H1, big_M30, big_M15, big_M5 = step1.load_prepared_data(self.__dataset_name)
        #
        data_all_df_bigs = {'H1': [big_H1],
                            'M30': [big_M30],
                            'M15': [big_M15],
                            'M5': [big_M5]}
        _all_df_bigs = pandas.DataFrame(data_all_df_bigs)
        #
        self.raw_data = _all_df_bigs
    #
    # Add if not present and set field 'target' to  __df_raw_data => __df_raw_data_with_target
    #
    def compute_target_additional_params(self, step2_params):
        self.__step2_params = step2_params
    #
    def compute_target(self):
        #
        import step2_dataset_prepare_target_data as step2
        #
        output_step2_data_with_target = step2.prepare_target_data_with_define_target(
            self.raw_data,
            self.__step2_params['step2_profondeur_analyse'],
            self.__step2_params['step2_target_period'],
            self.__step2_params['step2_symbol_for_target'],
            self.__step2_params['step2_targetLongShort'],
            self.__step2_params['step2_ratio_coupure'],
            self.__step2_params['step2_targets_classes_count'],
            self.__step2_params['step2_target_class_col_name'],
            self.__step2_params['step2_use_ATR'])
        #
        print(self.__step2_params['step2_profondeur_analyse'])
        print(output_step2_data_with_target)
        self.raw_data_with_target = output_step2_data_with_target
    #
    # Generate __df_learning_data for networks with Flat entry
    #
    def compute_learning_data_flat(self):
        raise NotImplementedError("Please Implement this method")
    #
    # Generate __df_learning_data for networks with LSTM/GRU or Conv1D entry
    #
    def compute_learning_data_GRU_LSTM_Conv1D_additional_params(self, step3_params):
        self.__step3_params = step3_params
    #
    def compute_learning_data_GRU_LSTM_Conv1D(self):
        return self.__create_step3_data()
    #
    def __create_step3_data(self):
        #
        import step3_dataset_prepare_learning_input_data as step3
        #
        learning_data = learning_data_template.copy()
        #
        output_step2_data_with_target = self.raw_data_with_target
        print(output_step2_data_with_target)
        #
        # initialisé, puis dans la boucle ssuivante incrémenté à chaque appel à step3.generate_learning_data_dynamic3
        idx_start = self.__step3_params['step3_idx_start']
        #
        for phase in ('test2', 'test1', 'val', 'train'):
            #
            samples_by_class = 0
            if phase == 'test2':
                samples_by_class =self.__step3_params['step3_tests_by_class']
            elif phase == 'test1':
                samples_by_class = self.__step3_params['step3_tests_by_class']
            elif phase == 'val':
                samples_by_class = self.__step3_params['step3_samples_by_class'] * 0.2  # 20 %
            elif phase == 'train':
                samples_by_class = self.__step3_params['step3_samples_by_class'] * 0.8  # 80 %
            #
            recouvrement = samples_by_class  # pb ici du fait que toutes les générations sont dans la même boucle
            #
            resOK, idx_start, np_X, df_y_Nd, df_y_1d, df_atr = step3.generate_learning_data_dynamic3(
                output_step2_data_with_target,
                idx_start,
                recouvrement,
                samples_by_class,
                self.__step3_params['step3_time_depth'],
                self.__step3_params['step3_column_names_to_scale'],
                self.__step3_params['step3_column_names_not_to_scale'])
            #
            if resOK == False:
                raise RuntimeError('Error during step3.generate_learning_data_dynamic3')
            #
            learning_data_base = learning_data_base_template.copy()
            learning_data_base['np_X'] = np_X
            learning_data_base['df_y_Nd'] = df_y_Nd
            learning_data_base['df_y_1d'] = df_y_1d
            learning_data_base['df_atr'] = df_atr
            learning_data[phase] = learning_data_base
        #
        return learning_data