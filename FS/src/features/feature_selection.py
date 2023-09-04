import numpy as np
import sage
import xgboost as xgb
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
from sklearn.metrics import accuracy_score 
from sklearn.inspection import permutation_importance
import time
import matplotlib.pyplot as plt
import shap
import os
import torch
import math

class sageValues:
    """ A class for methods related to the Sage feature selection method
    Args:
        data_dict (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        model (obj): a trained XGBoost model
        seed (int): a random state
    """
    def __init__(self, data_dict, pred_type, model, seed):
        self.data_dict = data_dict
        self.pred_type = pred_type.lower()
        self.model = model
        self.sage_val = None
        self.feature_names = self.data_dict['X_val'].columns.to_list()
        self.seed = seed

    def compute_sage_val(self):
        """ A method for computing feature importance using the Sage method
        Returns:
            sage_features (list): ranked features
            values (list): ranked feature importances
        """
        # Calculate sage values
        imputer = sage.MarginalImputer(self.model, self.data_dict["X_train"][:512].values)
        estimator = sage.KernelEstimator(imputer, 'cross entropy' if self.pred_type == 'classification' else 'mse', random_state=self.seed)
        self.sage_val = estimator(self.data_dict["X_val"].values, self.data_dict["y_val"].values, thresh=0.025)
        # Order sage values
        values = self.sage_val.values
        argsort = np.argsort(values)[::-1]
        values = values[argsort]
        sage_features = np.array(self.feature_names)[argsort]
        return sage_features, values

    def sage_plot(self):
        # Plot sage values
        self.sage_val.plot(self.feature_names)
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=8)
        plt.show()

def sage_importance(model, data, pred_type, seed, target_colname):
    """ A method that calls the sage class to extract the features, feature scores, and total runtime
    Args:
        model (obj): a trained XGBoost model
        data (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
    Returns:
        sage_features (list): ranked features
        sage_feature_scores (list): ranked feature importances
        total_time (float): total runtime for the Sage feature selection process
    """ 
    start_time = time.time()
    # Generate the Sage Values
    sv1 = sageValues(
        data_dict = data,
        pred_type = pred_type,
        model = model,
        seed = seed
    )
    sage_features, sage_feature_scores = sv1.compute_sage_val()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nRuntime: {total_time:.2f} seconds")
    display(sv1.sage_plot())

    return sage_features, sage_feature_scores, total_time

def permutation_test(model, data, pred_type, seed, target_colname):
    """ A method that extracts the features, feature scores, and total runtime
    Args:
        model (obj): a trained XGBoost model
        data (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
    Returns: 
        feature_names_sorted (list): ranked features
        premu_test.importances_mean[sorted_idx] (list): ranked feature importances
        total_time_permu (float): total runtime for the Sage feature selection process
    """      
    start_time = time.time()
    premu_test = permutation_importance(model, data["X_val"], data["y_val"],  random_state = seed)
    sorted_idx = premu_test.importances_mean.argsort()[::-1]

    feature_names = data['X_val'].columns.to_list()
    feature_names_sorted = [feature_names[i] for i in sorted_idx]

    end_time = time.time()
    total_time_permu = end_time - start_time

    plt.barh(feature_names_sorted, premu_test.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")

    return feature_names_sorted, premu_test.importances_mean[sorted_idx], total_time_permu

def xgb_importance(model, data, pred_type, seed, target_colname):
    """ A method that extracts the features, feature scores, and total runtime
    Args:
        model (obj): a trained XGBoost model
        data (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
    Returns:
        top_features (list): ranked features
        top_scores (list): ranked feature importances
        total_time (float): total runtime for the Sage feature selection process
    """     
    # make predictions for val data
    y_pred = model.predict(data['X_val'])
    # round predictions
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(predictions, data['y_val'])

    start_time = time.time()
    # Calculate feature importance scores
    feature_importances=model.get_booster().get_score(importance_type='weight')
    # Sort the feature importance scores
    sorted_idx = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    # Extract the top feature names and scores
    top_features, top_scores = zip(*sorted_idx[:-1])
    end_time = time.time()
    total_time = end_time - start_time

    return top_features, top_scores, total_time

def shap_importance(model, data, pred_type, seed, target_colname):
    """ A method that extracts the features, feature scores, and total runtime
    Args:
        model (obj): a trained XGBoost model
        data (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
    Returns:
        sorted_features (list): ranked features
        feature_score_shap (list): ranked feature importances
        total_time (float): total runtime for the Sage feature selection process
    """        
    start_time = time.time()
    explainer = shap.Explainer(model)
    shap_values = explainer(data["X_val"])
    #shap.plots.waterfall(shap_values[0])
    end_time = time.time()
    total_time = end_time - start_time

    mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
    features = np.array(data['X_train'].columns.to_list())
    # Sort the features based on their mean absolute SHAP values
    argsort = np.argsort(mean_abs_shap_values)[::-1]
    sorted_features = features[argsort]
    print("Top features:", sorted_features)
    shap.plots.beeswarm(shap_values, max_display=25)
    feature_score_shap = np.mean(shap_values.values, axis=0)[np.argsort(np.mean(shap_values.values, axis=0))[::-1]]
    
    return sorted_features, feature_score_shap, total_time


def cae_importance(data, pred_type, seed, target_colname):
    """ A method that extracts the features, feature scores, and total runtime
    Args:
        model (obj): a trained XGBoost model
        data (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
    Returns:
        sorted_features (list): ranked features
        feature_scores (list): ranked feature importances
        total_time (float): total runtime for the Sage feature selection process
    """     
    X_train = data['X_train'].values
    y_train = data['y_train'].values
    X_val = data['X_val'].values
    y_val = data['y_val'].values
    (x_train, y_train), (x_val, y_val) = (X_train, y_train), (X_val, y_val)

    x_train = np.reshape(x_train, (len(x_train), -1))
    x_val = np.reshape(x_val, (len(x_val), -1))
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    start_time = time.time()

    setup_seed(seed)
    selector_supervised = ConcreteAutoencoderFeatureSelector(K = len(data['X_val'].columns), rand_seed=seed, output_function = g, num_epochs = 3)
    selector_supervised.fit(x_train, y_train, x_val, y_val)

    end_time = time.time()
    total_time = end_time - start_time

    feature_importances=selector_supervised.get_support(indices = True)
    argsort = np.argsort(feature_importances)[::-1]
    features = np.array(data['X_train'].columns.to_list())
    sorted_features = features[argsort]
    feature_scores = feature_importances[argsort]

    return sorted_features, feature_scores, total_time

    
class ConcreteSelect(Layer):
    def __init__(self, output_dim, rand_seed, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        self.rand_seed = rand_seed

    def build(self, input_shape):
        self.temp = self.add_weight(name='temp', shape=[], initializer=Constant(self.start_temp), trainable=False)
        self.logits = self.add_weight(name='logits', shape=[self.output_dim, input_shape[1]], initializer=glorot_normal(seed=self.rand_seed), trainable=True)
        super(ConcreteSelect, self).build(input_shape)

    def call(self, X, training=None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0, seed=self.rand_seed)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        #noisy_logits = (self.logits) / temp
        #samples = K.softmax(noisy_logits)
        samples = K.sigmoid(noisy_logits)

        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])

        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))

        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class StopperCallback(EarlyStopping):

    def __init__(self, mean_max_target = 0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)

    def on_epoch_begin(self, epoch, logs = None):
        print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature', K.get_value(self.model.get_layer('concrete_select').temp))

    def get_monitor_value(self, logs):
        #monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        monitor_value = K.get_value(K.mean(K.max(K.sigmoid(self.model.get_layer('concrete_select').logits))))
        return monitor_value

class ConcreteAutoencoderFeatureSelector():

    def __init__(self, K, output_function, num_epochs = 300, batch_size = None, learning_rate = 0.001, start_temp = 10.0, min_temp = 0.1, tryout_limit = 5, rand_seed=42):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.rand_seed = rand_seed

    def fit(self, X, Y = None, val_X = None, val_Y = None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)

        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size

        for i in range(self.tryout_limit):

            K.set_learning_phase(1)
            inputs = Input(shape = X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, self.rand_seed, self.start_temp, self.min_temp, alpha, name = 'concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features)

            self.model = Model(inputs, outputs)

            self.model.compile(Adam(self.learning_rate), loss = 'binary_crossentropy')

            print(self.model.summary())

            stopper_callback = StopperCallback()

            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose = 1, callbacks = [stopper_callback], validation_data = validation_data)#, validation_freq = 10)

            #if K.get_value(K.mean(K.max(K.softmax(self.concrete_select.logits, axis = -1)))) >= stopper_callback.mean_max_target:
            if K.get_value(K.mean(K.max(K.sigmoid(self.concrete_select.logits)))) >= stopper_callback.mean_max_target:
                break

        num_epochs *= 2

        #self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.probabilities = K.get_value(K.sigmoid(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

        return self

    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits), self.model.get_layer('concrete_select').logits.shape[1]), axis = 0))

    def transform(self, X):
        return X[self.get_indices()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices = False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model

def setup_seed(seed):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def g(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(2, activation='sigmoid')(x)
    return x
