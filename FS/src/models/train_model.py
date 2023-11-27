# Author: JYang
# Last Modified: Nov-27-2023
# Description: This script provides the method(s) for generating the trained XGBoost model

import xgboost as xgb
import torch
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from feature_selection_timeseries.src.models.utils import setup_seed

class generateModel:
    """ A class with a method for generating a trained model
    Args:
        data_dict (dict): a dictionary containing dataframes of the train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
    """
    def __init__(self, data_dict, pred_type, seed):
        self.data_dict = data_dict
        self.pred_type = pred_type.lower()
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup_seed(self.seed)

    def get_model(self):
        """ Train an XGBoost model and return it
        Returns: 
            model (obj): A trained XGBoost model
        """
        param = {
            "objective": "binary:logistic" if self.pred_type.lower() == 'classification' else "reg:squarederror",
            "eval_metric": "error" if self.pred_type.lower() == 'classification' else "rmse",
            #"max_depth": 5,
            "tree_method": "hist",
            "device": self.device,
            #"eta": 0.05,
            "seed": self.seed#,
            #"min_child_weight": 2,
            #"learning_rate": 0.3
        }
        print(f"Using Params: {param}")
        # Extract features and labels from the data dictionary
        X, y = np.array(self.data_dict["X_train"]), np.array(self.data_dict["y_train"])
        # Create a DMatrix for XGBoost training
        dtrain = xgb.DMatrix(X, label=y, feature_names=list(self.data_dict["X_train"].columns))
        # Train the XGBoost model
        model = xgb.train(param, dtrain)
        return model

## With hyperparam tuning 
# class generateModel_test:
#     """ A class with a method for generating a trained model that includes hyperparam tuning for the test data
#     Args:
#         data_dict (dict): a dictionary containing dataframes of the train and validation data
#         pred_type (str): a string indicating the type of prediction problem: classification or regression
#         seed (int): a random state
#     """
#     def __init__(self, data_dict, pred_type, seed):
#         self.data_dict = data_dict
#         self.pred_type = pred_type.lower()
#         self.seed = seed
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def get_model(self):
#         """ Train an XGBoost model and return it
#         Returns: 
#             model (obj): A trained XGBoost model
#         """
#         # Extract features and labels from the data dictionary
#         X, y = np.array(self.data_dict["X_train"]), np.array(self.data_dict["y_train"])

#         param = {
#             "objective": "binary:logistic" if self.pred_type.lower() == 'classification' else "reg:squarederror",
#             "eval_metric": "error" if self.pred_type.lower() == 'classification' else "rmse",
#             #"max_depth": 5,
#             "tree_method": "hist",
#             "device": self.device,
#             #"eta": 0.05,
#             "seed": self.seed#,
#             #"min_child_weight": 2,
#             #"learning_rate": 0.3
#         }

#         param_grid = {
#             'max_depth': [10, 20, 50],
#             'learning_rate': [0.01, 0.3, 0.5],
#             'min_child_weight': [2, 5, 10]
#         }

#         xgb_model = xgb.XGBClassifier() if self.pred_type == 'classification' else xgb.XGBRegressor()
#         tscv = TimeSeriesSplit(n_splits=3)
#         grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv)
#         grid_search.fit(X, y)
#         best_params = grid_search.best_params_
#         print(f"Best Hyperparameters: {best_params}")
#         param.update(best_params)
#         print(f"Using Params: {param}")

#         # Create a DMatrix for XGBoost training
#         dtrain = xgb.DMatrix(X, label=y, feature_names=list(self.data_dict["X_train"].columns))
#         # Train the XGBoost model
#         model = xgb.train(param, dtrain)

#         return model


class generateModel_backup:
    """ A class with a method for generating a trained model
    Args:
        data_dict (dict): a dictionary containing dataframes of the train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
    """
    def __init__(self, data_dict, pred_type, seed):
        self.data_dict = data_dict
        self.pred_type = pred_type.lower()
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self):
        """ Train an XGBoost model and return it
        Returns: 
            model (obj): A trained XGBoost model
        """
        param = {
            "objective": "binary:logistic" if self.pred_type.lower() == 'classification' else "reg:squarederror",
            "eval_metric": "error" if self.pred_type.lower() == 'classification' else "rmse",
            "max_depth": 10,
            "tree_method": "hist",
            "device": self.device,
            "eta": 0.05,
            "seed": self.seed
        }
        # Extract features and labels from the data dictionary
        X, y = np.array(self.data_dict["X_train"]), np.array(self.data_dict["y_train"])
        # Create a DMatrix for XGBoost training
        dtrain = xgb.DMatrix(X, label=y, feature_names=list(self.data_dict["X_train"].columns))
        # Train the XGBoost model
        model = xgb.train(param, dtrain)

        return model


