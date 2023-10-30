# Author: JYang
# Last Modified: Oct-26-2023
# Description: This script provides the method(s) for generating the trained XGBoost model

import xgboost as xgb
import torch
import numpy as np

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
        if self.pred_type.lower() == 'classification':
            model = xgb.XGBClassifier(
                max_depth=10,
                objective='binary:logistic',
                nthread=4,
                random_state=self.seed,
                eval_metric='error',
                n_estimators=100,
                tree_method = "hist", 
                device = self.device
            )
        else:
            model = xgb.XGBRegressor(
                max_depth=10,
                objective='reg:squarederror',
                nthread=4,
                random_state=self.seed,
                eval_metric='error',
                n_estimators=100,
                tree_method = "hist", 
                device = self.device
            )

        # Set up data for xgboost model
        X_train = np.array(self.data_dict["X_train"])
        y_train = np.array(self.data_dict["y_train"])

        # Train model using xgb.fit
        #model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        model.fit(X_train, y_train, verbose=False)

        return model


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

    def get_model(self):
        """ Train an XGBoost model and return it
        Returns: 
            model (obj): A trained XGBoost model
        """

        param = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "max_depth": 10,
            #"num_round": 500,
            "tree_method": "hist",
            "device": self.device,
            "eta": 0.05,
        }

        X, y = np.array(self.data_dict["X_train"]), np.array(self.data_dict["y_train"])

        dtrain = xgb.DMatrix(X, label=y, feature_names=list(self.data_dict["X_train"].columns))

        model = xgb.train(param, dtrain)

        return model



