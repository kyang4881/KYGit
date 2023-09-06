# Author: JYang
# Last Modified: Sept-05-2023
# Description: This script provides the method(s) for generating the trained XGBoost model

import xgboost as xgb

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
                n_estimators=100  
            )
        else:
            model = xgb.XGBRegressor(
                max_depth=10,
                objective='reg:squarederror',
                nthread=4,
                random_state=self.seed,
                eval_metric='error',
                n_estimators=100  
            )

        # Set up data for xgboost model
        X_train = self.data_dict["X_train"]
        y_train = self.data_dict["y_train"]

        # Train model using xgb.fit
        #model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        model.fit(X_train, y_train, verbose=False)

        return model

