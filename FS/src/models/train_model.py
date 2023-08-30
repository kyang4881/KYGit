import xgboost as xgb


class generateModel:
    """ ...
    Args:
        data_dict (dict):
        pred_type (str):
        seed (int):
    """
    def __init__(self, data_dict, pred_type, seed):
        self.data_dict = data_dict
        self.pred_type = pred_type.lower()
        self.seed = seed

    def get_model(self):
        """ ...
        Returns:
        """
        if self.pred_type.lower() == 'classification':
            model = xgb.XGBClassifier(
                max_depth=10,
                objective='binary:logistic',
                nthread=4,
                random_state=self.seed,
                eval_metric='error'#,
                #num_round=50  # Equivalent to num_round in xgb.train
            )
        else:
            model = xgb.XGBRegressor(
                max_depth=10,
                objective='reg:squarederror',
                nthread=4,
                random_state=self.seed,
                eval_metric='error'#,
                #n_estimators=50  # Equivalent to num_round in xgb.train
            )

        # Set up data for xgboost model
        X_train = self.data_dict["X_train"]
        y_train = self.data_dict["y_train"]

        # Train model using xgb.fit
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

        return model

