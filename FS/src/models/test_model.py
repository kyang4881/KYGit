# Author: JYang
# Last Modified: Oct-09-2023
# Description: This script provides the method(s) for evaluating model performance on the test data

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score
from feature_selection_timeseries.src.models.train_model import generateModel
from feature_selection_timeseries.src.models.utils import check_column_types

class computeTestScore:
    """ A class for computing the model scores based on a given set of features on the test set
    Args:
        selected_features (list): a lsit of features to use
        data (dataframe): the test data
    """
    def __init__(self, selected_features, train_data, test_data, pred_type, seed, scaler_saved, encoder_saved, printout=False):
        self.selected_features = selected_features
        self.X_train_data = train_data.iloc[:, :-1]
        self.X_test_data = test_data.iloc[:, :-1]
        self.y_train_data = train_data.iloc[:, -1]
        self.y_test_data = test_data.iloc[:, -1]
        self.pred_type = pred_type
        self.seed = seed     
        self.scaler = scaler_saved
        self.encoder = encoder_saved
        self.printout = printout
        self.cat_cols, self.num_cols = check_column_types(self.X_train_data)       
        
    def encode_norm(self, X_train, X_test):
        """ A method for data transformation
        Args:
            X_train (dataframe): a dataframe containing train data
            X_test (dataframe): a dataframe containing test data
        Returns:
             X_train_transformed (dataframe): a dataframe containing transformed train data
             X_test_transformed (dataframe): a dataframe containing transformed test data
        """
        if self.printout:
            print("X_train")
            display(X_train.head())
            print("X_test")
            display(X_test.head())        
        
        # Normalize numerical variables
        X_train_scaled = self.scaler.transform(X_train[self.num_cols])
        X_test_scaled = self.scaler.transform(X_test[self.num_cols])

        # Encode categorical variables
        X_train_encoded = self.encoder.transform(X_train[self.cat_cols])
        X_test_encoded = self.encoder.transform(X_test[self.cat_cols])

        # Extract feature names
        num_feature_names = [str(f) for f in self.scaler.get_feature_names_out().tolist()]
        cat_feature_names = [str(f) for f in self.encoder.get_feature_names_out().tolist()]

        # Combine normalized and encoded features
        X_train_transformed = pd.concat([
            pd.DataFrame(X_train_scaled, columns=num_feature_names),
            pd.DataFrame(X_train_encoded.toarray(), columns=cat_feature_names)
        ], axis=1)
        
        X_test_transformed = pd.concat([
            pd.DataFrame(X_test_scaled, columns=num_feature_names),
            pd.DataFrame(X_test_encoded.toarray(), columns=cat_feature_names)
        ], axis=1)

        return X_train_transformed, X_test_transformed
    
    def filter_data(self, X_train_transformed, X_test_transformed):
        """ A method for filtering dataframes based on a selected list of features
        Returns:
            X_train_filtered (dataframe): train data with filtered features only
            X_test_filtered (dataframe): test data with filtered features only
        """
        X_train_filtered = X_train_transformed[self.selected_features]
        X_test_filtered = X_test_transformed[self.selected_features] 
        
        if self.printout:
            print("Selected Features:\n", self.selected_features, "\n")
            print("X_train_transformed columns:\n", list(X_train_transformed.columns), "\n")
            print("X_test_transformed columns:\n", list(X_test_transformed.columns), "\n")
            print("Selected features not in X_train_transformed columns:\n", [item for item in self.selected_features if item not in list(X_train_transformed.columns)], "\n")
            print("Selected features not in X_test_transformed columns:\n", [item for item in self.selected_features if item not in list(X_test_transformed.columns)], "\n")

        return X_train_filtered, X_test_filtered
        
    def pred(self):
        """ A method for generating prediction metrics on the test data
        Returns:
            metrics (dict): a dictionary containing scoring metrics 
        """
        X_train_transformed, X_test_transformed = self.encode_norm(X_train=self.X_train_data, X_test=self.X_test_data)
        
        if self.printout:
            print("X_train_transformed")
            display(X_train_transformed.head())
            print("X_test_transformed")
            display(X_test_transformed.head())

        X_train_filtered, X_test_filtered = self.filter_data(X_train_transformed=X_train_transformed, X_test_transformed=X_test_transformed)
        
        if self.printout:
            print("X_train_filtered")
            display(X_train_filtered.head())
            print("X_test_filtered")
            display(X_test_filtered.head())
        
        data_dict = {
            "X_train": X_train_filtered,
            "y_train": self.y_train_data,
            "X_test": X_test_filtered,
            "y_test": self.y_test_data
        }
        
        trained_model = generateModel(data_dict=data_dict, pred_type=self.pred_type, seed=self.seed).get_model()
        y_pred = trained_model.predict(data_dict['X_test'])
        y_true = data_dict['y_test']
        
        if self.pred_type == "classification":
            y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
            score = f1_score(y_true, y_pred) #accuracy_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cm = confusion_matrix(y_true, y_pred)
            metrics = {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "total_positive": np.sum(data_dict['y_test'] == 1),
                "total_negative": np.sum(data_dict['y_test'] == 0),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "accuracy": score
            }
        return metrics

    
    
    
