# Author: JYang
# Last Modified: Sept-06-2023
# Description: This script provides the method(s) for data preprocessing 

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

class Preprocess:
    """ A class that contains a method for data transformation and train/validation split
    Args:
        data (dataframe): input data for transformation/split
        target (str): a string indicating the name of the target variable column
        cat_cols (list): a list of categorical features
        num_cols (list): a list of numerical features
        num_cv_splits (int): an integer indicating the number of cross validation fold
    """
    def __init__(self, data, target, cat_cols, num_cols, num_cv_splits):
        self.data = data
        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.num_cv_splits = num_cv_splits
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    def encode_norm(self, X_train, X_val):
        """ A method for data transformation
        Args:
            X_train (dataframe): a dataframe containing train data
            X_val (dataframe): a dataframe containing validation data
        Returns:
             X_train_data_transformed (dataframe): a dataframe containing transformed train data
             X_val_data_transformed (dataframe): a dataframe containing transformed validation data
        """
        # Normalize numerical variables
        X_train_scaled = self.scaler.fit_transform(X_train[self.num_cols])
        X_val_scaled = self.scaler.transform(X_val[self.num_cols])
        
        # Encode categorical variables
        X_train_encoded = self.encoder.fit_transform(X_train[self.cat_cols])
        X_val_encoded = self.encoder.transform(X_val[self.cat_cols])

        # Extract feature names
        num_feature_names = [str(f) for f in self.scaler.get_feature_names_out().tolist()]
        cat_feature_names = [str(f) for f in self.encoder.get_feature_names_out().tolist()]

        # Combine normalized and encoded features
        X_train_data_transformed = pd.concat([
            pd.DataFrame(X_train_scaled, columns=num_feature_names),
            pd.DataFrame(X_train_encoded.toarray(), columns=cat_feature_names)
        ], axis=1)

        X_val_data_transformed = pd.concat([
            pd.DataFrame(X_val_scaled, columns=num_feature_names),
            pd.DataFrame(X_val_encoded.toarray(), columns=cat_feature_names)
        ], axis=1)

        return X_train_data_transformed, X_val_data_transformed

    def split_data(self):
        """ A method for splitting the original data into train/validation sets
        Returns:
            df_compiled (dict): a dictionary containing the train and validation data
        """
        # Define number of splits for cross validation
        tscv = TimeSeriesSplit(max_train_size=None, n_splits=self.num_cv_splits)            
        # Dictionary containing all cv splits
        df_compiled = {'X_train': [], 'X_val': [], 'y_train': [], 'y_val': []}
        # For each cv split, normalize and encode the data
        for train_index, test_index in tscv.split(self.data):
            # CV splits
            X_train, X_val = self.data.iloc[train_index, :-1], self.data.iloc[test_index, :-1] 
            y_train, y_val = self.data.iloc[train_index, -1], self.data.iloc[test_index, -1]
            # Transform data
            X_train_data_transformed, X_val_data_transformed = self.encode_norm(X_train, X_val)
            # Append to dictionary
            df_compiled['X_train'].append(X_train_data_transformed)
            df_compiled['X_val'].append(X_val_data_transformed)
            df_compiled['y_train'].append(y_train)
            df_compiled['y_val'].append(y_val)
        
        return df_compiled, self.scaler, self.encoder
