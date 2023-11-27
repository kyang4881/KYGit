# Author: JYang
# Last Modified: Nov-27-2023
# Description: This script provides the method(s) for data preprocessing 

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np

class Preprocess:
    """ A class that contains a method for data transformation and train/validation split
    Args:
        data (dataframe): input data for transformation/split
        target (str): a string indicating the name of the target variable column
        cat_cols (list): a list of categorical features
        num_cols (list): a list of numerical features
        label_cols (list): a list of columns to label encode
        do_not_encode_cols (list): a list of columns to not encode
        num_cv_splits (int): an integer indicating the number of cross validation fold
        train_examples (int): number of train examples in each cv split
        test_examples (int): number of test examples in each cv split
    """
    def __init__(self, data, target, cat_cols, num_cols, label_cols, do_not_encode_cols, num_cv_splits, train_examples, test_examples):
        self.data = data
        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.label_cols = label_cols 
        self.do_not_encode_cols = do_not_encode_cols
        self.num_cv_splits = num_cv_splits
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.label_encoder = LabelEncoder()
        self.split_model = TimeSeriesSplitImproved(num_cv_splits)

    def encode_norm(self, X_train, X_val):
        """ A method for data transformation
        Args:
            X_train (dataframe): a dataframe containing train data
            X_val (dataframe): a dataframe containing validation data
        Returns:
             X_train_data_transformed (dataframe): a dataframe containing transformed train data
             X_val_data_transformed (dataframe): a dataframe containing transformed validation data
        """
        X_train_data_transformed = pd.DataFrame()
        X_val_data_transformed = pd.DataFrame()

        # Normalize numerical variables
        if len(self.num_cols) > 0:
            X_train_scaled = self.scaler.fit_transform(X_train[self.num_cols])
            X_val_scaled = self.scaler.transform(X_val[self.num_cols])
            num_feature_names = [str(f) for f in self.scaler.get_feature_names_out().tolist()]
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train_scaled, columns=num_feature_names)], axis=1)
            X_val_data_transformed = pd.concat([X_val_data_transformed, pd.DataFrame(X_val_scaled, columns=num_feature_names)], axis=1)
            print("Added Transformed Numerical Features")
            display(X_train_data_transformed.head())

        # Encode categorical variables
        if len(self.cat_cols) > 0:
            X_train_encoded = self.encoder.fit_transform(X_train[self.cat_cols]).toarray()
            X_val_encoded = self.encoder.transform(X_val[self.cat_cols]).toarray()
            cat_feature_names = [str(f) for f in self.encoder.get_feature_names_out().tolist()]
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train_encoded, columns=cat_feature_names)], axis=1)
            X_val_data_transformed = pd.concat([X_val_data_transformed, pd.DataFrame(X_val_encoded, columns=cat_feature_names)], axis=1)
            print("Added Transformed Categorical Features")
            display(X_train_data_transformed.head())

        # Label Encode variables
        if len(self.label_cols) > 0:
            X_train_label_encoded = self.label_encoder.fit_transform(X_train[self.label_cols])
            X_val_label_encoded = self.label_encoder.transform(X_val[self.label_cols])
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train_label_encoded, columns=self.label_cols)], axis=1)
            X_val_data_transformed = pd.concat([X_val_data_transformed, pd.DataFrame(X_val_label_encoded, columns=self.label_cols)], axis=1)
            print("Added Transformed Label Features")
            display(X_train_data_transformed.head())

        # Features that do not require transformation
        if len(self.do_not_encode_cols) > 0:
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train[self.do_not_encode_cols].values, columns=self.do_not_encode_cols)], axis=1)
            X_val_data_transformed = pd.concat([X_val_data_transformed, pd.DataFrame(X_val[self.do_not_encode_cols].values, columns=self.do_not_encode_cols)], axis=1)
            print("Added Non-Transformed Features")
            display(X_train_data_transformed.head())

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
        # Dictionary containing indices of train and test sets
        train_test_index = {'train_index':[], 'test_index':[]}
        # For each cv split, normalize and encode the data
        for train_index, test_index in self.split_model.split(self.data, fixed_length=True, train_examples=self.train_examples, test_examples=self.test_examples):
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
            train_test_index['train_index'].append(train_index)
            train_test_index['test_index'].append(test_index)
        return df_compiled, self.scaler, self.encoder, self.label_encoder, train_test_index

    
class TimeSeriesSplitImproved(TimeSeriesSplit):
    """ A class that contains a method for applying cross-validation split"""
    
    def split(self, X, fixed_length=False, train_examples=1, test_examples=1):
        """ A method that applies rolling window time series cross validation split
        Args:
            X (dict): a dictionary containing the train and validation data
            fixed_length (bool): whether to apply the expanding window cross validation split
            train_examples (int): number of train examples in each cv split
            test_examples (int): number of test examples in each cv split           
        """
        n_samples = _num_samples(X)
        n_splits = self.n_splits  # Inherited from TimeSeriesSplit  
        train_start = 0
        
        print("\n--------------------------Cross-Validation--------------------------")
        for i in range(n_splits):                  
            train_end = train_start + train_examples 
            test_start = train_end                
            test_end = test_start + test_examples     
            
            # Ensure that the indices do not exceed the size of the input dataframe
            train_end = min(train_end, n_samples)
            test_start = min(test_start, n_samples)
            test_end = min(test_end, n_samples)
            
            print(f"train_end: {train_end}, test_start: {test_start}, test_end: {test_end}") 

            train_return, test_return = np.arange(train_start, train_end), np.arange(test_start, test_end)
            if len(train_return) == train_examples and len(test_return) == test_examples:
                yield (train_return, test_return)
                print(f"(Split #{i+1}) Cross-validation indices: train ([{train_return[0]}, {train_return[-1]}]), test ([{test_return[0]}, {test_return[-1]}])")
            else:
                if len(train_return) == 0 and len(test_return) == 0: 
                    print(f"(Split #{i+1}) Warning: Indices not included due to incompleteness: train ([]), test ([])")
                if len(train_return) > 0 and len(test_return) == 0: 
                    print(f"(Split #{i+1}) Warning: Indices not included due to incompleteness: train ([{train_return[0]}, {train_return[-1]}]), test ([])")
                if len(test_return) > 0 and len(test_return) > 0:
                    print(f"(Split #{i+1}) Warning: Indices not included due to incompleteness: train ([{train_return[0]}, {train_return[-1]}]), test ([{test_return[0]}, {test_return[-1]}])")
            
            train_start = train_end
            
            
