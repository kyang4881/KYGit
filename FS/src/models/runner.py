# Author: JYang
# Last Modified: Nov-27-2023
# Description: This script provides the method(s) that consolidate multiple methods into a wrapped run function to execute the pipeline

import numpy as np
import pandas as pd
from collections import Counter
from feature_selection_timeseries.src.models.predict_model import run_scoring_pipeline
from feature_selection_timeseries.src.models.utils import create_df, add_to_dataframe, check_column_types
from feature_selection_timeseries.src.preprocessing.preprocessor import Preprocess
from feature_selection_timeseries.src.models.train_model import generateModel
from feature_selection_timeseries.src.features.feature_selection import featureValues
from feature_selection_timeseries.src.visualization.visualize import plot_ts
from feature_selection_timeseries.src.models.test_model import computeTestScore
import time
import joblib
import ast

def get_metrics_df(seed, target_colname, data, data_original, full_df, method_name, dataset_name, pred_type, cv_iteration, train_examples,
                   test_examples, num_cv_splits, rebalance=False, rebalance_type=None, append_to_full_df=False, n_features=None, feature_direction=None):
    """ A methold for generating the model results
    Args:
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
        data (dict): a dictionary containing train (rebalanced, if applicable) and validation data
        data_original (dict): a dictionary containing train and validation data
        full_df (dataframe): a dataframe containing currently tracked model results
        method_name (str): name of the feature selection model
        dataset_name (str): name of the dataset
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        cv_iteration (int): an integer indicating the cross validation split iteration
        train_examples (int): number of train examples in each cv split
        test_examples (int): number of test examples in each cv split
        num_cv_splits (int): an integer indicating the number of cross validation splits
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        append_to_full_df (bool): a boolean indicating whether to append model results to the existing tracking table
        n_features (int): number of features to use; used for top n and bottom n features
        feature_direction (str): a string indicating whether to evaluate top/bottom/both ranking features
    Returns:    
        full_df (dataframe): a dataframe containing all currently tracked model results
    """   
    # Retrieve the trained model
    model = generateModel(
        data_dict = data,
        pred_type = pred_type,
        seed = seed
    ).get_model()
        
    feature_values = featureValues(data_dict=data, pred_type=pred_type, model=model, seed=seed, target_colname=target_colname, n_features=n_features)
        
    # Generate feature ranking and other input variables
    if method_name.lower() == "permutation":
        sorted_features, feature_scores, total_time = feature_values.permutation_test()

    if method_name.lower() == "xgboost":
        sorted_features_xgb, feature_scores, total_time = feature_values.xgb_importance()
        sorted_features = list(sorted_features_xgb) + [f for f in list(data['X_train'].columns) if f not in sorted_features_xgb]

    if method_name.lower() == "shap":
        sorted_features, feature_scores, total_time = feature_values.shap_importance()

    if method_name.lower() == "boruta":
        sorted_features, feature_scores, total_time = feature_values.boruta_importance()
        
    if method_name.lower() == "sage":
        sorted_features, feature_scores, total_time = feature_values.sage_importance()
        
    if method_name.lower() == "cae":
        sorted_features, feature_scores, total_time = feature_values.cae_importance()
    
    if method_name.lower() == "dynamic":
        sorted_features, feature_scores, total_time = feature_values.dynamic_selection_importance()
        
    if method_name.lower() == "stg":
        sorted_features, feature_scores, total_time = feature_values.stg_importance()
    
    if method_name.lower() == "lasso":
        sorted_features, feature_scores, total_time = feature_values.lasso_importance()
    
    if method_name.lower() == "cart":
        sorted_features, feature_scores, total_time = feature_values.cart_importance()
        
    if method_name.lower() == "svm":
        sorted_features, feature_scores, total_time = feature_values.svm_importance()
        
    if method_name.lower() == "rf":
        sorted_features, feature_scores, total_time = feature_values.randomforest_importance()
        
    # Generate the scoring metrics
    all_scores, all_scores_reverse, all_features, all_features_reverse, cm_val, cm_val_reversed  = run_scoring_pipeline(
        feature_impt = sorted_features,
        n_features = n_features,
        input_data_dict = data,
        pred_type = pred_type,
        rebalance=rebalance,
        rebalance_type=rebalance_type,
        seed=seed,
        feature_direction=feature_direction
    )

    X_train_shape = np.shape(data["X_train"])
    X_val_shape = np.shape(data["X_val"])

    if pred_type == "classification":
        y_train_dict = dict(sorted(Counter(data["y_train"]).items()))
        y_val_dict = dict(sorted(Counter(data["y_val"]).items()))
        y_classes_dict = {0: y_train_dict[0] + y_val_dict[0], 1: y_train_dict[1] + y_val_dict[1]}
        best_score = [num_index == all_scores.index(max(all_scores)) for num_index in range(len(all_scores))]
    else:
        y_train_dict = {}
        y_val_dict = {}
        y_classes_dict = {}
        best_score = [num_index == all_scores.index(min(all_scores)) for num_index in range(len(all_scores))]
    
    # Compile dataframe containing scoring metrics for all feature subsets
    results_df = create_df(
        all_features = all_features,
        all_scores = all_scores,
        all_features_rev = all_features_reverse,
        all_scores_rev = all_scores_reverse,
        dataset_size = str(np.shape(data_original)),
        total_time = total_time,
        method_name = method_name,
        dataset_name = dataset_name,
        pred_type = pred_type,
        feature_score = feature_scores,
        cm_val = cm_val,
        cm_val_reversed = cm_val_reversed,
        rebalance = rebalance,
        rebalance_type = rebalance_type,
        data_shape = str({
            "cv_train_size": train_examples,
            "cv_test_size": test_examples,
            "num_cv_split": num_cv_splits,
            "X_train (instance/feature)": X_train_shape,
            "X_val (instance/feature)": X_val_shape,
            "y_train (class/count)": y_train_dict,
            "y_val (class/count)": y_val_dict,
            "X_total (instance/features)": (X_train_shape[0] + X_val_shape[0], X_train_shape[1]),
            "y_total (class/count)": y_classes_dict            # If KeyError => data has a missing class; not enough data 
        }),
        is_best_score = best_score,
        cv_iteration = cv_iteration
    )

    display(results_df.head())
    
    if append_to_full_df:
        full_df = pd.concat([results_df, full_df])
        
    return full_df

def compile_one_sweep(label_cols, do_not_encode_cols, seed, target_colname, data, full_df, method_name, dataset_name, pred_type, num_cv_splits=5, rebalance=False, rebalance_type=None, append_to_full_df=False, train_examples=1, test_examples=1, n_features=0, feature_direction="both"):
    """ A method that runs through 1 sweep of the entire pipeline by wrapping the required methods
    Args:
        label_cols (list): list of columns to label encode
        do_not_encode_cols (list): list of columns to not encode
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
        data (dict): a dictionary containing train and validation data
        full_df (dataframe): a dataframe containing all currently tracked model results
        method_name (str): name of the feature selection model
        dataset_name (str): name of the dataset
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        num_cv_splits (int): an integer indicating the number of cross validation splits
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        append_to_full_df (bool): a boolean indicating whether to append model results to the existing tracked results
        train_examples (int): number of train examples in each cv split
        test_examples (int): number of test examples in each cv split
        n_features (int): number of features to use; used for top n and bottom n features
        feature_direction (str): a string indicating whether to evaluate top/bottom/both ranking features
    Returns:      
        full_df (dataframe): a dataframe containing all currently tracked model results
        scaler_saved (object): an object for rescaling numerical features
        encoder_saved (object): an object for encoding categorical features
        label_encoder_saved (obj): an object for label encoding
    """   
    # Extract categorical and numerical features
    categorical_cols, numerical_cols = check_column_types(data.iloc[:,:-1], label_cols, do_not_encode_cols)
    print("Categorical Columns: ", categorical_cols)
    print("Numerical Columns: ", numerical_cols)
    print("Label Encode Columns: ", label_cols)
    print("Do Not Encode Columns: ", do_not_encode_cols)
    
    # Preprocessing the data
    processor1 = Preprocess(
        data = data,
        target = target_colname,
        cat_cols = categorical_cols,
        num_cols = numerical_cols, 
        label_cols = label_cols,  #  changed nov 20
        do_not_encode_cols = do_not_encode_cols,  #  changed nov 20
        num_cv_splits = num_cv_splits,
        train_examples = train_examples,
        test_examples = test_examples
    )
    # Dictionary containing all cross validation splits
    compiled_data, scaler_saved, encoder_saved, label_encoder_saved, train_test_index = processor1.split_data()
    # The number of cv splits
    cv_iteration = len(compiled_data['X_train'])
    
    # Loop for all cv splits and compute scoring metrics
    for i in range(cv_iteration):
        selected_cv_dict = {}
        # Iterate through the original dictionary
        for key, df_list in compiled_data.items():
            selected_cv_dict[key] = df_list[i]

        #display(selected_cv_dict)
        print('\nX_train', np.shape(selected_cv_dict['X_train']))
        print('X_val', np.shape(selected_cv_dict['X_val']))
        print('y_train', np.shape(selected_cv_dict['y_train']))
        print('y_val', np.shape(selected_cv_dict['y_val']), "\n")
        print(f"Running Cross-Validation Split: train_index=[{train_test_index['train_index'][i][0]}, {train_test_index['train_index'][i][-1]}], test_index=[{train_test_index['test_index'][i][0]}, {train_test_index['test_index'][i][-1]}]\n")

        # Compute metrics
        full_df = get_metrics_df(
            seed=seed, 
            target_colname=target_colname, 
            data_original=data,
            data=selected_cv_dict, 
            full_df=full_df, 
            method_name=method_name, 
            dataset_name=dataset_name, 
            pred_type=pred_type, 
            cv_iteration=i,
            train_examples=train_examples,
            test_examples=test_examples,
            num_cv_splits=num_cv_splits,
            rebalance=rebalance, 
            rebalance_type=rebalance_type, 
            append_to_full_df=append_to_full_df,
            n_features=n_features,
            feature_direction=feature_direction
        )
    return full_df, scaler_saved, encoder_saved, label_encoder_saved

class run:
    """
    Class for running a set of experiments on different methods, datasets, and rebalance types.
    
    Attributes:
        save_output_file (bool): Whether to save the training and testing outputs.
        train_df (pd.DataFrame): The training dataset.
        y (pd.DataFrame): Contains the true values.
        train_test_list (list): List of tuples representing train and test split details.
        methods (list): List of methods to evaluate.
        datasets (list): List of datasets to use for evaluation.
        rebalance_type (list): List of rebalance types to consider.
        label_cols (list): List of columns containing labels.
        do_not_encode_cols (list): List of columns not to be encoded.
        seed (int): Seed for reproducibility.
        target_colname (str): Column name for the target variable.
        dataset_name (str): Name of the dataset.
        pred_type (str): Type of prediction.
        append_to_full_df (bool): Whether to append results to the full dataframe.
        n_features (int): Number of features to consider.
        feature_direction (str): Direction of feature selection.
        train_outputs_file_name (str): File name for saving training outputs.
        current_date (str): Current date for timestamping files.
        scaler_filename (str): File name for saving scaler object.
        encoder_filename (str): File name for saving encoder object.
        label_encoder_filename (str): File name for saving label encoder object.
        test_output_file_name (str): File name for the test data outputs
        test_pred_file_name (str): File name for the test data predictions
    """
    def __init__(self, save_output_file, train_df, y, train_test_list, methods, datasets, rebalance_type, label_cols, 
                  do_not_encode_cols, seed, target_colname, dataset_name, pred_type, append_to_full_df, n_features, 
                  feature_direction, train_outputs_file_name, current_date, scaler_filename, encoder_filename, 
                  label_encoder_filename, test_output_file_name, test_pred_file_name
                 ):
        self.save_output_file = save_output_file
        self.train_df = train_df
        self.y = y
        self.train_test_list = train_test_list
        self.methods = methods
        self.datasets = datasets
        self.rebalance_type = rebalance_type
        self.label_cols = label_cols
        self.do_not_encode_cols = do_not_encode_cols
        self.seed = seed
        self.target_colname = target_colname
        self.dataset_name = dataset_name
        self.pred_type = pred_type
        self.append_to_full_df = append_to_full_df
        self.n_features = n_features
        self.feature_direction = feature_direction
        self.train_outputs_file_name = train_outputs_file_name
        self.current_date = current_date
        self.scaler_filename = scaler_filename
        self.encoder_filename = encoder_filename
        self.label_encoder_filename = label_encoder_filename
        self.test_output_file_name = test_output_file_name
        self.test_pred_file_name = test_pred_file_name

        self._test_df = None
        self._train_df = None
        self.scaler_saved = None
        self.encoder_saved = None
        self.label_encoder_saved = None

        # An empty dataframe to store results
        self.full_df = pd.DataFrame({
            "timestamp": [],
            "feature": [],
            "score": [],
            "feature_reversed": [],
            "score_reversed_rank": [],
            "method": [],
            "dataset": [],
            "dataset_size": [],
            "runtime_sec": [],
            "prediction_type": [],
            "feature_score": [],
            "cm_val": [],
            "cm_val_reversed":[],
            "rebalance": [],
            "rebalance_type": [],
            "data_shape": [],
            "is_best_score": [],
            "cv_iteration": []
        })

    def train(self):
        """
        Performs the training and evaluation process for different combinations of methods, datasets, and rebalance types.
        """
        # Indices based on cv splits
        train_start = 0
        train_end = test_start = self.train_test_list[0][0]*self.train_test_list[0][2] + self.train_test_list[0][1]
        test_end =  self.train_test_list[0][0]*self.train_test_list[0][2] + self.train_test_list[0][1]*2
        # Filter for the neceassary training and holdout sample data
        self.train_df = self.train_df.iloc[-test_end:,:]
        # Update index
        self.train_df.index = range(0, test_end)
        # Training and holdout set
        self._test_df = self.train_df.iloc[-self.train_test_list[0][1]:, :]
        self._train_df = self.train_df.iloc[train_start:train_end, :]

        for n in range(len(self.train_test_list)):
            start_time = time.time()
            for method in self.methods:
                for dataset in self.datasets:
                    for rt in self.rebalance_type:
                        rebalance = False if rt=="None" else True
                        print(f"\n\nMethod: {method}\nDataset: {dataset}\nRebalance_type: {rt}\nRebalance: {rebalance}\n")
                        print(f"Combination={n}, Train_start={train_start}, Train_end={train_end-1}, Hold_out_test_start={test_start}, Hold_out_test_end={test_end}\n")
                        self.full_df, self.scaler_saved, self.encoder_saved, self.label_encoder_saved = compile_one_sweep(
                            label_cols=self.label_cols, 
                            do_not_encode_cols=self.do_not_encode_cols, 
                            seed=self.seed,
                            target_colname=self.target_colname,
                            data=eval("self._" + dataset +"_df"),
                            full_df=self.full_df,
                            method_name=method,
                            dataset_name=self.dataset_name,
                            pred_type=self.pred_type,
                            num_cv_splits=self.train_test_list[n][2],
                            rebalance=rebalance,
                            rebalance_type=rt,
                            append_to_full_df=self.append_to_full_df,
                            train_examples=self.train_test_list[n][0],
                            test_examples=self.train_test_list[n][1],
                            n_features=self.n_features,
                            feature_direction=self.feature_direction
                        )
                        print(np.shape(self.full_df))
                        if self.save_output_file: 
                            self.full_df.to_excel(f"{self.train_outputs_file_name}{self.current_date}.xlsx")
                            print(f"Train outputs saved to: {self.train_outputs_file_name}{self.current_date}.xlsx")

            end_time = time.time()
            total_time = end_time - start_time
            print(f"\nTotal Runtime: {total_time:.2f} seconds")

        joblib.dump(self.scaler_saved, self.scaler_filename)
        joblib.dump(self.encoder_saved, self.encoder_filename)
        joblib.dump(self.label_encoder_saved, self.label_encoder_filename)

    def test(self):
        """
        Performs the testing and evaluation process for the trained combinations of methods, datasets, and rebalance types.
        """
        file_path = f"{self.train_outputs_file_name}{self.current_date}.xlsx"
        # import feature selection trained results
        test_input_df = pd.read_excel(file_path, sheet_name="Sheet1") 

        test_input_df['feature_list'] = test_input_df['feature'].apply(ast.literal_eval)
        selected_features = list(test_input_df['feature_list'])

        compiled_metrics = []
        test_pred_full = pd.DataFrame()
                
        for i in range(len(selected_features)):
            eval_record = eval(test_input_df['data_shape'][i])
            cv_train_size = eval_record['cv_train_size']
            cv_test_size = eval_record['cv_test_size']
            num_cv_split = eval_record['num_cv_split']
            cur_method = test_input_df['method'][i]
            cur_cv_iter = str(test_input_df['cv_iteration'][i])
            cur_num_feature = str(len(eval(test_input_df['feature'][i])))
            cur_features = test_input_df['feature'][i]

            computerA = computeTestScore(
                label_cols=self.label_cols, 
                do_not_encode_cols=self.do_not_encode_cols,
                selected_features=selected_features[i],
                train_data=self._train_df,
                test_data=self._test_df,
                pred_type=self.pred_type,
                seed=self.seed,
                printout=False,
                scaler_saved=self.scaler_saved,
                encoder_saved=self.encoder_saved,
                label_encoder_saved=self.label_encoder_saved
            )

            # metrics: A dict of containing RMSE and num of features, results: a dataframe of true values and preds
            metrics, results = computerA.pred()
            compiled_metrics.append(metrics)

            other_fields_df = {
                "num_cv_split": [cv_test_size]*cv_test_size,
                "num_cv_split": [num_cv_split]*cv_test_size,
                "method": [cur_method]*cv_test_size,
                "cv_iteration": [cur_cv_iter]*cv_test_size,
                "feature": [cur_features]*cv_test_size,
                "num_feature": [cur_num_feature]*cv_test_size
            }

            test_pred = pd.concat([self.y.iloc[-cv_test_size:, :-1].reset_index(drop=True), results, pd.DataFrame(other_fields_df)], axis=1)
            # Plot time series
            print(f"Method: {cur_method}, Cv_iteration: {cur_cv_iter}, Num_features: {cur_num_feature}")
            plot_ts(test_pred)

            test_pred_full = pd.concat([test_pred_full, test_pred], axis=0)
            print(f"test_pred_full shape: {np.shape(test_pred_full)}")
            compiled_metrics_df = pd.DataFrame(compiled_metrics)
            compiled_metrics_df = compiled_metrics_df.applymap(lambda x: x[0] if isinstance(x, list) else x)
            test_output_df = pd.concat([test_input_df, compiled_metrics_df], axis=1)

        # Export test data results
        if self.save_output_file:
            # Save test results
            test_output_df.to_excel(f"{self.test_output_file_name}{self.current_date}.xlsx", index=False)
            print(f"Test outputs saved to: {self.test_output_file_name}{self.current_date}.xlsx")
            # Save predictions
            test_pred_full.to_excel(f"{self.test_pred_file_name}{self.current_date}.xlsx", index=False)
            print(f"Test preds saved to: {self.test_pred_file_name}{self.current_date}.xlsx")
            # Prediction Correlation
            correlation_df = test_pred_full.groupby(['ticker', 'cv_iteration', 'method', 'num_feature']).apply(lambda x: x['y_true'].corr(x['y_pred'])).reset_index(name='correlation')
            correlation_df.to_excel(f"{self.test_pred_file_name}correlation_{self.current_date}.xlsx", index=False)
            print(f"Test outputs with the correlation field saved to: {self.test_pred_file_name}correlation_{self.current_date}.xlsx")



