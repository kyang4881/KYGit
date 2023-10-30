# Author: JYang
# Last Modified: Oct-24-2023
# Description: This script provides the method(s) that consolidate multiple methods into a wrapped run function to execute the pipeline

import numpy as np
import pandas as pd
from collections import Counter
from feature_selection_timeseries.src.models.predict_model import run_scoring_pipeline
from feature_selection_timeseries.src.models.utils import create_df, add_to_dataframe, check_column_types
from feature_selection_timeseries.src.preprocessing.preprocessor import Preprocess
from feature_selection_timeseries.src.models.train_model import generateModel
from feature_selection_timeseries.src.features.feature_selection import featureValues

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
        cv_iteration (int): an integer indicating the cross validation iteration
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        append_to_full_df (bool): a boolean indicating whether to append model results to the existing tracked results
        n_features (int): number of features to use; used for top n and bottom n features
        feature_direction (str): a string indicating whether to evaluate top/bottom/both ranking features
    Returns:    
        full_df (dataframe): a dataframe containing all currently tracked model results
    """   
    model = generateModel(
        data_dict = data,
        pred_type = pred_type,
        seed = seed
    ).get_model()
        
    feature_values = featureValues(data_dict=data, pred_type=pred_type, model=model, seed=seed, target_colname=target_colname)
        
    # Generate ranked features and other input variables
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
    y_train_dict = dict(sorted(Counter(data["y_train"]).items()))
    y_val_dict = dict(sorted(Counter(data["y_val"]).items()))
    
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
            "y_total (class/count)": {0: y_train_dict[0] + y_val_dict[0], 1: y_train_dict[1] + y_val_dict[1]}            # If KeyError => data has a missing class; not enough data 
        }),
        is_max_acc = [num_index == all_scores.index(max(all_scores)) for num_index in range(len(all_scores))],
        cv_iteration = cv_iteration
    )

    display(results_df.head())
    
    # Append scoring metrics for feature subsets into a dataframe
    #if append_to_full_df:
    #    full_df = pd.concat([add_to_dataframe(
    #        df = results_df,
    #        import_name = "feature_selection_timeseries/data/external/empty.xlsx",
    #        export = False,
    #        export_name = None
    #    ), full_df])

    if append_to_full_df:
        full_df = pd.concat([results_df, full_df])
        
    return full_df


def run(seed, target_colname, data, full_df, method_name, dataset_name, pred_type, num_cv_splits=5, rebalance=False, rebalance_type=None, append_to_full_df=False, train_examples=1, test_examples=1, n_features=0, feature_direction="both"):
    """ A method that runs through the entire pipeline by wrapping the required methods
    Args:
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
        data (dict): a dictionary containing train and validation data
        full_df (dataframe): a dataframe containing all currently tracked model results
        method_name (str): name of the feature selection model
        dataset_name (str): name of the dataset
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        num_cv_splits (int): an integer indicating the number of cross validation fold
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        append_to_full_df (bool): a boolean indicating whether to append model results to the existing tracked results
        n_splits (int): number of cross-validation splits
        train_examples (int): number of train examples in each cv split
        test_examples (int): number of test examples in each cv split
        n_features (int): number of features to use; used for top n and bottom n features
        feature_direction (str): a string indicating whether to evaluate top/bottom/both ranking features
    Returns:      
        full_df (dataframe): a dataframe containing all currently tracked model results
    """   
    # Extract categorical and numerical features
    categorical_cols, numerical_cols = check_column_types(data.iloc[:,:-1])
    print("Categorical Columns:", categorical_cols)
    print("Numerical Columns:", numerical_cols)
    
    # Preprocessing the data
    processor1 = Preprocess(
        data = data,
        target = target_colname,
        cat_cols = categorical_cols,
        num_cols = numerical_cols, 
        num_cv_splits = num_cv_splits,
        train_examples = train_examples,
        test_examples = test_examples
    )
    # Dictionary containing all cross validation splits
    compiled_data, scaler_saved, encoder_saved, train_test_index = processor1.split_data()
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
    return full_df, scaler_saved, encoder_saved
