import numpy as np
import pandas as pd
from collections import Counter
from feature_selection_timeseries.src.models.predict_model import run_scoring_pipeline
from feature_selection_timeseries.src.models.utils import create_df, add_to_dataframe, check_column_types
from feature_selection_timeseries.src.features.feature_selection import sage_importance, permutation_test, xgb_importance, shap_importance, cae_importance
from feature_selection_timeseries.src.preprocessing.preprocessor import Preprocess
from feature_selection_timeseries.src.models.predict_model import generateModel


def get_metrics_df(seed, target_colname, data, data_original, full_df, method_name, dataset_name, pred_type, cv_iteration, rebalance=False, rebalance_type=None, append_to_full_df=False):
    """ ...
    Args:
        seed (int):
        target_colname (str):
        data (dict):
        data_original (dict):
        full_df (dataframe):
        method_name (str):
        dataset_name (str):
        pred_type (str):
        cv_iteration (int):
        rebalance (bool):
        rebalance_type (str):
        append_to_full_df (str):
    Returns:      
    """   
    model = generateModel(
        data_dict = data,
        pred_type = pred_type,
        seed = seed
    ).get_model()
        
    # Generate ranked features and other input variables
    if method_name.lower() == "sage":
        sorted_features, feature_scores, total_time = sage_importance(model=model, data=data, pred_type=pred_type, seed=seed, target_colname=target_colname)

    if method_name.lower() == "permutation":
        sorted_features, feature_scores, total_time = permutation_test(model=model, data = data, pred_type = pred_type, seed=seed, target_colname=target_colname)

    if method_name.lower() == "xgboost":
        sorted_features_xgb, feature_scores, total_time = xgb_importance(model=model, data=data, pred_type=pred_type, seed=seed, target_colname=target_colname)
        sorted_features = list(sorted_features_xgb) + [f for f in list(data['X_train'].columns) if f not in sorted_features_xgb]

    if method_name.lower() == "shap":
        sorted_features, feature_scores, total_time = shap_importance(model=model, data = data, pred_type = pred_type, seed=seed, target_colname=target_colname)

    if method_name.lower() == "cae":
        sorted_features, feature_scores, total_time = cae_importance(data = data, pred_type = pred_type, seed=seed, target_colname=target_colname)

    # Generate the scoring metrics
    all_scores, all_scores_reverse, all_features, all_features_reverse, cm_val, cm_val_reversed  = run_scoring_pipeline(
        feature_impt = sorted_features,
        input_data_dict = data,
        pred_type = pred_type,
        rebalance=rebalance,
        rebalance_type=rebalance_type,
        seed=seed
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
            "X_train (instance/feature)": X_train_shape,
            "X_val (instance/feature)": X_val_shape,
            "y_train (class/count)": y_train_dict,
            "y_val (class/count)": y_val_dict,
            "X_total (instance/features)": (X_train_shape[0] + X_val_shape[0], X_train_shape[1]),
            "y_total (class/count)": {0: y_train_dict[0] + y_val_dict[0], 1: y_train_dict[1] + y_val_dict[1]}
        }),
        is_max_acc = [num_index == all_scores.index(max(all_scores)) for num_index in range(len(all_scores))],
        cv_iteration = cv_iteration
    )

    display(results_df.head())

    # Append scoring metrics for feature subsets into a dataframe
    if append_to_full_df:
        full_df = pd.concat([add_to_dataframe(
            df = results_df,
            import_name = "feature_selection/data/external/empty.xlsx",
            export = False,
            export_name = None
        ), full_df])

    return full_df


def run(seed, target_colname, data, full_df, method_name, dataset_name, pred_type, num_cv_splits, rebalance=False, rebalance_type=None, append_to_full_df=False):
    """ ...
    Args:
        seed (int):
        target_colname (str):
        data (dict):
        full_df (dataframe):
        method_name (str):
        dataset_name (str):
        pred_type (str):
        num_cv_splits (int):
        rebalance (bool):
        rebalance_type (str):
        append_to_full_df (str):
    Returns:      
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
        num_cv_splits = num_cv_splits
    )
    # Dictionary containing all cross validation splits
    compiled_data = processor1.split_data()
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
            rebalance=rebalance, 
            rebalance_type=rebalance_type, 
            append_to_full_df=append_to_full_df
        )
    return full_df