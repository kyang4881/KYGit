import pandas as pd
import numpy as np
import datetime
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, BorderlineSMOTE


def check_column_types(df):
    """...
    Args:
        df (dataframe):
    Returns:   
    """
    categorical_columns = []
    numerical_columns = []
    # Check and return the categorical and numerical columns
    for column in df.columns:
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            numerical_columns.append(column)
    return categorical_columns, numerical_columns

def create_df(all_features, all_scores, all_features_rev, all_scores_rev, dataset_size, total_time, method_name, dataset_name, pred_type, feature_score, cm_val, cm_val_reversed, rebalance, rebalance_type, data_shape, is_max_acc, cv_iteration):
    """...
    Args:
        all_features (list):
        all_scores (list):
        all_features_rev (list):
        all_scores_rev (list):
        dataset_size (tuple):
        total_time (float):
        method_name (str):
        dataset_name (str):
        pred_type (str):
        feature_score (float):
        cm_val (dict):
        cm_val_reversed (dict):
        rebalance (bool):
        rebalance_type (str):
        data_shape (dict):
        is_max_acc (bool):
        cv_iteration (int):
    Returns:   
    """
    df_len = len(all_scores)
    # Generate a dataframe containing the metrics for all feature subsets
    results_df = pd.DataFrame({
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]*df_len,
        "feature": all_features,
        "score": all_scores,
        "feature_reversed" : all_features_rev,
        "score_reversed_rank": all_scores_rev,
        "method": [method_name]*df_len,
        "dataset": [dataset_name]*df_len,
        "dataset_size": [dataset_size]*df_len,
        "runtime_sec": [total_time]*df_len,
        "prediction_type": [pred_type]*df_len,
        "feature_score": list(feature_score) + [0]*(df_len - len(feature_score)),
        "cm_val": cm_val,
        "cm_val_reversed": cm_val_reversed,
        "rebalance": [rebalance]*df_len,
        "rebalance_type": [rebalance_type]*df_len,
        "data_shape": [data_shape]*df_len,
        "is_max_acc": is_max_acc,
        "cv_iteration": [cv_iteration]*df_len
    })
    return results_df

def add_to_dataframe(df, import_name, export, export_name=None):
    """...
    Args:
        df (dataframe):
        import_name (str):
        export (bool):
        export_name (bool):
    Returns:
    """
    # Read the import file
    merged_results_import = pd.read_excel(f"./{import_name}")
    merged_results_import.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    # Add index to the dataframe
    index_df = pd.DataFrame({"index" : np.arange(0, np.shape(df)[0]).tolist()})
    df_indexed_df = pd.concat([index_df, df], axis=1)

    print("Shape (Imported df): ", np.shape(merged_results_import))
    display(merged_results_import.head())

    print("Shape (df): ", np.shape(df_indexed_df))
    display(df_indexed_df.head())

    # Merge dataframes, convert feature names to string, remove duplicates
    merged_results_import_updated = pd.concat([merged_results_import, df_indexed_df])
    merged_results_import_updated['feature'] = merged_results_import_updated['feature'].map(str)
    merged_results_import_updated['feature_reversed'] = merged_results_import_updated['feature_reversed'].map(str)
    merged_results_import_updated = merged_results_import_updated.drop_duplicates()

    if export:
        merged_results_import_updated.to_excel(f"./{export_name}.xlsx", index=False)
        print(f"Exported: {export_name}.xlsx")

    return merged_results_import_updated

def rebalance_data(data, rebalance_type, seed):
    """...
    Args:
        data (dict):
        rebalance_type (str):
        seed (int):
    Returns:   
    """
    print(f"X_train shape before resmapling. X_train: {np.shape(data['X_train'])} y_train: {np.shape(data['X_train'])}")
    print(f"y_train classes before resampling: {dict(Counter(data['y_train']))}")

    X_data_df = data['X_train']
    y_data_df = data['y_train']

    # Various methods for resampling
    if rebalance_type.lower() == "random_over_sampler":
        oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        oversampler.fit(X_data_df, y_data_df)
        X_resampled, y_resampled = oversampler.fit_resample(X_data_df, y_data_df)
        
    if rebalance_type.lower() == "smoten":
        sampler = SMOTEN(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    if rebalance_type.lower() == "smote":
        sampler = SMOTE(random_state=seed)
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    if rebalance_type.lower() == "smotenc":
        sampler = SMOTENC(random_state=seed)
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    if rebalance_type.lower() == "borderlinesmote":
        sampler = BorderlineSMOTE(random_state=seed)        
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    data['X_train'] = X_resampled
    data['y_train'] = y_resampled

    print(f"X_train shape after resmapling. X_train: {np.shape(data['X_train'])} y_train: {np.shape(data['X_train'])}")
    print(f"y_train classes after resampling: {dict(Counter(data['y_train']))}")
    return data

