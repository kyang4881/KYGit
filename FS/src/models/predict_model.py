# Author: JYang
# Last Modified: Oct-24-2023
# Description: This script provides the method(s) for computing evaluation metrics

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score
from feature_selection_timeseries.src.models.train_model import generateModel
from feature_selection_timeseries.src.models.utils import rebalance_data
from feature_selection_timeseries.src.visualization.visualize import plotScore
import torch

class computeScore_backup:
    """ A class that contains two methods, filter_data is a method for trimming the features 
        of the dataset based on feature selection methods and pred_score is a method for
        generating the scoring metrics based on the predictions using subsets of features.
        
    Args:
        data_dict (dict): a dictionary containing dataframes of the train and validation data
        keep_cols (list): a list of columns to filter for
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
    """
    def __init__(self, data_dict, keep_cols, pred_type, seed):
        self.data_dict = data_dict
        self.keep_cols = keep_cols 
        self.data_dict_new = {}
        self.pred_type = pred_type.lower()
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def filter_data(self):
        """A method for filtering dataframes based on a selected list of features"""
        # Extract features and labels
        features = [k for k in self.data_dict.keys() if "X_" in k]
        labels = [k for k in self.data_dict.keys() if "y_" in k]
        # Filter features
        for f, l in zip(features, labels):
            if self.data_dict[f] is None:
                self.data_dict_new[f] = []
                self.data_dict_new[l] = []
            else:
                self.data_dict_new[f] = self.data_dict[f][self.keep_cols]
                self.data_dict_new[l] = self.data_dict[l]

    def pred_score(self):
        """ A method for generating prediction metrics on the validation data
        Returns: 
            score (float): model prediction accuracy
            cm_val (str): other scoring metrics 
        """
        # Apply feature filter
        self.filter_data()
        y_val_true = self.data_dict_new['y_val']
        # Generate trained XGBoost model
        self.trained_model = generateModel(data_dict=self.data_dict_new, pred_type=self.pred_type, seed=self.seed).get_model()
        # Generate predictions
        y_pred = self.trained_model.predict(self.data_dict_new['X_val'])
        # Compute and save scoring metrics
        if self.pred_type == "classification":
            y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
            score = f1_score(y_val_true, y_pred) #accuracy_score(y_val_true, y_pred)      # Changed Oct 09
            tn, fp, fn, tp = confusion_matrix(y_val_true, y_pred).ravel()
            cm = confusion_matrix(y_val_true, y_pred)
            cm_val = {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "total_positive": np.sum(self.data_dict_new['y_val'] == 1),
                "total_negative": np.sum(self.data_dict_new['y_val'] == 0),
                "precision": precision_score(y_val_true, y_pred, zero_division=0),
                "recall": recall_score(y_val_true, y_pred),
                "f1_score": f1_score(y_val_true, y_pred),
                "accuracy": accuracy_score(y_val_true, y_pred)
                }
            return score, str(cm_val)
        else:
            score = mean_squared_error(y_val_true, y_pred)
            return score

def run_scoring_pipeline_backup_original(feature_impt, input_data_dict, pred_type, rebalance, rebalance_type, seed):
    """ A method for generating model prediction scoring metrics
    Args:
        feature_impt (list): list of top features to use for model prediction
        input_data_dict (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        seed (int): a random state
    Returns:    
        all_scores (list): a list of prediction accuracies for each feature subset
        all_scores_reverse (list): a list of prediction accuracies for each feature subset with top features subsets in reversed order
        all_features (list): a list of top features subsets
        all_features_reverse (list): a list of top features subsets in reversed order
        all_cm_val (list): a list of other scoring metrics
        all_cm_val_reverse (list): a list of other scoring metrics for top features subsets in reversed order
    """
    # Rebalance the dataset
    if rebalance:
        input_data_dict_rebalanced = rebalance_data(data=input_data_dict, rebalance_type=rebalance_type, seed=seed)
   
    all_scores = []
    all_features = []
    all_cm_val = []
    # For each top n features, compute and save the scoring metrics
    for i in range(1, len(feature_impt)+1):        
        compute_score_1 = computeScore(
            data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
            keep_cols=feature_impt[:i] if i < len(feature_impt) else list(input_data_dict["X_train"].keys()),
            pred_type=pred_type,
            seed=seed
          )
        score, cm_val = compute_score_1.pred_score()
        all_scores.append(score)
        all_features.append(feature_impt[:i])
        all_cm_val.append(cm_val)

    all_scores_reverse = []
    all_features_reverse = []
    all_cm_val_reverse = []
    # For each top n features in reversed order, compute and save the scoring metrics
    for i in range(1, len(feature_impt)+1):
        compute_score_1 = computeScore(
            data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
            keep_cols=feature_impt[::-1][:i] if i < len(feature_impt) else list(input_data_dict["X_train"].keys()),
            pred_type=pred_type,
            seed=seed
        )
        score, cm_val = compute_score_1.pred_score()
        all_scores_reverse.append(score)
        all_features_reverse.append(feature_impt[::-1][:i])
        all_cm_val_reverse.append(cm_val)

    # Plot the scores
    plotter = plotScore(data=all_scores, feature_impt=feature_impt, pred_type=pred_type)
    plt.close()
    display(plotter.score_plot())
    print("Rank Reversed:\n")
    plt.close()
    # Plot the scores for the reversed feature order
    plotter_reversed = plotScore(data=all_scores_reverse, feature_impt=feature_impt[::-1], pred_type=pred_type)
    display(plotter_reversed.score_plot())

    return all_scores, all_scores_reverse, all_features, all_features_reverse, all_cm_val, all_cm_val_reverse


def run_scoring_pipeline_backup_2(feature_impt, n_features, input_data_dict, pred_type, rebalance, rebalance_type, seed):
    """ A method for generating model prediction scoring metrics
    Args:
        feature_impt (list): list of top features to use for model prediction
        n_features (int): number of features to use; used for top n and bottom n features
        input_data_dict (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        seed (int): a random state
    Returns:    
        all_scores (list): a list containing the prediction score for the top n features
        all_scores_reverse (list): a list containing the prediction score for the bottom n features
        all_features (list): a list of top n features subsets
        all_features_reverse (list): a list of bottom n features subsets
        all_cm_val (list): a list of other scoring metrics for top n features
        all_cm_val_reverse (list): a list of other scoring metrics for bottom n features
    """
        
    # Rebalance the dataset
    if rebalance:
        input_data_dict_rebalanced = rebalance_data(data=input_data_dict, rebalance_type=rebalance_type, seed=seed)
   
    all_scores = []
    all_features = []
    all_cm_val = []
    # For top n features, compute and save the scoring metrics
    compute_score_1 = computeScore(
        data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
        keep_cols=feature_impt[:n_features],
        pred_type=pred_type,
        seed=seed
      )
    score, cm_val = compute_score_1.pred_score()
    all_scores.append(score)
    all_features.append(feature_impt[:n_features])
    all_cm_val.append(cm_val)

    all_scores_reverse = []
    all_features_reverse = []
    all_cm_val_reverse = []
    
    # For bottom n features, compute and save the scoring metrics
    compute_score_1 = computeScore(
        data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
        keep_cols=feature_impt[::-1][:n_features],
        pred_type=pred_type,
        seed=seed
    )
    score, cm_val = compute_score_1.pred_score()
    all_scores_reverse.append(score)
    all_features_reverse.append(feature_impt[::-1][:n_features])
    all_cm_val_reverse.append(cm_val)

    # Plot the scores
    plotter = plotScore(data=all_scores, feature_impt=feature_impt, pred_type=pred_type)
    plt.close()
    display(plotter.score_plot())
    print("Rank Reversed:\n")
    plt.close()
    # Plot the scores for the reversed feature order
    plotter_reversed = plotScore(data=all_scores_reverse, feature_impt=feature_impt[::-1], pred_type=pred_type)
    display(plotter_reversed.score_plot())

    return all_scores, all_scores_reverse, all_features, all_features_reverse, all_cm_val, all_cm_val_reverse


def run_scoring_pipeline(feature_impt, n_features, input_data_dict, pred_type, rebalance, rebalance_type, seed, feature_direction):
    """ A method for generating model prediction scoring metrics
    Args:
        feature_impt (list): list of top features to use for model prediction
        n_features (int): number of features to use; used for top n and bottom n features
        input_data_dict (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        seed (int): a random state
        feature_direction (str): a string indicating whether to evaluate top ranking features or both top and bottom ranking features: top or both
    Returns:    
        all_scores (list): a list containing the prediction score for the top n features
        all_scores_reverse (list): a list containing the prediction score for the bottom n features
        all_features (list): a list of top n features subsets
        all_features_reverse (list): a list of bottom n features subsets
        all_cm_val (list): a list of other scoring metrics for top n features
        all_cm_val_reverse (list): a list of other scoring metrics for bottom n features
    """
    # Rebalance the dataset
    if rebalance:
        input_data_dict_rebalanced = rebalance_data(data=input_data_dict, rebalance_type=rebalance_type, seed=seed)
   
    all_scores = []
    all_features = []
    all_cm_val = []
    
    # Top n and all features
    use_num_features = sorted([int(i) for i in set([min(n_features, len(feature_impt)), len(feature_impt)])])
    
    # For top n and all features, compute and save the scoring metrics
    if feature_direction in ['top', 'both']:
        for i in use_num_features:        
            compute_score_1 = computeScore(
                data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
                keep_cols=feature_impt[:i] if i < len(feature_impt) else list(input_data_dict["X_train"].keys()),
                pred_type=pred_type,
                seed=seed
              )
            score, cm_val = compute_score_1.pred_score()
            all_scores.append(score)
            all_features.append(feature_impt[:i])
            all_cm_val.append(cm_val)

    all_scores_reverse = []
    all_features_reverse = []
    all_cm_val_reverse = []
    
    # For each top n and all features in reversed order, compute and save the scoring metrics
    if feature_direction in ['both']:
        for i in use_num_features:     
            compute_score_1 = computeScore(
                data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
                keep_cols=feature_impt[::-1][:i] if i < len(feature_impt) else list(input_data_dict["X_train"].keys()),
                pred_type=pred_type,
                seed=seed
            )
            score, cm_val = compute_score_1.pred_score()
            all_scores_reverse.append(score)
            all_features_reverse.append(feature_impt[::-1][:i])
            all_cm_val_reverse.append(cm_val)
        
    # Plot the scores
    if feature_direction in ['top', 'both']:
        print("\n\n-----------------------------------Evaluate Features From Highest Importance-----------------------------------\n")
        plt.close()
        plotter = plotScore(data=all_scores, feature_impt=feature_impt, pred_type=pred_type, use_num_features=use_num_features)
        display(plotter.score_plot())
    
    # Plot the scores for the reversed feature order
    if feature_direction in ['both']: 
        print("\n\n-----------------------------------Evaluate Features From Lowest Importance-----------------------------------\n")
        plt.close()
        plotter_reversed = plotScore(data=all_scores_reverse, feature_impt=feature_impt[::-1], pred_type=pred_type, use_num_features=use_num_features)
        display(plotter_reversed.score_plot())

    return all_scores, all_scores_reverse, all_features, all_features_reverse, all_cm_val, all_cm_val_reverse




class computeScore:
    """ A class that contains two methods, filter_data is a method for trimming the features 
        of the dataset based on feature selection methods and pred_score is a method for
        generating the scoring metrics based on the predictions using subsets of features.
        
    Args:
        data_dict (dict): a dictionary containing dataframes of the train and validation data
        keep_cols (list): a list of columns to filter for
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
    """
    def __init__(self, data_dict, keep_cols, pred_type, seed):
        self.data_dict = data_dict
        self.keep_cols = keep_cols 
        self.data_dict_new = {}
        self.pred_type = pred_type.lower()
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def filter_data(self):
        """A method for filtering dataframes based on a selected list of features"""
        # Extract features and labels
        features = [k for k in self.data_dict.keys() if "X_" in k]
        labels = [k for k in self.data_dict.keys() if "y_" in k]
        # Filter features
        for f, l in zip(features, labels):
            if self.data_dict[f] is None:
                self.data_dict_new[f] = []
                self.data_dict_new[l] = []
            else:
                self.data_dict_new[f] = self.data_dict[f][self.keep_cols]
                self.data_dict_new[l] = self.data_dict[l]

    def pred_score(self):
        """ A method for generating prediction metrics on the validation data
        Returns: 
            score (float): model prediction accuracy
            cm_val (str): other scoring metrics 
        """
        # Apply feature filter
        self.filter_data()
        y_val_true = self.data_dict_new['y_val']
        # Generate trained XGBoost model
        self.trained_model = generateModel(data_dict=self.data_dict_new, pred_type=self.pred_type, seed=self.seed).get_model()
        # Generate predictions
        dval = xgb.DMatrix(np.array(self.data_dict_new['X_val']), feature_names=list(self.data_dict_new['X_val'].columns))
        y_pred = self.trained_model.predict(dval)
        # Compute and save scoring metrics
        if self.pred_type == "classification":
            y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
            score = f1_score(y_val_true, y_pred) #accuracy_score(y_val_true, y_pred)      # Changed Oct 09
            tn, fp, fn, tp = confusion_matrix(y_val_true, y_pred).ravel()
            cm = confusion_matrix(y_val_true, y_pred)
            cm_val = {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "total_positive": np.sum(self.data_dict_new['y_val'] == 1),
                "total_negative": np.sum(self.data_dict_new['y_val'] == 0),
                "precision": precision_score(y_val_true, y_pred, zero_division=0),
                "recall": recall_score(y_val_true, y_pred),
                "f1_score": f1_score(y_val_true, y_pred),
                "accuracy": accuracy_score(y_val_true, y_pred)
                }
            return score, str(cm_val)
        else:
            score = mean_squared_error(y_val_true, y_pred)
            return score








