<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/flow_diagram.png" width="1200" />
</p>

---

This project involves the development of a Python pipeline encompassing data ingestion, feature selection, model optimization, and prediction using the XGBoost algorithm. It emphasizes adaptability in selecting variable numbers of features, considering computational costs, eliminating irrelevant features, and mitigating overfitting. The pipeline's foundation lies in a thorough literature review that identifies and implements literature state-of-the-art feature selection methods, as well as traditional feature selection methods that are compatible with the XGBoost architecture. The dataset encompasses both standard datasets and a custom financial time series dataset created to simulate real-world conditions. The pipeline integrates rebalancing techniques and data transformation processes. The goal is to achieve robust model performance through back testing and hyperparameter tuning, paving the way for continuous improvement and future iterations of the project.

---

## Pseudocode

```python
For each feature selection method:

    For each moving window:
    
        For each hyperparam in hyperparameter combinations:
        
            For each bt_split in backtesting splits:
            
                feature_importance = feature_selection_method(bt_split_train_data)       
                
                For num_feature in list of possible num_features:
                
					feature_subset = select_features(feature_importance, num_feature)
                  
					model = train_model(bt_split_train_data[:, feature_subset], hyperparam)
                  
					predictions = model.predict(bt_split_validation_data)
                  
					score = RMSE(predictions, bt_split_validation_data_true_values)

            average_rmse = average rmse of all bt splits by possible num_features
            updated_features = average feature importance ranked and filtered by possible num_features
        
        best_hyperparam, features = hyperparam with min(average_rmse) and corresponding updated_features by possible num_features
        best_hyperparam_model = train_model(all_train_data[:, features], best_hyperparam)
        best_hyperparam_predictions = best_hyperparam_model.predict(test_data)
        best_hyperparam_score = RMSE(best_hyperparam_predictions, test_data_true_values)
        
    avg_method_score = average(best_hyperparam_scores) across all moving windows
    
pick the method with min(avg_method_score)

```