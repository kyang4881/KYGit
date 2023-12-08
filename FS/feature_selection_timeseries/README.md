<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/flow_diagram.png" width="1200" />
</p>

---

This project involves the development of a Python pipeline encompassing data ingestion, feature selection, model optimization, and prediction using the XGBoost algorithm. It emphasizes adaptability in selecting variable numbers of features, considering computational costs, eliminating irrelevant features, and mitigating overfitting. The pipeline's foundation lies in a thorough literature review that identifies and implements literature state-of-the-art feature selection methods, as well as traditional feature selection methods that are compatible with the XGBoost architecture. The dataset encompasses both standard datasets and a custom financial time series dataset created to simulate real-world conditions. The pipeline integrates rebalancing techniques and data transformation processes. The goal is to achieve robust model performance through back testing and hyperparameter tuning, paving the way for continuous improvement and future iterations of the project.

---

<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/py_ver.png" width="150" />
</p>


## Pseudocode

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/pseudocode.png" width="1200" />
</p>

---

## Rolling Window Back Testing

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/rolling_bt.png" width="1200" />
</p>

---

## How Features Are Aggregated

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/feature_selection_example.png" width="1200" />
</p>

---

## Train-Validation Splits (All Features)

The same concept applies for the top 50 subset features.

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/bt_all_features.png" width="1200" />
</p>

---

## Notebook

Import the necessary libraries in the requirements.txt file.

Change the file directory to the path where the feature_selection_timeseries folder is located.

```python
import os
directory_path = input("Enter your file directory: ")
os.chdir(directory_path)
```

Import other necessary libraries.

```python
from feature_selection_timeseries.src.models.pipeline import run
from feature_selection_timeseries.src.models.utils import create_time_feature, tune_cv_split, convert_to_sample 
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
import csv
```

Defining the file names and paths where the original data is stored and where the data filtered by the date threshold will be saved to.

```python
year = "2006"
date_threshold = year + '-01-01'  # filter for data with date >=
sub_folder = "sp500_subset"

# path of the file containing the features
x_filename_in = 'stock_x.csv'
x_filename_out = f'stock_x_sample_regression_{year}_filtered_11_stocks.csv'
x_in_path = f'./feature_selection_timeseries/data/raw/{sub_folder}/'
x_out_path = f'./feature_selection_timeseries/data/raw/{sub_folder}/stocks/'

# path of the file containing the label
y_filename_in = 'stock_y_ret84.csv'
y_filename_out = f'stock_y_ret84_sample_regression_{year}_filtered_11_stocks.csv'
y_in_path = x_in_path
y_out_path = x_out_path
```

If you need to filter the original data by certain stocks, add the stock tickers to the list below and run the convert_to_sample() function.

```python
stocks=[
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"
]

convert_to_sample(
    path_in=x_in_path,
    path_out=x_out_path,
    filename_in = x_filename_in,
    filename_out = x_filename_out,
    date_threshold=date_threshold,
    filter_type="combined stocks",
    stocks=stocks
)

convert_to_sample(
    path_in=y_in_path,
    path_out=y_out_path,
    filename_in = y_filename_in,
    filename_out = y_filename_out,
    date_threshold=date_threshold,
    filter_type="combined stocks",
    stocks=stocks
)
```

Import the files containing the features and target values. Then create a set of new date features. Finally merge the features and label into a single dataframe.

```python
# Specify the path to your CSV file
x_path = f'{x_in_path}{x_filename_out}'
y_path = f'{y_in_path}{y_filename_out}'
# Import data
sp500_df = pd.read_csv(x_path)
y = pd.read_csv(y_path)
# Adjust labels
y['target'] = y['ret_fwd_84']
# Create additional time features
sp500_df = create_time_feature(sp500_df)
# Combine features and target
sp500_df = pd.concat([sp500_df.iloc[:, 1:], y['target']], axis=1)
```

As an illustration, to keep execution time manageable a single stock will be used in this example. Additionally, only two sets of hyperparameters are considered. Run the codes below to filter for the ticker ADBE (Adobe Inc).  

```python
sp500_df = sp500_df[sp500_df['ticker'] == 'ADBE']
y = y[y['ticker'] == 'ADBE']
```

In the train_model.py file.

```python
param_grid = {
   'min_child_weight': [0.5, 1],
   'gamma': [0.01],  
   'max_depth': [6], 
   'learning_rate': [0.3], 
   'alpha': [0]  
}
    
```

View the updated dataframes.

```python
display(y.head())
print(f"Dataset Shape: {np.shape(y)}")
```
<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/adbe_table_y.png" width="300" />
</p>


```python
display(sp500_df.head())
print(f"Dataset Shape: {np.shape(sp500_df)}")
```
<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/adbe_table_x.png" width="1200" />
</p>

Initializing the pipeline.

Compute the list of possible splits and their corresponding train-validation sizes. 5-fold splits will be used in this example, as specified. Note that the first term is not used, as the rolling window size and train-validation split size are computed separately, but the validation/test size is indicated by the middle term, and the number of splits for both the back test rolling window and train-validation parts is the same.

```python
# Possible train validation splits
train_test_list = [tune_cv_split(
    sp500_df.iloc[-np.shape(sp500_df)[0]:,:],
    val_test_prop_constraint = 0.2, # Size of validation set relative to the train set
    num_split_constraint = 5 # Number of splits
)[-1]]

keep_data_index = train_test_list[0][0]*train_test_list[0][2] + 2*train_test_list[0][1]
print(f"\nUsing Split: {train_test_list}")
```

<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/train_val_split.png" width="300" />
</p>


Initialize the other arguments.

```python
r1 = run(
    cross_validation_type= "moving window", 
    save_output_file = True, 
    raw_df = sp500_df.iloc[-keep_data_index:, :].reset_index(drop=True), 
    y = y.iloc[-keep_data_index:, :].reset_index(drop=True),
    train_test_list = train_test_list, 
    methods = ["xgboost"], 
    rebalance_type = ["None"], 
    label_cols = [], 
    do_not_encode_cols = ["dayofmonth", "dayofweek", "quarter", "month", "year", "dayofyear", "weekofyear"], 
    seed = 42,
    target_colname = "target",
    dataset_name = "sp500",
    pred_type = "regression",
    append_to_full_df = False,
    n_features = 50,  
    feature_direction = "top", 
    train_outputs_file_name = None,
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    scaler_filename = "./feature_selection_timeseries/data/processed/scaler_saved.save",
    encoder_filename = "./feature_selection_timeseries/data/processed/encoder_saved.save",
    label_encoder_filename = "./feature_selection_timeseries/data/processed/lalbel_encoder_saved.save",
    test_output_file_name = f"./feature_selection_timeseries/data/delete/Consolidated_Stocks_FS_timeseries_sp500_outputs_test_results_",
    test_pred_file_name = f"./feature_selection_timeseries/data/delete/Consolidated_Stocks_FS_timeseries_sp500_outputs_test_preds_",
    print_outputs_train = True,
    print_outputs_test = True
)
```

Train the model. Average the feature scores and validation scores, and determine the optimal hyperparameters. 

```python
r1.train()
```

Train Results Example: 50 Features

<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/train_50.png" width="800" />
</p>

Test Results Example: All Features

<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/train_all.png" width="800" />
</p>


Set the optimal hyperparameters, and apply the optimal models on the hold out test data.

```python
r1.test()
```

Test Results Example: 50 Features

<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/test_50.png" width="800" />
</p>

Test Results Example: All Features

<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/images/test_all.png" width="800" />
</p>


---

## Citations



            