<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/flow_diagram.png" width="1200" />
</p>

---

This project involves the development of a Python pipeline encompassing data ingestion, feature selection, model optimization, and prediction using the XGBoost algorithm. It emphasizes adaptability in selecting variable numbers of features, considering computational costs, eliminating irrelevant features, and mitigating overfitting. The pipeline's foundation lies in a thorough literature review that identifies and implements literature state-of-the-art feature selection methods, as well as traditional feature selection methods that are compatible with the XGBoost architecture. The dataset encompasses both standard datasets and a custom financial time series dataset created to simulate real-world conditions. The pipeline integrates rebalancing techniques and data transformation processes. The goal is to achieve robust model performance through back testing and hyperparameter tuning, paving the way for continuous improvement and future iterations of the project.

---

## Pseudocode

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/pseudocode.png" width="1200" />
</p>

---

## Rolling Window Back Testing

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/rolling_bt.png" width="1200" />
</p>

---

## How Features Are Aggregated

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/feature_selection_example.png" width="1200" />
</p>

---

## 1 Fixed Window Back Testing For All Features

<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/bt_all_features.png" width="1200" />
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

As an illustration, to keep execution time manageable a single stock will be used in this example. Run the codes below to filter for the ticker ADBE (Adobe Inc).

```python
sp500_df = sp500_df[sp500_df['ticker'] == 'ADBE']
y = y[y['ticker'] == 'ADBE']
```

View the updated dataframes.

```python
display(y.head())
print(f"Dataset Shape: {np.shape(y)}")
```
<p align="left">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/adbe_table_y.png" width="300" />
</p>


```python
display(sp500_df.head())
print(f"Dataset Shape: {np.shape(sp500_df)}")
```
<p align="center">
  <img src="https://raw.githubusercontent.com/kyang4881/KYGit/master/FS/feature_selection_timeseries/docs/artwork/adbe_table_x.png" width="1200" />
</p>


---

## Citations
