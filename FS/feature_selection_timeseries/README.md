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

---

## Citations
