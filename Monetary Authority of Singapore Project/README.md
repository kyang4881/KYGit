## MAS Project

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/py_ver.png" width="150" />
</p>

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/flow_diagram.png" width="1200" />
</p>

---

## Introduction

The Monetary Authority of Singapore (MAS) is guided by a set of values and a comprehensive code of conduct that underpin its mission to promote sustained non-inflationary economic growth and maintain a sound and progressive financial center. Upholding values such as integrity, commitment, enterprise, and teamwork, MAS fosters a culture that maintains zero tolerance towards fraud and misconduct. The code of conduct emphasizes the highest standards of personal and professional behavior, duty of confidentiality, avoidance of conflicts of interest, and responsible use of MAS' resources.

Functioning as Singapore's central bank and integrated financial regulator, MAS plays a pivotal role in promoting sustainable economic growth through monetary policy formulation and oversight of financial institutions. Its functions include acting as the central bank, conducting integrated supervision of financial services, managing official foreign reserves, and developing Singapore as an international financial center. The MAS Act, enacted in 1970, granted MAS the authority to regulate the financial services sector, marking a significant evolution in its role as a regulator. Over the years, MAS has adapted to the demands of a complex financial environment, leading to the formation of the institution on January 1, 1971. Since then, MAS has played a crucial role in regulating various sectors, including insurance and securities, and assumed additional responsibilities such as currency issuance following its merger with the Board of Commissioners of Currency in 2002.

## Problem Statement

---

In the complex landscape of stock market analysis, the dynamics of price changes are shaped by a multitude of factors including historical data, fundamental indicators, and the intricate psychology of investors. However, the diversity of these features poses challenges in attaining heightened prediction accuracy. To address this, a strategic feature selection process may be imperative prior to deploying machine learning models for predictions. This process should not only focus on key features but also serves to mitigate issues such as irrelevant variables, computational costs, and overfitting, thereby enhancing the overall performance of the machine learning models. Care must be taken to strike the right balance when selecting features. Opting for too few features may leave the model with insufficient information for accurate predictions, while an excessive number of features can increase runtime and lead to a deterioration in generalization performance due to the curse of dimensionality. Hence, the focus should be on pinpointing the most impactful features to ensure a good balance between the accuracy of the predictions and time complexity of the models.

The overarching goal revolves around choosing the best feature selection method to aid in constructing a stock portfolio that outperforms the SP500 index. To meet this objective, a versatile pipeline will be developed using Python, encompassing data ingestion, feature selection, and prediction using the XGboost to align with existing infrastructure. Various feature selection methods are to be tested and their performance benchmarked. This pipeline should offer adaptability in selecting a variable number of features, whether fixed or optimally determined. Furthermore, the inclusion of visualization and summarization components enhances the interpretability and practical utility of the entire stock market prediction process.

---

## Traditional Feature Selection Methods

Feature selection is the process of choosing a subset of relevant features from a larger feature set within a dataset. The motivation behind this practice includes simplifying models for interpretability, reducing algorithm run-time, eliminating irrelevant features, mitigating the curse of dimensionality, and enhancing model generalization abilities. There are two main types of feature selection methods: unsupervised and supervised. Unsupervised methods, such as correlation-based approaches, do not rely on the target variable and focus on removing redundant variables. On the other hand, the supervised methods, which are the focus of this project, involve the use of the target variable for evaluation, this category includes:

1. Filter Methods: These assess features based on intrinsic properties and statistical measures, independent of the specific machine learning algorithm chosen. The goal is to select the most relevant features before applying any learning algorithm.
2. Wrapper Methods: These methods involve an iterative process where different subsets of features are evaluated in terms of their impact on model performance. This evaluation is done by training and testing the model with each subset, aiming to find the most informative feature set.
3. Intrinsic (Embedded) Methods: These are algorithms that automatically perform feature selection during the training process. The selection is integrated into the learning algorithm itself, eliminating the need for a separate feature selection step.

Each type of feature selection method has its strengths and weaknesses, and the choice of method depends on factors such as the dataset characteristics, the goals of the analysis, and computational considerations. Some examples of traditional feature selection methods that were explored and implemented into the pipeline are:

* Permutation Importance: stands out for its speed and model-agnostic nature, making it suitable for baseline comparison, though it may struggle with intricate feature interactions.
* SHAP Values: while comprehensive, are computationally intensive, making them valuable for detailed baseline assessments on smaller datasets.
* Boruta: commonly applied in financial data analysis, is well-known and benefits from numerous benchmark results, enhancing its suitability for baseline comparisons.
* XGBoost Importance: offers a built-in, fast solution, but its limitation lies in potential oversight of intricate feature relationships.
* Feature Selection with Annealing (FSA): emerges as a highly efficient and model-agnostic method, particularly adept at handling nonlinearity and providing guarantees for feature recovery and convergence.
* FeatBoost: while model-agnostic, prioritizes accuracy improvement by being stringent in sample weight assignment.
* LASSO Regression Feature Importance: introduces a penalty factor to prevent overfitting and implicitly performs feature selection but struggles with correlated features and biases coefficients.
* CART Feature Importance: being less sensitive to outliers and requiring minimal supervision, effectively models feature interactions but necessitates pruning to mitigate overfitting and exhibits sensitivity to certain data characteristics.

---
## Literature Review on State-of-the-Art Feature Selection Methods

The literature review on state-of-the-art feature selection methods aimed to establish a comprehensive understanding of feature selection and subsequently identify recent, high performing approaches compatible with the XGBoost architecture. Over 70 research papers from reputable sources such as NIPS, ICML, JML, ScienceDirect, Arxiv, and Springer were examined. From this extensive review, methods feasible for integration with XGBoost were selected. A few noteworthy approaches with expected state-of-the-art performance that were implemented are summarized below.

* Shapley Additive Global Importance (SAGE): SAGE focuses on quantifying the global importance of features in machine learning models and achieves state-of-the-art performance (Figure 11). Motivated by a recent emphasis on local interpretability, SAGE introduces model-based and universal predictive power notions. This method addresses challenges posed by feature interactions, aiming to develop a model-agnostic approach that efficiently accounts for future interactions. SAGE employs Shapley values and an efficient sampling-based algorithm to approximate feature importance. The advantages of SAGE include its ability to identify important and corrupted features swiftly (Figure 12), outperforming the widely used SHAP method, and exhibiting top-tier performance across various datasets.

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig11.png" width="800" />
</p>

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig12.png" width="800" />
</p>


* Dynamic Feature Selection (DFS): DFS is an algorithm designed to select features with minimal budget while maximizing predictive accuracy. It performs approximation of the greedy policy trained using amortized optimization and focuses on the variational perspective of the Conditional Mutual Information (CMI) for the greedy policy, which is leveraged to train a network that directly predicts the optimal selection given the current features. It incorporates "reward shaping" in the training process, which outperforms static feature selection methods, achieving state-of-the-art performance (Figure 14). Its effectiveness lies in the strategic selection of features through repeated calls to the policy network and predictions made after each selection using the predictor network (Figure 13).

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig13.png" width="800" />
</p>

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig14.png" width="800" />
</p>


* Stochastic Gates Feature Selection (STG): STG introduces an embedded feature selection approach tailored for nonlinear models like neural networks. It achieves high sparsity without compromising performance by leveraging the probabilistic relaxation of the ℓ₀ norm of features. Stochastic gates, drawn from the STG approximation of the Bernoulli distribution, are obtained through the application of the hard-sigmoid function to a mean-shifted Gaussian random variable (Figure 15). The resulting stochastic gate, attached to the input feature, is controlled by the trainable parameter μ𝑑, providing a nuanced solution to feature selection challenges in complex, nonlinear models. As a result, STG is an embedded feature selection method that achieves state-of-art performance (Figure 16).

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig15.png" width="800" />
</p>

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig16.png" width="800" />
</p>

---

## Data Gathering and Processing

Established datasets from the University of California's [archive](https://archive.ics.uci.edu/datasets) were utilized to implement and evaluate various feature selection methods. This approach was adopted as the final dataset for the project was still under development. The chosen datasets covered diverse topics for classification and regression tasks, the datasets were: bike sharing, wine quality, census, spam, breast cancer, Polish companies bankruptcy, chess, and cars. This strategy allowed for the assessment of feature selection methods in terms of their ability to generalize across different conditions, including varying data volumes and feature set complexities.

---

## Custom Dataset

The University of California datasets provided a valuable benchmark for assessing various feature selection methods. However, limitations arose when attempting to find a suitable time series dataset from financial sources for testing existing methods. Given the scarcity of freely available financial time series datasets meeting project requirements, a pragmatic decision was made to create a mock time series dataset. This mock dataset aimed to replicate an experiment closely aligned with the anticipated actual dataset.

Leveraging financial data from the S&P 500 stock index, the mock dataset was enriched with additional features from diverse sources. The target variable is binary, either up or down (1 or 0), based on the 20-days forward returns of the S&P500 index. Numerical features included the Dollar Index, MSCI Emerging Market Index, VIX, Gold Price, Crude Oil WTI Spot Price, Treasury 2Y Yield, Treasury 10Y Yield, S&P 500 Daily Volume, 20-days lagged moving average returns, and 20 randomly generated noise features. The additional categorical features encompassed daily changes in currencies like SGDTWD Daily Chg, JPYCAD Daily Chg, NZDAUD Daily Chg, EURCNY Daily Chg, Nordpool Power Price Daily Chg, and STXE 600 Daily Chg. Notably, the 20 randomly generated noise features played an important role. They served to identify the feature selection methods that were adept at filtering out irrelevant features and provided a quantitative basis for comparing the performance of these methods. This approach addresses the challenge of limited availability of suitable financial time series datasets, enabling the project to proceed with more robust experiments and refinement of the feature selection methodologies using a dataset that is meant to resemble the actual dataset more closely, until it becomes available.

---

## Rebalancing

The issue of class imbalance was observed in some of the alternative datasets, where certain datasets exhibited uneven class distribution. Class imbalance occurs when one class (minority class) has significantly fewer instances than another (majority class), impacting machine learning model performance. Imbalances lead to biases favoring the majority class, poor generalization to the minority class, misleading evaluation metrics, and biased feature importance estimates.

To address these challenges, resampling techniques were employed for skewed datasets. These techniques aim to mitigate bias, improve generalization, provide more informative evaluation metrics, and alleviate biases in feature importance estimates caused by imbalanced datasets.

1. Random Over Sampling: Involves randomly duplicating samples from the minority class to balance class representation.
2. SMOTE (Synthetic Minority Over-sampling Technique): Selects minority examples close in feature space, then generates new samples along the line connecting them.
3. SMOTEN: An extension of SMOTE for categorical data, generating new samples with feature values corresponding to the most common category among neighbors.
4. BorderlineSMOTE: Generates synthetic samples for minority class instances near the borderline between minority and majority classes.

---

## Transformation

To prepare data for the feature selection methods and model prediction, a data transformation pipeline was established to standardize the features. Depending on the dataset, three potential transformation methods could be applied:

1. One-Hot Encode: Represents each category as a binary vector, ensuring appropriate numerical representation for models without introducing ordinal relationships.
2. Label Encode: Assigns a unique integer to each category, providing a numerical representation suitable for ordinal data where the order matters.
3. Normalization: Utilizes Z-score normalization, scaling features to have a mean of 0 and a standard deviation of 1. This ensures consistent scales among numerical features in the dataset.

To ensure that all features in the dataset are consistently transformed for the train, validation, and test data, the sklearn data preprocessing objects, OneHotEncoder, StandardScaler, and LabelEncoder were defined for the training set, and applied on all three sets separately. This ensured that the sklearn transformation objects are consistent and that data leakage would not occur. However, it’s important to note that any new categorical feature values found in the validation or test set that were not observed in the train set, will default to zero by setting the parameter, handle_unknown='ignore', for the one hot encoder, a necessary measure to prevent an error from getting raised.

---

## Model Development

### Pipeline Configuration

For project configuration, to enhance pipeline flexibility for various model setups, the pipeline was configured to accommodate both classification and regression tasks. Additionally, the use of modularization and object-oriented programming in the source code aimed to create a scalable and easy to maintain pipeline. To complement that, the Cookiecutter project template, a commonly used structure for machine learning, was used as it has all the necessary folder and subfolder directories and has strong support in Python. As a result of these configurations, most adjustments need only be done at initialization, such as specifying the feature selection methods, task type, number of back testing windows or train-validation splits, number of top features to select, preferred rebalancing methods, and other relevant variables, contributing to adaptability and simplicity in project management.

### Time Series

The computational resources available are a local machine with only 8 GB RAM and 12.7 GB via Colab. Considering memory and runtime constraints, the original dataset has been scaled down to a single stock (Adobe Inc.). This filtering process results in about 4100 instances. To understand why downsizing is a necessary requirement with the hardware available; with 5 feature selection methods to be tested, each method will undergo 5 back testing windows, each back testing window contains 5 train-validation splits, and each of these combinations will be subjected to 2 hyperparameters combinations for the 2 categories on the number of features (top 50 and all). The purpose of having many back testing windows and train-validation splits is to establish a model with better generalization for such high frequency and volatility data. In total, there are 5x5x5x2x2 = 500 combinations or models to train based on this one configuration. As the number of hyperparameters scales up, the cost of execution time will increase substantially.

To address the temporal nature of the dataset, the date field has been transformed into additional features, including dayofweek, dayofmonth, dayofyear, weekofyear, month, quarter, and year. Additionally, financial indicators such as moving average and exponential moving average features of various time periods are leveraged to capture historical trends in the data. Given that the ticker field contains string values, it will be subjected to one-hot encoding. Consequently, the final input dataset encompasses a total of 407 features, spanning the date range from 2006 to 2022. Normalization will be applied to all numerical features except for the newly introduced date fields, which will remain in their original form.

This preprocessing strategy not only optimizes the dataset to fit within resource constraints but also incorporates essential temporal features and prepares categorical data for effective use in machine learning models. The transformation ensures that the resulting input is well-structured and relevant for subsequent analysis and model training.

### Evaluation

The choice of metrics is contingent upon the nature of the problem, whether it's a classification or regression task. For classification tasks, common metrics include accuracy, precision, recall, F1 score, and AUC. On the other hand, regression tasks, like the one at hand, often utilize metrics such as mean squared error (MSE), mean absolute error (MAE), root mean squared error (RMSE), mean percentage error (MPE), and R-square (R^2). The selection of a specific metric is guided by the unique characteristics of the data and the overarching goals of the analysis.

### Metrics

Given the context of predicting a continuous target variable, specifically the 84-day forward returns in this regression problem, the Root Mean Squared Error (RMSE) was chosen as the primary evaluation metric. RMSE stands out due to its widespread applicability, interpretability (as it shares the same unit as the target variable), and suitability for scenarios where larger errors should be penalized more significantly than smaller errors. This choice is particularly pertinent in the financial domain, where the implications of prediction inaccuracies can have a substantial impact on decision-making and strategic planning. The emphasis on RMSE underscores a commitment to robustly assessing model performance and ensuring that predictions align closely with the real-world implications of financial data.

### Back testing and Hyperparameter Tuning

In contrast to typical cross-validation procedures for non-temporal datasets, time series’ equivalent cross-validation is called back testing, which necessitates preserving the chronological order of the data. Two methods for splitting time series data were considered:
* 1. Back testing with a sliding window: This method involves iteratively moving a window through the time series, with each pass updating the training set and predicting on subsequent data points, as shown in Figure 2.
    * a. Training window size: the number of data points included in a training pass.
    * b. Test window size: the number of data points to include for prediction.
    * c. Sliding steps: the number of data points skipped from one pass to another.
* 2. Back testing with an expanding window: This method requires four parameters: starting window size, ending window size, test window size, and expanding steps.
    * a. Starting window size: the number of data points included in the first training pass.
    * b. Ending window size: the number of data points included in the last training pass.
    * c. Test window size: number of data points to include for prediction.
    * d. Expanding steps: the number of data points added to the training time series from one pass to another.

However, only the sliding window approach was tested due to the additional computational cost associated with having larger training sets for the expanding window approach. Moreover, the sliding window method is better aligned with business objectives, given the high-frequency characteristics of the dataset. This decision reflects a choice made in consideration of both computational efficiency and relevance to the project's overarching goals.

In an ideal scenario, the size parameters for the back testing windows should be fine-tuned. However, due to computational and memory constraints, only a set of fixed values could be tested. The number of rolling windows was limited to 5, and the size of the test set (holdout) was configured to be the same as that of the validation set. The validation set exists within each rolling window preceded by the training set. An illustrative example of a 5-fold train-validation split is shown in Figure 3.

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig3.png" width="800" />
</p>


Due to the complicated nature of combining feature selection with validation splits and hyperparameter tuning, the set of features selected within each train-validation split needs to be handled appropriately for generalization. Since we have 5-fold validation, each split would have a different set of features and feature scores, thus the features need to be aggregated and the feature scores averaged. The resulting features with their average scores are then sorted, and the top predefined number of features are returned, representing the feature set for all 5-fold validation splits, an example on the computation is shown in Figure 4. This procedure is iterated for all feature selection methods, rolling windows, hyperparameter combinations, and number of features. The pseudocode for the process is shown in Figure 5.

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig4.png" width="800" />
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/fig5.png" width="800" />
</p>

With so many combinations, to overcome the computation constraints, only two test cases for the number of features will be used, top 50 features and the full feature set, rather than comparing the performance from 1 feature to all n features. The full feature set provides the baseline performance, while the subset of features is used to gauge the effectiveness of the feature selection methods. The objective is to have fewer features but capture the variances in the data for generalization capabilities, thus having fewer features yet achieves performance as good as having all features is a good measure, while having a better performance than with all features is an indication of excellent performance. Other benefits of models with fewer features are that they are cheaper to train and easier to interpret.

For all those combinations previously described, the hyperparameter combinations that give the best averaged RMSE are retained as the best model. The best hyperparameters are then used for retraining, and predictions are made on the hold-out test data. The RMSE scores are then averaged across all rolling windows for all feature selection methods to determine the method that has the best test score, Figure 5 and 6.


## Pseudocode

A high-level view for the pipeline processes.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/pseudocode.png" width="1200" />
</p>

---

## Rolling Window Back Testing

A view of a single back testing window broken out by its train-validation splits. 

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/rolling_bt.png" width="1200" />
</p>

---

## Feature Score Aggregation

How feature scores are generalized (feature scores averaged) across multiple train-validation splits.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/feature_selection_example.png" width="1200" />
</p>

---

## Train-Validation Splits (All Features)

A view of a single back testing window with the train-validation splits and scoring metrics for different combinations of hyperparameters. The same concept applies for the top 50 subset features.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/bt_all_features.png" width="1200" />
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

Alternatively, using the 11 stocks files, which can be downloaded via the URL link below (request for access is required):

https://drive.google.com/drive/folders/1yN-JTu9pvL8Tm2xTSiK5L8F_L1tIWxoQ?usp=sharing

After completing the downloads, add the files to the directory: ./feature_selection_timeseries/data/raw/sp500_subset/

```python
year = "2006"
date_threshold = year + '-01-01'  # filter for data with date >=
sub_folder = "sp500_subset"

# path of the file containing the features
x_filename_in = 'stock_x.csv'
x_filename_out = f'stock_x_sample_regression_{year}_filtered_11_stocks.csv'
x_in_path = f'./feature_selection_timeseries/data/raw/{sub_folder}/'
x_out_path = f'./feature_selection_timeseries/data/raw/{sub_folder}/'

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

As an illustration, to keep execution time manageable a single stock will be used in this example. Additionally, two sets of hyperparameters and five feature selection methods (XGBoost, Permutation, SHAP, Cart, and Lasso) are considered. Run the codes below to filter for the ticker ADBE (Adobe Inc).  

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
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/adbe_table_y.png" width="300" />
</p>


```python
display(sp500_df.head())
print(f"Dataset Shape: {np.shape(sp500_df)}")
```
<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/adbe_table_x.png" width="1200" />
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
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/train_val_split.png" width="300" />
</p>


Initialize the other arguments.

```python
r1 = run(
    cross_validation_type= "moving window", # or "expanding window"
    save_output_file = True, # Whether to save test outputs
    raw_df = sp500_df.iloc[-keep_data_index:, :].reset_index(drop=True), # Discard extra data instances from the beginning of the time series rather than the end
    y = y.iloc[-keep_data_index:, :].reset_index(drop=True), # Discard extra data instances from the beginning of the time series rather than the end
    train_test_list = train_test_list, # A list of possible list of train and validation size, and number of splits
    methods = ["xgboost", "permutation", "shap", "lasso", "cart"], # Available methods: ["xgboost", "cae", "permutation", "shap", "boruta", "sage", "lasso", "cart", "svm", "rf", "stg", "dynamic"]
    rebalance_type = ["None"], # ["borderlinesmote", "smoten", "random_over_sampler", "smote", "None"]
    label_cols = [], # Columns to label encode
    do_not_encode_cols = ["dayofmonth", "dayofweek", "quarter", "month", "year", "dayofyear", "weekofyear"], # These fields are not transformed
    seed = 42,
    target_colname = "target", # The name of the field that holds the true values
    dataset_name = "sp500",
    pred_type = "regression",
    append_to_full_df = False,
    n_features = 50,  # The number of top features to filter for
    feature_direction = "top", # Feature order based on their scores in descending order
    train_outputs_file_name = None,
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    scaler_filename = "./feature_selection_timeseries/data/processed/scaler_saved.save",
    encoder_filename = "./feature_selection_timeseries/data/processed/encoder_saved.save",
    label_encoder_filename = "./feature_selection_timeseries/data/processed/lalbel_encoder_saved.save",
    test_output_file_name = f"./feature_selection_timeseries/data/experiment/Consolidated_Stocks_FS_timeseries_sp500_outputs_test_results_",
    test_pred_file_name = f"./feature_selection_timeseries/data/experiment/Consolidated_Stocks_FS_timeseries_sp500_outputs_test_preds_",
    print_outputs_train = True,
    print_outputs_test = True
)
```

Train the model. Average the feature scores and validation scores, and determine the optimal hyperparameters. 

The back testing rolling windows are: 

```python
"""
Train: [[0, 3300], [150, 3450], [300, 3600], [450, 3750], [600, 3900]]
Test:  [[3300, 3450], [3450, 3600], [3600, 3750], [3750, 3900], [3900, 4050]]
"""
```

Within each rolling window, there are 5-fold train-validation splits. 

```python
"""
1. Train:      [[0, 630], [630, 1260], [1260, 1890], [1890, 2520], [2520, 3150]]
   Validation: [[630, 780], [1260, 1410], [1890, 2040], [2520, 2670], [3150, 3300]]

2. Train:      [[150, 780], [780, 1410], [1410, 2040], [2040, 2670], [2670, 3300]]
   Validation: [[780, 930], [1410, 1560], [2040, 2190], [2670, 2820], [3300, 3450]]

3. Train:      [[300, 930], [930, 1560], [1560, 2190], [2190, 2820], [2820, 3450]]
   Validation: [[930, 1080], [1560, 1710], [2190, 2340], [2820, 2970], [3450, 3600]]

4. Train:      [[450, 1080], [1080, 1710], [1710, 2340], [2340, 2970], [2970, 3600]]
   Validation: [[1080, 1230], [1710, 1860], [2340, 2490], [2970, 3120], [3600, 3750]]

5. Train:      [[600, 1230], [1230, 1860], [1860, 2490], [2490, 3120], [3120, 3750]]
   Validation: [[1230, 1380], [1860, 2010], [2490, 2640], [3120, 3270], [3750, 3900]]
"""
```

Train models and obtain the optimal hyperparamters.

```python
r1.train()
```

Set the optimal hyperparameters and apply the optimal models on the hold out test data.

```python
r1.test()
```

**Test Results:**

CART 50 Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/cart_test_50_bt.png" width="1200" />
</p>

CART All Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/cart_test_all_bt.png" width="1200" />
</p>

LASSO 50 Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/lasso_test_50_bt.png" width="1200" />
</p>

LASSO All Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/lasso_test_all_bt.png" width="1200" />
</p>

Permutation 50 Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/permu_test_50_bt.png" width="1200" />
</p>

Permutation All Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/permu_test_all_bt.png" width="1200" />
</p>

SHAP 50 Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/shap_test_50_bt_v2.png" width="1200" />
</p>

SHAP All Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/shap_test_all_bt_v2.png" width="1200" />
</p>

XGBoost 50 Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/xgb_test_50_bt.png" width="1200" />
</p>

XGBoost All Features

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/xgb_test_all_bt.png" width="1200" />
</p>

Average Test Scores (RMSE) of All Back Testing Windows For Each Method and Number of Features:

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Monetary%20Authority%20of%20Singapore%20Project/feature_selection_timeseries/docs/images/avg_test_rmse_all_bt_windows_v2.png" width="600" />
</p>

From the average test scores (RMSE) of all back testing windows for each method, it's evident that the SHAP feature selection method generated better subset of features than the other methods in this experiment.

---

## Sources

1. *SAGE:* Ian Covert, Scott Lundberg, Su-In Lee. "Understanding Global Feature Contributions With Additive Importance Measures." NeurIPS 2020 <https://github.com/iancovert/sage>

2. *Dynamic:* Ian Covert, Wei Qiu, Mingyu Lu, Nayoon Kim, Nathan White, Su-In Lee. "Learning to Maximize Mutual Information for Dynamic Feature Selection." ICML, 2023. <https://github.com/iancovert/dynamic-selection>

3. *CAE:* Abid, Abubakar, Muhammad Fatih Balin, and James Zou. "Concrete autoencoders for differentiable feature selection and reconstruction." arXiv preprint arXiv:1901.09346 (2019). <https://github.com/mfbalin/Concrete-Autoencoders/tree/master>

4. *STG:* Yamada, Yutaro, et al. "Feature selection using stochastic gates." International Conference on Machine Learning. PMLR, 2020 <https://github.com/runopti/stg>

5. *XGBoost:* Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016. <https://xgboost.readthedocs.io/en/stable/index.html>

6. *SHAP:* Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems 30 (2017). <https://github.com/shap/shap>

7. *Boruta:* Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. Journal of Statistical Software, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11 <https://github.com/scikit-learn-contrib/boruta_py>

8. *Permutation:* <https://scikit-learn.org/stable/modules/permutation_importance.html>

9. *LASSO:* <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>

10. *CART:* <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>

11. *SVM:* <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>

12. *RF:* <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
