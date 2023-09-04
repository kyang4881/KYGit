import pandas as pd
import sage
from scipy.io import arff
import numpy as np

def importer(file_path, dataset, seed, sample=False, sample_size=100):
    """ A method for importing datasets from multiple sources
    Args:
        file_path (str): the path of the dataset
        dataset (dict): the name of the dataset to import
        sample (bool): whether to take a sample from the dataset
        sample_size (int): the size of the sample to take
    Returns:   
        df (dataframe): a dataframe containing data
    """
    if dataset.lower() == "credit":
        df_path = file_path + '/credit.data'
        df = pd.read_csv(df_path, delimiter=' ', header=None)
        df.columns = df.columns.astype(str)
        df = df.rename(columns={df.columns[-1]: 'target'})
        df['target'] = df['target'].replace({1: 0, 2: 1})
        
    if dataset.lower() == "madelon":
        madelon_train = file_path + '/madelon_train.data'
        madelon_valid = file_path + '/madelon_valid.data'
        madelon_train_labels = file_path + '/madelon_train.labels'
        madelon_valid_labels = file_path + '/madelon_valid.labels'

        madelon_train_df = pd.read_csv(madelon_train, delimiter=' ', header=None)
        madelon_valid_df = pd.read_csv(madelon_valid, delimiter=' ', header=None)

        madelon_train_labels_df = pd.read_csv(madelon_train_labels, delimiter=' ', header=None)
        madelon_valid_labels_df = pd.read_csv(madelon_valid_labels, delimiter=' ', header=None)

        madelon_train_df['target'] = madelon_train_labels_df
        madelon_valid_df['target'] = madelon_valid_labels_df

        df = pd.concat([madelon_train_df, madelon_valid_df])
        df.columns = df.columns.astype(str)
        df = df.fillna(0)
        df['target'] = df['target'].replace({-1: 0, 1: 1})
        
    if dataset.lower() == "bike":
        # Load data
        df = sage.datasets.bike()
        df = pd.concat([df.iloc[:, :-3], df.iloc[:, -1]], axis=1)
        feature_names = df.columns.tolist()[:-1]
        df.rename(columns={"Count": "target"}, inplace=True)
        df.head()
        
    if dataset.lower() == "wine":
        redwine_ = file_path + '/winequality-red.csv'
        redwine_df = pd.read_csv(redwine_, delimiter=';')

        whitewine_ = file_path + '/winequality-white.csv'
        whitewine_df = pd.read_csv(whitewine_, delimiter=';')

        redwine_df['target'] = 0
        whitewine_df['target'] = 1

        df = pd.concat([redwine_df, whitewine_df])
        
    if dataset.lower() == "census":
        adult_ = file_path + '/adult.data'
        df = pd.read_csv(adult_, delimiter=',', header=None)
        df.rename(columns={14: 'target'}, inplace=True)
        df['target'] = df['target'].map({' <=50K': 0, ' >50K': 1})
        df.columns = df.columns.astype(str)
        
    if dataset.lower() == "spam":
        spam_ = file_path + '/spambase.data'
        df = pd.read_csv(spam_, delimiter=',', header=None)
        df.rename(columns={57: 'target'}, inplace=True)
        df.columns = df.columns.astype(str)
        
    if dataset.lower() == "breast_cancer":
        breast_cancer_ = file_path + '/wdbc.data'
        df = pd.read_csv(breast_cancer_, delimiter=',', header=None)
        df.rename(columns={1: 'target'}, inplace=True)
        df.columns = df.columns.astype(str)

        breast_cancer_df_target = df.iloc[:, 1]
        breast_cancer_df_feature = df.iloc[:, 2:]

        df = pd.concat([breast_cancer_df_feature, breast_cancer_df_target], axis=1)
        df['target'] = df['target'].map({'B': 0, 'M': 1})

    if dataset.lower() == "polish":
        file_path = file_path + '/1year.arff'
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        df.rename(columns={"class": 'target'}, inplace=True)
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({b'0': 0, b'1': 1})
        
    if dataset.lower() == "chess":
        chess_ = file_path + '/chess.data'
        df = pd.read_csv(chess_, delimiter=',', header=None)
        df.rename(columns={36: 'target'}, inplace=True)
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({"nowin": 0, "won": 1})

    if dataset.lower() == "car":
        car_ = file_path + '/car.data'
        df = pd.read_csv(car_, delimiter=',', header=None)
        df.rename(columns={6: 'target'}, inplace=True)
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({"unacc": 0, "acc": 0, "good": 1, "vgood": 1})
        
    if dataset.lower() == "ibeacon":
        ibeacon_ = file_path + '/iBeacon_RSSI_Labeled.csv'
        df_x = pd.read_csv(ibeacon_, delimiter=',')
        df_x.columns = df_x.columns.astype(str)
        df_x.rename(columns={'location': 'target'}, inplace=True)
        df = pd.concat([df_x.iloc[:, 1:], df_x.iloc[:, 0]], axis=1)
        df = df[df['target'].isin(["K04", "J04"])]
        df['target'] = df['target'].map({"K04": 0, "J04": 1})
        df = df.iloc[:, 1:]
        
    if dataset.lower() == "ai4i":
        ai4i = file_path + '/ai4i.csv'
        df = pd.read_csv(ai4i, delimiter=',')
        df.columns = df.columns.astype(str)
        df = df[df['target'].isin(["L", "H"])]
        df['target'] = df['target'].map({"L": 0, "H": 1})
        df = df.iloc[:, 3:]
        
    if dataset.lower() == "eye":
        file_path = file_path + '/EEG Eye State.arff'
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        df.rename(columns={"eyeDetection": 'target'}, inplace=True)
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({b'0': 0, b'1': 1})
        #df = df.iloc[:5000, :]
        
    if dataset.lower() == "strawberry":
        file_path = file_path + '/Strawberry_TRAIN.arff'
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({b'1': 0, b'2': 1})
        
    if dataset.lower() == "strawberry_test":
        file_path = file_path + '/Strawberry_TEST.arff'
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({b'1': 0, b'2': 1})
        
    if dataset.lower() == "ford":
        file_path = file_path + "/FordA_TRAIN.arff"
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({b'-1': 0, b'1': 1})
        
    if dataset.lower() == "ford_test":
        file_path = file_path + "/FordA_TEST.arff"
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        df.columns = df.columns.astype(str)
        df['target'] = df['target'].map({b'-1': 0, b'1': 1})
        
    if sample:
        np.random.seed(seed)
        df = df.sample(n=sample_size)
        
    print(np.shape(df))
    display(df.head())
        
    return df
