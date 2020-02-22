import distutils
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
import numpy as np
import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.utils import shuffle

# Check current working directory
os.getcwd()

# Add files to current working directory and import to Python
fake_data_file = open("C:\\Users\\kyang\\PycharmProjects\\NeuroNet\\clean_fake.txt", "r")
real_data_file = open("C:\\Users\\kyang\\PycharmProjects\\NeuroNet\\clean_real.txt", "r")

# Print data
print(fake_data_file.read())
print(real_data_file.read())

with open("C:\\Users\\kyang\\PycharmProjects\\NeuroNet\\clean_fake.txt", "r") as f:
    fake_list = f.readlines()  # Import data as list
    fake_list = [items.strip() for items in fake_list]   # Remove '\n'

with open("C:\\Users\\kyang\\PycharmProjects\\NeuroNet\\clean_real.txt", "r") as f:
    real_list = f.readlines()
    real_list = [items.strip() for items in real_list]   # Remove '\n'

print(fake_list)
print(real_list)

# Close files
fake_data_file.close()
real_data_file.close()

# Method to vectorize list of data
fake_vec = CountVectorizer()
real_vec = CountVectorizer()

# Vectorize data
fake_vectorized = fake_vec.fit_transform(fake_list)
real_vectorized = real_vec.fit_transform(real_list)

print(fake_vec.get_feature_names())
print(real_vec.get_feature_names())

print(fake_vectorized.toarray())
print(real_vectorized.toarray())

# The dimension of the array
np.shape(fake_vectorized.toarray())   # (1298, 3689)
np.shape(real_vectorized.toarray())   # (1968, 3587)

fake_vectorized_array = fake_vectorized.toarray()
real_vectorized_array = real_vectorized.toarray()

fake_vectorized_labels_array = np.array([0]*(np.shape(fake_vectorized_array)[0]))
real_vectorized_labels_array = np.array([1]*(np.shape(real_vectorized_array)[0]))

### Combine Fake and Real

len(fake_list) #1298
len(real_list) #1968

df_features = fake_list + real_list
df_labels = [0]*(np.shape(fake_vectorized_array)[0]) + [1]*(np.shape(real_vectorized_array)[0])

# Convert text to vector of frequencies
df_vec = CountVectorizer()
df_features_vectorized = df_vec.fit_transform(df_features)
print(df_vec.get_feature_names())
print(df_features_vectorized.toarray())  # (3266, 5799)

df_features_vectorized_array = df_features_vectorized.toarray()
df_label_array = np.array(df_labels)

# Combine features with labels then shuffle the data
data = np.hstack((df_features_vectorized_array, np.array([df_label_array]).T))
np.random.seed(123)
data_rand = shuffle(data)

###########################################################################################################################

# Split data into training and testing sets
data_rand_train = data_rand[0:round(0.7*np.shape(data_rand)[0]), ] # (2286, 5800)
data_rand_validation = data_rand[round(0.7*np.shape(data_rand)[0]): round(0.85*np.shape(data_rand)[0])] # (490, 5800)
data_rand_test = data_rand[round(0.85*np.shape(data_rand)[0]):] # (490, 5800)

# Separate data into features and labels for both training and testing sets
data_rand_train_features = np.array([i[:-1] for i in data_rand_train])  # (2286, 5799)
data_rand_validation_features = np.array([i[:-1] for i in data_rand_validation])  # (490, 5799)
data_rand_test_features = np.array([i[:-1] for i in data_rand_test])  # (490, 5799)
data_rand_train_labels = np.array([i[-1] for i in data_rand_train])  # (2286,)
data_rand_validation_labels = np.array([i[-1] for i in data_rand_validation])  # (490,)
data_rand_test_labels = np.array([i[-1] for i in data_rand_test])  # (490,)

# Fit model on training set
knn = KNeighborsClassifier()
knn.fit(data_rand_train_features, data_rand_train_labels)

# Check accuracy of default n on validation set
predictions = list()
for i in range(np.shape(data_rand_validation)[0]):
    predictions.append(int(knn.predict([data_rand_validation_features[i]])))
data_rand_validation_predicted = np.hstack((data_rand_validation_features, np.array([predictions]).T))
sum(data_rand_validation_predicted[:, -1] == data_rand_validation_labels) /np.shape(data_rand_validation_labels)[0] # Validation_accuracy 

# Iterate through multiple n values
accuracy = []
k_num = []

for k in range(3, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_rand_train_features, data_rand_train_labels)

    predictions = list()
    for i in range(np.shape(data_rand_validation)[0]):
        predictions.append(int(knn.predict([data_rand_validation_features[i]])))

    data_rand_validation_predicted = np.hstack((data_rand_validation_features, np.array([predictions]).T))

    accuracy.append(sum(data_rand_validation_predicted[:, -1] == data_rand_validation_labels)/np.shape(data_rand_validation_labels)[0]) 
    k_num.append(k)


# Check accuracy of default n on testing set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_rand_train_features, data_rand_train_labels)
predictions = list()
for i in range(np.shape(data_rand_test)[0]):
    predictions.append(int(knn.predict([data_rand_test_features[i]])))
data_rand_test_predicted = np.hstack((data_rand_test_features, np.array([predictions]).T))
sum(data_rand_test_predicted[:, -1] == data_rand_test_labels) /np.shape(data_rand_test_labels)[0] # test_accuracy


# Using grid search to determine optimal k value
# Separate data into features and labels for both training and testing sets
data_rand_features = np.array([i[:-1] for i in data_rand])
data_rand_labels = np.array([i[-1] for i in data_rand])

param_grid = dict(n_neighbors=list(range(3, 10)))  # Iterate for k = [3,10)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')   # 10 folds for each iteration
grid.fit(data_rand_features, data_rand_labels)
grid.cv_results_  # optimal k = 4

###########################################################################################################################

class knn_model:

    def __init__(self, k, train_percentage, validation_percentage, test_percentage, data_set, cv, grid_lower_k, grid_upper_k, scoring):
        self.k = k
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.data_set = data_set
        self.cv = cv
        self.train_data_filter()
        self.validation_data_filter()
        self.test_data_filter()
        self.data_features = np.array([i[:-1] for i in self.data_set])
        self.data_labels = np.array([i[-1] for i in self.data_set])
        self.grid_lower_k = grid_lower_k
        self.grid_upper_k = grid_upper_k
        self.scoring = scoring

    def train_data_filter(self):
        self.df_train = self.data_set[0:round(self.train_percentage * np.shape(self.data_set)[0]), ]
        self.train_features = np.array([i[:-1] for i in self.df_train])
        self.train_labels = np.array([i[-1] for i in self.df_train])
        return self.train_features, self.train_labels

    def validation_data_filter(self):
        self.df_validation = self.data_set[round(self.train_percentage * np.shape(self.data_set)[0]): round(
            (self.train_percentage + self.validation_percentage) * np.shape(self.data_set)[0])]
        self.validation_features = np.array([i[:-1] for i in self.df_validation])
        self.validation_labels = np.array([i[-1] for i in self.df_validation])
        return self.validation_features, self.validation_labels

    def test_data_filter(self):
        self.df_test = self.data_set[round((self.train_percentage + self.validation_percentage) * np.shape(self.data_set)[0]):]
        self.test_features = np.array([i[:-1] for i in self.df_test])
        self.test_labels = np.array([i[-1] for i in self.df_test])
        return self.test_features, self.test_labels

    def train(self):
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.fit = self.knn.fit(self.train_features, self.train_labels)
        return self.knn, self.fit

    def grid_search(self):
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.fit = self.knn.fit(self.train_features, self.train_labels)
        self.param_grid = dict(n_neighbors=list(range(self.grid_lower_k, self.grid_upper_k)))
        self.grid = GridSearchCV(self.knn, self.param_grid, cv=self.cv, scoring=self.scoring)
        self.grid.fit(self.data_features, self.data_labels)
        return self.grid

    def run(self, input_feature_vector, input_label_vector):
        predictions = list()
        for i in range(np.shape(input_feature_vector)[0]):
            predictions.append(int(self.knn.predict([input_feature_vector[i]])))
        data_label_predicted = np.hstack((input_feature_vector, np.array([predictions]).T))
        accuracy = sum(data_label_predicted[:, -1] == input_label_vector) / np.shape(input_label_vector)[0]
        return accuracy, data_label_predicted


run_knn = knn_model(k=4, train_percentage=0.7, validation_percentage=0.15, test_percentage=0.15, data_set=data_rand, cv=10, 
                    grid_lower_k=3, grid_upper_k=10, scoring='accuracy')

run_knn.train()
validation_df_true_label = run_knn.validation_data_filter()[1]
validation_results = run_knn.run(run_knn.validation_data_filter()[0], validation_df_true_label)
validation_accuracy = validation_results[0]
validation_predictions = validation_results[1]

grid_results = run_knn.grid_search()
grid_results.cv_results_   # optimal k = 4

test_df_true_label = run_knn.test_data_filter()[1]
test_results = run_knn.run(run_knn.test_data_filter()[0], test_df_true_label)
test_accuracy = test_results[0]
test_predictions = test_results[1]

