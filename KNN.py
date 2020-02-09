import distutils
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
import random as rd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


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
validation_accuracy = []
predictions = list()
for i in range(np.shape(data_rand_validation)[0]):
    predictions.append(int(knn.predict([data_rand_validation_features[i]])))
data_rand_validation_predicted = np.hstack((data_rand_validation_features, np.array([predictions]).T))
validation_accuracy.append(sum(data_rand_validation_predicted[:, -1] == data_rand_validation_labels) /np.shape(data_rand_validation_labels)[0]) # Validation_accuracy = 67%

# Iterate through multiple n values
accuracy = []
k_num = []

for k in range(3, 25):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_rand_train_features, data_rand_train_labels)

    predictions = list()
    for i in range(np.shape(data_rand_validation)[0]):
        predictions.append(int(knn.predict([data_rand_validation_features[i]])))

    data_rand_validation_predicted = np.hstack((data_rand_validation_features, np.array([predictions]).T))

    accuracy.append(sum(data_rand_validation_predicted[:, -1] == data_rand_validation_labels)/np.shape(data_rand_validation_labels)[0]) # k=3 => 73%
    k_num.append(k)


# Check accuracy of default n on testing set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_rand_test_features, data_rand_test_labels)
test_accuracy = []
predictions = list()
for i in range(np.shape(data_rand_test)[0]):
    predictions.append(int(knn.predict([data_rand_test_features[i]])))
data_rand_test_predicted = np.hstack((data_rand_test_features, np.array([predictions]).T))
test_accuracy.append(sum(data_rand_test_predicted[:, -1] == data_rand_test_labels) /np.shape(data_rand_test_labels)[0]) # test_accuracy = 77%
