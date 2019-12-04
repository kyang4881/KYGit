import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
import random as rd
import pandas as pd
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

class Neural_Network:

    def __init__(self,
             num_instances,
             num_features,
             num_hidden_nodes,
             num_output_nodes,
             learning_rate,
             data_set,
             train_percentage,
             train_iterations):
        self.num_instances = num_instances
        self.num_features = num_features
        self.num_output_nodes = num_output_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.learning_rate = learning_rate
        self.data_set = data_set
        self.train_percentage = train_percentage
        self.train_iterations = train_iterations
        self.train_data_filter()
        self.rand_weights()
        self.test_data_filter()

    def train_data_filter(self):
        #Training set
        self.input_df_train = self.data_set[:int(np.round(len(self.data_set) * (self.train_percentage / 100))), :4].astype('float')
        self.outcome_df_train = np.array(pd.get_dummies(self.data_set[:int(np.round(len(self.data_set) * (self.train_percentage / 100))), 4]))
        return self.input_df_train, self.outcome_df_train

    def test_data_filter(self):
        #Testing set
        #self.input_df_test = self.data_set[int(np.round(len(self.data_set) * (self.train_percentage / 100))):len(self.data_set), :(self.data_set.shape[1] - 1)].astype('float')  #(Train on subset, Test all full set)
        #self.outcome_df_test = np.array(pd.get_dummies(self.data_set[int(np.round(len(self.data_set) * (self.train_percentage / 100))):len(self.data_set), (self.data_set.shape[1] - 1)]))  #(Train on subset, Test all full set)

        self.input_df_test = self.data_set[int(np.round(len(self.data_set) * (self.train_percentage / 100))):len(self.data_set), : (self.data_set.shape[1] - 1)].astype('float')
        self.outcome_df_test = np.array(pd.get_dummies(self.data_set[int(np.round(len(self.data_set) * (self.train_percentage / 100))):len(self.data_set), (self.data_set.shape[1] - 1)]))
        return self.input_df_test, self.outcome_df_test

    def rand_weights(self):
        np.random.seed(123)
        self.weights_hidden = np.random.rand(self.num_features, self.num_hidden_nodes)
        self.bias_hidden = np.random.randn(self.num_hidden_nodes)
        self.weights_output = np.random.rand(self.num_hidden_nodes, self.num_output_nodes)
        self.bias_output = np.random.randn(self.num_output_nodes)

    def train(self):
        for iterations in range(self.train_iterations):
        #Forward Propagration
            #First Layer
            xh = np.dot(self.input_df_train, self.weights_hidden) + self.bias_hidden
            yh = sigmoid(xh)                                                            #First Layer Outputs

            #Second Layer
            xo = np.dot(yh, self.weights_output) + self.bias_output
            yo = softmax(xo)                                                            #Second Layer Outputs

        #Back Propagation
            #Second Layer
            dcost_dxo = yo - self.outcome_df_train    #Error =  Predicted - True labels
            dxo_dweights_output = yh
            dcost_weights_output = np.dot(dxo_dweights_output.T, dcost_dxo)
            dcost_bias_output = dcost_dxo

            #First Layer
            dxo_dyh = self.weights_output
            dcost_dyh = np.dot(dcost_dxo, dxo_dyh.T)
            dyh_dxh = sigmoid_derivative(xh)
            dxh_dweights_hidden = self.input_df_train
            dcost_weights_hidden = np.dot(dxh_dweights_hidden.T, dyh_dxh * dcost_dyh)
            dcost_bias_hidden = dcost_dyh * dyh_dxh

            #Updating Weights
            self.weights_hidden -= self.learning_rate * dcost_weights_hidden
            self.bias_hidden -= self.learning_rate * dcost_bias_hidden.sum(axis=0)
            self.weights_output -= self.learning_rate * dcost_weights_output
            self.bias_output -= self.learning_rate * dcost_bias_output.sum(axis=0)

    def run(self, input_vector):
        #First Layer
        xh_inputs = np.dot(input_vector, self.weights_hidden) + self.bias_hidden
        yh_outputs = sigmoid(xh_inputs)

        #Second Layer
        xo_inputs = np.dot(yh_outputs, self.weights_output) + self.bias_output
        yo_outputs = softmax(xo_inputs)

        return yo_outputs


###################################### Example 1: Python Built-in Dataset (Iris) #################################

iris = datasets.load_iris()
iris_features_df = iris.data
iris_feature_names_df = iris.feature_names
iris_target_df = iris.target
iris_names = iris.target_names

#Add target labels to features array
iris_df = np.hstack((iris_features_df, np.array([iris_target_df]).T))

np.random.seed(123)
#Randomize dataset
iris_df = shuffle(iris_df)


#Determine Optimal Learning Rate
for lr in range(1, 6, 1):
    accuracy_df = []
    #Determine Optimal Training Proportion
    for train_per in range(10, 100, 10):
        #Run Neural Networks Model
        simple_network = Neural_Network(num_instances = iris_df.shape[0],
                                        num_features = iris_df.shape[1] - 1,
                                        num_hidden_nodes = 4,  #iris_df.shape[1] - 1,
                                        num_output_nodes = len(np.unique(iris_df[:,4])),   #len(np.unique(iris_df[:int(np.round(len(iris_df) * (train_per / 100))), (iris_df.shape[1] - 1)])),
                                        learning_rate = (10**-lr),
                                        data_set = iris_df,
                                        train_percentage = train_per,
										train_iterations = 10000)

        #Training the model
        simple_network.train()
        #Testing data with true labels
        test_df_true_label = simple_network.test_data_filter()[1]
        #Run trained model on testing data to get predicted labels
        test_df_pred_label_rounded = np.round(simple_network.run(simple_network.test_data_filter()[0]))

        #Compare predicted labels against true labels
        compare_pred_actuals = test_df_true_label == test_df_pred_label_rounded

        #Count number of correct predictions
        correct_cnt = 0
        for i in range(len(compare_pred_actuals)):
            if all(compare_pred_actuals[i,:] == True):
                correct_cnt += 1

        #Print Accuracy of Model:
        print("The accuracy is: " + str(round((correct_cnt/len(test_df_pred_label_rounded))*100, 1)) + "%. " + str(correct_cnt) + "/" + str(len(test_df_pred_label_rounded)) + " testing set records are predicted correctly. " +
              "The learning rate is: " + str(10**-lr) + ". " +
              "The training/testing proportions: " + str(train_per) + "/" + str(100 - train_per) + "%.")

        #percentage of correct predictions
        accuracy_df.append(round((correct_cnt/len(test_df_pred_label_rounded))*100,1))

    plt.plot(accuracy_df, label = (10**-lr))
    plt.ylabel('Accuracy Percentage (%)')
    plt.xlabel('Training Set Percentage (x10%)')
    plt.legend()

