class initialize_data:
    
    def __init__(self, file_path, file_tab):
        self.file_path = file_path
        self.file_tab = file_tab
    
    def file_import(self):
        """Import training and testing data"""
        return pd.read_excel(self.file_path, sheet_name = self.file_tab)


class preprocess_data:
    
    def __init__(self, target, features, features_to_process, data, initial_data):
        self.target = target
        self.features = features
        self.features_to_process = features_to_process
        self.data = data
        self.initial_data = initial_data
    
    def label_encode(self):
        """Encode the categorical data for which rank is relevant"""
        encoder = preprocessing.LabelEncoder()
        
        for i in range(np.shape(self.data)[1]):
            
            data_features = list(self.data.columns)
            
            if data_features[i] in self.features_to_process:
                encoder.fit(self.data[data_features[i]])
                print(data_features[i] + " Processed Classes: \n ")
                print(list(encoder.classes_))
                self.initial_data[data_features[i]] = encoder.transform(self.data[data_features[i]])
            else:
                self.initial_data[data_features[i]] = self.data[data_features[i]]
                
        return self.initial_data
    
    def one_hot_encode(self):
        """Encode the categorical data for which rank is irrelevant"""
        df_dummies = pd.get_dummies(self.data) 
        self.initial_data = pd.concat([self.initial_data, df_dummies], axis = 1) 
        
        return self.initial_data
    
    def concat_numerical(self):
        """Concatenate data that doesn't need to be encoded"""
        return pd.concat([self.initial_data, self.data], axis = 1) 
    
    def standardize_data(self):
        """Normalize scaler values"""
        scaler = preprocessing.StandardScaler().fit(self.data)
        scaled_data = pd.DataFrame(scaler.transform(self.data))
        scaled_data.columns = self.data.columns
        
        return scaled_data

    def align_train_test(self, train_data, test_data):
        """Align the features between the training and testing sets"""
        all_columns = list(set(list(train_data.columns) + list(test_data.columns)))

        test_extra_col = list(set(test_data.columns) - set(train_data.columns)) # Extra features from the testing set
        train_extra_col = list(set(train_data.columns) - set(test_data.columns)) # Extra features from the training set

        train_test_dict = {}
        
        if len(test_extra_col) > 0:
            train_add_col = pd.DataFrame(0, index=np.arange(len(train_data)), columns=test_extra_col) # Add extra testing features to training set

            train_test_dict['train'] = pd.concat([train_data, train_add_col], axis = 1)[all_columns] # Return new training set
             
        else:
            train_test_dict['train'] = train_data[all_columns]

        if len(train_extra_col) > 0: # If the training set has extra features
            test_add_col = pd.DataFrame(0, index=np.arange(len(test_data)), columns=train_extra_col) # Add extra training features to testing set

            train_test_dict['test'] = pd.concat([test_data, test_add_col], axis = 1)[all_columns] # Return new testing set  
            
        else:
            train_test_dict['test'] = test_data[all_columns]
            
            
        return train_test_dict
