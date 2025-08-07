

# import pandas as pd
# import numpy as np
import data_loading as data_loading
from preprocess import preprocess_XZ  
from sklearn.model_selection import train_test_split


class DataLoader:
    
    def __init__(self, config, seed):
        """
        Initializes the DataLoader class with the given configuration and seed.
        Parameters:
        config (dict): Configuration dictionary containing parameters for loading and processing the data.
        seed (int): Random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        self.data = None
        self.S_features = None
        self.X_features = None
        self.Z_features = None
        self.Y_target = None
        self.Zstandardized = None
        self.X_transform = None
        self.X_preprocessor = None
        self.datasplit = {}
        
    
    def load_data(self):
        """
        Loads the dataset based on the configuration provided.
        """
        if self.config['data_name'] == 'german':
            self.data = data_loading.load_german_data(self.config['datapath'], log_transform=True, model_interactions=self.config['model_interactions'], plot_cov=self.config['plot_cov'])
        else:
            self.data = data_loading.load_synthetic(self.config['datapath'], model_interactions=self.config['model_interactions'], plot_cov=self.config['plot_cov'])
            
    
    def extract_features(self):
        """
        Extracts sensitive attributes, covariates, treatments, and target from the dataset.
        """
        num_sens = self.config['num_sens']
        
        if self.config['model_interactions']:
            num_sens += 1
            
        num_covariates = self.config['num_covariates']
        
        # print(f"num covariates {num_covariates} and num sens {num_sens}")
      
        num_treatments = self.config['num_treatments']
        covariate_idxs = list(range(num_sens, num_sens + num_covariates))
        treatment_idxs = list(range(num_sens + num_covariates, num_sens + num_covariates + num_treatments))
        self.S_features = self.data.iloc[:, :num_sens]
        
        self.X_features = self.data.iloc[:, covariate_idxs]
        self.Z_features = self.data.iloc[:, treatment_idxs]
        self.Y_target = self.data.iloc[:, -1]
        
    def preprocess_features(self):
        """
        Preprocess covariates and treatments (standardization and transformation).
        """
        self.Zstandardized = preprocess_XZ(treatment_Z=self.Z_features, data_name=self.config['data_name']) #return of a dictionary  of the mean and sigma
        self.X_transform, self.X_preprocessor = preprocess_XZ(covariates_X=self.X_features, data_name=self.config['data_name'])
    
    def split_data(self):
        """
        Splits the data into training and test sets.
        """
        if self.config['data_name'] == 'german':
            X_train, X_test, S_train, S_test, Z_train, Z_test, y_train, y_test = train_test_split(
                self.X_transform, self.S_features, self.Zstandardized['Z_scaled'], self.Y_target,
                test_size=self.config['testsize'], random_state=self.seed)
            
            self.datasplit = {
                'all_X': self.X_transform, 'all_S': self.S_features, 'all_Z': self.Zstandardized['Z_scaled'], 'all_Y': self.Y_target,
                'train_X': X_train, 'test_X': X_test, 'train_S': S_train, 'test_S': S_test,
                'train_Z': Z_train, 'test_Z': Z_test, 'train_Y': y_train, 'test_Y': y_test}
            
        else:
            X_train, X_test, S_train, S_test, Z_train, Z_test, y_train, y_test = train_test_split(self.X_features,
                  self.S_features, self.Z_features, self.Y_target, test_size=self.config['testsize'],
                  random_state=self.seed)
            
            self.datasplit = {
                'all_X': self.X_features, 'all_S': self.S_features, 'all_Z': self.Z_features, 'all_Y': self.Y_target,
                'train_X': X_train, 'test_X': X_test, 'train_S': S_train, 'test_S': S_test,
                'train_Z': Z_train, 'test_Z': Z_test, 'train_Y': y_train, 'test_Y': y_test}
            
        
    def load(self):
        """
        Main method to load, preprocess, and split the dataset.
        """
        self.load_data()
        self.extract_features()
        
        if self.config['data_name'] == 'german':
            self.preprocess_features()
            self.split_data()
            return self.datasplit, self.Zstandardized, self.X_preprocessor
        else:
            self.split_data()
            return self.datasplit














