import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = None
        self.label_encoders = {}
        self.outlier_bounds_ = {}
        
    def fit(self, X, y=None):        
        num_columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                      'Guest_Popularity_percentage', 'Number_of_Ads']
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(X[num_columns])
        
        ordinal_columns = ['Episode_Sentiment', 'Publication_Day', 'Publication_Time', 
                          'Podcast_Name', 'Genre']
        
        for col in ordinal_columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col])
                self.label_encoders[col] = le
                logging.info(f"Fitted LabelEncoder for {col}")
        
        outlier_cols = ['Episode_Length_minutes', 'Number_of_Ads']
        for col in outlier_cols:
            if col in X.columns:
                data_std = np.std(X[col])
                data_mean = np.mean(X[col])
                anomaly_cut_off = data_std * 3
                lower_limit = round(data_mean - anomaly_cut_off, 2)
                upper_limit = round(data_mean + anomaly_cut_off, 2)
                self.outlier_bounds_[col] = (lower_limit, upper_limit)
        
        logging.info("CustomImputer fitting completed")
        return self
    
    def transform(self, X):
        logging.info("Transforming data...")
        X_copy = X.copy()
        
        X_copy = self._handle_missing_values(X_copy)
        
        X_copy = self._handle_percentage_outliers(X_copy)
        
        X_copy = self._handle_numeric_outliers(X_copy)
        
        X_copy = self._extract_episode_number(X_copy)
        
        X_copy = self._encode_categorical(X_copy)
        
        X_copy = self._create_features(X_copy)
        
        logging.info(f"Transformation completed. New shape: {X_copy.shape}")
        return X_copy
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def _handle_missing_values(self, X):
        X_copy = X.copy()
        
        if 'Number_of_Ads' in X_copy.columns:
            median_ads = X_copy['Number_of_Ads'].median()
            X_copy['Number_of_Ads'] = X_copy['Number_of_Ads'].fillna(median_ads)
        
        if 'Guest_Popularity_percentage' in X_copy.columns:
            X_copy['Guest_Popularity_percentage'] = X_copy['Guest_Popularity_percentage'].fillna(0)
        
        num_columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                      'Guest_Popularity_percentage', 'Number_of_Ads']
        
        if self.imputer is not None:
            existing_num_cols = [col for col in num_columns if col in X_copy.columns]
            if existing_num_cols:
                X_copy[existing_num_cols] = self.imputer.transform(X_copy[existing_num_cols])
        
        return X_copy
    
    def _handle_percentage_outliers(self, X):
        X_copy = X.copy()
        percentage_cols = ['Guest_Popularity_percentage', 'Host_Popularity_percentage']
        
        for col in percentage_cols:
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], 0, 100)
        
        return X_copy
    
    def _handle_numeric_outliers(self, X):
        X_copy = X.copy()
        
        for col, (lower_limit, upper_limit) in self.outlier_bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower_limit, upper_limit)
                
                if col == 'Number_of_Ads':
                    X_copy[col] = X_copy[col].round(0)
        
        return X_copy
    
    def _extract_episode_number(self, X):
        X_copy = X.copy()
        
        if 'Episode_Title' in X_copy.columns:
            X_copy['Episode_Title'] = X_copy['Episode_Title'].str.extract(r'(\d+)').astype(int)
        
        return X_copy
    
    def _encode_categorical(self, X):
        X_copy = X.copy()
        ordinal_columns = ['Episode_Sentiment', 'Publication_Day', 'Publication_Time', 
                          'Podcast_Name', 'Genre']
        
        for col in ordinal_columns:
            if col in X_copy.columns and col in self.label_encoders:
                try:
                    X_copy[f'{col}_Encoded'] = self.label_encoders[col].transform(X_copy[col])
                except ValueError as e:
                    logging.warning(f"Error encoding {col}: {e}")
                    X_copy[f'{col}_Encoded'] = -1
        
        columns_to_drop = [col for col in ordinal_columns if col in X_copy.columns]
        X_copy = X_copy.drop(columns=columns_to_drop, errors='ignore')
        
        return X_copy
    
    def _create_features(self, X):
        X_copy = X.copy()
        
        # Ads_per_Popularity
        if all(col in X_copy.columns for col in ['Number_of_Ads', 'Host_Popularity_percentage']):
            X_copy['Ads_per_Popularity'] = (X_copy['Number_of_Ads'] / 
                                          (X_copy['Host_Popularity_percentage'] + 0.1))
        
        # Length_Ads_Ratio
        if all(col in X_copy.columns for col in ['Episode_Length_minutes', 'Number_of_Ads']):
            X_copy['Length_Ads_Ratio'] = (X_copy['Episode_Length_minutes'] / 
                                        (X_copy['Number_of_Ads'] + 1))
        
        # Length_Popularity_Interaction
        if all(col in X_copy.columns for col in ['Episode_Length_minutes', 
                                               'Host_Popularity_percentage', 
                                               'Guest_Popularity_percentage']):
            X_copy['Length_Popularity_Interaction'] = (
                X_copy['Episode_Length_minutes'] * 
                (X_copy['Host_Popularity_percentage'] + X_copy['Guest_Popularity_percentage'])
            )
        
        return X_copy
        