import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
import gc

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_best
from sklearn.model_selection import train_test_split

@dataclass
class PredictionsConfig:
    sample_submission_path: str=os.path.join('artifacts', "sample_submission.csv")
    config_dir: str = 'artifacts/configs'
class ModelTrainer:
    def __init__(self):
        self.prediction_config=PredictionsConfig()
    def initiate_model_trainer(self, train, test):
        try:
            logging.info("Splitting training and test input data")
            print(type(train))
            X_train = train.drop(['id', 'Listening_Time_minutes'], axis=1)
            y_train = train['Listening_Time_minutes']
            X_test = test.drop('id', axis=1)
            best_params = evaluate_best(X_train, y_train)
            # Creating one base config
            base_config = {
                'model_type': 'RandomForestRegressor',
                'base_params': best_params
            }
            
            # Saving config
            models_dir = self.prediction_config.config_dir
            n_models = 30
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(base_config, f'{models_dir}/base_config.joblib')
            
            # Saving only random_state for each model
            for i in range(n_models):
                model_info = {
                    'random_state': i,
                    'n_jobs': 3
                }
                
                filename = f'{models_dir}/model_{i}.joblib'
                joblib.dump(model_info, filename)
                
                logging.info(f"Model config {i+1} saved")
            
            logging.info(f"All {n_models} model configurations saved")
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            final_prediction_val = self.model_prediction(X_tr, y_tr, X_val, models_dir)
            rmse = np.sqrt(mean_squared_error(y_val, final_prediction_val))
            
            prediction = self.model_prediction(X_train, y_train, X_test, models_dir)
            results = pd.DataFrame({
                    'id': range(750000, 750000 + len(prediction)),
                    'Listening_Time_minutes': prediction
            })

            results.to_csv(self.prediction_config.sample_submission_path, index=False, float_format='%.3f')
            logging.info("Predictions were saved in predictions.csv")
            return rmse
        except Exception as e:
            raise CustomException(e, sys)
        
    def model_prediction(self, X_train, y_train, X_test, models_dir):
        predictions = []
        model_files = []
        logging.info("Reading model configs")
        for file in os.listdir(models_dir):
            if file.startswith('model_') and file.endswith('.joblib'):
                model_files.append(file)
            
        model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        base_config = joblib.load(f'{models_dir}/base_config.joblib')    

        for i, file in enumerate(model_files):
            model_info = joblib.load(f'{models_dir}/model_{i}.joblib')
                
            params = {**base_config['base_params'], 
                    'random_state': model_info['random_state'],
                    'n_jobs': model_info.get('n_jobs', 3)}
            forest = RandomForestRegressor(**params)
            forest.fit(X_train, y_train)
            pred = forest.predict(X_test)
            predictions.append(pred)
                
            logging.info(f"Model {i+1} trained and predicted")
                
            del forest, params
            gc.collect()
            
        final_prediction = np.mean(predictions, axis=0)
        return final_prediction