import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.components.custom_imputer import CustomImputer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
        
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        #This function is responsible for data transformation
        try:
            data_pipeline = Pipeline([
                ('custom_imputer', CustomImputer()),
            ])

            return data_pipeline
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "Listening_Time_minutes"


            logging.info(f"Applying preprocessing object on train and test dataframe")

            input_feature_train = preprocessing_obj.fit_transform(train_df)
            input_feature_test = preprocessing_obj.transform(test_df)

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                input_feature_train,
                input_feature_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)