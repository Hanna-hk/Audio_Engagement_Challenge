import os
import sys
import dill

from src.exception import CustomException
from sklearn.ensemble import RandomForestRegressor

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_best(X_tr, y_tr):
    logging.info("Starting evaluation of the best parameters")
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [10, 15, None],
        'min_samples_split': randint(2, 10),
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=8, cv=3,
        n_jobs=3, verbose=1, random_state=42,
        return_train_score=True
    )
    random_search.fit(X_tr, y_tr)
    best_rf_params = random_search.best_params_
    logging.info("Evaluated best params for model")
    return best_rf_params
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)