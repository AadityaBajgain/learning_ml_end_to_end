import os
import sys
import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")
    

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        logging.info("model training initiated")
        try:
            logging.info("Splitting training and testing data...")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor()
            }
            
            # params = {
                
            # }
            
            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test= X_test, y_test = y_test, models=models)
        except Exception as e:
            raise CustomException(e, sys)


