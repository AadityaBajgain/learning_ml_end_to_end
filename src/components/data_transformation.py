import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact","preprocessor.pkl")
    
class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            num_columns = ["math_score", "writing_score"]
            
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level__of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Numerical Scaling completed")
            logging.info("Categorical Encoding and Scaling completed")
            
            
            preprocessor  = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)