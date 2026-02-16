import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.utils import save_obj

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
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_set = pd.read_csv(train_path)
            test_set = pd.read_csv(test_path)
            
            logging.info("Completed reading train and test data")
            
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            input_train_feature_df = train_set.drop(target_column_name, axis=1)
            input_train_target_df = train_set[target_column_name]
            
            
            input_test_feature_df = test_set.drop(target_column_name, axis=1)
            input_test_target_df = test_set[target_column_name]
            
            logging.info("Applying preprocessing object on train and test set")
            
            input_train_feature_arr = preprocessing_obj.fit_transform(input_train_feature_df)
            input_test_feature_arr = preprocessing_obj.transform(input_test_feature_df)
            
            
            train_arr = np.c_(input_train_feature_arr, np.array(input_train_feature_df))
            
            
            test_arr = np.c_(input_test_feature_arr, np.array(input_test_feature_df))
            
            
            logging.info("Saved preprocessing object")
            
            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj()
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)