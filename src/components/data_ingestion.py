import os 
import sys
from src.exception import CustomException
from src.logger import logging


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass

class DataIngestionConfig:
    train_data_path : str = os.path.join("artifact", "train.csv")
    test_data_path : str = os.path.join("artifact", "test.csv")
    raw_data_path : str = os.path.join("artifact", "data.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initialize_data_ingestion(self):
        logging.info("Initiated the data ingestion method or component")
        
        try:
            df = pd.read_csv("notebooks/data/stud.csv")
            logging.info("Read the dataset as dataframe")
        except:
            pass