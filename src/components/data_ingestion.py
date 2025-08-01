import os
import sys
from src.exception import Custom_Exception
from src.logger import logging
from src.components.data_transformation import Data_Transformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Entered data ingestion method or component")
            df=pd.read_csv('Notebook\data\stud.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise Custom_Exception(e,sys)
        

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=Data_Transformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path=train_data,test_path=test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_train(train_arr,test_arr))
    