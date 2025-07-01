import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocess_file_path=os.path.join('artifacts',"preprocessor.pkl")

class Data_Transformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformer_data_object(self):
        try:
            num_columns=['reading_score','writing_score']  
            cat_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]          

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('Onehotencoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical and Numerical pipeline created")

            preprocessor=ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,num_columns),
                ('cat_pipeline',cat_pipeline,cat_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise Custom_Exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Train and test data read completed')
            logging.info('Preprocessing the data')

            preprocess_obj=self.get_transformer_data_object()

            target_column_name='math_score'
            num_columns=['reading_score','writing_score']

            input_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_train_df=train_df[target_column_name]

            input_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_train_arr=preprocess_obj.fit_transform(input_train_df)
            input_test_arr=preprocess_obj.transform(input_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocess_file_path,
                obj=preprocess_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_file_path,
            )
        except Exception as e:
            raise Custom_Exception(e,sys)