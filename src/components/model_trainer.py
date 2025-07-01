import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
    trained_model_config=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_train(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'Linear Regressor': LinearRegression(),
                'Catboost Regression': CatBoostRegressor(verbose=False),
                'Xgboost Regression': XGBRFRegressor(),
                'Random Forest Regression': RandomForestRegressor(),
                'Ada Boost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'K Neighbors Regressor': KNeighborsRegressor()
            }

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Custom_Exception("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_config,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise Custom_Exception(e,sys)

        