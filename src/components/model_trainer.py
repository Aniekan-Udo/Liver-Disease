from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object,evalute_model,save_object

from dataclasses import dataclass

resampler_path = "artifacts/resampler.pkl"
resampler = load_object(file_path=resampler_path)

@dataclass
class ModelTrainerConfig:
    model_trainer_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            X_train,y_train=resampler.fit_resample(X_train,y_train)
            X_test,y_test=resampler.fit_resample(X_test,y_test)

            models={
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression()
            }

            params={
                
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression": {
                    'penalty': ['l2'],
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }

                
                
            }

            model_result:dict=evalute_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            ## To get best model score from dict
            best_model_score=max(sorted(model_result.values()))

            ## To get best model name from dict

            best_model_name = list(model_result.keys())[
                list(model_result.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.model_trainer_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            evaluation_metric = f1_score(y_test, predicted)
            return evaluation_metric


        except Exception as e:
            raise CustomException(e,sys)