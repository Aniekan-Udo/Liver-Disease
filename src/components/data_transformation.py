from src.exception import CustomException
from src.logger import logging

import os
import sys

from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path_obj=os.path.join('artifacts','preprocessor.pkl')
    resampler_file_path=os.path.join("artifacts","resampler.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_resampler_obj(self):
        try:
            """
            This method will undersample the dataset in other to help reduce overfitting
            """
            resampler = RandomUnderSampler(random_state=42)

            return resampler
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def get_data_transformer_obj(self):
        try:
            
            preprocessor=StandardScaler()
           
            return preprocessor
                    
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            

            logging.info("Sample the training and test data")
            train_df=train_df.sample(random_state=42, frac=1).reset_index(drop=True)
            test_df=test_df.sample(random_state=42, frac=1).reset_index(drop=True)

            target_column_name='Diagnosis'
            numerical_columns=["BMI", "AlcoholConsumption", "PhysicalActivity", "LiverFunctionTest"]


            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            train_target_df=train_df[target_column_name]
            

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            test_target_df=test_df[target_column_name]

            logging.info("Applying resample object on train and test data")
            resampler_obj=self.get_data_resampler_obj()
            
            input_feature_train_df,train_target_df=resampler_obj.fit_resample(input_feature_train_df,train_target_df)
            input_feature_test_df,test_target_df=resampler_obj.fit_resample(input_feature_test_df,test_target_df)
            

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            preprocessing_obj=self.get_data_transformer_obj()

            train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr=preprocessing_obj.transform(input_feature_test_df)


            # # Combine input features and target feature for the training set
            train_arr = np.c_[train_arr, np.array(train_target_df)]

            # # Combine input features and target feature for the test set
            test_arr = np.c_[test_arr, np.array(test_target_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.transformation_config.preprocessor_file_path_obj,
                obj=preprocessing_obj

            )

            save_object(

                file_path=self.transformation_config.resampler_file_path,
                obj=resampler_obj
                
                )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_file_path_obj,
                self.transformation_config.resampler_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)