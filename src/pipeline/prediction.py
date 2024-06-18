import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            resampler_path = "artifacts/resampler.pkl"
            preprocessor_path = 'artifacts/preprocessor.pkl'
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            #resampled_data = resampler.fit_resample(features)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(  self,
        Age: str,
        Gender: str,
        BMI: str,
        AlcoholConsumption: str,
        Smoking: str,
        GeneticRisk: str,
        PhysicalActivity: str,
        Diabetes: str,
        Hypertension: str,
        LiverFunctionTest: str
        
        ):

        self.Age = Age

        self.Gender = Gender

        self.BMI = BMI

        self.AlcoholConsumption = AlcoholConsumption

        self.Smoking = Smoking

        self.GeneticRisk = GeneticRisk

        self.PhysicalActivity = PhysicalActivity

        self.Diabetes = Diabetes

        self.Hypertension = Hypertension

        self.LiverFunctionTest = LiverFunctionTest


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "BMI": [self.BMI],
                "AlcoholConsumption": [self.AlcoholConsumption],
                "Smoking": [self.Smoking],
                "GeneticRisk": [self.GeneticRisk],
                "PhysicalActivity": [self.PhysicalActivity],
                "Diabetes": [self.Diabetes],
                "Hypertension": [self.Hypertension],
                "LiverFunctionTest": [self.LiverFunctionTest],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
