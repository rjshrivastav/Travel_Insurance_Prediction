import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_model


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            model = load_model(model_path)
            preprocessor = load_model(preprocessor_path)
            data_scale = preprocessor.transform(features)
            pred = model.predict_proba(data_scale)[:,1]
            return pred
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Age: int,
                 Employment_Type: str,
                 GraduateOrNot: str,
                 AnnualIncome: int,
                 FamilyMembers: int,
                 ChronicDiseases: int,
                 FrequentFlyer: str,
                 EverTravelledAbroad: str):
        self.Age = Age

        self.Employment_Type = Employment_Type

        self.GraduateOrNot = GraduateOrNot

        self.AnnualIncome = AnnualIncome

        self.FamilyMembers = FamilyMembers

        self.ChronicDiseases = ChronicDiseases

        self.FrequentFlyer = FrequentFlyer

        self.EverTravelledAbroad = EverTravelledAbroad


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age': [self.Age],
                'Employment_Type': [self.Employment_Type],
                'GraduateOrNot': [self.GraduateOrNot],
                'AnnualIncome': [self.AnnualIncome],
                'FamilyMembers': [self.FamilyMembers],
                'ChronicDiseases': [self.ChronicDiseases],
                'FrequentFlyer': [self.FrequentFlyer],
                'EverTravelledAbroad': [self.EverTravelledAbroad]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
