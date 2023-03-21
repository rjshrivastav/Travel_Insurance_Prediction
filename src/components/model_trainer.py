import os
import sys
from dataclasses import dataclass

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging
from optuna import trial

from src.utils import save_object, evaluate_model, model_param

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('spliting training and test input data')
            x_train, y_train, x_test, y_test = (train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                                )
            
            model = {
                "Lightgbm": LGBMClassifier()
            }

            params={
                "Lightgbm":{
                    'reg_alpha' : trial.Trial.suggest_loguniform('reg_alpha', 0.001, 0.1, 0.1),
                    'reg_lambda' : trial.Trial.suggest_loguniform('reg_lambda',0.001, 0.1),
                    'learning_rate' : trial.Trial.suggest_uniform('learning_rate', 0.03 , 0.07),
                    'max_depth' : trial.Trial.suggest_int('max_depth', 1 , 20),
                    'n_estimators' : trial.Trial.suggest_int('n_estimators', 100 , 20000)
                }}
            
            best_params = model_param(x_train,y_train,x_test,y_test,model,params)
            print(best_params)

            best_model = evaluate_model(x_train,y_train,x_test,y_test,model,best_params)
            
            logging.info('saving model')
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            
            predicted = best_model.predict(x_test)
            roc_curve = roc_auc_score(y_test, predicted)
            return best_model, roc_curve
        except Exception as e:
            raise CustomException(e,sys)
