import os
import sys
from dataclasses import dataclass

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging
from optuna.trial import Trial
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
            
            model = LGBMClassifier()

            best_params = model_param(x_train,y_train,x_test,y_test,model)
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
