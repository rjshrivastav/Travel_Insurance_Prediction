import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import roc_auc_score
import optuna

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def model_param(X_train,y_train,X_test,y_test, models):
    try:
        def fit_lgb(X_train,y_train,X_test,y_test,models, param):
            models.set_params(**param)
            models.fit(X_train, y_train,eval_set=[(X_test,y_test)])
    
            y_train_pred = models.predict_proba(X_train)[:,1]
            
            y_test_pred = models.predict_proba(X_test)[:,1]
            y_train_pred = np.clip(y_train_pred, 0.1, None)
            y_test_pred = np.clip(y_test_pred, 0.1, None)
            
            log = {
                "train roc_curve": roc_auc_score(y_train, y_train_pred),
                "valid roc_curve": roc_auc_score(y_test, y_test_pred)
            }
            
            return models, log
        
        def objective(trial):
            acc = 0
            params={
                    'reg_alpha' : trial.suggest_uniform('reg_alpha', 0.001, 0.1),
                    'reg_lambda' : trial.suggest_uniform('reg_lambda',0.001, 0.1),
                    'learning_rate' : trial.suggest_uniform('learning_rate', 0.03 , 0.07),
                    'max_depth' : trial.suggest_int('max_depth', 1 , 20),
                    'n_estimators' : trial.suggest_int('n_estimators', 100 , 20000)
                }
            model, log = fit_lgb(X_train, y_train, X_test, y_test, models, params)
            acc += log['valid roc_curve']
                
            return acc
        
        study = optuna.create_study(direction = 'maximize')
        study.optimize(objective,n_trials=15)
        lgb_params = study.best_params
        return lgb_params
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_model(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test, models, param):
    try:
        models.set_params(**param)
        models.fit(X_train,y_train)

        y_train_pred = models.predict(X_train)
        y_test_pred = models.predict(X_test)

        train_model_score = roc_auc_score(y_train,y_train_pred)
        test_model_score = roc_auc_score(y_test,y_test_pred)

        return models
    except Exception as e:
        raise CustomException(e, sys)