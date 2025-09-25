import yaml
import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle as pkl
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def write_yaml_file(file_path:str,content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w")as file:
            yaml.dump(content,file)        
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def save_numpy_array_data(file_path:str,array:np.array):
    """
    Save numpy array data to file
    file_path:str location of file save
    array: np.array data to save
    """ 
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb")as file_obj:
         np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def save_object(file_path:str,obj:object)->None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pkl.dump(obj,file_obj)
        logging.info("Exited the save_object method of MainUtils class")    
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exits")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pkl.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def load_numpy_array(file_path:str)->np.array:
    """
    load numpy array data from file
    file_path:str location of filr to load
    return: np.array data loaded
    """

    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    try:
        report = {}
        best_models = {}

        for model_name, model_instance in models.items():
            model_params = param.get(model_name, {})

            # If params exist, run GridSearch
            if model_params:
                gs = GridSearchCV(model_instance, model_params, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
            else:
                # No tuning
                model_instance.fit(X_train, y_train)
                best_model = model_instance
                best_params = {}

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Scores
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            # Store everything
            report[model_name] = {
                "best_params": best_params,
                "train_score": train_score,
                "test_score": test_score
            }
            best_models[model_name] = best_model

        return report, best_models

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
