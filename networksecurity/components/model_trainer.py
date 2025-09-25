import sys
import os
import numpy as np
import pandas as pd
import mlflow

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import(
    DataTranformationArtifacts,
    ModelTrainerArtifact
    )

from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
   AdaBoostClassifier,
   GradientBoostingClassifier,
   RandomForestClassifier,
)
import joblib
import mlflow
import dagshub

dagshub.init(repo_owner='harshitdubey7896', repo_name='Network-Security')
mlflow.set_tracking_uri("https://dagshub.com/harshitdubey7896/Network-Security.mlflow") 
mlflow.set_experiment("Network-Security-Experiment")




class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTranformationArtifacts):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
          raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self, best_model, train_metric, test_metric):
     with mlflow.start_run():
        # Log metrics
        mlflow.log_metric("train_f1_score", train_metric.f1_score)
        mlflow.log_metric("train_precision", train_metric.precision_score)
        mlflow.log_metric("train_recall", train_metric.recall_score)
        mlflow.log_metric("test_f1_score", test_metric.f1_score)
        mlflow.log_metric("test_precision", test_metric.precision_score)
        mlflow.log_metric("test_recall", test_metric.recall_score)

        # Save model locally
        model_path = "best_model.pkl"
        joblib.dump(best_model, model_path)

        # Log as artifact (works on DagsHub)
        mlflow.log_artifact(model_path, artifact_path="model")

    
                

        
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,64,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,0.5,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        # Evaluate all models
        model_report, best_models = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=x_test, y_test=y_test,
            models=models, param=params
        )

        # Select best model by highest test_score
        best_model_name = max(model_report, key=lambda k: model_report[k]["test_score"])
        best_model_info = model_report[best_model_name]
        best_model = best_models[best_model_name]

        # Log / print details
        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"Best Params: {best_model_info['best_params']}")
        logging.info(f"Train Score: {best_model_info['train_score']}")
        logging.info(f"Test Score: {best_model_info['test_score']}")

        print(f"Best Model: {best_model_name}")
        print(f"   Best Params: {best_model_info['best_params']}")
        print(f"   Train Score: {best_model_info['train_score']}")
        print(f"   Test Score: {best_model_info['test_score']}")

        # Metrics
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=best_model.predict(X_train))
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=best_model.predict(x_test))

        self.track_mlflow(best_model, classification_train_metric, classification_test_metric)

        # Save final model with preprocessor
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        
        save_object("final_model/model.pkl",best_model)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)




        print(f"   Train: {classification_train_metric}")
        print(f"   Test : {classification_test_metric}")   
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
           train_file_path=self.data_transformation_artifact.transformed_train_file_path
           test_file_path=self.data_transformation_artifact.transformed_test_file_path
           
           #Loading Training array and testing array
           train_arr= load_numpy_array(train_file_path)
           test_arr= load_numpy_array(test_file_path)
           X_train,y_train,x_test,y_test=(
              train_arr[:,:-1],
              train_arr[:,-1],
              test_arr[:,:-1],
              test_arr[:,-1],
           )

           model_trainer_artifact=self.train_model(X_train,y_train,x_test,y_test)
           return model_trainer_artifact
        except Exception as e:
           raise NetworkSecurityException(e,sys)