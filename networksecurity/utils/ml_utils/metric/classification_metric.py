from networksecurity.entity.artifact_entity import ClaassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from sklearn.metrics import f1_score,precision_score,recall_score
import sys

def get_classification_score(y_true,y_pred)->ClaassificationMetricArtifact:
   try:  
        model_f1_score=f1_score(y_true,y_pred)
        model_recall_score=recall_score(y_true,y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric = ClaassificationMetricArtifact(f1_score=model_f1_score,
                                                            precision_score=model_precision_score,recall_score=model_recall_score)
        return classification_metric
   except Exception as e:
        raise NetworkSecurityException(e,sys)

