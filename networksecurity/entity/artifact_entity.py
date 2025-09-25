from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    trained_file_path:str
    test_file_path:str

@dataclass
class DatavalidationArtifacts:
        validation_status:bool
        valid_train_file_path:str
        valid_test_file_path:str
        invalid_train_file_path:str
        invalid_test_file_path:str
        drift_report_file_path:str

@dataclass
class DataTranformationArtifacts:
      transformed_object_file_path:str
      transformed_train_file_path:str
      transformed_test_file_path:str

@dataclass
class ClaassificationMetricArtifact:
      f1_score:float
      precision_score:float
      recall_score:float


@dataclass
class ModelTrainerArtifact:
      train_model_file_path:str
      train_metric_artifact:ClaassificationMetricArtifact
      test_metric_artifact:ClaassificationMetricArtifact    

      
