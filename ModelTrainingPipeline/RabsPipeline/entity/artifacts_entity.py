from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_zip_file_path:str
    feature_store_path:str

@dataclass
class DataValidationArtifact:
    validation_status:str


@dataclass
class ModelTrainerArtifact:
    model_trainer_dir:str
    trained_model_file_path:str
    model_trainer_dir:str



    
