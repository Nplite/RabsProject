import os
from dataclasses import dataclass
from datetime import datetime
from RabsPipeline.constant.training_pipeline import *




@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR



training_pipeline_config:TrainingPipelineConfig = TrainingPipelineConfig() 



@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME)

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR)

    data_download_url: str = DATA_DOWNLOAD_URL
    roboflow_api_key: str = ROBOFLOW_API_KEY
    data_format_model: str = DATA_FORMAT_MODEL
    roboflow_project_name: str = ROBOFLOW_PROJECT_NAME
    roboflow_workspace_name: str = ROBOFLOW_WORKSPACE_NAME
    roboflow_dataset_version: int = ROBOFLOW_DATASET_VERSION


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME)

    valid_status_file_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_STATS_DIR)

    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME)

    weight_name = MODEL_TRAINER_PRETRAINED_WEIGHT

    existing_train_dir = EXISTING_TRAIN_DIR

    existing_train_obb_dir = EXISTING_TRAIN_OBB_DIR

    yaml_file_path = YAML_FILE_PATH

    base_directory = BASE_DIR

    trained_best_weight_filepath = TRAINED_BEST_MODEL_PATH

    no_epochs = MODEL_TRAINER_NO_EOPCHS

    batch_size = MODEL_TRAINER_BATCH_SIZE