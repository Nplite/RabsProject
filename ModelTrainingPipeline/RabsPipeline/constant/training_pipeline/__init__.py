import os

ARTIFACTS_DIR: str = "artifacts"


#############################################################################
"""Data Ingestion related constant start with DATA_INGESTION VAR NAME"""
#############################################################################


DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

ROBOFLOW_API_KEY: str = "RcIHX2vbIwfmpdWiOEtM"

DATA_FORMAT_MODEL : str = "yolov8"

ROBOFLOW_WORKSPACE_NAME: str = "rabsproject"

ROBOFLOW_PROJECT_NAME: str = "truckdetection-vuldr"

ROBOFLOW_DATASET_VERSION: int = 1


                

# DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1cP_aB2vL8GK1rjKTc1OvkRSNPckpDrTp/view?usp=sharing"

DATA_DOWNLOAD_URL: str = ""



#############################################################################
"""Data Validation related constant start with DATA_VALIDATION VAR NAME"""
#############################################################################


DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATS_DIR : str = "status.txt"

DATA_VALIDATION_ALL_REQUIRED_FILES : str = ['train', 'test', 'valid', "data.yaml"]



############################################################################
"""Model trainer related constant start with MODEL_TRAINER VAR NAME"""
#############################################################################

MODEL_TRAINER_DIR_NAME : str = "model_trainer"

BASE_DIR = r"/home/ai/Desktop/RABs/ModelTrainingPipeline/artifacts/data_ingestion/feature_store"
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "train/images")
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, "valid/images")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test/images")
YAML_FILE_PATH = os.path.join(BASE_DIR, "data.yaml")

EXISTING_TRAIN_DIR: str = "/home/ai/Desktop/RABs/ModelTrainingPipeline/runs/train"

EXISTING_TRAIN_OBB_DIR: str = "runs/detect/obb_exp"

MODEL_TRAINER_PRETRAINED_WEIGHT: str = "yolov8s.pt"

DATA_CLASS_NAMES = ["Truck"]

NUMBER_OF_CLASS = 1

TRAINED_BEST_MODEL_PATH: str = '/home/ai/Desktop/RABs/ModelTrainingPipeline/runs/train/weights/best.pt'

# TRAINED_BEST_MODEL_PATH: str = 'runs/detect/obb_exp/weights/best.pt'

MODEL_TRAINER_NO_EOPCHS: int = 1

MODEL_TRAINER_BATCH_SIZE: int = 6


