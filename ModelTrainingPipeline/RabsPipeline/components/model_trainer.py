import os
import sys
import yaml
from RabsPipeline.utils.main_utils import read_yaml_file
from RabsPipeline.logger import logging
from RabsPipeline.exception import RabsException
from RabsPipeline.entity.config_entity import   ModelTrainerConfig
from RabsPipeline.entity.artifacts_entity import  ModelTrainerArtifact
from RabsPipeline.constant.training_pipeline import *
from ultralytics import YOLO
from IPython.display import display, Image
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

import os
import shutil
import sys
from ultralytics import YOLO
from RabsPipeline.entity.artifacts_entity import ModelTrainerArtifact
from RabsPipeline.exception import RabsException
from RabsPipeline.logger import logging
from RabsPipeline.utils.main_utils import update_yaml_file




class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:

            logging.info("Initializing YOLO model for fine-tuning")
            train_directory = self.model_trainer_config.existing_train_dir or self.model_trainer_config.existing_train_obb_dir
            if os.path.exists(train_directory):
                print(f"Directory {train_directory} already exists. Deleting it.")
                shutil.rmtree(train_directory)
            else:
                print(f"Directory {train_directory} does not exist. Proceeding with training.")

            model = YOLO(self.model_trainer_config.weight_name)
            update_yaml_file(file_path=self.model_trainer_config.yaml_file_path)

            model.train(
                data = self.model_trainer_config.yaml_file_path,
                epochs=self.model_trainer_config.no_epochs, 
                batch = self.model_trainer_config.batch_size,
                imgsz=640, 
                project= "/home/ai/Desktop/RABs/ModelTrainingPipeline/runs",
                plots=True )

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            if not os.path.exists(self.model_trainer_config.trained_best_weight_filepath):
                raise FileNotFoundError(f"Model file not found at {self.model_trainer_config.trained_best_weight_filepath}")

            shutil.copy(self.model_trainer_config.trained_best_weight_filepath, self.model_trainer_config.model_trainer_dir, )

            # os.system(f"cp {self.model_trainer_config.trained_best_weight_filepath} {self.model_trainer_config.model_trainer_dir}/")
            # model_trainer_artifact = ModelTrainerArtifact(
            #     trained_model_file_path=f"{self.model_trainer_config.model_trainer_dir}/best.pt",  )

            model_trainer_artifact = ModelTrainerArtifact(
                model_trainer_dir=self.model_trainer_config.model_trainer_dir,
                trained_model_file_path=f"{self.model_trainer_config.model_trainer_dir}/best.pt")
            
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise RabsException(e, sys)




