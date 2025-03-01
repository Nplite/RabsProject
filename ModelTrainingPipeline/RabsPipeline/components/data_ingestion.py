import os
import sys
import zipfile
import gdown
import shutil
from roboflow import Roboflow
from RabsPipeline.logger import logging
from RabsPipeline.exception import RabsException
from RabsPipeline.entity.config_entity import DataIngestionConfig
from RabsPipeline.entity.artifacts_entity import DataIngestionArtifact



class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
           raise RabsException(e, sys)
        

        
    def download_data(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.data_ingestion_config.data_download_url
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            data_file_name = "data.zip"
            zip_file_path = os.path.join(zip_download_dir, data_file_name)
            logging.info(f"Downloading data from {dataset_url} into file {zip_file_path}")


            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_file_path)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_file_path}")

            return zip_file_path

        except Exception as e:
            raise RabsException(e, sys)

    def download_roboflow_dataset(self, roboflow_feature_store_path):
            try:
                rf = Roboflow(api_key= self.data_ingestion_config.roboflow_api_key)
                project = rf.workspace(self.data_ingestion_config.roboflow_workspace_name).project(self.data_ingestion_config.roboflow_project_name)
                version = project.version(self.data_ingestion_config.roboflow_dataset_version)
                dataset = version.download(self.data_ingestion_config.data_format_model)
                downloaded_folder = os.path.abspath(dataset.location)
                os.makedirs(roboflow_feature_store_path, exist_ok=True)
                required_files = ["data.yaml", "train", "test", "valid"]

                for item in required_files:
                    source_path = os.path.join(downloaded_folder, item)
                    target_path = os.path.join(roboflow_feature_store_path, item)

                    if os.path.exists(source_path):
                        if os.path.isdir(source_path):
                            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(source_path, target_path)
                        print(f"Moved: {item} -> {roboflow_feature_store_path}")
                    else:
                        print(f"Warning: {item} not found in the downloaded dataset.")
                print("Dataset successfully moved to feature store path.")
                
                if os.path.exists(downloaded_folder):
                    shutil.rmtree(downloaded_folder)
                    print(f"Deleted temporary downloaded dataset folder: {downloaded_folder}")

            except Exception as e:
                print(f"Error downloading or moving dataset: {str(e)}")

    
    def extract_zip_file(self,zip_file_path: str)-> str:
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(feature_store_path)
            logging.info(f"Extracting zip file: {zip_file_path} into dir: {feature_store_path}")

            return feature_store_path

        except Exception as e:
            raise RabsException(e, sys)
        


    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            feature_store_path = None
            if self.data_ingestion_config.data_download_url != "":
                zip_file_path = self.download_data()
                feature_store_path = self.extract_zip_file(zip_file_path)

                data_ingestion_artifact = DataIngestionArtifact(
                    data_zip_file_path = zip_file_path,
                    feature_store_path = feature_store_path
                )

                logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
                logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

                return data_ingestion_artifact
            
            else:
                self.download_roboflow_dataset(roboflow_feature_store_path = self.data_ingestion_config.feature_store_file_path)
                data_ingestion_artifact = DataIngestionArtifact(
                    data_zip_file_path = None,
                    feature_store_path = feature_store_path )

                logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
                logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
                print("DATA INGESTION ROBOFLOW\n", data_ingestion_artifact)

                return data_ingestion_artifact

        except Exception as e:
            raise RabsException(e, sys)
        
