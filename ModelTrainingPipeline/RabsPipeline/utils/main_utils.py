import os.path
import sys
import yaml
import base64
from RabsPipeline.exception import RabsException
from RabsPipeline.logger import logging
from RabsPipeline.constant.training_pipeline import *



def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise RabsException(e, sys) from e
    



def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
            logging.info("Successfully write_yaml_file")

    except Exception as e:
        raise RabsException(e, sys)
    
def update_yaml_file(file_path):
    try:
        import yaml
        import logging
        import sys

        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            
        base_dir = BASE_DIR
        base_dir = base_dir.replace("\\", "/")
        data['path'] = base_dir
        data['train'] = TRAINING_DATA_PATH
        data['val'] = VALIDATION_DATA_PATH
        data['test'] = TEST_DATA_PATH
        data['names'] = DATA_CLASS_NAMES
        data['nc'] = NUMBER_OF_CLASS
        class CustomDumper(yaml.Dumper):
            def increase_indent(self, flow=False, indentless=False):
                return super(CustomDumper, self).increase_indent(flow, False)

        with open(file_path, 'w') as file:
            yaml.dump(
                data,
                file,
                default_flow_style=False,
                Dumper=CustomDumper,
                sort_keys=False
            )
            logging.info(f"Updated paths and added class details in {file_path}")
    except Exception as e:
        raise RabsException(e, sys) from e


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

    
    