import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import yaml
import zipfile
from sklearn.model_selection import train_test_split
import shutil
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories, get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import logger


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        print("Config: ", self.config)
        self.params = read_yaml(params_filepath)
        print("Params: ", self.params)
        create_directories([self.config.data_ingestion.root_dir])

    def copy_zip_data(self):
        source_path = self.config.data_ingestion.source
        destination_path = self.config.data_ingestion.local_data_file
        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)
            logger.info(
                f"Zip file copied from {source_path} to {destination_path}.")
        else:
            logger.info(f"File {destination_path} already exists.")

    def unzip_data(self):
        try:
            with zipfile.ZipFile(self.config.data_ingestion.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(self.config.data_ingestion.unzip_dir)
            logger.info("Zip file extracted successfully.")
        except FileNotFoundError:
            logger.error(
                f"Zip file not found at path: {self.config.data_ingestion.local_data_file}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while extracting zip file: {e}")
            raise

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.data_ingestion

        create_directories([config.root_dir, params.working_dir])
        self.copy_zip_data()
        self.unzip_data()

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source=config.source,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            source_csv=config.source_csv,
            train_size=params.train_size,
            test_size=params.test_size,
            validation_size=params.validation_size,
            random_state=params.random_state,
            max_samples=params.max_samples,
            min_samples=params.min_samples,
            img_size=params.img_size,
            working_dir=params.working_dir
        )

        return data_ingestion_config
