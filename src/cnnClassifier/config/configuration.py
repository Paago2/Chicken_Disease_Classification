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
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig, PrepareBaseModelConfig)
from cnnClassifier import logger
from keras.applications import EfficientNetB5
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adamax
from keras import regularizers
import tensorflow as tf


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([
            self.config.data_ingestion.root_dir,
            self.config.artifacts_root,
            self.config.prepare_base_model.root_dir
        ])

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

    def get_prepare_base_model_config(self):
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.base_model.image_size,
            params_learning_rate=self.params.learning_rate,
            params_include_top=self.params.base_model.include_top,
            params_weights=self.params.base_model.weights,
            params_classes=self.params.classes,
            params_model_name=self.params.base_model.model_name,
            params_pooling=self.params.base_model.pooling,
            params_dropout_rate1=self.params.base_model.dropout_rate1,
            params_dropout_rate2=self.params.base_model.dropout_rate2,
            params_dense_1024_regularizer_l2=self.params.base_model.dense_1024_regularizer_l2,
            params_dense_1024_regularizer_l1=self.params.base_model.dense_1024_regularizer_l1,
            params_dense_128_regularizer_l2=self.params.base_model.dense_128_regularizer_l2,
            params_dense_128_regularizer_l1=self.params.base_model.dense_128_regularizer_l1
        )

        return prepare_base_model_config
