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


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_csv(self):
        logger.info("Loading CSV file.")
        df = pd.read_csv(self.config.source_csv)
        df.columns = ['filepaths', 'labels']
        df['filepaths'] = df['filepaths'].apply(
            lambda x: os.path.join(self.config.root_dir, x))
        logger.info(f"CSV file loaded successfully with {df.shape[0]} rows.")
        return df

    def split_data(self, df):
        logger.info(
            "Splitting data into training, testing, and validation sets.")
        trsplit = self.config.train_size
        vsplit = self.config.validation_size
        dsplit = vsplit / (1 - trsplit)
        strat = df['labels']
        train_df, dummy_df = train_test_split(
            df, train_size=trsplit, shuffle=True, random_state=self.config.random_state, stratify=strat)
        strat = dummy_df['labels']
        test_df, valid_df = train_test_split(
            dummy_df, train_size=dsplit, shuffle=True, random_state=self.config.random_state, stratify=strat)
        logger.info("Data split successfully.")
        logger.info(f"Train set size: {train_df.shape[0]}")
        logger.info(f"Test set size: {test_df.shape[0]}")
        logger.info(f"Validation set size: {valid_df.shape[0]}")

        return train_df, test_df, valid_df

    def class_distribution(self, train_df):
        logger.info("Getting class distribution.")
        groups = train_df.groupby('labels')
        print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
        for label in train_df['labels'].unique():
            print('{0:^30s} {1:^13d}'.format(
                label, len(groups.get_group(label))))
        print('\n')

    def trim(self, train_df):
        logger.info("Trimming classes.")
        max_size = self.config.max_samples
        min_size = self.config.min_samples
        column = 'labels'
        train_df = train_df.copy()
        original_class_count = len(list(train_df[column].unique()))
        logger.info('Original Number of classes in dataframe: %s',
                    original_class_count)
        sample_list = []
        groups = train_df.groupby(column)
        for label in train_df[column].unique():
            group = groups.get_group(label)
            sample_count = len(group)
            if sample_count > max_size:
                strat = group[column]
                samples, _ = train_test_split(
                    group, train_size=max_size, shuffle=True, random_state=self.config.random_state, stratify=strat)
                sample_list.append(samples)
            elif sample_count >= min_size:
                sample_list.append(group)
        train_df = pd.concat(sample_list, axis=0).reset_index(drop=True)
        final_class_count = len(list(train_df[column].unique()))
        if final_class_count != original_class_count:
            logger.warning(
                '*** WARNING***  dataframe has a reduced number of classes')
        balance = list(train_df[column].value_counts())
        logger.info('Class balance: %s', balance)
        return train_df

    def prepare_working_dir(self):
        logger.info("Preparing working directory.")
        os.makedirs(self.config.working_dir, exist_ok=True)
        logger.info("Working directory prepared.")

    def execute(self):
        logger.info("Starting data ingestion.")
        df = self.load_csv()
        train_df, test_df, valid_df = self.split_data(df)
        self.class_distribution(train_df)
        train_df = self.trim(train_df)
        self.prepare_working_dir()

        # save the dataframes to CSV files
        print("About to write train.csv")
        train_df.to_csv(os.path.join(
            self.config.working_dir, 'train.csv'), index=False)
        print("Finished writing train.csv")

        test_df.to_csv(os.path.join(
            self.config.working_dir, 'test.csv'), index=False)
        valid_df.to_csv(os.path.join(
            self.config.working_dir, 'valid.csv'), index=False)

        logger.info("Data ingestion completed successfully.")
        return train_df, test_df, valid_df
