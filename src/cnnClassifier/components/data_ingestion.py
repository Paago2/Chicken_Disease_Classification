import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import yaml
import zipfile
from sklearn.model_selection import train_test_split
import shutil
from keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories, get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_images(self):
        logger.info("Loading image file paths and labels.")
        filepaths = []
        labels = []
        for dirpath, dirnames, filenames in os.walk(self.config.root_dir):
            for filename in filenames:
                filepaths.append(os.path.join(dirpath, filename))
                labels.append(os.path.basename(dirpath))
        df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
        logger.info(
            f"Image file paths and labels loaded successfully with {df.shape[0]} rows.")
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

    def save_data(self, df, path):
        logger.info("Saving dataframe to %s.", path)
        df.to_csv(path, index=False)
        logger.info("Dataframe saved.")

    def create_generators(self, train_df, valid_df, test_df, img_size=(224, 224), batch_size=20):
        logger.info("Creating data generators.")

        trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                                   height_shift_range=.2, zoom_range=.2)
        t_and_v_gen = ImageDataGenerator()

        train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                              class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
        valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                                    class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

        length = len(test_df)
        test_batch_size = sorted([int(length/n) for n in range(1, length+1)
                                 if length % n == 0 and length/n <= 80], reverse=True)[0]
        test_steps = int(length/test_batch_size)
        test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                                   class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

        logger.info("Data generators created successfully.")
        return train_gen, valid_gen, test_gen, test_batch_size, test_steps

    def execute(self):
        logger.info("Starting data ingestion.")
        df = self.load_images()
        train_df, test_df, valid_df = self.split_data(df)
        self.class_distribution(train_df)
        train_df = self.trim(train_df)
        self.prepare_working_dir()

        # save the dataframes to corresponding directories
        self.save_data(train_df, os.path.join(
            self.config.working_dir, 'train'))
        self.save_data(test_df, os.path.join(self.config.working_dir, 'test'))
        self.save_data(valid_df, os.path.join(
            self.config.working_dir, 'valid'))

        train_gen, valid_gen, test_gen, test_batch_size, test_steps = self.create_generators(
            train_df, test_df, valid_df)

        classes = list(train_gen.class_indices.keys())
        class_indices = list(train_gen.class_indices.values())
        class_count = len(classes)
        labels = test_gen.labels
        logger.info('test batch size: %s, test steps: %s, number of classes : %s',
                    test_batch_size, test_steps, class_count)
        logger.info('{0:^25s}{1:^12s}'.format('class name', 'class index'))
        for klass, index in zip(classes, class_indices):
            logger.info(f'{klass:^25s}{str(index):^12s}')

        logger.info("Data ingestion completed successfully.")
        return train_df, test_df, valid_df, train_gen, valid_gen, test_gen, test_batch_size, test_steps
