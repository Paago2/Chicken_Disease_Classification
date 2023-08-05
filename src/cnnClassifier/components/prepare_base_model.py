from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from keras.applications import EfficientNetB5
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adamax
from keras import regularizers
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def prepare_base_model(self):
        img_shape = (*self.config.params_image_size, 3)
        base_model = EfficientNetB5(include_top=self.config.params_include_top,
                                    weights=self.config.params_weights,
                                    input_shape=img_shape,
                                    pooling=self.config.params_pooling)

        base_model.trainable = True

        x = base_model.output
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Dense(1024,
                  kernel_regularizer=regularizers.l2(
                      l=self.config.params_dense_1024_regularizer_l2),
                  activity_regularizer=regularizers.l1(
                      self.config.params_dense_1024_regularizer_l1),
                  bias_regularizer=regularizers.l1(
                      self.config.params_dense_1024_regularizer_l1),
                  activation='relu')(x)
        x = Dropout(rate=self.config.params_dropout_rate1, seed=123)(x)
        x = Dense(128,
                  kernel_regularizer=regularizers.l2(
                      l=self.config.params_dense_128_regularizer_l2),
                  activity_regularizer=regularizers.l1(
                      self.config.params_dense_128_regularizer_l1),
                  bias_regularizer=regularizers.l1(
                      self.config.params_dense_128_regularizer_l1),
                  activation='relu')(x)
        x = Dropout(rate=self.config.params_dropout_rate2, seed=123)(x)

        output = Dense(self.config.params_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)

        model.compile(Adamax(learning_rate=self.config.params_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def get_base_model(self):
        model = self.prepare_base_model()
        model.save(self.config.base_model_path)

    def update_base_model(self):
        model = tf.keras.models.load_model(self.config.base_model_path)
        model.save(self.config.updated_base_model_path)
