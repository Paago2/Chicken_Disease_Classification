from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source: str
    source_csv: str
    local_data_file: Path
    unzip_dir: Path
    train_size: float = None
    test_size: float = None
    validation_size: float = None
    random_state: int = None
    max_samples: int = None
    min_samples: int = None
    img_size: list = None
    working_dir: str = None
    batch_size: int = None
    color_mode: str = 'rgb'
    class_mode: str = 'categorical'
    horizontal_flip: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    zoom_range: float = 0.2


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_model_name: str
    params_pooling: str
    params_dropout_rate1: float
    params_dropout_rate2: float
    params_dense_1024_regularizer_l2: float
    params_dense_1024_regularizer_l1: float
    params_dense_128_regularizer_l2: float
    params_dense_128_regularizer_l1: float
