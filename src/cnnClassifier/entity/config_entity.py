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
