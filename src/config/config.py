from dataclasses import dataclass
from typing import Tuple, Dict, Any
import yaml

@dataclass
class ModelConfig:
    yolo_model_path: str = "models/weights/yolov5s.pt"
    confidence_threshold: float = 0.5
    device: str = "cuda"  # veya "cpu"

@dataclass
class ImageConfig:
    min_resolution: Tuple[int, int] = (720, 1280)
    max_resolution: Tuple[int, int] = (1080, 1920)
    supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')

@dataclass
class OCRConfig:
    languages: Tuple[str, ...] = ('en',)
    number_min_confidence: float = 0.7
    number_max_length: int = 2

@dataclass
class FieldConfig:
    width_meters: float = 105.0
    height_meters: float = 68.0
    penalty_area_width: float = 40.32
    penalty_area_height: float = 16.5

@dataclass
class SystemConfig:
    model: ModelConfig = ModelConfig()
    image: ImageConfig = ImageConfig()
    ocr: OCRConfig = OCRConfig()
    field: FieldConfig = FieldConfig()
    max_players: int = 22
    debug_mode: bool = False

class ConfigManager:
    def __init__(self, config_path: str = None):
        self.config = SystemConfig()
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            self._update_config(config_dict)

    def _update_config(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def save_config(self, config_path: str) -> None:
        config_dict = {
            'model': self.config.model.__dict__,
            'image': self.config.image.__dict__,
            'ocr': self.config.ocr.__dict__,
            'field': self.config.field.__dict__,
            'max_players': self.config.max_players,
            'debug_mode': self.config.debug_mode
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

    def get_config(self) -> SystemConfig:
        return self.config 