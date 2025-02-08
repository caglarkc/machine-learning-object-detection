import cv2
import numpy as np
from typing import Optional
from datetime import datetime
import logging
from ..config.config import SystemConfig
from ..models.player_detector import PlayerDetector
from ..models.number_reader import NumberReader
from ..models.coordinate_transformer import CoordinateTransformer
from ..visualization.visualizer import Visualizer
from ..models.data_models import MatchFrame

class FootballAnalyzer:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # Alt sistemleri başlat
        self.player_detector = PlayerDetector(config.model)
        self.number_reader = NumberReader(config.ocr)
        self.coordinate_transformer = CoordinateTransformer(config.field)
        self.visualizer = Visualizer(config.field)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def process_image(self, image_path: str, field_corners: Optional[list] = None) -> Optional[MatchFrame]:
        """Görüntüyü işle ve sonuçları döndür"""
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image could not be loaded")

            # Görüntü boyutlarını kontrol et
            if not self._validate_image_size(image):
                image = self._resize_image(image)

            # Oyuncuları tespit et
            players = self.player_detector.detect(image)
            self.logger.info(f"Detected {len(players)} players")

            # Numaraları oku
            players = self.number_reader.read_numbers(image, players)
            self.logger.info("Completed number reading")

            # Saha köşeleri verildiyse koordinat dönüşümünü yap
            if field_corners:
                self.coordinate_transformer.calculate_transformation_matrix(image, field_corners)
                players = self.coordinate_transformer.transform_coordinates(players)
                players = self.coordinate_transformer.validate_coordinates(players)
                self.logger.info("Completed coordinate transformation")

            # MatchFrame oluştur
            frame = MatchFrame(
                frame_id=hash(image_path),
                timestamp=datetime.now(),
                players=players,
                original_image=image
            )

            # Görselleştirme
            if self.config.debug_mode:
                visualization = self.visualizer.draw_frame(frame)
                frame.processed_image = visualization
                self.logger.info("Created visualization")

            return frame

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None

    def _validate_image_size(self, image: np.ndarray) -> bool:
        """Görüntü boyutlarının minimum gereksinimleri karşılayıp karşılamadığını kontrol et"""
        height, width = image.shape[:2]
        min_height, min_width = self.config.image.min_resolution
        return height >= min_height and width >= min_width

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü minimum boyutlara yeniden boyutlandır"""
        min_height, min_width = self.config.image.min_resolution
        height, width = image.shape[:2]
        
        # En-boy oranını koru
        aspect_ratio = width / height
        new_width = max(min_width, width)
        new_height = int(new_width / aspect_ratio)
        
        if new_height < min_height:
            new_height = min_height
            new_width = int(new_height * aspect_ratio)
            
        return cv2.resize(image, (new_width, new_height))

    def save_results(self, frame: MatchFrame, output_dir: str) -> None:
        """Sonuçları kaydet"""
        try:
            # Görselleştirmeyi kaydet
            if frame.processed_image is not None:
                visualization_path = f"{output_dir}/visualization_{frame.frame_id}.jpg"
                self.visualizer.save_visualization(frame.processed_image, visualization_path)
                self.logger.info(f"Saved visualization to {visualization_path}")

            # Oyuncu verilerini JSON olarak kaydet
            import json
            player_data = [player.to_dict() for player in frame.players]
            json_path = f"{output_dir}/players_{frame.frame_id}.json"
            with open(json_path, 'w') as f:
                json.dump(player_data, f, indent=4)
            self.logger.info(f"Saved player data to {json_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}") 