import cv2
import numpy as np
from typing import List, Tuple
from ..models.data_models import Player, MatchFrame
from ..config.config import FieldConfig
import matplotlib.pyplot as plt
import logging

class Visualizer:
    def __init__(self, config: FieldConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.field_template = self._create_field_template()
        self.colors = {
            'team1': (255, 0, 0),    # Kırmızı
            'team2': (0, 0, 255),    # Mavi
            'unknown': (128, 128, 128)  # Gri
        }

    def _create_field_template(self) -> np.ndarray:
        """2D futbol sahası şablonu oluştur"""
        # Piksel cinsinden saha boyutları
        width_px = 800
        height_px = int(width_px * (self.config.height_meters / self.config.width_meters))
        
        # Boş saha oluştur
        field = np.ones((height_px, width_px, 3), dtype=np.uint8) * 50  # Koyu yeşil
        
        # Saha çizgilerini çiz
        self._draw_field_lines(field)
        
        return field

    def _draw_field_lines(self, field: np.ndarray) -> None:
        """Saha çizgilerini çiz"""
        h, w = field.shape[:2]
        
        # Kenar çizgileri
        cv2.rectangle(field, (0, 0), (w-1, h-1), (255, 255, 255), 2)
        
        # Orta çizgi
        cv2.line(field, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        
        # Orta daire
        center = (w//2, h//2)
        radius = int(9.15 / self.config.width_meters * w)
        cv2.circle(field, center, radius, (255, 255, 255), 2)
        
        # Ceza sahaları
        penalty_width = int(self.config.penalty_area_width / self.config.width_meters * w)
        penalty_height = int(self.config.penalty_area_height / self.config.height_meters * h)
        
        # Sol ceza sahası
        cv2.rectangle(field, (0, h//2-penalty_height//2), 
                     (penalty_width, h//2+penalty_height//2), (255, 255, 255), 2)
        
        # Sağ ceza sahası
        cv2.rectangle(field, (w-penalty_width, h//2-penalty_height//2), 
                     (w-1, h//2+penalty_height//2), (255, 255, 255), 2)

    def draw_frame(self, frame: MatchFrame) -> np.ndarray:
        """Maç karesini görselleştir"""
        try:
            result = self.field_template.copy()
            
            # Her oyuncuyu çiz
            for player in frame.players:
                if player.position_2d is None:
                    continue
                    
                # Koordinatları piksel koordinatlarına dönüştür
                x, y = self._convert_to_pixels(player.position_2d)
                
                # Oyuncu rengini belirle
                color = self.colors.get(player.team, self.colors['unknown'])
                
                # Oyuncuyu çiz
                cv2.circle(result, (int(x), int(y)), 5, color, -1)
                
                # Numarayı yaz
                if player.number:
                    cv2.putText(result, str(player.number), 
                              (int(x+10), int(y+10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (255, 255, 255), 1)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error drawing frame: {str(e)}")
            return self.field_template.copy()

    def _convert_to_pixels(self, coords: Tuple[float, float]) -> Tuple[int, int]:
        """Gerçek koordinatları piksel koordinatlarına dönüştür"""
        h, w = self.field_template.shape[:2]
        x = int(coords[0] / self.config.width_meters * w)
        y = int(coords[1] / self.config.height_meters * h)
        return x, y

    def save_visualization(self, image: np.ndarray, path: str) -> None:
        """Görselleştirmeyi kaydet"""
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            self.logger.error(f"Error saving visualization: {str(e)}") 