import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch
from ultralytics import YOLO
import logging
import sys
import os

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config.config import ModelConfig

class SelfLearningDetector:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = self._initialize_model()
        self.confidence_threshold = 0.2  # Daha da düşük güven skoru
        self.min_player_height = 50  # Minimum oyuncu yüksekliği (piksel)
        self.max_player_height = 400  # Maximum oyuncu yüksekliği (piksel)
        
    def _initialize_model(self):
        """Initialize YOLO model"""
        try:
            model = YOLO('yolov8n.pt')
            return model
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Gelişmiş görüntü ön işleme"""
        try:
            # Boyut kontrolü ve yeniden boyutlandırma
            height, width = image.shape[:2]
            if width > 1920:  # Full HD'den büyükse küçült
                scale = 1920 / width
                new_width = 1920
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            # Kontrastı artır (CLAHE)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # Gürültü azaltma
            denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

            # Keskinleştirme
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # Renk doygunluğunu artır
            hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)  # Doygunluğu %20 artır
            final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return final

        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {str(e)}")
            return image

    def detect_and_learn(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Gelişmiş insan tespiti"""
        try:
            # Görüntü ön işleme
            processed_image = self._preprocess_image(image)
            
            # Çoklu ölçek tespiti
            detections = []
            scales = [1.0, 0.8, 1.2]  # Farklı ölçekler
            
            for scale in scales:
                # Görüntüyü ölçekle
                if scale != 1.0:
                    height, width = processed_image.shape[:2]
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    scaled_image = cv2.resize(processed_image, (new_width, new_height))
                else:
                    scaled_image = processed_image

                # YOLO tespiti
                results = self.model(scaled_image, classes=[0], verbose=False)
                current_detections = self._process_results(results)
                
                # Ölçek düzeltmesi
                if scale != 1.0:
                    for det in current_detections:
                        x1, y1, x2, y2 = det['bbox']
                        det['bbox'] = (
                            int(x1 / scale),
                            int(y1 / scale),
                            int(x2 / scale),
                            int(y2 / scale)
                        )
                
                detections.extend(current_detections)
            
            # Tespitleri iyileştir
            refined_detections = self._refine_all_detections(detections)
            
            # Görselleştirme
            annotated_img = image.copy()
            self._draw_detections(annotated_img, refined_detections)
            
            return refined_detections, annotated_img
            
        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}")
            return [], image

    def _process_results(self, results) -> List[Dict]:
        """YOLO sonuçlarını işle"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf)
                    
                    # Boyut kontrolü
                    height = y2 - y1
                    if self.min_player_height <= height <= self.max_player_height:
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'height': height
                        }
                        detections.append(detection)
        
        return detections

    def _refine_all_detections(self, detections: List[Dict]) -> List[Dict]:
        """Tüm tespitleri iyileştir"""
        # Önce NMS uygula
        nms_detections = self._apply_nms(detections)
        
        # Sonra örtüşmeleri kontrol et
        refined_detections = self._refine_overlapping_detections(nms_detections)
        
        # Boy oranı kontrolü
        valid_detections = []
        for det in refined_detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width
            
            # İnsan için makul boy oranı kontrolü (1.5 ile 4.0 arası)
            if 1.5 <= aspect_ratio <= 4.0:
                valid_detections.append(det)
        
        return valid_detections

    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Non-Maximum Suppression uygula"""
        if not detections:
            return []

        # Güven skoruna göre sırala
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        kept_detections = []
        while detections:
            best = detections.pop(0)
            kept_detections.append(best)
            
            detections = [
                det for det in detections
                if self._calculate_iou(best['bbox'], det['bbox']) < iou_threshold
            ]
        
        return kept_detections

    def _refine_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """Örtüşen tespitleri iyileştir"""
        refined_detections = []
        
        # Boyuta göre sırala
        sorted_detections = sorted(detections, 
                                 key=lambda x: x['height'],
                                 reverse=True)
        
        for det in sorted_detections:
            should_add = True
            x1, y1, x2, y2 = det['bbox']
            
            for ref_det in refined_detections:
                rx1, ry1, rx2, ry2 = ref_det['bbox']
                
                # Dikey örtüşme kontrolü
                vertical_overlap = min(y2, ry2) - max(y1, ry1)
                min_height = min(y2-y1, ry2-ry1)
                
                if vertical_overlap > 0.7 * min_height:  # Dikey örtüşme çok fazlaysa
                    # Yatay mesafe kontrolü
                    horizontal_distance = min(abs(x1-rx2), abs(rx1-x2))
                    if horizontal_distance < 0.3 * min(x2-x1, rx2-rx1):
                        should_add = False
                        break
            
            if should_add:
                refined_detections.append(det)
        
        return refined_detections

    def _draw_detections(self, image: np.ndarray, detections: List[Dict]):
        """Tespitleri görselleştir"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Güven skoruna göre renk değiştir (kırmızı->yeşil)
            color = (
                0,
                int(255 * conf),  # Yeşil kanal
                int(255 * (1-conf))  # Kırmızı kanal
            )
            
            # Dikdörtgen çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Güven skorunu yaz
            label = f"Person: {conf:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """IoU hesapla"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0 