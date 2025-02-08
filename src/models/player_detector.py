import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict
from ..models.data_models import Player, BoundingBox
from ..config.config import ModelConfig
import logging

class PlayerDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize the player detector with YOLO model."""
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5
        self.logger = logging.getLogger(__name__)

    def detect_players(self, image: np.ndarray) -> List[Dict]:
        """
        Detect players in the image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Dict]: List of detected players with their bounding boxes and confidence scores
        """
        results = self.model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id
                    }
                    detections.append(detection)
        
        return detections

    def filter_overlapping_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Filter overlapping detections using Non-Maximum Suppression.
        
        Args:
            detections (List[Dict]): List of detected players
            iou_threshold (float): IoU threshold for filtering
            
        Returns:
            List[Dict]: Filtered detections
        """
        if not detections:
            return []

        # Convert to format suitable for NMS
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Perform NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            iou_threshold
        )
        
        return [detections[i] for i in indices.flatten()]

    def get_player_positions(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Get center positions of detected players.
        
        Args:
            detections (List[Dict]): List of detected players
            
        Returns:
            List[Tuple[int, int]]: List of (x, y) coordinates for player centers
        """
        positions = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            positions.append((center_x, center_y))
        
        return positions

    def detect(self, image: np.ndarray) -> List[Player]:
        try:
            results = self.model(image)
            players = []
            
            for idx, detection in enumerate(results.xyxy[0]):
                if detection[4] >= self.confidence_threshold:
                    bbox = BoundingBox(
                        x1=float(detection[0]),
                        y1=float(detection[1]),
                        x2=float(detection[2]),
                        y2=float(detection[3])
                    )
                    
                    player = Player(
                        id=idx,
                        bbox=bbox,
                        confidence=float(detection[4])
                    )
                    players.append(player)
            
            return players
        
        except Exception as e:
            self.logger.error(f"Error during player detection: {str(e)}")
            raise

    def _filter_overlapping_detections(self, players: List[Player]) -> List[Player]:
        """Üst üste binen tespitleri filtrele"""
        filtered_players = []
        for i, player1 in enumerate(players):
            should_add = True
            for player2 in filtered_players:
                if self._calculate_iou(player1.bbox, player2.bbox) > 0.5:
                    should_add = False
                    break
            if should_add:
                filtered_players.append(player1)
        return filtered_players

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """İki bounding box arasındaki IoU hesapla"""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0 