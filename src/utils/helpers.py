import cv2
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

def load_image(path: str) -> np.ndarray:
    """Görüntüyü yükle ve kontrol et"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    return image

def detect_field_corners(image: np.ndarray) -> List[Tuple[float, float]]:
    """Saha köşelerini otomatik tespit et"""
    # Bu fonksiyon geliştirilecek
    # Şu an için manuel girdi gerekiyor
    raise NotImplementedError("Automatic field corner detection not implemented yet")

def calculate_iou(box1: Tuple[float, float, float, float], 
                 box2: Tuple[float, float, float, float]) -> float:
    """İki bounding box arasındaki IoU hesapla"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Görüntü kalitesini artır"""
    try:
        # Kontrast artırma
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Gürültü azaltma
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced)

        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image 