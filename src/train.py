import argparse
import logging
import os
import cv2
import json
import numpy as np
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.models.self_learning_detector import SelfLearningDetector
from src.config.config import ConfigManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train football analysis model')
    parser.add_argument('--labeled_dir', type=str, required=True,
                       help='Directory containing labeled training images')
    parser.add_argument('--unlabeled_dir', type=str, required=True,
                       help='Directory containing unlabeled test images')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    return parser.parse_args()

def process_labeled_data(detector: SelfLearningDetector, labeled_dir: str, output_dir: str):
    """Etiketli verilerden öğren"""
    logger = logging.getLogger(__name__)
    logger.info("Learning from labeled data...")
    
    # Etiketli görüntüleri işle
    labeled_results = []
    image_files = [f for f in os.listdir(labeled_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(labeled_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Görüntüden öğren
        patterns = detector.learn_from_labeled_image(image)
        
        # Sonuçları topla
        labeled_results.append({
            'image': img_file,
            'patterns': patterns,
            'total_patterns': len(patterns)
        })
        
        logger.info(f"Learned from {img_file}: Found {len(patterns)} patterns")
    
    # Öğrenilen desenleri JSON olarak kaydet
    with open(os.path.join(output_dir, 'learned_patterns.json'), 'w') as f:
        json.dump(labeled_results, f, indent=2)
    
    return detector

def process_unlabeled_data(detector: SelfLearningDetector, unlabeled_dir: str, output_dir: str):
    """Etiketsiz verileri işle"""
    logger = logging.getLogger(__name__)
    logger.info("Processing unlabeled data...")
    
    # Etiketsiz görüntüleri işle
    unlabeled_results = []
    image_files = [f for f in os.listdir(unlabeled_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(unlabeled_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Öğrenilen desenleri kullanarak tespit yap
        detections, annotated_image = detector.detect_using_learned_patterns(image)
        
        # Sonuçları kaydet
        result_path = os.path.join(output_dir, f'detected_{img_file}')
        cv2.imwrite(result_path, annotated_image)
        
        # Sonuçları topla
        unlabeled_results.append({
            'image': img_file,
            'detections': detections,
            'total_players': len(detections)
        })
        
        logger.info(f"Processed {img_file}: Found {len(detections)} players")
    
    # Tespit sonuçlarını JSON olarak kaydet
    with open(os.path.join(output_dir, 'detection_results.json'), 'w') as f:
        json.dump(unlabeled_results, f, indent=2)

def main():
    # Setup
    logger = setup_logging()
    args = parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize detector
        detector = SelfLearningDetector(config.model)
        
        # Etiketli verilerden öğren
        detector = process_labeled_data(detector, args.labeled_dir, args.output)
        
        # Etiketsiz verileri işle
        process_unlabeled_data(detector, args.unlabeled_dir, args.output)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 