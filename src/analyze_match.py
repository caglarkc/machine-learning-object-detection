import cv2
import argparse
import logging
import sys
import os
import json

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.self_learning_detector import SelfLearningDetector
from src.config.config import ConfigManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze football match')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to match image')
    parser.add_argument('--config', type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml'),
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--feedback', type=str,
                       help='Path to feedback JSON file with error locations')
    return parser.parse_args()

def load_feedback(feedback_path: str):
    """Geri bildirim dosyasını yükle"""
    try:
        with open(feedback_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading feedback file: {str(e)}")
        return None

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
        
        # Read image
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f"Could not read image: {args.image}")
        
        # Process image
        logger.info("Processing image...")
        detections, annotated_image = detector.detect_and_learn(image)
        
        # Geri bildirim varsa uygula
        if args.feedback and os.path.exists(args.feedback):
            feedback = load_feedback(args.feedback)
            if feedback and 'error_locations' in feedback:
                logger.info("Applying feedback and reprocessing...")
                for error_loc in feedback['error_locations']:
                    detections, annotated_image = detector.apply_feedback(
                        image, detections, (error_loc['x'], error_loc['y'])
                    )
        
        # Save results
        output_path = os.path.join(args.output, 'result.jpg')
        cv2.imwrite(output_path, annotated_image)
        
        # Save detections
        detections_path = os.path.join(args.output, 'detections.json')
        with open(detections_path, 'w') as f:
            json.dump({
                'total_detections': len(detections),
                'detections': detections,
                'confidence_threshold': detector.confidence_threshold
            }, f, indent=2)
        
        # Print results
        logger.info(f"Found {len(detections)} players")
        for i, det in enumerate(detections):
            logger.info(f"Player {i+1}: Confidence {det['confidence']:.2f}")
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Detections saved to {detections_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 