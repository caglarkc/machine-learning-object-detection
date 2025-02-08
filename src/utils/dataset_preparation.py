import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
import shutil
from pathlib import Path
import yaml

class DatasetPreparation:
    def __init__(self, config_path: str = 'config/dataset_config.yaml'):
        self.config = self._load_config(config_path)
        self.classes = ['player', 'ball', 'referee']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load dataset configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_image(self, image_path: str, output_path: str) -> Tuple[str, List[Dict]]:
        """
        Prepare single image and its annotations
        
        Args:
            image_path: Path to source image
            output_path: Path to save processed image
            
        Returns:
            Tuple of (saved image path, list of annotations)
        """
        # Read and resize image if needed
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Resize if needed
        if self.config['resize_images']:
            image = cv2.resize(image, 
                             (self.config['image_width'], 
                              self.config['image_height']))
            
        # Save processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        
        return output_path

    def create_yolo_dataset(self, source_dir: str, split_ratio: float = 0.8):
        """
        Create YOLO format dataset from source images
        
        Args:
            source_dir: Directory containing source images and annotations
            split_ratio: Train/validation split ratio
        """
        # Create dataset directories
        dataset_path = Path('data/dataset')
        train_img_dir = dataset_path / 'images' / 'train'
        val_img_dir = dataset_path / 'images' / 'val'
        train_label_dir = dataset_path / 'labels' / 'train'
        val_label_dir = dataset_path / 'labels' / 'val'
        
        # Create directories if they don't exist
        for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(Path(source_dir).glob('*.jpg')) + list(Path(source_dir).glob('*.png'))
        np.random.shuffle(image_files)
        
        # Split into train and validation
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Process training images
        print("Processing training images...")
        for img_path in train_files:
            self._process_image_and_annotation(img_path, train_img_dir, train_label_dir)
            
        # Process validation images
        print("Processing validation images...")
        for img_path in val_files:
            self._process_image_and_annotation(img_path, val_img_dir, val_label_dir)
        
        # Create dataset YAML file
        self._create_dataset_yaml(dataset_path)
        
        print(f"Dataset created: {len(train_files)} training images, {len(val_files)} validation images")

    def _process_image_and_annotation(self, img_path: Path, img_dir: Path, label_dir: Path):
        """Process single image and its annotation"""
        # Copy image
        shutil.copy(img_path, img_dir / img_path.name)
        
        # Process annotation if exists
        ann_path = img_path.with_suffix('.txt')
        if ann_path.exists():
            shutil.copy(ann_path, label_dir / ann_path.name)
        else:
            print(f"Warning: No annotation file for {img_path}")

    def _create_dataset_yaml(self, dataset_path: Path):
        """Create YAML file for YOLO training"""
        yaml_content = {
            'path': str(dataset_path.absolute()),
            'train': str(dataset_path / 'images' / 'train'),
            'val': str(dataset_path / 'images' / 'val'),
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        with open(dataset_path / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f)

    def create_annotation(self, image_path: str, boxes: List[Dict], output_path: str):
        """
        Create YOLO format annotation file
        
        Args:
            image_path: Path to image
            boxes: List of bounding boxes
            output_path: Path to save annotation file
        """
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        with open(output_path, 'w') as f:
            for box in boxes:
                # Convert to YOLO format
                x_center = (box['x1'] + box['x2']) / (2 * width)
                y_center = (box['y1'] + box['y2']) / (2 * height)
                w = (box['x2'] - box['x1']) / width
                h = (box['y2'] - box['y1']) / height
                
                # Write to file
                class_id = self.classes.index(box['class'])
                f.write(f"{class_id} {x_center} {y_center} {w} {h}\\n") 