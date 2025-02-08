import easyocr
import numpy as np
from typing import List, Tuple, Optional
import cv2
from ..models.data_models import Player
from ..config.config import OCRConfig
import logging

class NumberReader:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reader = self._initialize_reader()
        self.min_confidence = 0.6

    def _initialize_reader(self):
        try:
            return easyocr.Reader(self.config.languages)
        except Exception as e:
            self.logger.error(f"Error initializing EasyOCR: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better number recognition.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Noise removal
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def read_number(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """
        Read jersey number from the given player bounding box.
        
        Args:
            image (np.ndarray): Full image
            bbox (Tuple[int, int, int, int]): Player bounding box (x1, y1, x2, y2)
            
        Returns:
            Optional[int]: Detected jersey number or None if not found
        """
        try:
            x1, y1, x2, y2 = bbox
            player_img = image[y1:y2, x1:x2]
            
            # Preprocess the cropped image
            processed_img = self.preprocess_image(player_img)
            
            # Read text
            results = self.reader.readtext(processed_img)
            
            # Filter and process results
            numbers = []
            for (bbox, text, prob) in results:
                if prob >= self.min_confidence:
                    # Try to convert text to number
                    try:
                        num = int(''.join(filter(str.isdigit, text)))
                        if 1 <= num <= 99:  # Valid jersey numbers range
                            numbers.append((num, prob))
                    except ValueError:
                        continue
            
            # Return the number with highest confidence
            if numbers:
                numbers.sort(key=lambda x: x[1], reverse=True)
                return numbers[0][0]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error reading number: {str(e)}")
            return None

    def read_multiple_numbers(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[int]]:
        """
        Read jersey numbers for multiple players.
        
        Args:
            image (np.ndarray): Full image
            bboxes (List[Tuple[int, int, int, int]]): List of player bounding boxes
            
        Returns:
            List[Optional[int]]: List of detected jersey numbers
        """
        return [self.read_number(image, bbox) for bbox in bboxes]

    def read_numbers(self, image: np.ndarray, players: List[Player]) -> List[Player]:
        for player in players:
            try:
                number = self.read_number(image, player.bbox)
                player.number = number
            except Exception as e:
                self.logger.warning(f"Error reading number for player {player.id}: {str(e)}")
                continue
        return players 