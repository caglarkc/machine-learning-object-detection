import numpy as np
import cv2
from typing import List, Tuple, Optional
from ..models.data_models import Player, BoundingBox
from ..config.config import FieldConfig
import logging

class CoordinateTransformer:
    def __init__(self, config: FieldConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transformation_matrix = None
        
        # Output image dimensions for 2D view (10 pixels per meter)
        self.output_width = int(self.config.width_meters * 10)
        self.output_height = int(self.config.height_meters * 10)

    def detect_field_lines(self, image: np.ndarray) -> np.ndarray:
        """Detect field lines in the image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get white lines
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            kernel = np.ones((3,3), np.uint8)
            lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return lines
        except Exception as e:
            self.logger.error(f"Error detecting field lines: {str(e)}")
            return None

    def calculate_transformation_matrix(self, image: np.ndarray, field_corners: Optional[List[Tuple[float, float]]] = None) -> None:
        """Calculate perspective transformation matrix."""
        try:
            if field_corners is None:
                # Try to automatically detect field corners using line detection
                lines = self.detect_field_lines(image)
                if lines is None:
                    raise ValueError("Could not detect field lines")
                
                lsd = cv2.createLineSegmentDetector(0)
                detected_lines, _, _, _ = lsd.detect(lines)
                
                if detected_lines is None:
                    raise ValueError("Could not detect line segments")
                
                # Use detected lines to estimate corners
                field_corners = self._estimate_corners_from_lines(detected_lines)
            
            if len(field_corners) != 4:
                raise ValueError("Exactly 4 field corners are required")

            # Real world coordinates (in meters)
            real_corners = np.float32([
                [0, 0],
                [self.config.width_meters, 0],
                [self.config.width_meters, self.config.height_meters],
                [0, self.config.height_meters]
            ])

            # Image coordinates
            image_corners = np.float32(field_corners)
            
            self.transformation_matrix = cv2.getPerspectiveTransform(image_corners, real_corners)
            
        except Exception as e:
            self.logger.error(f"Error calculating transformation matrix: {str(e)}")
            raise

    def _estimate_corners_from_lines(self, lines: np.ndarray) -> List[Tuple[float, float]]:
        """Estimate field corners from detected lines."""
        # This is a simplified implementation
        # In a real system, you'd need more sophisticated corner detection
        corners = [
            [lines[0][0][0], lines[0][0][1]],  # Top-left
            [lines[0][0][2], lines[0][0][1]],  # Top-right
            [lines[0][0][2], lines[0][0][3]],  # Bottom-right
            [lines[0][0][0], lines[0][0][3]]   # Bottom-left
        ]
        return corners

    def transform_coordinates(self, players: List[Player]) -> List[Player]:
        """Transform player coordinates to 2D field coordinates."""
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix not calculated")

        for player in players:
            try:
                # Get player's foot position (bottom center of bounding box)
                foot_position = (
                    (player.bbox.x1 + player.bbox.x2) / 2,
                    player.bbox.y2
                )

                # Transform coordinates
                transformed_point = cv2.perspectiveTransform(
                    np.array([[foot_position]], dtype=np.float32),
                    self.transformation_matrix
                )

                player.position_2d = (
                    float(transformed_point[0][0][0]),
                    float(transformed_point[0][0][1])
                )

            except Exception as e:
                self.logger.error(f"Error transforming coordinates for player {player.id}: {str(e)}")
                player.position_2d = None

        return players

    def validate_coordinates(self, players: List[Player]) -> List[Player]:
        """Validate transformed coordinates are within field boundaries."""
        valid_players = []
        for player in players:
            if player.position_2d is None:
                continue

            x, y = player.position_2d
            if (0 <= x <= self.config.width_meters and 
                0 <= y <= self.config.height_meters):
                valid_players.append(player)
            else:
                self.logger.warning(f"Player {player.id} coordinates out of bounds")

        return valid_players

    def create_field_template(self) -> np.ndarray:
        """Create a 2D football field template."""
        # Create blank image
        field = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        # Draw field (green background)
        field[:, :] = (0, 128, 0)  # Dark green
        
        # Draw field lines (white)
        # Field border
        cv2.rectangle(field, (0, 0), (self.output_width-1, self.output_height-1), (255, 255, 255), 2)
        
        # Center line
        cv2.line(field, 
                 (self.output_width//2, 0),
                 (self.output_width//2, self.output_height),
                 (255, 255, 255), 2)
        
        # Center circle
        center = (self.output_width//2, self.output_height//2)
        radius = int(9.15 * 10)  # 9.15m radius in pixels
        cv2.circle(field, center, radius, (255, 255, 255), 2)
        
        # Penalty areas
        penalty_width = int(16.5 * 10)  # 16.5m in pixels
        penalty_height = int(40.32 * 10)  # 40.32m in pixels
        
        # Left penalty area
        cv2.rectangle(field, 
                     (0, self.output_height//2 - penalty_height//2),
                     (penalty_width, self.output_height//2 + penalty_height//2),
                     (255, 255, 255), 2)
        
        # Right penalty area
        cv2.rectangle(field,
                     (self.output_width - penalty_width, self.output_height//2 - penalty_height//2),
                     (self.output_width - 1, self.output_height//2 + penalty_height//2),
                     (255, 255, 255), 2)
        
        return field 