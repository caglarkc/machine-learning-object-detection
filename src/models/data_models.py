from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1

@dataclass
class Player:
    id: int
    bbox: BoundingBox
    number: Optional[str] = None
    team: Optional[str] = None
    position_2d: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    
    def to_dict(self):
        return {
            'id': self.id,
            'number': self.number,
            'position': self.position_2d,
            'team': self.team,
            'confidence': self.confidence
        }

@dataclass
class MatchFrame:
    frame_id: int
    timestamp: datetime
    players: List[Player]
    original_image: np.ndarray
    processed_image: Optional[np.ndarray] = None
    
    def get_team_positions(self, team: str) -> List[Tuple[float, float]]:
        return [p.position_2d for p in self.players if p.team == team]
    
    def get_player_by_number(self, number: str) -> Optional[Player]:
        for player in self.players:
            if player.number == number:
                return player
        return None 