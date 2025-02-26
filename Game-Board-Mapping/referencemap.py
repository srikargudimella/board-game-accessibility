import cv2
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from shapely.geometry import Polygon, box

from typing import NamedTuple

class GameSquare(NamedTuple):
    """
    Represents a square on the game board.
    
    Attributes:
        id (int): Unique identifier for the square
        name (str): Descriptive name
        points (list): Corner coordinates defining the square
        center (tuple): Center coordinates (x,y)
        color (str): Color of the square
        contains_dragon (bool): Whether square has dragon
        landing_square (int): Final square ID after any special moves
    """
    id: int
    name: str
    points: list
    center: tuple
    color: str
    contains_dragon: bool
    landing_square: int

def get_center(points):
    """
    Calculate center point from list of coordinates.
    
    Args:
        points (list): List of (x,y) coordinates
        
    Returns:
        tuple: Center point (x,y)
    """
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (center_x, center_y) 


class ReferenceMap:
    """
    Manages the reference game board image and square annotations.
    """

    def __init__(self, reference_image_path, annotations_path):
        """
        Initialize with reference image and annotations.
        
        Args:
            reference_image_path (str): Path to reference board image
            annotations_path (str): Path to XML annotations file
        """
        self.reference_image = cv2.imread(reference_image_path)
        self.reference_squares = self._load_annotations(annotations_path)
        self.reference_corners = np.float32([
            [0, 0],               # top-left
            [2883, 0],           # top-right
            [2883, 2550],        # bottom-right
            [0, 2550]            # bottom-left
        ])

    def _load_annotations(self, annotations_path):
        """
        Load and parse square annotations from XML file.
        
        Args:
            annotations_path (str): Path to XML file
            
        Returns:
            list: List of GameSquare objects
        """
        tree = ET.parse(annotations_path)
        root = tree.getroot()
        
        coordinates = {}
        self.original_boxes = {}  # Store original box information
        
        game_squares = []

        for idx, polygon in enumerate(root.findall(".//polygon")):

            label = polygon.get('label')
            points = polygon.get('points')
            
            # Convert points string to list of coordinates
            # Points are typically stored as "x1,y1;x2,y2;x3,y3;..."
            point_pairs = points.split(';')
            coordinates = []
            for pair in point_pairs:
                x, y = map(float, pair.split(','))
                coordinates.append((x, y))
            square_color = label.split("_")[0]
            square_name = f'{idx}_{label}'
            square_center = get_center(coordinates)
            if square_name == "2_Yellow_Square":
                landing_square = 33
            elif square_name == "17_Green_square":
                landing_square = 27
            else:
                landing_square = idx
            contains_dragon = "Dragon" in square_name
            game_square = GameSquare(idx, square_name, coordinates, square_center, square_color, contains_dragon, landing_square)
            game_squares.append(game_square)

        return game_squares
    

    def show_map_with_annotations(self):
        """
        Display reference map with annotated squares and coordinates.
        """
        img_copy = self.reference_image.copy()
        
        for square in self.reference_squares:
            # Get the points for the current square
            points = square.points  # Access the points from the GameSquare
            
            # Draw the polygon for the square
            cv2.polylines(img_copy, [np.int32(points)], isClosed=True, color=(255, 255, 255), thickness=4)  # Green polygon
            
            # Draw the annotated coordinates on the image
            cv2.circle(img_copy, (int(square.center[0]), int(square.center[1])), 5, (0, 255, 0), -1)  # Green dot for coordinates
            
            # Add label for the square
            cv2.putText(img_copy, square.name, (int(square.center[0]), int(square.center[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Annotated Reference Map', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

