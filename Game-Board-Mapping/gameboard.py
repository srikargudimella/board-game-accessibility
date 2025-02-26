import cv2
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from referencemap import ReferenceMap, GameSquare, get_center
from dataclasses import dataclass
import time

@dataclass
class GamePiece:
    name: str
    box: list[int]
    center: list[int]
    current_square: int
    auto_controlled: bool

class GameBoard:

    def __init__(self, reference_image_path, annotations_path, pieces=[], human_controlled_pieces=[]):
        """
        Initialize with reference image and XML annotations
        
        Args:
            reference_image_path: Path to the reference Candyland board image
            annotations_path: Path to the XML file containing square coordinates
            pieces: list of piece names that are auto controlled
        """

        self.reference_map = ReferenceMap(reference_image_path, annotations_path)
        
        # Store reference corner coordinates (clockwise from top-left)
        self.corner_detector = YOLO('corner_detector.pt')
        self.piece_detector = YOLO('game_piece_detector.pt')
        self.pieces = {}
        for piece in pieces:
            self.pieces[piece] = GamePiece(name=piece, box=[], center=[], current_square=0, auto_controlled=False)
        for piece in human_controlled_pieces:
            self.pieces[piece].auto_controlled = True

    def _detect_corners(self, camera_image):
        results = self.corner_detector.predict(camera_image, show=False, verbose=False)
        corners = results[0].boxes.xyxy.numpy()
        self.corner_boxes = corners
        # Calculate centers for each detected corner box
        centers = []
        for box in corners:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append(([center_x, center_y]))
        
        centers = np.float32(centers)
        # print("centers", centers)
        # if len(centers) != 4:
        #     raise ValueError("Error: Detected corners do not match expected number of corners")
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        centers = sorted(centers, key=lambda point: (point[1], point[0]))  # Sort by y first, then x
        # print("sorted centers", centers)
        top_row = sorted(centers[:2], key=lambda point: point[0]) or []  # Sort top two by x
        # print("top row", top_row)
        bottom_row = sorted(centers[2:], key=lambda point: point[0]) or []  # Sort bottom two by x
        if bottom_row:
            bottom_row.reverse()
        # print("bottom row", bottom_row)
        top_row.extend(bottom_row)

        ordered_corners = np.array(top_row)
        # print("ordered corners", ordered_corners)
        return ordered_corners    

    def _get_game_square_coordinates(self, camera_corners, camera_image):
        """
        Get the coordinates of all game squares in the camera image
        
        Args:
            camera_image: Current frame from the overhead camera
        Returns:
            transformed_coordinates: Dictionary mapping square IDs to their box coordinates
        """
        # Hardcoded camera image corners (clockwise from top-left)
        # Calculate homography
        H, _ = cv2.findHomography(self.reference_map.reference_corners, camera_corners)
        
        # Transform all reference coordinates to camera coordinates
        camera_squares = []
        for square in self.reference_map.reference_squares:
            polygon = square.points
            
            # Convert polygon points to homogeneous coordinates
            square_points =  np.float32([[x, y, 1] for x, y in polygon])
            
            # Transform each point using homography
            transformed_points = []
            for point in square_points:
                transformed_point = np.dot(H, point)
                transformed_point = transformed_point / transformed_point[2]
                transformed_points.append(transformed_point[:2])
            transformed_center = get_center(transformed_points)
            camera_square = GameSquare(
                id= square.id,
                name= square.name,
                points= transformed_points,
                center= transformed_center,
                color= square.color,
                contains_dragon=square.contains_dragon,
                landing_square= square.landing_square
            )

            camera_squares.append(camera_square)
            

        return camera_squares
    

    
    def _piece_in_square(self, piece_box, square):
        
        return True
    
    def _update_piece_location(self, piece_name, piece_box):
        if piece_name in self.pieces: 
            self.pieces[piece_name].box = piece_box
            # Calculate center directly from box coordinates
            center_x = (piece_box[0] + piece_box[2]) / 2
            center_y = (piece_box[1] + piece_box[3]) / 2
            self.pieces[piece_name].center = [center_x, center_y]
            return True
        # else:
        #     # Calculate center directly from box coordinates
        #     center_x = (piece_box[0] + piece_box[2]) / 2
        #     center_y = (piece_box[1] + piece_box[3]) / 2
        #     piece = GamePiece(name=piece_name, box=piece_box, center=[center_x, center_y], current_square=-1, auto_controlled=False)
        #     self.pieces[piece_name] = piece
        #     return True


    def _detect_pieces(self, camera_image):
        results = self.piece_detector.predict(camera_image, verbose=False)
        piece_classes = results[0].boxes.cls.numpy()
        names = results[0].names
        piece_names = [names[int(cls)] for cls in piece_classes]
        piece_boxes = results[0].boxes.xyxy.numpy()

        for piece_name, piece_box in zip(piece_names, piece_boxes):

            self._update_piece_location(piece_name, piece_box)


    
    def update_game_board(self, camera_image):
        # cv2.imshow('camera_image', camera_image)
        # cv2.waitKey(0)  # Display frame for 1ms instead of waiting for keypress
        # cv2.destroyAllWindows()

        self.latest_game_frame = camera_image
        self.corners = self._detect_corners(camera_image)
        if len(self.corners) != 4:
            raise ValueError("Error: Detected corners do not match expected number of corners")
        self.game_squares = self._get_game_square_coordinates( self.corners, camera_image)
        self._detect_pieces(camera_image)

    def _find_next_color_square(self, curr_square_id, color, index):
        # print(f"Finding next {color} square at index {index} starting from square {curr_square_id}")
        num_seen = 0
        # First check the current square
        if self.reference_map.reference_squares[curr_square_id].color == color:
            num_seen += 1
          
        while num_seen < index:
            curr_square_id += 1
            # print(f"Checking square {curr_square_id} and square: {self.reference_map.reference_squares[curr_square_id]}")
            if self.reference_map.reference_squares[curr_square_id].color == color:
                num_seen += 1
        return curr_square_id
    
    def generate_path(self, piece_name, target_square_id):
        """
        Generate a path from current piece position to target square that follows the game board
        and avoids other pieces.
        
        Args:
            piece_name: Name of the piece to move
            target_square_id: ID of the target square
        
        Returns:
            List of (x,y) coordinates representing waypoints along the path
        """
        current_square = self.pieces[piece_name].current_square
        path_points = []
        
        # Add starting position
        path_points.append(self.pieces[piece_name].center)
        
        # Generate intermediate waypoints through squares between current and target
        current_id = current_square
        while current_id < target_square_id:
            current_id += 1
            # Use direct indexing with bounds check
            if 0 <= current_id < len(self.game_squares):
                square = self.game_squares[current_id]
                path_points.append(square.center)
        
        # Add target position
        if 0 <= target_square_id < len(self.game_squares):
            target_square = self.game_squares[target_square_id]
            # print(f"Target square: {target_square}")
            path_points.append(target_square.center)
            if target_square.landing_square != target_square_id:
                # print(f"Landing square: {target_square.landing_square}")
                path_points.append(self.game_squares[target_square.landing_square].center)
        # print(f"Path points: {path_points}")
        # Adjust path points to avoid other pieces
        if len(self.pieces[piece_name].box) > 0:
            adjusted_path = self._adjust_path_for_collisions(path_points, piece_name)
        else:
            adjusted_path = path_points
        
        return adjusted_path

    def _calculate_avoidance_points(self, start, end, obstacle_center, box):
        """
        Calculate points to avoid obstacle by following its corners and edges.
        Returns two points that define the path around the obstacle.
        """
        start = np.array(start)
        end = np.array(end)
        
        # Define box corners (clockwise from top-left)
        corners = np.array([
            [box[0], box[1]],  # top-left (0)
            [box[2], box[1]],  # top-right (1)
            [box[2], box[3]],  # bottom-right (2)
            [box[0], box[3]]   # bottom-left (3)
        ])
        
        # Calculate path direction and perpendicular vector
        path_vector = end - start
        path_direction = path_vector / np.linalg.norm(path_vector)
        perp_vector = np.array([-path_direction[1], path_direction[0]])
        
        # Determine which side of the path the obstacle center is on
        to_obstacle = np.array(obstacle_center) - start
        side = np.dot(to_obstacle, perp_vector)
        
        # Fixed clearance distance
        clearance = 40  # pixels
        
        # Determine primary movement direction and select corners
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > dy:  # Primarily horizontal movement
            if path_vector[0] > 0:  # Moving right
                if side > 0:  # Obstacle is above
                    entry_corner = corners[3]  # bottom-left
                    exit_corner = corners[2]   # bottom-right
                else:  # Obstacle is below
                    entry_corner = corners[0]  # top-left
                    exit_corner = corners[1]   # top-right
            else:  # Moving left
                if side > 0:  # Obstacle is above
                    entry_corner = corners[2]  # bottom-right
                    exit_corner = corners[3]   # bottom-left
                else:  # Obstacle is below
                    entry_corner = corners[1]  # top-right
                    exit_corner = corners[0]   # top-left
        else:  # Primarily vertical movement
            if path_vector[1] > 0:  # Moving down
                if side > 0:  # Obstacle is to the right
                    entry_corner = corners[3]  # bottom-left
                    exit_corner = corners[0]   # top-left
                else:  # Obstacle is to the left
                    entry_corner = corners[2]  # bottom-right
                    exit_corner = corners[1]   # top-right
            else:  # Moving up
                if side > 0:  # Obstacle is to the right
                    entry_corner = corners[0]  # top-left
                    exit_corner = corners[3]   # bottom-left
                else:  # Obstacle is to the left
                    entry_corner = corners[1]  # top-right
                    exit_corner = corners[2]   # bottom-right
        
        # Calculate offset vector based on movement direction
        if dx > dy:  # Horizontal movement
            offset_vector = np.array([0, clearance if side > 0 else -clearance])
        else:  # Vertical movement
            offset_vector = np.array([clearance if side > 0 else -clearance, 0])
        
        # Calculate avoidance points with offset
        avoidance_point1 = entry_corner + offset_vector
        avoidance_point2 = exit_corner + offset_vector
        
        return avoidance_point1, avoidance_point2

    def _adjust_path_for_collisions(self, path_points, moving_piece_name):
        """
        Adjust path points to avoid collisions with other pieces.
        """
        adjusted_path = []
        adjusted_path.append(path_points[0])
        
        for i in range(len(path_points) - 1):
            start = np.array(path_points[i])
            end = np.array(path_points[i + 1])
            
            # Check for collisions with other pieces
            segment_points = [start]
            intersect_found = False
            for piece_name, piece in self.pieces.items():
                if piece_name == moving_piece_name:
                    continue
                if len(piece.box) < 4:
                    continue
                # print(f"Checking collision with piece: {piece}")
                # Create a slightly larger bounding box for clearance
                margin = 20  # pixels
                box = [
                    piece.box[0] - margin,
                    piece.box[1] - margin,
                    piece.box[2] + margin,
                    piece.box[3] + margin
                ]
                
                # Check if path segment intersects with piece bounding box
                if self._line_intersects_box(start, end, box):
                    intersect_found = True
                    # Calculate avoidance points
                    avoidance_points = self._calculate_avoidance_points(start, end, piece.center, box)
                    if avoidance_points:
                        segment_points.extend(avoidance_points)
                        path_points[i+1] = segment_points[-1]
            
            if not intersect_found:
                segment_points.append(end)
            adjusted_path.extend(segment_points[1:])  # Don't add start point again
        
        return adjusted_path

    def _line_intersects_box(self, start, end, box):
        """Check if line segment intersects with bounding box."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # Box corners
        box_corners = [
            [box[0], box[1]],  # top-left
            [box[2], box[1]],  # top-right
            [box[2], box[3]],  # bottom-right
            [box[0], box[3]]   # bottom-left
        ]
        
        # Check intersection with each edge of the box
        for i in range(4):
            box_start = box_corners[i]
            box_end = box_corners[(i + 1) % 4]
            
            # Check if the lines intersect
            if (ccw(start, box_start, box_end) != ccw(end, box_start, box_end) and
                ccw(start, end, box_start) != ccw(start, end, box_end)):
                return True
        
        return False

    def find_target_square(self, piece_name, card_name):
        current_square = self.pieces[piece_name].current_square
        if card_name == "JaqGus":
            return 20
        elif card_name == "Sebastian":
            return 32
        elif card_name == "Lumiere":
            return 51
        elif card_name == "Dopey":
            return 66
        elif card_name == "Gus":
            return 53
        elif card_name == "Jaq":
            return 54
        else:
            num_square = int(card_name.split("_")[0])
            card_color = card_name.split("_")[1].capitalize()
            print(f"Finding target square for {piece_name} with card {card_name} on square {current_square}")
            return self._find_next_color_square(current_square+1, card_color, num_square)
    
    def update_piece_position(self, piece_name, target_square_id):
        self.pieces[piece_name].current_square = target_square_id
        return True
    
    def show_game_board_with_annotations(self):
        """
        Visualize boxes on the image with labels
        
        Args:
            image: Image to draw on
            transformed_coordinates: Dictionary mapping square IDs to their corner coordinates
            color: BGR color tuple (default: red)
            thickness: Line thickness in pixels
        Returns:
            Image with boxes and labels drawn
        """
        annotated_frame = self.annotate_frame(self.latest_game_frame)

        cv2.namedWindow('Annotated Gameboard', cv2.WINDOW_NORMAL)  # Create resizable window
        cv2.resizeWindow('Annotated Gameboard', 880, 560)  # Set window size to 640x360
        resized_frame = cv2.resize(annotated_frame, (880, 560))  # Resize the frame to 640x360

        cv2.imshow('Annotated Gameboard', resized_frame)
        cv2.waitKey(2000)
        time.sleep(0.2)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def annotate_frame(self, frame):
        img_copy = frame.copy()
        for square in self.game_squares:
            # Get the points for the current square
            points = square.points  # Access the points from the GameSquare
            
            red = (0, 0, 255)
            purple = (255, 0, 255)
            yellow = (0, 255, 255)  
            blue = (255, 0, 0)
            orange = (0, 165, 255)
            green = (0, 255, 0)

            if square.color == "Yellow":
                color = yellow
            elif square.color == "Green":
                color = green
            elif square.color == "Blue":
                color = blue
            elif square.color == "Purple":
                color = purple
            elif square.color == "Orange":
                color = orange  
            elif square.color == "Red":
                color = red
            else:
                color = (255,255,255)

            if square.contains_dragon:
                text = square.name
            elif square.color not in ["Yellow", "Green", "Purple", "Orange", "Red", "Blue"]:
                text = square.name
            else:
                text = str(square.id)
            # Draw the polygon outline in white
            cv2.polylines(img_copy, [np.int32(points)], isClosed=True, color=(0, 0, 0), thickness=4)  # White outline
            
            # Draw the polygon interior in the specified color
            cv2.polylines(img_copy, [np.int32(points)], isClosed=True, color=color, thickness=2)  # Colored interior            
            # Draw the annotated coordinates on the image
            cv2.circle(img_copy, (int(square.center[0]), int(square.center[1])), 5, green, -1)  # Green dot for coordinates
            
            # Draw a black rectangle behind the text for highlighting
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = int(square.center[0]) - text_size[0] // 2
            text_y = int(square.center[1]) - 10 - text_size[1] // 2
            cv2.rectangle(img_copy, (text_x - 5, text_y - text_size[1] - 5), 
                           (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)  # Black rectangle
            
            cv2.putText(img_copy, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)            
        
        for piece, game_piece in self.pieces.items():      
            box = game_piece.box
            if len(box) == 0:
                continue
            cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 5)  # Fixed alignment by removing the stray '0'
            text_size = cv2.getTextSize(str(piece), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = int(box[0]) + (int(box[2]) - int(box[0])) // 2 - text_size[0] // 2  # Centering the text horizontally
            text_y = int(box[1]) - 10 - text_size[1] // 2
            cv2.rectangle(img_copy, (text_x - 5, text_y - text_size[1] - 5), 
                           (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)  # Black rectangle

            cv2.putText(img_copy, str(piece), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)  # Updated to use text_x for alignment
        return img_copy

    def plot_path(self, path_points, frame=None):
        """
        Plot the generated path on the game board image.
        
        Args:
            path_points: List of (x,y) coordinates representing the path
            frame: Optional frame to plot on. If None, uses latest game frame
        
        Returns:
            Frame with plotted path
        """
        if frame is None:
            frame = self.latest_game_frame.copy()
        else:
            frame = frame.copy()
        
        # First draw the regular annotations
        frame = self.annotate_frame(frame)
        
        # Convert path points to integer coordinates for drawing
        path_points = np.array(path_points, dtype=np.int32)
        
        # Draw lines connecting path points
        for i in range(len(path_points) - 1):
            start = tuple(path_points[i])
            end = tuple(path_points[i + 1])
            
            # Draw thick white line as background
            cv2.line(frame, start, end, (255, 255, 255), 4)
            # Draw thinner blue line on top
            cv2.line(frame, start, end, (255, 165, 0), 2)
            
            # Draw circles at waypoints
            cv2.circle(frame, start, 5, (0, 255, 0), -1)  # Green filled circle
        
        # Draw circle at final point
        if len(path_points) > 0:
            cv2.circle(frame, tuple(path_points[-1]), 5, (0, 255, 0), -1)
        
        return frame

    def show_path(self, path_points):
        """
        Display the path visualization in a window.
        
        Args:
            path_points: List of (x,y) coordinates representing the path
        """
        frame = self.plot_path(path_points)
        cv2.namedWindow('Path Visualization', cv2.WINDOW_NORMAL)  # Create resizable window
        cv2.resizeWindow('Path Visualization', 880, 560)  # Set window size to 640x360

        resized_frame = cv2.resize(frame, (880, 560))  # Resize the frame to 640x360
        cv2.imshow('Path Visualization', resized_frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        
