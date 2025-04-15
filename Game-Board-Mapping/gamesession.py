import cv2
from gameboard import GameBoard
import tkinter as tk
from tkinter import ttk, messagebox
import random
from carddetection.card_color_detection import process_card
import time
import json
import sys
import socket
import pickle
import struct
import asyncio
import websockets
import base64
import io
import numpy as np
# import pygame  # Commented out pygame import
import serial  # Uncommented serial import
import keyboard  # Import the keyboard library

show_cv = True

X_MAX = 340
Y_MAX = 315
class GameSession:
    """
    Manages a game session including webcam input, piece tracking, and turn management.
    Handles communication with the frontend client through websockets.
    """

    def __init__(self, reference_map_path, annotations_path, config_file=None, enable_streaming=False):
        """
        Initialize a new game session.

        Args:
            reference_map_path (str): Path to the reference game board image
            annotations_path (str): Path to XML file containing board annotations
            config_file (str, optional): Path to JSON config file for game setup
            enable_streaming (bool): Whether to enable websocket streaming to frontend
        """
        # Initialize webcam
        self.overhead_cam = cv2.VideoCapture(1)
        # self.overhead_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.overhead_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        if not self.overhead_cam.isOpened():
            raise Exception("Error: Could not open webcam")
        
        self.card_reader_cam = cv2.VideoCapture(0)
        if not self.card_reader_cam.isOpened():
            raise Exception("Error: Could not open card reader webcam")
        
        # Initialize serial connection to CNC controller
        try:
            self.serial_conn = serial.Serial(
                port='/dev/cu.usbmodem2101',  # or 'COM3' on Windows
                baudrate=115200,
                timeout=1,
                writeTimeout=1
            )
            # Wait for Grbl to initialize
            time.sleep(2)
            self.serial_conn.flushInput()
            print("Serial connection to CNC controller established")
            
            # Home the controller
            # Read initial position report before homing
            # self._send_g_code_command(b'?\n')  # Request position report
            # time.sleep(0.5)  # Wait for response
            
            # print("Position before homing:")
            # while self.serial_conn.in_waiting > 0:
            #     response = self.serial_conn.readline().decode().strip()
            #     print(f"  {response}")
            self._send_g_code_command(b'$H\n')           
            # self._send_g_code_command(b'$X\n')

            # self._send_g_code_command(b'?\n')  # Request position report
            # time.sleep(0.5)  # Wait for response
            
            # print("Position before homing:")
            # while self.serial_conn.in_waiting > 0:
            #     response = self.serial_conn.readline().decode().strip()
            #     print(f"  {response}")

            
            # Set units to millimeters and absolute positioning
            self._send_g_code_command(b'G21\n')  # Set units to millimeters
            
            # Set work coordinates
            move_to_origin_string = f"G91 G0 X-{X_MAX} Y0\n"
            self._send_g_code_command(move_to_origin_string.encode())  # Move to position
            self._send_g_code_command(b'G10 L20 P1 X0 Y0 Z0\n')  # Set work coordinates
                          
        except serial.SerialException as e:
            print(f"Warning: Could not open serial port: {e}")
            self.serial_conn = None
        
        self.r_pressed = False
        self.m_pressed = False
        # Create popup window for piece selection or load from config
        self.pieces = []
        self.assisted_controlled_input = []
        if config_file:
            self._load_config(config_file)
        else:
            self._show_setup_popup()
            
        self.game_board = GameBoard(reference_map_path, annotations_path, self.pieces, self.assisted_controlled_input)
        self.game_board.corner_detector.predict(source=0, stream=True, show=True)
        self.turn_number = 0
        self.stream_enabled = enable_streaming
        if enable_streaming:
            self.websocket_server = None
            self.websocket_clients = set()
            # Start websocket server in a separate thread
            import threading
            self.ws_thread = threading.Thread(target=self._start_websocket_server)
            self.ws_thread.daemon = True
            self.ws_thread.start()
        else:
            self.client_socket = None
        # pygame.init()  # Commented out pygame initialization
        # self.screen = pygame.display.set_mode((640, 480))  # Commented out window creation
        # pygame.display.set_caption('Game Session')  # Commented out window title setting

    def _load_config(self, config_file):
        """
        Load game configuration from a JSON file.
        
        Args:
            config_file (str): Path to JSON config file containing piece and player settings
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Validate config structure
            required_keys = ['pieces', 'assisted_controlled']
            if not all(key in config for key in required_keys):
                raise ValueError("Config file must contain 'pieces' and 'assisted_controlled' keys")
                
            # Validate number of players (1-4)
            if not 1 <= len(config['pieces']) <= 4:
                raise ValueError("Number of players must be between 1 and 4")
                
            # Validate number of assisted players (0-2)
            if not 0 <= len(config['assisted_controlled']) <= 2:
                raise ValueError("Number of assisted controlled players must be between 0 and 2")
                
            # Validate that assisted_controlled pieces exist in pieces list
            if not all(piece in config['pieces'] for piece in config['assisted_controlled']):
                raise ValueError("All assisted controlled pieces must be in the pieces list")
                
            # Validate piece uniqueness
            if len(set(config['pieces'])) != len(config['pieces']):
                raise ValueError("Each piece can only be selected once")
                
            self.pieces = config['pieces']
            self.assisted_controlled_input = config['assisted_controlled']
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in config file: {config_file}")

    def _show_setup_popup(self):
        """
        Display initial game setup window for selecting number of players.
        Creates a Tkinter window for user input.
        """
        try:
            selection_window = tk.Toplevel()
        except:
            root = tk.Tk()
            root.withdraw()
            selection_window = tk.Toplevel()
        
        selection_window.title("Game Setup")
        selection_window.geometry("300x150")
        selection_window.transient()
        selection_window.grab_set()
        
        # First screen - Number of players selection
        ttk.Label(selection_window, 
                 text="How many players? (1-4)",
                 wraplength=250).pack(pady=10)
        
        num_players_var = tk.StringVar()
        num_players_combo = ttk.Combobox(selection_window, 
                                       textvariable=num_players_var,
                                       values=["1", "2", "3", "4"],
                                       state="readonly")
        num_players_combo.pack(pady=10)
        num_players_combo.set("2")  # Default value
        
        def show_piece_selection():
            num_players = int(num_players_var.get())
            selection_window.destroy()
            self._show_player_piece_selection(num_players)
        
        ttk.Button(selection_window, text="Next", 
                  command=show_piece_selection).pack(pady=20)
        
        selection_window.wait_window(selection_window)

    def _show_player_piece_selection(self, num_players):
        """
        Display window for selecting game pieces for each player.
        
        Args:
            num_players (int): Number of players in the game
        """
        piece_window = tk.Toplevel()
        piece_window.title("Select Player Pieces")
        piece_window.geometry("400x400")
        piece_window.transient()
        piece_window.grab_set()
        
        ttk.Label(piece_window, 
                 text="Select piece for each player:",
                 wraplength=350).pack(pady=10)
        
        # Available pieces
        piece_options = ["Purple Square", "Striped Triangle", "Green Square", "Yellow Circle"]
        player_vars = []
        
        # Create dropdown for each player
        for i in range(num_players):
            frame = ttk.Frame(piece_window)
            frame.pack(pady=5, padx=10, fill="x")
            
            ttk.Label(frame, text=f"Player {i + 1}:").pack(side="left", padx=5)
            player_var = tk.StringVar()
            player_vars.append(player_var)
            
            dropdown = ttk.Combobox(frame, 
                                  textvariable=player_var,
                                  values=piece_options,
                                  state="readonly")
            dropdown.pack(side="left", padx=5, fill="x", expand=True)
        
        def validate_and_close():
            # Get selected pieces
            selected_pieces = [var.get() for var in player_vars]
            
            # Check if all players selected a piece
            if "" in selected_pieces:
                messagebox.showerror("Error", "Please select a piece for each player")
                return
            
            # Check for duplicate selections
            if len(set(selected_pieces)) != len(selected_pieces):
                messagebox.showerror("Error", "Each piece can only be selected once")
                return
            
            # Store pieces in order of play
            self.pieces = selected_pieces
            piece_window.destroy()
            self._show_control_selection(num_players)
        
        ttk.Button(piece_window, text="Next", 
                  command=validate_and_close).pack(pady=20)
        
        piece_window.wait_window(piece_window)

    def _show_control_selection(self, num_players):
        """
        Display window for selecting which players are assisted controlled.
        Maximum of 2 players can be assisted controlled.
        
        Args:
            num_players (int): Number of players in the game
        """
        control_window = tk.Toplevel()
        control_window.title("Player Control Selection")
        control_window.geometry("400x400")
        control_window.transient()
        control_window.grab_set()
        
        ttk.Label(control_window, 
                 text="Select which players are assisted controlled (max 2):",
                 wraplength=350).pack(pady=10)
        
        # Create checkboxes for each player
        assisted_controls = []
        for i in range(num_players):
            var = tk.BooleanVar()
            assisted_controls.append(var)
            ttk.Checkbutton(control_window, 
                           text=f"Player {i + 1} ({self.pieces[i]})",
                           variable=var).pack(pady=5)
        
        def validate_and_close():
            # Count selected human players
            assisted_players = sum(var.get() for var in assisted_controls)
            
            if assisted_players > 2:
                messagebox.showerror("Error", "Please select at most 2 assisted players")
                return
            
            # Store human-controlled piece names
            self.assisted_controlled_input = [self.pieces[i] for i, var in enumerate(assisted_controls) if var.get()]
            control_window.destroy()
        
        ttk.Button(control_window, text="Start Game", 
                  command=validate_and_close).pack(pady=20)
        
        control_window.wait_window(control_window)

    def generate_move_instructions(self, piece_name, target_square):
        """
        Generate path instructions for moving a piece to target square.
        
        Args:
            piece_name (str): Name of the piece to move
            target_square (int): ID of the target square
            
        Returns:
            list: List of path points for piece movement
        """
        # TO DO: Add code to generate move instructions
        path_points = self.game_board.generate_path(piece_name, target_square)
        if show_cv:
            self.game_board.show_path(path_points)
        return path_points

    def _generate_front_end_path(self, piece_name, target_square):
        """
        Generate path information formatted for frontend visualization.
        
        Args:
            piece_name (str): Name of the piece to move
            target_square (int): ID of the target square
            
        Returns:
            tuple: (path_points, landing_square) where path_points is list of coordinates
                  and landing_square is final square ID after any special moves
        """
        front_end_path = []
        for square in self.game_board.reference_map.reference_squares[self.game_board.pieces[piece_name].current_square : target_square+1]:
            front_end_path.append([int(square.center[0]), int(square.center[1])])   
        if (self.game_board.reference_map.reference_squares[target_square].landing_square != target_square):
            landing_square = self.game_board.reference_map.reference_squares[target_square].landing_square
            front_end_path.append([int(self.game_board.reference_map.reference_squares[landing_square].center[0]), int(self.game_board.reference_map.reference_squares[landing_square].center[1])])
        else:
            landing_square = target_square
        return front_end_path, landing_square
    
    def execute_move(self, move_instructions, frame):
        """
        Execute physical movement of a game piece using the CNC XY plotter.
        Transforms image coordinates to physical coordinates and generates G-code.
        
        Args:
            move_instructions (list): List of path points in image coordinates
        """
        if not move_instructions or len(move_instructions) < 2:
            print("Warning: No valid move instructions provided")
            return
        print("move_instructions", move_instructions)
        # Transform image coordinates to physical coordinates
        physical_coordinates = self._transform_to_physical_coordinates(move_instructions, frame)
        
        # Generate G-code for the movement path
        gcode_path = self._generate_gcode(physical_coordinates)
        
        # Send G-code to the CNC controller
        self._send_gcode_to_controller(gcode_path)
        
    def _transform_to_physical_coordinates(self, image_coordinates, frame):
        """
        Transform image coordinates to physical coordinates using calibration pattern.
        
        Args:
            image_coordinates (list): List of [x,y] coordinates in image space
            frame: Current camera frame
            
        Returns:
            list: List of [x,y] coordinates in physical space (mm)
        """
        print("transform_to_physical_coordinates")
        # Detect calibration pattern in the current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern_size = (3, 38)  # Number of inner corners in the calibration pattern
        
        # Try to find the chessboard corners
        print("finding chessboard corners")
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if not ret:
            print("Warning: Could not detect calibration pattern")
            # Fallback to a simple scaling factor if pattern not detected
            return self._apply_simple_scaling(image_coordinates)
        print("detected chessboard corners")
        # Visualize detected corners for debugging
        if ret and show_cv:
            img_with_corners = frame.copy()
            cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, ret)
            cv2.imshow('Detected Corners', img_with_corners)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        
        # Known physical dimensions of the calibration pattern (in mm)
        square_size = 10  # Size of each square in mm
        
        # Create array of physical coordinates for the pattern
        # The corners are ordered row by row, so we need to match that ordering
        physical_corners = np.zeros((pattern_size[0] * pattern_size[1], 2), np.float32)
        for i in range(pattern_size[1]):  # rows
            for j in range(pattern_size[0]):  # columns
                # Calculate index in the flattened array
                idx = i * pattern_size[0] + j
                # Assign physical coordinates (in mm)
                physical_corners[idx] = [j * square_size - 90 , i * square_size - 60]
        # Extract detected corner points correctly
        detected_points = np.array([corner[0] for corner in corners])
        if detected_points[0][1] > detected_points[-1][1]:
            detected_points= detected_points[::-1]
        for idx, point in enumerate(detected_points):
            print("Image point:", point, " - physical point:", physical_corners[idx])
        # Calculate homography matrix
        H, status = cv2.findHomography(detected_points, physical_corners, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Failed to calculate homography matrix")
            return self._apply_simple_scaling(image_coordinates)
            
        # For debugging, transform a few test points to verify the homography
        test_points = [
            [0, 0],  # Origin
            [frame.shape[1], 0],  # Top right
            [0, frame.shape[0]],  # Bottom left
            [frame.shape[1], frame.shape[0]]  # Bottom right
        ]
        print("Test image points:", test_points)
        test_physical = []
        for point in test_points:
            homogeneous = np.array([point[0], point[1], 1])
            transformed = np.dot(H, homogeneous)
            physical = transformed[:2] / transformed[2]
            test_physical.append(physical)
        print("Test physical points:", test_physical) 

        
        # Transform image coordinates to physical coordinates
        physical_coords = []
        for point in image_coordinates:
            # Convert to homogeneous coordinates
            homogeneous = np.array([point[0], point[1], 1])
            # Apply transformation
            transformed = np.dot(H, homogeneous)
            # Convert back from homogeneous coordinates
            physical = transformed[:2] / transformed[2]
            physical[0] = max(1, min(physical[0], X_MAX))
            physical[1] = max(1, min(physical[1], Y_MAX))
            physical_coords.append(physical)
            
        print("Image coordinates:", image_coordinates)
        print("Physical coordinates:", physical_coords)
        
        return physical_coords
    
    def _apply_simple_scaling(self, image_coordinates):
        """
        Apply a simple scaling transformation as fallback when calibration pattern isn't detected.
        
        Args:
            image_coordinates (list): List of [x,y] coordinates in image space
            
        Returns:
            list: List of [x,y] coordinates in physical space (mm)
        """
        # Estimated physical dimensions of the game board in mm
        physical_width = 500  # Adjust based on your actual board size
        physical_height = 500
        
        # Image dimensions
        image_height, image_width = self.game_board.latest_game_frame.shape[:2]
        
        # Calculate scaling factors
        scale_x = physical_width / image_width
        scale_y = physical_height / image_height
        
        # Apply scaling
        physical_coords = []
        for point in image_coordinates:
            physical_x = point[0] * scale_x
            physical_y = point[1] * scale_y
            physical_coords.append([physical_x, physical_y])
            
        return physical_coords
    
    def _generate_gcode(self, physical_coordinates):
        """
        Generate G-code for the CNC plotter to follow the path.
        
        Args:
            physical_coordinates (list): List of [x,y] coordinates in physical space (mm)
            
        Returns:
            str: G-code commands for the movement
        """
        gcode = []
        
        # Add header - just set units and coordinate mode
        gcode.append("G21 ; Set units to millimeters")
        gcode.append("G54 ; Select work coordinate system")
        
        # Since the controller is already attached to the piece,
        # we can start moving directly through the waypoints
        for i, point in enumerate(physical_coordinates):
            # First point might need a different speed
            if i == 0:
                gcode.append(f"G90 G1 X{point[0]:.2f} Y{point[1]:.2f} F3000 ; Move to waypoint {i}")
            else:
                gcode.append(f"G90 G1 X{point[0]:.2f} Y{point[1]:.2f} F2000 ; Move to waypoint {i}")
        
        
        return "\n".join(gcode)
    
    def _send_g_code_command(self, cmd):
        """
        Send a G-code command to the CNC controller and wait for 'ok' response.
        
        Args:
            cmd (bytes): Command to send to the controller
            
        Returns:
            bool: True if response received, False otherwise
        """
        if self.serial_conn is None or not self.serial_conn.is_open:
            print("Warning: Serial connection not available")
            return False
            
        self.serial_conn.write(cmd)
        print(f"Sent: {cmd.decode().strip()}")
        
        # Wait for response with timeout
        response_received = False
        start_time = time.time()
        timeout = 60  # 60 seconds timeout for response
        
        while not response_received and time.time() - start_time < timeout:
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode().strip()
                
                if response.startswith('error:'):
                    print(f"Error from Grbl: {response}")
                    response_received = True
                elif response == 'ok':
                    print("Received: ok")
                    response_received = True
                else:
                    # Log other responses (like position reports)
                    print(f"Grbl: {response}")
            else:
                # Small delay to prevent CPU hogging while waiting for response
                time.sleep(0.01)
        
        if not response_received:
            print(f"Warning: No response from Grbl for command: {cmd.decode().strip()}")
        
        return response_received

    def _send_gcode_to_controller(self, gcode):
        """
        Send G-code to the CNC controller line by line using serial communication.
        Implements the simple send-response protocol for Grbl.
        Also saves G-code to file for debugging.
        
        Args:
            gcode (str): G-code commands to send
        """
        # Save G-code to file for debugging
        gcode_file = "move_path.gcode"
        with open(gcode_file, "w") as f:
            f.write(gcode)
        
        print(f"G-code saved to {gcode_file}")
        
        # Function to establish/reestablish serial connection
        def ensure_connection():
            if self.serial_conn is None or not self.serial_conn.is_open:
                try:
                    # If connection exists but is closed, try to reopen
                    if self.serial_conn is not None:
                        try:
                            self.serial_conn.open()
                            print("Reopened existing serial connection")
                        except (serial.SerialException, ValueError):
                            # If reopening fails, create a new connection
                            self.serial_conn = None
                
                    # If no connection exists, create a new one
                    if self.serial_conn is None:
                        self.serial_conn = serial.Serial(
                            port='/dev/ttyUSB0',  # or 'COM3' on Windows
                            baudrate=115200,
                            timeout=1,
                            writeTimeout=1
                        )
                        print("Established new serial connection")
                
                    # Wait for Grbl to initialize and flush startup text
                    time.sleep(2)
                    self.serial_conn.flushInput()
                    return True
                except serial.SerialException as e:
                    print(f"Error establishing serial connection: {e}")
                    self.serial_conn = None
                    return False
            return True  # Connection already exists and is open
        
        # Ensure we have a valid connection before proceeding
        if not ensure_connection():
            print("Warning: Could not establish serial connection. G-code not sent.")
            return
        
        try:
            # Split gcode into lines and remove empty lines and comments
            lines = [line.strip() for line in gcode.splitlines()]
            lines = [line for line in lines if line and not line.startswith(';')]
            
            for line in lines:
                # Check connection before sending each line
                if not ensure_connection():
                    print("Serial connection lost and could not be re-established")
                    break
                
                # Send g-code line using the _send_g_code_command method
                cmd = (line.strip() + '\n').encode()
                success = self._send_g_code_command(cmd)
                
                if not success:
                    print(f"Failed to get response for command: {line}")
                    # Try to reestablish connection for next command
                    ensure_connection()
                
        except Exception as e:
            print(f"Error during G-code transmission: {e}")
            import traceback
            traceback.print_exc()

    def complete_turn(self, piece_name):
        """
        Execute a complete turn for the given piece including:
        - Drawing a card
        - Finding target square
        - Moving piece
        - Updating game state
        
        Args:
            piece_name (str): Name of the piece taking the turn
        """
        self.update_game_state()
        self._stream_game_state()
        if show_cv:
            self.game_board.show_game_board_with_annotations()
        time.sleep(0.4)
        card_name = self.draw_card()
        target_square = self.game_board.find_target_square(piece_name, card_name)
        front_end_path, landing_square = self._generate_front_end_path(piece_name, target_square)
        self._stream_message("card_drawn", {"card_name": card_name, "path_to_target": front_end_path, "landing_square": self.game_board.reference_map.reference_squares[landing_square].points})
        if self.game_board.pieces[piece_name].auto_controlled:
            print("Waiting for player to hit m to move...")
            # self._stream_message("target_square_selected", {"id": target_square, "center": self.game_board.reference_map.reference_squares[target_square].center, "points": self.game_board.reference_map.reference_squares[target_square].points})
            # self._stream_message("player_move_prompt", "Waiting for player to hit m to move...")
            move_instructions = self.generate_move_instructions(piece_name, target_square)
            while True:
                # print("self.m_pressed", self.m_pressed)
                # for event in pygame.event.get():  # Commented out pygame event handling
                #     if event.type == pygame.KEYDOWN:
                #         print(event.key)
                #         if event.key == pygame.K_m:
                #             self.m_pressed = True
                if keyboard.is_pressed('m'):  # Use keyboard library for 'm' key press
                    self.m_pressed = True
                if self.m_pressed:
                    print("m pressed")
                    self.m_pressed = False
                    break
        else:
            time.sleep(15)
        # print(self.game_board.pieces[piece_name].current_square)
        print("landing_square", landing_square)
        self.game_board.update_piece_position(piece_name, landing_square)
        self.execute_move(move_instructions, self.game_board.latest_game_frame)

        self.turn_number += 1
        self._stream_message("turn_complete", "Turn complete")
        self.update_game_state()
        if show_cv:
            self.game_board.show_game_board_with_annotations()

        self._stream_game_state()
    
    
    def update_game_state(self):
        """
        Update internal game state by processing latest webcam frame.
        Detects board corners and piece positions.
        
        Raises:
            ValueError: If board corners cannot be detected
        """
        self._stream_message("game_state_update_start", "Updating game state...")
        while True:
            ret, frame = self.overhead_cam.read()
            # Rotate the frame by 180 degrees
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if not ret:
                raise Exception("Error: Could not read frame from webcam")
            # frame = cv2.imread("image1.jpg")
            try:
                self.game_board.update_game_board(frame)
                time.sleep(0.2)
                
                annotated_frame = self.game_board.annotate_frame(frame)
                # cv2.imshow('frame', annotated_frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.waitKey(1)
                
                # If board moved or it's the first update, show alignment and wait for confirmation
                if self.game_board.board_moved or not hasattr(self, 'alignment_confirmed'):
                    self.alignment_confirmed = False
                    self._stream_message("game_board_alignment_display", "prepare to show display")
                    self._stream_frame(annotated_frame)
                    
                    # Wait for confirmation from frontend (with timeout)
                    confirmation_received = False
                    start_time = time.time()
                    timeout = 10  # 10 seconds timeout
                    
                    while not confirmation_received and time.time() - start_time < timeout:
                        # Check if confirmation has been received (set by websocket handler)
                        if hasattr(self, 'alignment_confirmed') and self.alignment_confirmed:
                            confirmation_received = True
                        time.sleep(0.1)
                    
                    # If timeout occurred without explicit rejection, assume alignment is good
                    if not confirmation_received:
                        print("Alignment confirmation timed out, assuming alignment is good")
                        self.alignment_confirmed = True
                
                break
                
            except ValueError as e:
                stream_sent = True
                for box in self.game_board.corner_boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                if self.stream_enabled:
                    self._stream_frame(frame)

        print("Game state update completed")
        self._stream_message("game_state_update_success", "Game state updated successfully")

    def _stream_game_state(self):
        """
        Send current game state to frontend clients via websocket.
        Includes turn number, current piece, and all piece positions.
        """
        game_state = {
            "turn_number": self.turn_number,
            "current_piece": self.pieces[self.turn_number % len(self.pieces)],
            "pieces": [{"piece_name": name, "current_square": {"id": self.game_board.pieces[name].current_square, "center": self.game_board.reference_map.reference_squares[self.game_board.pieces[name].current_square].center}, "auto_controlled": self.game_board.pieces[name].auto_controlled} for name in self.pieces],
        }
        # print(game_state)
        self._stream_message("game_state", game_state)

    def _stream_frame(self, frame):
        """
        Send a video frame to frontend clients via websocket as a single image.
        
        Args:
            frame: OpenCV image frame to stream
        """
        if not self.websocket_clients:
            return
        
        try:
            # Resize the frame to have a maximum dimension of 1200 pixels
            max_dimension = 1200
            height, width = frame.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_size = (int(width * scale), int(height * scale))
                frame = cv2.resize(frame, new_size)

            # Encode frame with good quality
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Convert to base64 string
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Send the image as a single message
            self._stream_message("game_board_frame", {
                "image": img_str,
                "timestamp": time.time()
            })
            
            print("Successfully sent the image frame to clients")
            
        except Exception as e:
            print(f"Error streaming frame: {e}")
            import traceback
            traceback.print_exc()

    def _stream_message(self, message_type, data):
        """
        Send a typed message to all websocket clients.
        
        Args:
            message_type (str): Type identifier for the message
            data: Message payload
        """
        message_json = {
            "type": message_type,
            "data": data
        }
        websockets.broadcast(self.websocket_clients, json.dumps(message_json))

    def _start_websocket_server(self):
        """
        Start websocket server for frontend communication.
        Runs in separate thread.
        """
        async def handler(websocket):
            self.websocket_clients.add(websocket)
            self._stream_game_state()
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        command = data.get('command')
                        print(command)
                        if command == 'draw_card':
                            self.r_pressed = True
                            print(self.r_pressed)
                            # print("handler_m_pressed:", self.m_pressed)
                        elif command == 'move':
                            self.m_pressed = True
                            # Process move command
                            # Move handling code...
                        
                        elif command == 'request_alignment_view':
                            # Send current alignment view
                            frame = cv2.imread("image.png")  # Or get from camera
                            annotated_frame = self.game_board.annotate_frame(frame)
                            self._stream_message("game_board_alignment_display", "manual alignment view")
                            self._stream_frame(annotated_frame)
                        
                        elif command == 'confirm_alignment':
                            # Mark alignment as confirmed
                            self.alignment_confirmed = True
                            print("Alignment confirmed by frontend")
                        
                        elif command == 'report_alignment_issue':
                            # Mark alignment as rejected
                            self.alignment_confirmed = False
                            print("Alignment issue reported by frontend")
                            # You could trigger a recalibration here
                            
                    except json.JSONDecodeError:
                        print(f"Invalid JSON received: {message}")
                    except Exception as e:
                        print(f"Error processing message: {e}")
                
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)

        async def start_server():
            self.websocket_server = await websockets.serve(
                lambda ws: handler(ws), 
                "localhost", 
                8089
            )
            await self.websocket_server.wait_closed()

        asyncio.run(start_server())

    def draw_card(self):
        """
        Process card draw using card reader camera.
        
        Returns:
            str: Card identifier (e.g. "1_Red", "2_Blue", etc)
        """
        # TO DO: Add code to communicate with microcontroller to draw a card
        # TO DO: Add code to read card using card reader camera
        # ret, frame = self.card_reader_cam.read()
        # if not ret:
        #     raise Exception("Error: Could not read frame from card reader webcam")
        # options = ["JaqGus", "Sebastian", "Lumiere", "Dopey", "Gus", "Jaq", "1_Red", "2_Red", "1_Purple", "2_Purple", "1_Yellow", "2_Yellow", "1_Blue", "2_Blue", "1_Orange", "2_Orange", "1_Green", "2_Green"]
        # options = ["1_Blue.jpg", "1_Green.jpg", "1_Orange.jpg", "1_Purple.jpg", "1_Red.jpg", "1_Yellow.jpg"]
        # card = random.choice(options)
        ret, frame = self.card_reader_cam.read()
        cv2.namedWindow('Captured Frame', cv2.WINDOW_NORMAL)  # Create resizable window
        cv2.resizeWindow('Captured Frame', 880, 560)  # Set window size to 640x360
        resized_frame = cv2.resize(frame, (880, 560))  # Resize the frame to 640x360
        if show_cv: 
            cv2.imshow('Captured Frame', resized_frame)
            cv2.waitKey(2000)  # Display the frame for 2 seconds
            cv2.destroyAllWindows()

        # card_color = process_card(frame)
        card_color =  "yellow" if self.turn_number == 0 else "green"
        return "1_" + card_color

    def main_loop(self):
        """
        Main game loop that processes turns until game ends.
        Waits for 'r' key press to start each turn.
        """
        running = True
        print("Starting main loop")
        print ("Waiting for r key press")
        while running:
            # Check for 'r' key press globally
            if keyboard.is_pressed('r'):
                self.r_pressed = True
                print("R key pressed")

            if self.r_pressed:
                self.r_pressed = False  # Reset after processing
                self.complete_turn(self.pieces[self.turn_number % len(self.pieces)])
                print("Turn complete")
                print("Waiting for r key press")
            
            # Add a small sleep to prevent high CPU usage
            time.sleep(0.1)

        # pygame.quit()  # Commented out pygame cleanup
    
    def __del__(self):
        # Release the webcam when the object is destroyed
        if hasattr(self, 'overhead_cam'):
            self.overhead_cam.release()
        if hasattr(self, 'card_reader_cam'):
            self.card_reader_cam.release()
        if hasattr(self, 'client_socket') and self.client_socket:
            self.client_socket.close()
        if hasattr(self, 'websocket_server') and self.websocket_server:
            self.websocket_server.close()
        if hasattr(self, 'serial_conn') and self.serial_conn:
            self.serial_conn.close()



if __name__ == "__main__":
    enable_streaming = '-s' in sys.argv
    config_file = None
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            config_file = arg
            break
    game = GameSession("ReferenceMap.png", "candylandMapAnnotations.xml", config_file, enable_streaming)
    game.main_loop()

