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

class GameSession:

    def __init__(self, reference_map_path, annotations_path, config_file=None, enable_streaming=False):
        # Initialize webcam
        self.overhead_cam = cv2.VideoCapture(1)
        # self.overhead_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.overhead_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        if not self.overhead_cam.isOpened():
            raise Exception("Error: Could not open webcam")
        
        self.card_reader_cam = cv2.VideoCapture(0)
        if not self.card_reader_cam.isOpened():
            raise Exception("Error: Could not open card reader webcam")
        
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

    def _load_config(self, config_file):
        """Load game configuration from JSON file"""
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
        # TO DO: Add code to generate move instructions
        path_points = self.game_board.generate_path(piece_name, target_square)
        self.game_board.show_path(path_points)
        return path_points

    def _generate_front_end_path(self, piece_name, target_square):
        front_end_path = []
        for square in self.game_board.reference_map.reference_squares[self.game_board.pieces[piece_name].current_square : target_square+1]:
            front_end_path.append([int(square.center[0]), int(square.center[1])])   
        if (self.game_board.reference_map.reference_squares[target_square].landing_square != target_square):
            landing_square = self.game_board.reference_map.reference_squares[target_square].landing_square
            front_end_path.append([int(self.game_board.reference_map.reference_squares[landing_square].center[0]), int(self.game_board.reference_map.reference_squares[landing_square].center[1])])
        else:
            landing_square = target_square
        return front_end_path, landing_square

    def complete_turn(self, piece_name):
        self.update_game_state()
        self._stream_game_state()
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
                key = input()
                if key == 'm':
                    break
        else:
            time.sleep(15)
        # print(self.game_board.pieces[piece_name].current_square)
        self.game_board.update_piece_position(piece_name, landing_square)
        # self.movement_controller.execute_move(move_instructions)

        self.turn_number += 1
        self._stream_message("turn_complete", "Turn complete")
        self.update_game_state()
        self.game_board.show_game_board_with_annotations()

        self._stream_game_state()
    
    
    def update_game_state(self):
        self._stream_message("game_state_update_start", "Updating game state...")
        stream_sent = False
        while True:
            ret, frame = self.overhead_cam.read()
            
            if not ret:
                raise Exception("Error: Could not read frame from webcam")
            try:
                self.game_board.update_game_board(frame)
                time.sleep(0.2)
                # print("Double check")
                self.game_board.update_game_board(frame)
                
                # Draw annotations
                annotated_frame = frame.copy()
                for box in self.game_board.corner_boxes:
                    cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                for piece, piece_info in self.game_board.pieces.items():
                    if len(piece_info.box) > 0:
                        cv2.rectangle(annotated_frame, (int(piece_info.box[0]), int(piece_info.box[1])), 
                                    (int(piece_info.box[2]), int(piece_info.box[3])), (0, 255, 0), 2)
                
                # cv2.imshow('frame', annotated_frame)
                if stream_sent:
                    for i in range(5):
                        ret, frame = self.overhead_cam.read()
                        if not ret:
                            raise Exception("Error: Could not read frame from webcam")
                        self.game_board.update_game_board(frame)
                        time.sleep(0.2)
                        # print("Double check2")
                        self.game_board.update_game_board(frame)
                        annotated_frame = frame.copy()
                        for box in self.game_board.corner_boxes:
                            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                        self._stream_frame(annotated_frame)
                    # End of if stream_sent block
                break  # Move break statement here to exit while True loop
                # cv2.waitKey(0)
            except ValueError as e:
                stream_sent = True
                for box in self.game_board.corner_boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                if self.stream_enabled:
                    self._stream_frame(frame)
                # cv2.imshow('frame', frame)
                # print(f"Error updating game board: {e}. Retrying...")
                # if cv2.waitKey(1) == ord('q'):
                #     break
        self._stream_message("game_state_update_success", "Game state updated successfully")

        # cv2.destroyAllWindows()
                
    def _stream_game_state(self):
        game_state = {
            "turn_number": self.turn_number,
            "current_piece": self.pieces[self.turn_number % len(self.pieces)],
            "pieces": [{"piece_name": name, "current_square": {"id": self.game_board.pieces[name].current_square, "center": self.game_board.reference_map.reference_squares[self.game_board.pieces[name].current_square].center}, "auto_controlled": self.game_board.pieces[name].auto_controlled} for name in self.pieces],
        }
        # print(game_state)
        self._stream_message("game_state", game_state)

    def _stream_frame(self, frame):
        """Send an OpenCV frame over websocket connection"""
        if not self.websocket_clients:
            return
        
        try:
            # Encode frame as JPEG with higher quality
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # Convert to base64 string
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Send with proper message structure
            self._stream_message("game_board_frame", {
                "image": img_str,
                "timestamp": time.time()
            })
            
        except Exception as e:
            print(f"Error streaming frame: {e}")

    def _stream_message(self, message_type, data):
        message_json = {
            "type": message_type,
            "data": data
        }
        websockets.broadcast(self.websocket_clients, json.dumps(message_json))

    def _start_websocket_server(self):
        async def handler(websocket):
            self.websocket_clients.add(websocket)
            self._stream_game_state()
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)

        async def start_server():
            self.websocket_server = await websockets.serve(handler, "localhost", 8089)
            await self.websocket_server.wait_closed()

        asyncio.run(start_server())

    def draw_card(self):
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
        cv2.imshow('Captured Frame', resized_frame)
        cv2.waitKey(2000)  # Display the frame for 2 seconds
        cv2.destroyAllWindows()

        card_color = process_card(frame)
        return "1_" + card_color

    def main_loop(self):
        while True:
            print(f"Press r to complete turn for {self.pieces[self.turn_number % len(self.pieces)]}...")
            while True:
                key = input()  # Wait for key input
                if key == 'r':
                    break  # Exit loop when 'r' is pressed
            self.complete_turn(self.pieces[self.turn_number % len(self.pieces)])
    
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



if __name__ == "__main__":
    enable_streaming = '-s' in sys.argv
    config_file = None
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            config_file = arg
            break
    game = GameSession("ReferenceMap.png", "candylandMapAnnotations.xml", config_file, enable_streaming)
    game.main_loop()

