import cv2
from gameboard import GameBoard


class GameBoardStream:
    def __init__(self, game_board, video_source=0, output_path='output.mp4', annotation_timestamps=None):
        """
        Initialize the game board stream
        
        Args:
            game_board: Instance of GameBoard class
            video_source: Camera index or video file path
            output_path: Path where the annotated video will be saved
            annotation_timestamps: List of timestamps (in seconds) where annotations should occur
        """
        self.game_board = game_board
        self.video_source = video_source
        self.output_path = output_path
        self.annotation_timestamps = sorted(annotation_timestamps) if annotation_timestamps else []
        self.cap = None
        
    def start(self):
        """Process the video and create annotated output"""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.video_source}")
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
        
        current_timestamp_idx = 0
        max_retries = 30  # Maximum number of frames to try before skipping a timestamp
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                current_time = current_frame / fps
                
                # Check if we need to annotate at this timestamp
                should_annotate = (
                    current_timestamp_idx < len(self.annotation_timestamps) and 
                    current_time >= self.annotation_timestamps[current_timestamp_idx]
                )
                
                if should_annotate:
                    retry_count = 0
                    success = False
                    
                    # Keep trying frames until we get a successful update or hit max retries
                    while retry_count < max_retries and not success:
                        try:
                            self.game_board.update_game_board(frame)
                            print(f"\nSuccessfully annotated at timestamp: {current_time:.2f}s")
                            
                            # Create annotated frame
                            annotated_frame = self.game_board.annotate_frame(frame.copy())
                            
                            # Optional: Display the annotated frame
                            cv2.imshow('Annotated Gameboard', annotated_frame)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                            # Write regular frame
                            out.write(frame)
                            freeze_frames = int(fps * 15) if current_timestamp_idx == 0 else int(fps * 5)
                            # Write annotated frame multiple times to create freeze effect
                            for _ in range(freeze_frames):
                                out.write(annotated_frame)
                                
                            success = True
                            current_timestamp_idx += 1
                            
                        except ValueError as e:
                            retry_count += 1
                            print(f"\rFailed attempt {retry_count}/{max_retries} at {current_time:.2f}s: {e}", end="")
                            
                            # Try to get next frame
                            ret, frame = self.cap.read()
                            if not ret:
                                break
                            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                            current_time = current_frame / fps
                    
                    if not success:
                        print(f"\nSkipping timestamp {self.annotation_timestamps[current_timestamp_idx]}s after {max_retries} failed attempts")
                        current_timestamp_idx += 1
                        out.write(frame)
                else:
                    # Write regular frame
                    out.write(frame)
                
                # Display progress
                progress = (current_frame / total_frames) * 100
                print(f"\rProgress: {progress:.1f}%", end="")
                
                cv2.imshow('Processing Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.stop()
            out.release()
            print("\nProcessing complete!")

    def stop(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()