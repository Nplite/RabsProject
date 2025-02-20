import logging
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from threading import Thread, Lock
import queue
from math import ceil, sqrt 
from ultralytics import YOLO
from vidgear.gears import CamGear
from Rabs.logger import logging
from Rabs.exception import RabsException
from Rabs.mongodb import MongoDBHandlerSaving  

class CameraStream:
    """Handles video streaming from RTSP cameras with improved error handling and frame management"""
    
    def __init__(self, rtsp_url: str, camera_id: int, buffer_size: int = 30):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.lock = Lock()
        self._initialize_stream()
        
    def _initialize_stream(self) -> None:
        """Initialize or reinitialize the camera stream"""
        try:
            self.cap = CamGear(source=self.rtsp_url, logging=True).start()
            self.fps = self.cap.stream.get(cv2.CAP_PROP_FPS)
            if not self.fps or self.fps <= 0:
                self.fps = 30
                logging.warning(f"Camera {self.camera_id}: Invalid FPS detected, defaulting to {self.fps}")
            logging.info(f"Camera {self.camera_id}: Initialized with FPS: {self.fps}")
        except RabsException as e:
            logging.error(f"Camera {self.camera_id}: Failed to initialize stream: {str(e)}")
            raise

    def start(self) -> 'CameraStream':
        """Start the frame capture thread"""
        self.capture_thread = Thread(target=self._update, daemon=True)
        self.capture_thread.start()
        return self

    def _update(self) -> None:
        """Continuously update frame buffer"""
        consecutive_failures = 0
        while not self.stopped:
            try:
                with self.lock:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    frame = self.cap.read()
                    if frame is not None:
                        self.frame_queue.put(frame)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 30:
                            logging.warning(f"Camera {self.camera_id}: Stream failure, attempting restart")
                            self._restart_stream()
                            consecutive_failures = 0
                            
                time.sleep(1 / self.fps)
                
            except RabsException as e:
                logging.error(f"Camera {self.camera_id}: Frame capture error: {str(e)}")
                time.sleep(1)

    def _restart_stream(self) -> None:
        """Restart the camera stream"""
        with self.lock:
            self.cap.stop()
            time.sleep(2)
            self._initialize_stream()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the most recent frame"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        """Stop the camera stream"""
        self.stopped = True
        with self.lock:
            self.cap.stop()
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)

class CameraProcessor:
    """Handles object detection processing for individual cameras"""
    
    def __init__(self, camera_id: int, rtsp_url: str, model_path: str = 'yolov8n.pt'):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.stream = CameraStream(rtsp_url, camera_id)
        self.window_name = f'Camera {self.camera_id}'
        self._initialize_model(model_path)

    def _initialize_model(self, model_path: str) -> None:
        """Initialize YOLO model"""
        try:
            self.model = YOLO(model_path)
            logging.info(f"Camera {self.camera_id}: Model initialized successfully")
        except RabsException as e:
            logging.error(f"Camera {self.camera_id}: Model initialization failed: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """Process a frame with object detection"""
        try:
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()
            detections = results[0].boxes.data.cpu().numpy()
            return annotated_frame, detections
        except RabsException as e:
            logging.error(f"Camera {self.camera_id}: Frame processing error: {str(e)}")
            return frame, []


class MultiCameraSystem:
    """Manages multiple camera streams and their processing"""
    
    def __init__(self, email: str):
        self.email = email
        self.camera_processors = {}
        self.is_running = False
        self.mongo_handler = MongoDBHandlerSaving()
        self.last_frames = {}  # Store last valid frame to avoid flickering
        self._initialize_cameras()

    def _initialize_cameras(self) -> None:
        """Fetch camera details from MongoDB and initialize processors"""
        camera_data = self.mongo_handler.fetch_camera_rtsp_by_email(self.email)

        if not camera_data:
            logging.error(f"No camera data found for email: {self.email}")
            return

        for camera in camera_data:
            try:
                camera_id = camera["camera_id"]
                rtsp_link = camera["rtsp_link"]

                processor = CameraProcessor(camera_id, rtsp_link)
                processor.stream.start()
                self.camera_processors[camera_id] = processor
                self.last_frames[camera_id] = None  # Initialize last valid frame storage
                logging.info(f"Camera {camera_id}: Initialized successfully from MongoDB")
            except RabsException as e:
                logging.error(f"Camera {camera_id}: Initialization failed: {str(e)}")

    def _process_cameras(self) -> None:
        """Main processing loop with automatic grid layout and anti-flicker adjustments"""
        logging.info("Starting camera processing")
        
        num_cameras = len(self.camera_processors)
        if num_cameras == 0:
            logging.error("No active cameras to process")
            return

        grid_cols = ceil(sqrt(num_cameras))  
        grid_rows = ceil(num_cameras / grid_cols)  

        logging.info(f"Grid Layout: {grid_rows} rows x {grid_cols} columns")

        while self.is_running:
            frames = []
            try:
                for camera_id, processor in self.camera_processors.items():
                    if processor.stream.stopped:
                        continue
                        
                    ret, frame = processor.stream.read()
                    if ret:
                        processed_frame, _ = processor.process_frame(frame)
                        self.last_frames[camera_id] = processed_frame  # Store latest valid frame
                    else:
                        processed_frame = self.last_frames.get(camera_id, None)  # Use last valid frame
                    
                    if processed_frame is not None:
                        frame_resized = cv2.resize(processed_frame, (320, 240))
                        frames.append(frame_resized)

                if len(frames) > 0:
                    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    while len(frames) < grid_rows * grid_cols:
                        frames.append(blank_frame)

                    rows = [np.hstack(frames[i * grid_cols:(i + 1) * grid_cols]) for i in range(grid_rows)]
                    grid_display = np.vstack(rows)

                    cv2.imshow("Multi-Camera View", grid_display)

                if cv2.waitKey(10) & 0xFF == ord('q'):  # Slight delay to reduce flickering
                    break
                    
            except RabsException as e:
                logging.error(f"Main processing loop error: {str(e)}")
                time.sleep(1)

    def start(self) -> None:
        """Start the camera system"""
        if not self.camera_processors:
            raise RuntimeError("No cameras were successfully initialized")
            
        self.is_running = True
        # self.processing_thread = Thread(target=self._process_cameras, daemon=True)
        # self.processing_thread.start()
        self._process_cameras()
        logging.info("Camera system started")

    def stop(self) -> None:
        """Stop the camera system"""
        self.is_running = False
        for processor in self.camera_processors.values():
            processor.stream.stop()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        logging.info("Camera system stopped")



# Example: Start camera system for a user
multi_camera_system = MultiCameraSystem(email="sd@example.com")
multi_camera_system.start()
