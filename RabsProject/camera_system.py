import logging
import cv2
import time
import numpy as np
from typing import Dict, Optional
from threading import Thread, Lock
import queue
import cv2
import numpy as np
import time
import os, sys
import csv
from datetime import datetime
from math import ceil, sqrt
import threading
from typing import List, Optional, Dict, Any
from ultralytics import YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING)
from vidgear.gears import CamGear
from RabsProject.logger import logging
from RabsProject.exception import RabsException
from RabsProject.mongodb import MongoDBHandlerSaving  


frame_queues = {}
MAX_QUEUE_SIZE = 30  # Adjust based on memory constraints and desired buffering




####################################################################################################################
                            ## Multicamera & Single Camera Procesing and Streaming ##
####################################################################################################################



class CameraStream:
    """Handles video streaming from RTSP cameras with improved error handling and frame management"""
    try:
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

    except Exception as e:
        raise RabsException(e, sys) from e


class CameraThread(threading.Thread):
    def __init__(self, email: str, camera_system: Any):
        threading.Thread.__init__(self)
        self.email = email
        self.camera_system = camera_system
        self.daemon = True  # Thread will exit when main program exits
        self.running = False
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        frame_queues[email] = self.frame_queue
    
    def run(self):
        """Thread's main execution function to capture frames and put them in queue"""
        self.running = True
        try:
            while self.running:
                frame = self.camera_system.get_next_frame()  # Assuming this method exists
                
                # If queue is full, remove oldest frame
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add new frame to queue
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error in camera thread for {self.email}: {str(e)}")
            self.running = False
    
    def stop(self):
        """Stop the thread gracefully"""
        self.running = False
        self.join(timeout=2.0)  # Wait for thread to finish, with timeout



# class CameraProcessor:
#     try:
#         """Handles object detection processing for individual cameras"""
        
#         def __init__(self, camera_id: int, rtsp_url: str, model_path: str):
#             self.camera_id = camera_id
#             self.rtsp_url = rtsp_url
#             self.stream = CameraStream(rtsp_url, camera_id)
#             self.window_name = f'Camera {self.camera_id}'
#             self._initialize_model(model_path)

#         def _initialize_model(self, model_path: str) -> None:
#             """Initialize YOLO model"""
#             try:
#                 self.model = YOLO(model_path)
#                 logging.info(f"Camera {self.camera_id}: Model initialized successfully")
#             except RabsException as e:
#                 logging.error(f"Camera {self.camera_id}: Model initialization failed: {str(e)}")
#                 raise

#         def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
#             """Process a frame with object detection"""
#             try:
#                 results = self.model(frame, verbose=False)
#                 annotated_frame = results[0].plot()
#                 detections = results[0].boxes.data.cpu().numpy()
#                 return annotated_frame, detections
#             except RabsException as e:
#                 logging.error(f"Camera {self.camera_id}: Frame processing error: {str(e)}")
#                 return frame, []

#     except Exception as e:
#         raise RabsException(e, sys) from e
    
import logging
import sys
import time
import numpy as np
import cv2
import winsound  # For Windows beep sound, replace for Linux/macOS
from ultralytics import YOLO

class CameraProcessor:
    """Handles fire detection and tracking using YOLO"""

    def __init__(self, camera_id: int, rtsp_url: str, model_path: str, output_path: str, fire_class_id: int = 0):
        """
        Initialize the camera processor.

        :param camera_id: Unique camera ID
        :param rtsp_url: RTSP stream URL
        :param model_path: Path to the trained YOLO model
        :param output_path: Path to save the processed video
        :param fire_class_id: Class ID for fire detection (default 0, update based on model)
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.fire_class_id = fire_class_id
        self.window_name = f'Camera {self.camera_id}'
        self._initialize_model(model_path)

        # Video Stream Setup
        self.cap = cv2.VideoCapture(rtsp_url)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def _initialize_model(self, model_path: str) -> None:
        """Initialize YOLO model"""
        try:
            self.model = YOLO(model_path)
            logging.info(f"Camera {self.camera_id}: Model initialized successfully")
        except Exception as e:
            logging.error(f"Camera {self.camera_id}: Model initialization failed: {str(e)}")
            raise RabsException(e, sys)

    def process_stream(self):
        """Process video stream, detect fire, and trigger alarm"""
        try:
            tracker_results = self.model.track(self.rtsp_url, stream=True, save=False)

            for result in tracker_results:
                frame = result.orig_img  # Get original frame
                fire_detected = False

                # Overlay "RABS INDUSTRIES" branding
                cv2.putText(frame, "RABS INDUSTRIES", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                for box in result.boxes:
                    cls = int(box.cls[0])  # Object class ID
                    confidence = box.conf[0]  # Confidence score
                    label = "Fire Detected"

                    if cls == self.fire_class_id and confidence > 0.5:  # Adjust confidence threshold if needed
                        fire_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red bounding box
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Trigger alarm if fire is detected
                if fire_detected:
                    logging.warning(f"🔥 Fire detected! Camera {self.camera_id}")
                    cv2.putText(frame, "FIRE DETECTED! ALARM!", (5, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.trigger_alarm()

                # Display frame
                cv2.imshow(f"Fire Detection - Camera {self.camera_id}", frame)
                self.out.write(frame)  # Save frame to video output

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Camera {self.camera_id}: Stream processing error: {str(e)}")

        finally:
            # Release resources
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()


    def trigger_alarm(self):
        """Trigger an alarm when fire is detected."""
        logging.warning("🚨 Alarm Triggered: Fire Detected!")
        try:
            os.system("beep -f 2000 -l 1000")  # Frequency: 2000Hz, Duration: 1000ms
        except:
            logging.warning("⚠️ 'beep' command not available. Try installing it using: sudo apt install beep")

        # 2. **Alternative: Play a sound file (WAV)**
        try:
            os.system("aplay alarm_sound.wav")  # Make sure you have an alarm sound file
        except:
            logging.warning("⚠️ 'aplay' command not available. Install it using: sudo apt install alsa-utils")

        # 3. **Log the alarm event**
        logging.info(f"Alarm Triggered for Camera {self.camera_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}")


class MultiCameraSystem:
    try:
        """Manages multiple camera streams and their processing"""

        def __init__(self, email: str, model_path:str):
            self.email = email
            self.model_path = model_path
            self.camera_processors = {}
            self.is_running = False
            self.mongo_handler = MongoDBHandlerSaving()
            self.last_frames = {}
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

                    processor = CameraProcessor(camera_id, rtsp_link, self.model_path)
                    processor.stream.start()
                    self.camera_processors[camera_id] = processor
                    self.last_frames[camera_id] = None
                    logging.info(f"Camera {camera_id}: Initialized successfully from MongoDB")
                except RabsException as e:
                    logging.error(f"Camera {camera_id}: Initialization failed: {str(e)}")

        def get_video_frames(self):
            """Generator function to yield video frames as bytes for HTTP streaming"""
            logging.info("Streaming multi-camera video frames")

            grid_cols = ceil(sqrt(len(self.camera_processors)))
            grid_rows = ceil(len(self.camera_processors) / grid_cols)
            blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)

            while True:
                frames = []
                for camera_id, processor in self.camera_processors.items():
                    if processor.stream.stopped:
                        continue
                    
                    ret, frame = processor.stream.read()
                    if ret:
                        processed_frame, _ = processor.process_frame(frame)
                        self.last_frames[camera_id] = processed_frame
                    else:
                        processed_frame = self.last_frames.get(camera_id, blank_frame)

                    frame_resized = cv2.resize(processed_frame, (320, 240))
                    frames.append(frame_resized)

                if frames:
                    while len(frames) < grid_rows * grid_cols:
                        frames.append(blank_frame)

                    rows = [np.hstack(frames[i * grid_cols:(i + 1) * grid_cols]) for i in range(grid_rows)]
                    grid_display = np.vstack(rows)

                    _, buffer = cv2.imencode(".jpg", grid_display)
                    yield (b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        buffer.tobytes() + b"\r\n")

                time.sleep(0.05)  # Small delay to control FPS

        def stop(self) -> None:
            """Stop the camera system"""
            self.is_running = False
            for processor in self.camera_processors.values():
                processor.stream.stop()
            cv2.destroyAllWindows()
            logging.info("Camera system stopped")

    except Exception as e:
        raise RabsException(e, sys) from e
    

class SingleCameraSystem:
    try:
        """Manages a single camera stream and its processing"""

        def __init__(self, camera_id: str, rtsp_link: str, email: str, model_path: str):
            self.email = email
            self.camera_id = camera_id
            self.rtsp_link = rtsp_link
            self.model_path = model_path
            self.mongo_handler = MongoDBHandlerSaving()
            self.is_running = False
            self.last_frame = None


            camera_data = self.mongo_handler.fetch_camera_rtsp_by_email(self.email)

            if not camera_data:
                logging.error(f"No camera data found for email: {self.email}")
                return


            for camera in camera_data:
                try:
                    camera_id = camera["camera_id"]
                    rtsp_link = camera["rtsp_link"]
                    
                    self.processor = CameraProcessor(camera_id, rtsp_link, self.model_path)

                    logging.info(f"Single Camera {self.camera_id}: Initialized successfully")
                except Exception as e:
                    logging.error(f"Single Camera {self.camera_id}: Initialization failed: {str(e)}")
                    self.processor = None

        def start(self):
            """Starts the single camera stream"""
            if not self.processor:
                logging.error(f"Single Camera {self.camera_id}: Cannot start, processor is not initialized")
                return
            
            try:
                self.processor.stream.start()
                self.is_running = True
                logging.info(f"Single Camera {self.camera_id}: Stream started")
            except Exception as e:
                logging.error(f"Single Camera {self.camera_id}: Failed to start: {str(e)}")
                self.is_running = False

        def get_video_frames(self):
            """Generator function to yield video frames as bytes for HTTP streaming"""
            logging.info(f"Streaming video frames for Single Camera {self.camera_id}")
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            while self.is_running:
                if self.processor and not self.processor.stream.stopped:
                    ret, frame = self.processor.stream.read()
                    if ret:
                        processed_frame, _ = self.processor.process_frame(frame)
                        self.last_frame = processed_frame
                    else:
                        logging.warning(f"Single Camera {self.camera_id}: Failed to read frame, using last frame")
                        processed_frame = self.last_frame if self.last_frame is not None else blank_frame
                else:
                    logging.error(f"Single Camera {self.camera_id}: Stream is stopped or processor is None")
                    processed_frame = blank_frame

                frame_resized = cv2.resize(processed_frame, (640, 480))
                _, buffer = cv2.imencode(".jpg", frame_resized)
                yield (b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    buffer.tobytes() + b"\r\n")

                time.sleep(0.05)  # Small delay to control FPS

        def stop(self):
            """Stops the camera stream and releases resources"""
            self.is_running = False
            if self.processor:
                self.processor.stream.stop()
            cv2.destroyAllWindows()
            logging.info(f"Single Camera {self.camera_id}: Stream stopped")

    except Exception as e:
        raise RabsException(e, sys) from e



# ##################################################################################################################
#                               # Truck Loading and Unloading #
# ##################################################################################################################


class MultiCameraSystemTruck:
    try:
        """Manages multiple camera streams and their processing using CameraProcessorV2."""
        def __init__(self, email: str, model_path=str, confidence=0.3, cooldown_period=60):
            self.email = email
            self.model_path = model_path
            self.confidence = confidence
            self.cooldown_period = cooldown_period
            self.camera_processors = {}
            self.is_running = False
            self.mongo_handler = MongoDBHandlerSaving()
            self.last_frames = {}
            self._initialize_cameras()

        def _initialize_cameras(self) -> None:
            """Fetch camera details from MongoDB and initialize processors with polygon regions."""
            camera_data = self.mongo_handler.fetch_camera_rtsp_by_email(self.email)

            if not camera_data:
                logging.error(f"No camera data found for email: {self.email}")
                return

            for camera in camera_data:
                try:
                    camera_id = camera["camera_id"]
                    rtsp_link = camera["rtsp_link"]
                    
                    # Parse polygon points from the database
                    polygon_points = None
                    if "polygonal_points" in camera and camera["polygonal_points"]:
                        try:
                            # Convert the string representation of polygon points to actual list of tuples
                            polygon_str = camera["polygonal_points"]
                            # Remove brackets and split by commas
                            points_str = polygon_str.strip('[]').split('), (')
                            
                            # Parse each point
                            polygon_points = []
                            for point_str in points_str:
                                point_str = point_str.replace('(', '').replace(')', '')
                                x, y = map(int, point_str.split(','))
                                polygon_points.append((x, y))
                                
                            logging.info(f"Camera {camera_id}: Parsed polygon points: {polygon_points}")
                        except Exception as e:
                            logging.error(f"Camera {camera_id}: Failed to parse polygon points: {str(e)}")
                            polygon_points = None

                    processor = CameraProcessorTruckYOLO(
                        camera_id, rtsp_link, self.model_path,
                        polygon_points=polygon_points,  # Pass the parsed polygon points
                        confidence=self.confidence, 
                        cooldown_period=self.cooldown_period)
                        
                    processor.stream.start()
                    self.camera_processors[camera_id] = processor
                    self.last_frames[camera_id] = None
                    logging.info(f"Camera {camera_id}: Initialized successfully from MongoDB")
                except RabsException as e:
                    logging.error(f"Camera {camera_id}: Initialization failed: {str(e)}")

        def get_video_frames(self):
            """Generator function to yield multi-camera video frames as bytes for HTTP streaming."""
            logging.info("Streaming multi-camera video frames")

            grid_cols = ceil(sqrt(len(self.camera_processors)))
            grid_rows = ceil(len(self.camera_processors) / grid_cols)
            blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)

            while True:
                frames = []
                for camera_id, processor in self.camera_processors.items():
                    if processor.stream.stopped:
                        continue

                    ret, frame = processor.stream.read()
                    frame = cv2.resize(frame, (1920,1080))
                    if ret:
                        processed_frame, _, _= processor.process_frame(frame)
                        self.last_frames[camera_id] = processed_frame
                    else:
                        processed_frame = self.last_frames.get(camera_id, blank_frame)

                    frame_resized = cv2.resize(processed_frame, (320, 240))
                    frames.append(frame_resized)

                if frames:
                    while len(frames) < grid_rows * grid_cols:
                        frames.append(blank_frame)

                    rows = [np.hstack(frames[i * grid_cols:(i + 1) * grid_cols]) for i in range(grid_rows)]
                    grid_display = np.vstack(rows)

                    _, buffer = cv2.imencode(".jpg", grid_display)
                    yield (b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        buffer.tobytes() + b"\r\n")

                time.sleep(0.05)  # Small delay to control FPS

        def stop(self) -> None:
            """Stop the multi-camera system."""
            self.is_running = False
            for processor in self.camera_processors.values():
                processor.stream.stop()
            cv2.destroyAllWindows()
            logging.info("Multi-camera system stopped")

        def get_all_statistics(self):
            """Retrieve tracking statistics for all cameras."""
            all_stats = {}
            for camera_id, processor in self.camera_processors.items():
                stats = processor.get_statistics()
                all_stats[camera_id] = stats
            return all_stats
    except Exception as e:
        raise RabsException(e, sys) from e


class CameraProcessorTruckYOLO:
    try:
        def __init__(self, camera_id: int, rtsp_url: str, model_path: str, 
                    polygon_points=None, confidence=0.25, cooldown_period=60, truck_class=7):
            self.camera_id = camera_id
            self.rtsp_url = rtsp_url
            self.stream = CameraStream(rtsp_url, camera_id)
            self.window_name = f'Camera {self.camera_id}'
            
            # Set polygon points from parameter or use default if None
            if polygon_points and len(polygon_points) >= 3:
                self.polygon = np.array(polygon_points, np.int32)
                logging.info(f"Camera {self.camera_id}: Using custom polygon: {polygon_points}")
            else:
                # Default polygon as fallback
                self.polygon = np.array([(571, 716), (825, 577), (1259, 616), (1256, 798)], np.int32)
                logging.info(f"Camera {self.camera_id}: Using default polygon")
                
            self.confidence = confidence
            self.truck_class = truck_class  # COCO class index for 'truck'
            
            # Timer settings
            self.cooldown_period = cooldown_period
            self.timer_active = False
            self.timer_start = None
            self.last_detection_time = None
            self.entry_times = {}
            self.tracking_data = []
            
            # Logging configuration
            self.log_file = f"tracking_log_camera_{self.camera_id}.csv"
            self.init_logging()
            
            self._initialize_model(model_path)

        def _initialize_model(self, model_path: str) -> None:
            """Initialize YOLO model"""
            try:
                self.model = YOLO(model_path)
                logging.info(f"Camera {self.camera_id}: Model initialized successfully")
            except RabsException as e:
                logging.error(f"Camera {self.camera_id}: Model initialization failed: {str(e)}")
                raise
        
        def init_logging(self):
            """Initialize logging to save tracking data to a CSV file."""
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Duration (seconds)"])

        def log_tracking_data(self, duration):
            """Log tracking data to a CSV file."""
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), round(duration, 2)])

        def is_point_in_polygon(self, point):
            """Check if a point is inside the polygon."""
            if self.polygon is None:
                return False
            return cv2.pointPolygonTest(self.polygon, point, False) >= 0

        def format_time(self, seconds):
            """Format seconds into HH:MM:SS."""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        def manage_timer(self, truck_detected_in_polygon, current_time):
            """Manage the timer state with cooldown period."""
            if truck_detected_in_polygon:
                if not self.timer_active:
                    self.timer_active = True
                    self.timer_start = current_time
                self.last_detection_time = current_time
            elif self.timer_active:
                if (current_time - self.last_detection_time) > self.cooldown_period:
                    duration = current_time - self.timer_start
                    self.tracking_data.append({'duration': duration, 'timestamp': datetime.now()})
                    self.log_tracking_data(duration)
                    self.timer_active = False
                    self.timer_start = None

        def draw_overlay(self, frame, boxes_info):
            """Draw visualization elements on the frame."""
            if self.polygon is not None:
                # Draw polygon
                pts = self.polygon.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Draw timer if active
            if self.timer_active and self.timer_start is not None:
                elapsed_time = time.time() - self.timer_start
                timer_text = f"Loading Time: {self.format_time(elapsed_time)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2

                (text_width, text_height), _ = cv2.getTextSize(timer_text, font, font_scale, thickness)
                cv2.rectangle(frame, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
                cv2.putText(frame, timer_text, (15, 35), font, font_scale, (255, 255, 255), thickness)

            # Draw detection boxes and their info
            for box_info in boxes_info:
                x1, y1, x2, y2 = box_info['bbox']
                is_inside = box_info['is_inside']
                track_id = box_info['track_id']
                center_point = box_info['center']

                color = (0, 255, 0) if is_inside else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return frame

        def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list, bool]:
            """Process a frame with object detection and truck tracking."""
            try:
                truck_in_polygon = False
                boxes_info = []
                
                # Run YOLO tracking
                results = self.model.track(frame, persist=True, conf=self.confidence, classes=[self.truck_class])
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            if hasattr(box, 'cls') and box.cls.cpu().numpy()[0] == self.truck_class:
                                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                center_point = (center_x, center_y)

                                track_id = int(box.id.cpu().numpy()[0]) if hasattr(box, 'id') and box.id is not None else -1

                                if track_id != -1:
                                    is_inside = self.is_point_in_polygon(center_point)
                                    if is_inside:
                                        truck_in_polygon = True

                                    boxes_info.append({
                                        'bbox': (x1, y1, x2, y2),
                                        'center': center_point,
                                        'track_id': track_id,
                                        'is_inside': is_inside
                                    })
                
                # Manage the timer state
                self.manage_timer(truck_in_polygon, time.time())
                
                # Create annotated frame with overlays
                annotated_frame = self.draw_overlay(frame.copy(), boxes_info)
                
                return annotated_frame, boxes_info, truck_in_polygon
                
            except Exception as e:
                logging.error(f"Camera {self.camera_id}: Frame processing error: {str(e)}")
                return frame, [], False
        
        def get_statistics(self):
            """Calculate and return time statistics."""
            if not self.tracking_data:
                return {
                    'total_sessions': 0,
                    'avg_duration': 0,
                    'min_duration': 0,
                    'max_duration': 0,
                    'std_duration': 0
                }

            durations = [data['duration'] for data in self.tracking_data]
            stats = {
                'total_sessions': len(durations),
                'avg_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations)
            }
            return stats
        
    except Exception as e:
        raise RabsException(e, sys) from e


class SingleCameraSystemTruck:
    try:
        """Manages a single camera stream and its processing"""
        def __init__(self, camera_id: int, rtsp_link: str, email: str, model_path: str, confidence=0.3, cooldown_period=60):
            self.email = email
            self.camera_id = camera_id
            self.rtsp_url = rtsp_link
            self.model_path = model_path
            self.confidence = confidence
            self.cooldown_period = cooldown_period
            self.mongo_handler = MongoDBHandlerSaving()
            self.is_running = False
            self.last_frame = None


            camera_data = self.mongo_handler.fetch_camera_rtsp_by_email(self.email)

            if not camera_data:
                logging.error(f"No camera data found for email: {self.email}")
                return

            for camera in camera_data:
                try:
                    camera_id = camera["camera_id"]
                    rtsp_link = camera["rtsp_link"]
                    
                    # Parse polygon points from the database
                    polygon_points = None
                    if "polygonal_points" in camera and camera["polygonal_points"]:
                        try:
                            # Convert the string representation of polygon points to actual list of tuples
                            polygon_str = camera["polygonal_points"]
                            # Remove brackets and split by commas
                            points_str = polygon_str.strip('[]').split('), (')
                            
                            # Parse each point
                            polygon_points = []
                            for point_str in points_str:
                                point_str = point_str.replace('(', '').replace(')', '')
                                x, y = map(int, point_str.split(','))
                                polygon_points.append((x, y))
                                
                            logging.info(f"Camera {camera_id}: Parsed polygon points: {polygon_points}")
                        except Exception as e:
                            logging.error(f"Camera {camera_id}: Failed to parse polygon points: {str(e)}")
                            polygon_points = None

                    self.processor = CameraProcessorTruckYOLO(
                        camera_id, rtsp_link, self.model_path,
                        polygon_points=polygon_points,  # Pass the parsed polygon points
                        confidence=self.confidence, 
                        cooldown_period=self.cooldown_period)

                    logging.info(f"Single Camera {self.camera_id}: Initialized successfully")
                except Exception as e:
                    logging.error(f"Single Camera {self.camera_id}: Initialization failed: {str(e)}")
                    self.processor = None

        def start(self):
            """Starts the single camera stream"""
            if not self.processor:
                logging.error(f"Single Camera {self.camera_id}: Cannot start, processor is not initialized")
                return
            
            try:
                self.processor.stream.start()
                self.is_running = True
                logging.info(f"Single Camera {self.camera_id}: Stream started")
            except Exception as e:
                logging.error(f"Single Camera {self.camera_id}: Failed to start: {str(e)}")
                self.is_running = False

        def get_video_frames(self):
            """Generator function to yield video frames as bytes for HTTP streaming"""
            logging.info(f"Streaming video frames for Single Camera {self.camera_id}")
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            while self.is_running:
                if self.processor and not self.processor.stream.stopped:
                    ret, frame = self.processor.stream.read()
                    if ret:
                        processed_frame, _, _ = self.processor.process_frame(frame)
                        self.last_frame = processed_frame
                    else:
                        logging.warning(f"Single Camera {self.camera_id}: Failed to read frame, using last frame")
                        processed_frame = self.last_frame if self.last_frame is not None else blank_frame
                else:
                    logging.error(f"Single Camera {self.camera_id}: Stream is stopped or processor is None")
                    processed_frame = blank_frame

                frame_resized = cv2.resize(processed_frame, (640, 480))
                _, buffer = cv2.imencode(".jpg", frame_resized)
                yield (b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    buffer.tobytes() + b"\r\n")

                time.sleep(0.05)  # Small delay to control FPS

        def stop(self):
            """Stops the camera stream and releases resources"""
            self.is_running = False
            if self.processor:
                self.processor.stream.stop()
            cv2.destroyAllWindows()
            logging.info(f"Single Camera {self.camera_id}: Stream stopped")


    except Exception as e:
        raise RabsException(e, sys) from e




####################################################################################################################
                                                ## END ##
####################################################################################################################

