import cv2
import numpy as np
import logging
from ultralytics import YOLO

class PPEVideoTester:
    def __init__(self, model_path: str, video_path: str):
        """Initialize the PPE detection model and video stream"""
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            logging.error("Error: Unable to open video file.")
            raise ValueError("Could not open video file.")
    
    def process_video(self):
        """Run PPE detection on the video"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  # End of video
            
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            cv2.imshow("PPE Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "ppe.pt"  # Update with actual model path
    video_path = "Videos/fire_car.mp4"  # Update with actual video file
    
    tester = PPEVideoTester(model_path, video_path)
    tester.process_video()





