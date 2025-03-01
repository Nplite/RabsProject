from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model
model = YOLO("FireDetection/fire_detection.pt")

# Define path to the video file
source = r"Videos/fre.mp4"

# Open video source
cap = cv2.VideoCapture(source)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video path
output_path = "D:/AlluviumIOT/RABSINDUSTRIES/processed_fire_tracking2.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame with YOLO tracking
tracker_results = model.track(source, stream=True, save=False)

for result in tracker_results:
    # Read the current frame
    frame = result.orig_img

    # Draw "RABS INDUSTRIES" text at the top center
    cv2.putText(frame, "RABS INDUSTRIES", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Check for fire detections
    fire_detected = False
    for box in result.boxes:
        cls = int(box.cls[0])  # Class index
        track_id = box.id  # Tracking ID of the object
        label = f"Fire Detected"  # Label for the detected object
        if cls == 0:  # Assuming 'fire' class index is 0
            fire_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # Add label

    if fire_detected:
        cv2.putText(frame, "FIRE DETECTED! ALARM!", (5, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame with overlays
    cv2.imshow("Fire Detection with Tracking", frame)
    out.write(frame)  # Save the frame to output video

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video with tracking saved at {output_path}")
