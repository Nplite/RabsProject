from ultralytics import YOLO
import cv2
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

model = YOLO("MODELS/SMOKE/combine_smoke/weights/best.pt")
source = r"Videos/unloading_video.mp4"

cap = cv2.VideoCapture(source)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tracker_results = model.track(source, stream=True, save=False)

for result in tracker_results:
    frame = result.orig_img
    smoke_detected = False
    for box in result.boxes:
        cls = int(box.cls[0])  
        track_id = box.id  
        label = f"Smoke Detected" 
        if cls == 0:  
            smoke_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 

    if smoke_detected:
        cv2.putText(frame, "SMOKE DETECTED! ALARM!", (5, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("smoke Detection with Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













