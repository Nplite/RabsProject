import sys,os
from RabsPipeline.pipeline.training_pipeline import TrainPipeline
from RabsPipeline.utils.main_utils import decodeImage, encodeImageIntoBase64






obj = TrainPipeline()
obj.run_pipeline()

# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model_path = "artifacts/model_trainer/best.pt"  # Update with your model's path
# model = YOLO(model_path)

# # Path to the input video
# input_video_path = 0  # Update with your input video path
# # output_video_path = "path/to/save/output/video.mp4"  # Path to save the output video

# # Open the input video
# cap = cv2.VideoCapture(input_video_path)

# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
# # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform object detection
#     results = model(frame)

#     # Visualize results on the frame
#     annotated_frame = results[0].plot()

#     # Write the frame to the output video
#     # out.write(annotated_frame)

#     # Optionally display the frame in a window
#     cv2.imshow("YOLOv8 Detection", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video objects
# cap.release()
# # out.release()
# cv2.destroyAllWindows()
