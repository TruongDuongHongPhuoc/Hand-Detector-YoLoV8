import cv2
import torch
from PIL import Image
from ultralytics import YOLO  # Assuming you have a custom YOLOv8 implementation
import time
# Load the YOLOv8 model
model = YOLO("runs/pose/train6/weights/best.pt")
modelName = model.names
# cap = cv2.VideoCapture(0)
# # while True:
results = model.predict(source=0, show = True, save = False)
# print(" Hello World" +results[0])
# results = model.predict(source="D:/Python/HandDetector/dataset/images/train/2_246.jpg", save = False)
# for r in results:
#         if r.boxes.cls:  # Check if there are any detections
#             first_detected_class = modelName[int(r.boxes.cls[0])]
#$########################################################################

# first_detected_class = None
# while first_detected_class is None:
#             # Capture a frame from the camera
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to capture image")
#                 continue

#             # Save the captured frame as an image
#             image_path = "captured_image.jpg"
#             cv2.imwrite(image_path, frame)

#             # Run detection on the saved image
#             results = model.predict(source="D:/Python/TestYoloPoseVer2/dataset/images/train/0_1.jpg", show=True, save=False, conf=0.5)
#             if len(results) > 0:
#                 for r in results:
#                     if r.boxes.cls.numel() > 0:  # Check if there are any detections
#                         first_detected_class = modelName[int(r.boxes.cls[0])]
#                         print(first_detected_class)
#                         break
#                     else:
#                         time.sleep(1)
                    
                    