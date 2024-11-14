from ultralytics import YOLO
dataset_yaml = """
train:  D:/Python/HandDetector/dataset/images/train
val: D:/Python/HandDetector/dataset/images/val
kpt_shape: [21, 2]  
flip_idx: [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
nc: 16
names:
  0: A
  1: B
  2: C
  3: D
  4: E
  5: F
  6: G
  7: H
  8: I  
  9: J
  10: K
  11: L
  12: M
  13: N
  14: O
  15: P
"""
with open("dataset.yaml", "w") as f:
    f.write(dataset_yaml)

# Load a model
# model = YOLO("yolov8n-pose.yaml")
# model.train(
#   data=" D:/Python/TestYoloPoseVer2/dataset.yaml",
#   model = 'yolov8n-pose.yaml',
#   workers = 6,
#   batch = 12,
#   dynamic = True,
#   nms = True,
#   pretrained = False,
#   epochs=5,
#   name = 'CustomizeYolow',
#   imgsz=640)

# model.train(
#     data="D:/Python/HandDetector/dataset.yaml",
#     workers=6,
#     batch=12,
#     epochs=5,
#     imgsz=640, 
#     pretrained=False,
#     name='YoloPoseHandGesture'
# )

model = YOLO("D:/Python/HandDetector/runs/pose/train5/weights/last.pt") 
model.train(data="D:/Python/HandDetector/dataset.yaml", epochs=10, imgsz=640 , pretrained = True, workers = 6)
# model = YOLO("yolov8n-pose.yaml").load("D:/Python/TestYoloPoseVer2/runs/pose/train12/weights/best.pt")
# model.train(data="D:/Python/TestYoloPoseVer2/dataset.yaml", epochs=1, imgsz=640, resume = True)
# model.train(resume = True)