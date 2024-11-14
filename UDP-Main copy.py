import socket
import time
import cv2
import json
from ultralytics import YOLO

UDP_IP = "127.0.0.1"
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

client_address = None
isDetect = False

# Load trained model
model = YOLO("runs/pose/train6/weights/best.pt")
modelName = model.names
previousMessage = None

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Wait for initial message from Unity to get client address
    if client_address is None:
        data, addr = sock.recvfrom(1024)
        client_address = addr
        print("Client address:", client_address)
        continue

    # Receive data from Unity
    data, addr = sock.recvfrom(1024)
    message = data.decode('utf-8').strip()
    print(f"Received message: '{message}' from {addr}")

    # Check the message and set detection flag
    if message == "True":
        isDetect = True
        print("Detection enabled (isDetect = True)")
    elif message == "False":
        isDetect = False
        print("Detection disabled (isDetect = False)")
    else:
        print(f"Unrecognized message: '{message}'")

    while isDetect:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        # Save the captured frame as an image
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)

        # Run detection on the saved image
        results = model.predict(source=image_path, show=False, save=False, conf=0.5)
        
        first_detected_class = None
        if len(results) > 0:
            for r in results:
                if r.boxes.cls.numel() > 0:  # Check if there are any detections
                    first_detected_class = modelName[int(r.boxes.cls[0])]
                    first_detected_class = "_" + first_detected_class + "_"
                    jsonData = json.dumps({"result": first_detected_class})
                    sock.sendto(jsonData.encode('utf-8'), client_address)
                    print(first_detected_class)
                    break

        if first_detected_class is None:
            first_detected_class = "_NULL_"
            jsonData = json.dumps({"result": first_detected_class})
            sock.sendto(jsonData.encode('utf-8'), client_address)
            print("Sent data to Unity:", jsonData)

        # After detecting one picture, break the inner loop
        isDetect = False
        print("Detection loop exited, waiting for next command.")
        time.sleep(1)
