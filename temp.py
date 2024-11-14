import socket
import time
import cv2
from ultralytics import YOLO 
UDP_IP = "127.0.0.1"
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# store client address
client_address = None

#Load trained Model
model = YOLO("runs/pose/train9/weights/best.pt")

#open the camera
cap = cv2.VideoCapture(0)

#Logic Loop
while True:
    data, addr = sock.recvfrom(1024)
    
    # Nếu chưa có địa chỉ client, lưu lại địa chỉ của client
    if client_address is None:
        client_address = addr
        print("Client address:", client_address)

    # Gửi dữ liệu liên tục cho client từ địa chỉ đã xác nhận
    while True:
        ret, frame = cap.read()
        results = model.predict(frame,conf = 0.5)
        # cv2.imshow("",frame)
        for result in results:
            labels = [model.names[int(cls)] for cls in result.boxes.cls]
            print("Detected labels:", labels)
            warp_packed = str(labels)
            if client_address:
            # Send the serialized message to Unity
                sock.sendto(warp_packed.encode('utf-8'), client_address)
                print("Sent to Unity:", warp_packed)
                time.sleep(5)  # Wait 1 second before sending the next data

        # message = model.predict(source=0, show = False, save = False, conf = 0.5)
        # warp_packed = str(message)
        # if client_address:
        #     # Send the serialized message to Unity
        #     sock.sendto(warp_packed.encode('utf-8'), client_address)
        #     print("Sent to Unity:", message)
        #     time.sleep(1)  # Wait 1 second before sending the next data
