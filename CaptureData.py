import cv2
import mediapipe as mp
import os
import keyboard
import random
from ultralytics import YOLO 
class HandDetectionApp:
    def __init__(self):
        self.mode = "detection"
        self.label = -1
        self.capture_counter = 0
        self.capture = cv2.VideoCapture(0)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create dataset directories
        self.dataset_dir = "./dataset"
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.labels_dir = os.path.join(self.dataset_dir, "labels")

        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_dir, split), exist_ok=True)

    def save_data_and_image(self, image, label, bbox, multi_hand_landmarks):
        # Split data into training and validation sets randomly
        split = "train" if random.random() < 0.8 else "val"

        # Define folder paths
        image_folder_path = os.path.join(self.images_dir, split)
        label_folder_path = os.path.join(self.labels_dir, split)

        # Get a unique filename for the image
        self.capture_counter, image_filename = self.get_unique_filename(image_folder_path, self.label)

        # Save image
        image_path = os.path.join(image_folder_path, image_filename)
        cv2.imwrite(image_path, image)

        # Save bounding box data
        data_path = os.path.join(label_folder_path, f"{self.label}_{self.capture_counter}.txt")
        with open(data_path, "w") as f:
            # Convert bbox to YOLO format (normalized x_center, y_center, width, height)
            h, w, _ = image.shape
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            width = bbox[2] / w
            height = bbox[3] / h
            f.write(f"{label} {x_center} {y_center} {width} {height}")
            for hand_landmarks in multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # Normalize and write x, y coordinates
                    x_normalized = landmark.x
                    y_normalized = landmark.y
                    if x_normalized >= 0.999:
                        x_normalized = 1
                    if y_normalized >= 0.999:
                        y_normalized = 1
                    f.write(f" {x_normalized} {y_normalized} ")
        self.capture_counter += 1
    
    def get_unique_filename(self, folder_path, label):
        counter = self.capture_counter
        while True:
            filename = f"{label}_{counter}.jpg"
            if not os.path.exists(os.path.join(folder_path, filename)):
                break
            counter += 1
        return counter, filename

    def run(self):
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to read frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            bbox = None

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                    y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                    x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                    y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                    h, w, _ = frame.shape
                    x_min = int(x_min * w)
                    y_min = int(y_min * h)
                    x_max = int(x_max * w)
                    y_max = int(y_max * h)

                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    #draw the Box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    #draw the LandMarks
                    # self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if self.mode == "detection":
                cv2.putText(frame, f"Mode: Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self.mode == "training":
                cv2.putText(frame, f"Mode: Training - Label: {self.label} {self.capture_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Hand Detection or Training', frame)
            
            if keyboard.is_pressed('/'):
                self.mode = "detection"
                print("Detection mode")
            elif keyboard.is_pressed('*'):
                self.mode = "training"
                print("Training mode")
            elif self.mode == "training" and keyboard.is_pressed('a'):
                key_event = 0
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('b'):
                key_event = 1
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('c'):
                key_event = 2
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('d'):
                key_event = 3
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('e'):
                key_event = 4
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('f'):
                key_event = 5
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('g'):
                key_event = 6
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('h'):
                key_event = 7
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('i'):
                key_event = 8
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('j'):
                key_event = 9
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('k'):
                key_event = 10
                self.label = key_event  
            elif self.mode == "training" and keyboard.is_pressed('l'):
                key_event = 11
                self.label = key_event     
            elif self.mode == "training" and keyboard.is_pressed('m'):
                key_event = 12
                self.label = key_event   
            elif self.mode == "training" and keyboard.is_pressed('n'):
                key_event = 13
                self.label = key_event 
            elif self.mode == "training" and keyboard.is_pressed('o'):
                key_event = 14
                self.label = key_event  
            elif self.mode == "training" and keyboard.is_pressed('p'):
                key_event = 15
                self.label = key_event   
            elif self.mode == "training" and keyboard.is_pressed('q'):
                key_event = 16
                self.label = key_event  
            elif self.mode == "training" and keyboard.is_pressed('r'):
                key_event = 17
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('s'):
                key_event = 18
                self.label = key_event   
            elif self.mode == "training" and keyboard.is_pressed('t'):
                key_event = 19
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('u'):
                key_event = 20
                self.label = key_event         
            elif self.mode == "training" and keyboard.is_pressed('v'):
                key_event = 21
                self.label = key_event  
            elif self.mode == "training" and keyboard.is_pressed('w'):
                key_event = 22
                self.label = key_event
            elif self.mode == "training" and keyboard.is_pressed('x'):
                key_event = 23
                self.label = key_event  
            elif self.mode == "training" and keyboard.is_pressed('y'):
                key_event = 24
                self.label = key_event  
            elif self.mode == "training" and keyboard.is_pressed('z'):
                key_event = 25
                self.label = key_event  
            elif self.mode == "training" and keyboard.is_pressed('+') and self.label != -1:
                if bbox:
                    self.save_data_and_image(frame, self.label, bbox, result.multi_hand_landmarks)
                    print("Image and data captured.")
                else:
                    print("No hand detected to save.")
            elif keyboard.is_pressed('esc'):
                break

            cv2.waitKey(1)

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandDetectionApp()
    app.run()
