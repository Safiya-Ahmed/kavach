import os
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime

# Tesseract configuration
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize YOLO model
model = YOLO('best.pt')

# Mouse event callback function for debugging
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Create a named window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture('mycarplate2.mp4')

# Read class names from the file
with open("coco1.txt", "r", encoding="utf-8") as my_file:
    class_list = my_file.read().split("\n")

# Define the area for license plate detection
area = [(27, 217), (16, 456), (1015, 451), (992, 217)]

count = 0
list1 = []
processed_numbers = set()

# Create directory for saving recognized plate images
if not os.path.exists('recognized_plates'):
    os.makedirs('recognized_plates')

# Open file for writing car plate data
with open("car_plate_data.txt", "a", encoding="utf-8") as file:
    file.write("NumberPlate\tDate\tTime\n")  # Writing column headers

while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        if result >= 0:
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 10, 20, 20)

            text = pytesseract.image_to_string(gray).strip()
            text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')
            if text not in processed_numbers:
                processed_numbers.add(text)
                list1.append(text)
                current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                filename = f'recognized_plates/{text}_{current_datetime}.png'
                cv2.imwrite(filename, crop)
                with open("car_plate_data.txt", "a", encoding="utf-8") as file:
                    file.write(f"{text}\t{current_datetime}\n")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.imshow('crop', crop)

    print(list1)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
