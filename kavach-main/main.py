import os
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import threading
from flask import jsonify

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best.pt')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.debug = True
socketio = SocketIO(app, cors_allowed_origins= "*")

CORS(app)

os.makedirs('recognized_plates', exist_ok=True)

def process_video():
    cap = cv2.VideoCapture(0)
    
    with open("coco1.txt", "r", encoding="utf-8") as my_file:
        class_list = my_file.read().split("\n")

    area = [(27, 217), (16, 456), (860, 456), (860, 217)]

    count = 0
    processed_numbers = set()

    with open("car_plate_data.txt", "a", encoding="utf-8") as file:
        file.write("NumberPlate\tDate\tTime\n")

    while True:    
        ret, frame = cap.read()
        count += 1
        if count % 4 != 0:
            continue
        if not ret:
            break

        frame = cv2.resize(frame, (888, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = class_list[d]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            if result >= 0:
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                text = pytesseract.image_to_string(gray).strip()
                text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')
                if text not in processed_numbers:
                    processed_numbers.add(text)
                    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    with open("car_plate_data.txt", "a", encoding="utf-8") as file:
                        file.write(f"{text}\t{current_datetime}\n")
                    image_filename = f"recognized_plates/{text}_{current_datetime}.jpg"
                    cv2.imwrite(image_filename, crop)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)


        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:

            jpeg_bytes = jpeg.tobytes()
            jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            socketio.emit('frame', jpeg_base64)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

@socketio.on('start')
def handle_connect():
    threading.Thread(target=process_video).start()

@app.get('/')
def handle_login():
    print('login')
    return jsonify({"message": "login"}) 

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

# import os
# import cv2
# import pandas as pd
# from ultralytics import YOLO
# import numpy as np
# import pytesseract
# from datetime import datetime
# from flask import Flask, jsonify
# from flask_socketio import SocketIO, emit
# from flask_cors import CORS
# import base64
# import threading

# # Set the path for Tesseract
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Load the YOLO model
# model = YOLO('best.pt')

# # Flask application setup
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# app.debug = True
# socketio = SocketIO(app, cors_allowed_origins="*")

# CORS(app)

# # Create directory for saving images
# os.makedirs('recognized_plates', exist_ok=True)

# # Function to process frames from a specific camera and emit to WebSocket clients
# def process_video():
#     cap = cv2.VideoCapture(0)  # Open the camera

#     # Read class names
#     with open("coco1.txt", "r", encoding="utf-8") as my_file:
#         class_list = my_file.read().split("\n")

#     # Define the polygon area
#     area = [(27, 217), (16, 456), (860, 456), (860, 217)]

#     count = 0
#     processed_numbers = set()

#     # Open file for writing car plate data
#     with open(f"car_plate_data.txt", "a", encoding="utf-8") as file:
#         file.write("NumberPlate\tDate\tTime\n")  # Writing column headers

#     while True:    
#         ret, frame = cap.read()
#         count += 1
#         if count % 4 != 0:
#             continue
#         if not ret:
#             break

#         frame = cv2.resize(frame, (888, 500))
#         results = model.predict(frame)
#         a = results[0].boxes.data
#         px = pd.DataFrame(a).astype("float")

#         for index, row in px.iterrows():
#             x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
#             d = int(row[5])
#             c = class_list[d]
#             cx = (x1 + x2) // 2
#             cy = (y1 + y2) // 2
#             result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
#             if result >= 0:
#                 crop = frame[y1:y2, x1:x2]
#                 gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#                 gray = cv2.bilateralFilter(gray, 10, 20, 20)

#                 text = pytesseract.image_to_string(gray).strip()
#                 text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')
#                 if text not in processed_numbers:
#                     processed_numbers.add(text)
#                     current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#                     with open(f"car_plate_data.txt", "a", encoding="utf-8") as file:
#                         file.write(f"{text}\t{current_datetime}\n")
#                     image_filename = f"recognized_plates/{text}_{current_datetime}.jpg"
#                     cv2.imwrite(image_filename, crop)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

#         # Draw polygon area on frame
#         cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)

#         # Encode frame to JPEG
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if ret:
#             # Convert to Base64 and send via WebSocket
#             jpeg_bytes = jpeg.tobytes()
#             jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
#             socketio.emit(f'frame_0', {'camera_index': 0, 'frame': 0})

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Define separate SocketIO events for each camera
# @socketio.on('start_camera_0')
# def handle_start_camera_0():
#     threading.Thread(target=process_video).start()

# @socketio.on('start_camera_1')
# def handle_start_camera_1():
#     threading.Thread(target=process_video).start()

# @socketio.on('start_camera_2')
# def handle_start_camera_2():
#     threading.Thread(target=process_video).start()

# @socketio.on('start_camera_3')
# def handle_start_camera_3():
#     threading.Thread(target=process_video).start()

# @app.route('/')
# def index():
#     return jsonify({"message": "Welcome to the multi-camera Flask app"})

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=5000)
