import cv2
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import json
from shapely import Polygon, box
import time
import mysql.connector

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'entry_db'
}

class VideoDetector:
    # -- Initialize Elements --
    def __init__(self, rtsp_url, camera_id):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.frame = None
        self.running = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        self.parking_boxes_data = []
        self.parking_boxes = []
        self.total_spaces = 0
        self.slot_status = {}
        self.log_messages = []
        self.available_slots = 0

        self.detection_threshold = 0.35

        self.reload_configuration()
        self.log_messages = self.load_logs_from_db()

    # -- Reload configuration without refresh --
    def reload_configuration(self):
        print(f"[{time.strftime('%H:%M:%S')}] Reloading configuration for camera {self.camera_id}...")
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
    
            cursor.execute("SELECT * FROM camera_settings WHERE camera_id = %s", (self.camera_id,))
            settings = cursor.fetchone()

            if settings:
                self.detection_threshold = settings.get('detection_threshold', 0.35)
                print(f"[{time.strftime('%H:%M:%S')}] Loaded settings: DET={self.detection_threshold}")

            cursor.execute("SELECT points, zone_name FROM bounding_boxes WHERE camera_id = %s ORDER BY box_index ASC", (self.camera_id,))
            result = cursor.fetchall()

            new_parking_data = [{'points': json.loads(row['points']), 'zone': row['zone_name']} for row in result]

            if new_parking_data != self.parking_boxes_data:
                print(f"[{time.strftime('%H:%M:%S')}] Change in bounding boxes detected. Resetting state.")

                if result:
                    self.parking_boxes_data = [{'points': json.loads(row['points']), 'zone': row['zone_name']} for row in result]
                    self.parking_boxes = [item['points'] for item in self.parking_boxes_data]
                else:
                    self.parking_boxes_data = []
                    self.parking_boxes = []

                self.total_spaces = len(self.parking_boxes)
                self.slot_status = {
                    idx: {
                        'entry': False, 
                        'last_changed': time.time(), 
                        'entry_time': None
                    } 
                    for idx in range(len(self.parking_boxes))}
                self.available_slots = self.total_spaces
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Settings updated. Preserving current occupancy state.")

            print(f"[{time.strftime('%H:%M:%S')}] Configuration reloaded. Found {self.total_spaces} spaces.")

            cursor.close()
            conn.close()

        except mysql.connector.Error as err:
            print(f"Error fetching bounding boxes for camera {self.camera_id}: {err}")
            self.parking_boxes = [] 

    # -- Load model trained on VisDrone Dataset --
    def load_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11) # 11 -> number of classes
        model.load_state_dict(torch.load('models/fasterrcnn_visdrone.pth', map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    # -- Do image transformation for prediction --
    def img_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img
    
    # -- Form predictions --
    def inference(self, img):
        with torch.no_grad():
            img = img.to(self.device)
            outputs = self.model([img])

            boxes = outputs[0]['boxes'].data.cpu().numpy()
            scores = outputs[0]['scores'].data.cpu().numpy()
            labels = outputs[0]['labels'].data.cpu().numpy()

            boxes = boxes[scores >= self.detection_threshold]
            labels = labels[scores >= self.detection_threshold]
            scores = scores[scores >= self.detection_threshold]

            return boxes, labels, scores
    
    # -- Check for intersection --
    def check_intersection(self, rect, poly_points):
        car_box = box(rect[0], rect[1], rect[2], rect[3]) # Detected object box
        parking_poly = Polygon(poly_points) # User-made bounding box

        return car_box.intersects(parking_poly)
    
    # -- Load Logs from Database --
    def load_logs_from_db(self):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            # Get the 100 most recent logs for this camera
            query = "SELECT message FROM logs WHERE camera_id = %s ORDER BY timestamp DESC LIMIT 100"
            cursor.execute(query, (self.camera_id,))
            logs = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return list(reversed(logs[-50:])) # Reverse to show oldest first
        
        except mysql.connector.Error as err:
            print(f"Error loading logs from DB: {err}")
            return []
        
    # -- Save Logs into Database --
    def save_logs_to_db(self, log_message):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            query = "INSERT INTO logs (camera_id, message) VALUES (%s, %s)"
            cursor.execute(query, (self.camera_id, log_message))
            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"Error saving log to DB: {err}")

    # -- Run the detection --
    def run(self):
        isConnected = False
        if '?rtsp_transport=' not in self.rtsp_url: # Quality control to force tcp for better performance
            self.rtsp_url += '?rtsp_transport=tcp'

        while self.running:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0) # Force FFMPEG and reduce buffer size for better performance

            if not cap.isOpened():
                if isConnected:
                    print(f"[{time.strftime('%H:%M:%S')}] Error: Could not open video stream. Retrying in 5 seconds...")
                    isConnected = False
                continue
            
            if not isConnected:
                print(f"[{time.strftime('%H:%M:%S')}] Successfully connected to video stream.")
                isConnected = True

            while self.running:
                success, frame = cap.read()
                # Automatically reconnect when timed out
                if not success:
                    if isConnected:
                        print(f"[{time.strftime('%H:%M:%S')}] Connection lost. Reconnecting...")
                        isConnected = False
                    break # Break out of loop to restart

                img_tensor = self.img_transform(frame)
                boxes, labels, scores = self.inference(img_tensor)
                car_indices = np.isin(labels, [4, 5, 6, 9]) # Class IDs for car, truck, etc.
                boxes = boxes[car_indices]

                current_time = time.time()
                for idx, points in enumerate(self.parking_boxes):
                    entry = False
                        
                    for bbox in boxes:
                        rect = (bbox[0], bbox[1], bbox[2], bbox[3])
                        
                        if self.check_intersection(rect, points):
                            entry = True
                            break

                    previous_entry = self.slot_status[idx]['entry']
                    self.slot_status[idx]['entry'] = entry
                    
                    zone_name = self.parking_boxes_data[idx].get('zone')
                    zone_prefix = f'{zone_name} - ' if zone_name else ''
                
                    if not previous_entry and entry:
                        log = f"[{time.strftime('%H:%M:%S')}] {zone_prefix}Slot {idx+1} ENTRY"
                        self.available_slots = max(0, self.available_slots - 1)
                        self.log_messages.append(log)
                        self.save_logs_to_db(log)
                        
                    if len(self.log_messages) > 50: # Limit number of logs at 50 for better memory efficiency
                        self.log_messages.pop(0)

                    # -- Coloring of bouding boxes for visualization of occupancy and vacancy --
                    color = (0, 255, 0) # Green (Vacant)
                    thickness = 4

                    if entry:
                        color = (0, 0, 255) # Red (Occupied)
                        thickness = 2
                    
                    pts = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
                
                self.frame = frame

            cap.release()
            time.sleep(5)
    
    # -- Generate video feed --
    def generate(self):
        try:
            while True:
                if self.frame is None:
                    time.sleep(0.03) # Interval of .03 seconds if failed
                    continue
                ret, buffer = cv2.imencode('.jpg', self.frame)
                frame_bytes = buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') # Yield feed
        except GeneratorExit:
            pass
        finally:
            pass
    
    # -- Stop video feed     --
    def stop(self):
        print('[INFO] Client disconnected. Stopping stream...')
        self.running = False