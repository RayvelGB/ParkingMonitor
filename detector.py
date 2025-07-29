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
    'database': 'parking_monitor_db'
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
        self.iou_threshold = 0.3
        self.use_intersection_only = False

        self.reload_configuration()

        self.CONFIRMATION_TIME = 5
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
                self.iou_threshold = settings.get('iou_threshold', 0.3)
                self.use_intersection_only = settings.get('use_intersection_only', False)
                print(f"[{time.strftime('%H:%M:%S')}] Loaded settings: DET={self.detection_threshold}, IOU={self.iou_threshold}, INTERSECT_ONLY={self.use_intersection_only}")

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
                        'occupied': False, 
                        'last_changed': time.time(), 
                        'entry_time': None, 
                        'detection_start_time': None
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
    
    # -- Calculation of Intersection over Union --
    def calculate_iou(self, rect, poly_points):
        car_box = box(rect[0], rect[1], rect[2], rect[3]) # Detected object box
        parking_poly = Polygon(poly_points) # User-made bounding box

        if not car_box.intersects(parking_poly):
            return 0.0
        
        inter_area = car_box.intersection(parking_poly).area
        union_area = car_box.union(parking_poly).area

        return inter_area / union_area
    
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
                self.available_slots = 0
                for idx, points in enumerate(self.parking_boxes):
                    occupied = False
                        
                    for bbox in boxes:
                        rect = (bbox[0], bbox[1], bbox[2], bbox[3])
                        
                        if self.use_intersection_only:
                            if self.check_intersection(rect, points):
                                occupied = True
                                break
                        else:
                            iou = self.calculate_iou(rect, points)
                            if iou > self.iou_threshold:
                                occupied = True
                                break
                    
                    zone_name = self.parking_boxes_data[idx].get('zone')
                    zone_prefix = f'{zone_name} - ' if zone_name else ''
                    
                    # -- Temporal Confirmation --
                    if occupied:
                        if self.slot_status[idx]['detection_start_time'] is None:
                            self.slot_status[idx]['detection_start_time'] = current_time
                        elif current_time - self.slot_status[idx]['detection_start_time'] >= self.CONFIRMATION_TIME:
                            if not self.slot_status[idx]['occupied']:

                                self.slot_status[idx]['occupied'] = True
                                self.slot_status[idx]['last_changed'] = current_time
                                self.slot_status[idx]['entry_time'] = current_time
                                log = f"[{time.strftime('%H:%M:%S')}] {zone_prefix}Slot {idx+1} OCCUPIED"
                                self.log_messages.append(log)
                                self.save_logs_to_db(log)

                    else:
                        self.slot_status[idx]['detection_start_time'] = None
                        if self.slot_status[idx]['occupied']:
                            duration = current_time - self.slot_status[idx]['entry_time'] if self.slot_status[idx]['entry_time'] else 0

                            if duration / 60 < 1:
                                log = f"[{time.strftime('%H:%M:%S')}] {zone_prefix}Slot {idx+1} VACATED - Occupied for {duration:.1f} seconds"
                            elif duration / 60 >= 1 and duration / 3600 < 1:
                                duration = duration / 60
                                log = f"[{time.strftime('%H:%M:%S')}] {zone_prefix}Slot {idx+1} VACATED - Occupied for {duration:.1f} minutes"
                            elif duration / 3600 >= 1:
                                duration = duration / 3600
                                log = f"[{time.strftime('%H:%M:%S')}] {zone_prefix}Slot {idx+1} VACATED - Occupied for {duration:.1f} hours"
                            self.log_messages.append(log)
                            self.save_logs_to_db(log)
                        
                        self.slot_status[idx]['occupied'] = False
                        self.slot_status[idx]['last_changed'] = current_time
                        self.slot_status[idx]['entry_time'] = None # Default back to None when vacated
                        
                    if len(self.log_messages) > 50: # Limit number of logs at 50 for better memory efficiency
                        self.log_messages.pop(0)
                        
                    if not occupied:
                        self.available_slots += 1
                        # Save number of available spaces

                    # -- Coloring of bouding boxes for visualization of occupancy and vacancy --
                    confirmed_occupancy = self.slot_status[idx]['occupied']
                    detection_start = self.slot_status[idx]['detection_start_time']
                    color = (0, 255, 0) # Green (Vacant)
                    thickness = 4

                    if confirmed_occupancy:
                        color = (0, 0, 255) # Red (Occupied)
                        thickness = 2
                    
                    elif detection_start is not None:
                        time_detected = current_time - detection_start
                        if time_detected < self.CONFIRMATION_TIME:
                            color = (0, 255, 255) # Yellow (Pending)
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