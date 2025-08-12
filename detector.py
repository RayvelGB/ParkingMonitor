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
from threading import Thread, Lock

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'entry_db'
}

class VideoDetector:
    # -- Initialize Elements --
    def __init__(self, rtsp_url, camera_id, event_callback=None):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.camera_name = 'Unnamed'
        self.event_callback = event_callback

        self.raw_frame = None
        self.processed_frame = None
        self.frame_lock = Lock()
        self.running = True

        if '?rtsp_transport=' not in self.rtsp_url:
            self.rtsp_url += '?rtsp_transport=tcp'
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        self.parking_boxes_data = []
        self.slot_status = {}
        self.log_messages = []
        self.detection_threshold = 0.35

        self.zone_counts = {}

        self.reload_configuration()
        self.log_messages = self.load_logs_from_db()

        self.reader_thread = Thread(target=self._frame_reader, daemon=True)
        self.reader_thread.start()

    def _frame_reader(self):
        print(f"[{time.strftime('%H:%M:%S')}] Cam {self.camera_id}: Starting frame reader thread.")
        while self.running:
            if not self.cap.isOpened():
                print(f"[{time.strftime('%H:%M:%S')}] Cam {self.camera_id}: Stream disconnected. Reconnecting...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                time.sleep(5)
                continue
            
            success, frame = self.cap.read()
            if success:
                with self.frame_lock:
                    self.raw_frame = frame
            else:
                self.cap.release()
        print(f"[{time.strftime('%H:%M:%S')}] Cam {self.camera_id}: Frame reader thread stopped.")

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

            cursor.execute("SELECT id, zone_name, total_spaces FROM zones WHERE camera_id = %s", (self.camera_id,))
            zones = cursor.fetchall()
            self.zone_counts = {
                zone['id']: {
                    'name': zone['zone_name'],
                    'total': zone['total_spaces'],
                    'available': zone['total_spaces']
                } for zone in zones
            }
            print(f"Loaded {len(self.zone_counts)} zones for camera {self.camera_id}.")

            cursor.execute("SELECT box_index, points, zone_id, mode FROM bounding_boxes WHERE camera_id = %s ORDER BY box_index ASC", (self.camera_id,))
            result = cursor.fetchall()

            new_parking_data = [{'points': json.loads(row['points']), 'zone_id': row['zone_id'], 'mode': row['mode']} for row in result]

            if new_parking_data != self.parking_boxes_data:
                print(f"[{time.strftime('%H:%M:%S')}] Change in bounding boxes detected. Resetting state.")
                self.parking_boxes_data = new_parking_data
                self.slot_status = {idx: {} for idx in range(len(self.parking_boxes_data))}

            else:
                print(f"[{time.strftime('%H:%M:%S')}] Settings updated. Preserving current occupancy state.")

            print(f"[{time.strftime('%H:%M:%S')}] Configuration reloaded.")

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

    # -- Run the detection --
    def run(self):
        print(f"[{time.strftime('%H:%M:%S')}] Cam {self.camera_id}: Starting processing loop.")
        while self.running:
            local_frame_copy = None
            with self.frame_lock:
                if self.raw_frame is not None:
                    local_frame_copy = self.raw_frame.copy()

            if local_frame_copy is None:
                time.sleep(0.05)
                continue

            img_tensor = self.img_transform(local_frame_copy)
            boxes, labels, _ = self.inference(img_tensor)
            car_indices = np.isin(labels, [4, 5, 6, 9])
            boxes = boxes[car_indices]

            for idx, box_data in enumerate(self.parking_boxes_data):
                points = box_data['points']
                box_mode = box_data['mode']

                is_occupied_now = any(self.check_intersection((b[0], b[1], b[2], b[3]), points) for b in boxes)
                previous_state = self.slot_status.get(idx, {}).get('occupied', False)
                
                # Event fires only on the frame the state changes
                if is_occupied_now and not previous_state:
                    if idx not in self.slot_status: self.slot_status[idx] = {}
                    self.slot_status[idx]['occupied'] = True

                    if box_mode == 'entry':
                        if self.event_callback:
                            self.event_callback(self.camera_id, idx, 'ENTRY')

                elif not is_occupied_now and previous_state:
                    if idx not in self.slot_status: self.slot_status[idx] = {}
                    self.slot_status[idx]['occupied'] = False

                    if box_mode == 'exit':
                        if self.event_callback:
                            self.event_callback(self.camera_id, idx, 'EXIT')

                color = (0, 0, 255) if is_occupied_now else (0, 255, 0)
                thickness = 2
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(local_frame_copy, [pts], isClosed=True, color=color, thickness=thickness)
            
            with self.frame_lock:
                self.processed_frame = local_frame_copy
            
            time.sleep(0.01)
    
    # -- Generate video feed --
    def generate(self):
        try:
            frame_to_send = None
            while self.running:
                with self.frame_lock:
                    if self.processed_frame is not None:
                        frame_to_send = self.processed_frame.copy()
                
                if frame_to_send is None:
                    time.sleep(0.3)
                    continue

                ret, buffer = cv2.imencode('.jpg', frame_to_send)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield(b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') # Yield feed
        except GeneratorExit:
            pass
        finally:
            pass
    
    # -- Stop video feed     --
    def stop(self):
        print(f'[INFO] Stopping all threads for camera {self.camera_id}...')
        self.running = False
        if self.reader_thread.is_alive():
            self.reader_thread.join()
        self.cap.release()