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

# DB_CONFIG = {
#     'host': '192.168.0.29',
#     'user': 'root',
#     'password': '1234',
#     'database': 'entry_db'
# }

# DGROUP_CONFIG = {
#     'host': '192.168.0.29',
#     'user': 'root',
#     'password': '1234',
#     'database': 'sap8di'
# }

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'entry_db'
}

class TripwireDetector:
    # -- Initialize Elements --
    def __init__(self, rtsp_url, camera_id, event_callback=None):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.camera_name = 'Unnamed'
        self.event_callback = event_callback
        self.detection_type = 'tripwire'

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

            query = """
                SELECT
                    z.id,
                    z.zone_name,
                    dg.jml AS total_spaces,
                    dg.cup,
                    dg.cdw
                FROM
                    zones z
                LEFT JOIN
                    sap8di.dgroup dg on z.id = dg.zone_id
                WHERE
                    camera_id = %s
                ORDER BY
                    z.id ASC
            """

            cursor.execute(query, (self.camera_id,))
            zones = cursor.fetchall()
            self.zone_counts = {
                zone['id']: {
                    'name': zone['zone_name'],
                    'total': zone['total_spaces'],
                    'available': zone['total_spaces'] - (zone['cdw'] - zone['cup'])
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
                time.sleep(0.01)
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
            
            time.sleep(0.005)
    
    # -- Generate video feed --
    def generate(self):
        try:
            frame_to_send = None
            while self.running:
                with self.frame_lock:
                    if self.processed_frame is not None:
                        frame_to_send = self.processed_frame.copy()
                
                if frame_to_send is None:
                    time.sleep(0.01)
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

class ParkingDetector:
    # -- Initialize Elements --
    def __init__(self, rtsp_url, camera_id):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.camera_name = 'Unnamed'
        self.detection_type = 'parking'

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
        self.iou_threshold = 0.3
        self.use_intersection_only = False
        self.confirmation_time = 10

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
                self.iou_threshold = settings.get('iou_threshold', 0.3)
                self.confirmation_time = settings.get('confirmation_time', 10)

            cursor.execute("SELECT id, zone_name FROM zones WHERE camera_id = %s", (self.camera_id,))
            zones = cursor.fetchall()
            
            cursor.execute("SELECT points, zone_id FROM bounding_boxes WHERE camera_id = %s ORDER BY box_index ASC", (self.camera_id,))
            self.parking_boxes_data = cursor.fetchall()
            
            new_zone_counts = {}
            for zone in zones:
                zone_id = zone['id']
                total_in_zone = sum(1 for box in self.parking_boxes_data if box['zone_id'] == zone_id)
                
                # Preserve the current available count if possible, otherwise reset it
                current_available = self.zone_counts.get(zone_id, {}).get('available', total_in_zone)

                new_zone_counts[zone_id] = {
                    'name': zone['zone_name'],
                    'total': total_in_zone,
                    'available': current_available
                }
            self.zone_counts = new_zone_counts
            
            self.slot_status = {
                idx: {'occupied': False, 'detection_start_time': None, 'entry_time': None} 
                for idx in range(len(self.parking_boxes_data))
            }
            
            print(f"Configuration reloaded. Found {len(self.parking_boxes_data)} boxes across {len(self.zone_counts)} zones.")
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"Error reloading configuration: {err}")
            self.parking_boxes_data = []

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

    def get_camera_ip(self, url):
        no_proto = url.split('://', 1)[1]
        if '@' in no_proto:
            no_proto = no_proto.split('@', 1)[1]

        camera_ip = no_proto.split(':')[0].split('/')[0]

        return camera_ip

    # -- Run the detection --
    def run(self):
        while self.running:
            local_frame_copy = None
            with self.frame_lock:
                if self.raw_frame is not None:
                    local_frame_copy = self.raw_frame.copy()

            if local_frame_copy is None:
                time.sleep(0.01)
                continue

            img_tensor = self.img_transform(local_frame_copy)
            boxes, labels, _ = self.inference(img_tensor)
            car_indices = np.isin(labels, [4, 5, 6, 9])
            boxes = boxes[car_indices]

            current_time = time.time()
            occupied_per_zone = {zone_id: 0 for zone_id in self.zone_counts}

            for idx, box_data in enumerate(self.parking_boxes_data):
                points = json.loads(box_data['points'])
                zone_id = box_data.get('zone_id')
                is_occupied_now = any(self.calculate_iou(b, points) > self.iou_threshold for b in boxes)
                
                status = self.slot_status[idx]

                if is_occupied_now:
                    if status['detection_start_time'] is None:
                        status['detection_start_time'] = current_time
                    elif current_time - status['detection_start_time'] >= self.confirmation_time:
                        if not status['occupied']:
                            status['occupied'] = True
                            status['entry_time'] = current_time
                            log = f"[{time.strftime('%H:%M:%S')}] '{self.zone_counts.get(zone_id, {}).get('name', 'N/A')}' - Slot {idx+1} OCCUPIED"
                            self.log_messages.append(log)
                            self.save_logs_to_db(log)
                else:
                    if status['occupied']:
                        status['occupied'] = False
                        duration = current_time - status['entry_time'] if status['entry_time'] else 0
                        log = f"[{time.strftime('%H:%M:%S')}] '{self.zone_counts.get(zone_id, {}).get('name', 'N/A')}' - Slot {idx+1} VACATED ({duration/60:.1f} mins)"
                        self.log_messages.append(log)
                        self.save_logs_to_db(log)
                    status['detection_start_time'] = None

                if status['occupied'] and zone_id in occupied_per_zone:
                    occupied_per_zone[zone_id] += 1
                
                # Drawing Logic...
                color = (0, 255, 0)
                if status['occupied']: color = (0, 0, 255)
                elif status['detection_start_time'] is not None: color = (0, 255, 255)
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(local_frame_copy, [pts], isClosed=True, color=color, thickness=2)

            # Update available counts for each zone
            for zone_id, counts in self.zone_counts.items():
                counts['available'] = counts['total'] - occupied_per_zone.get(zone_id, 0)
            
            with self.frame_lock:
                self.processed_frame = local_frame_copy

            time.sleep(0.005)
    
    # -- Generate video feed --
    def generate(self):
        try:
            frame_to_send = None
            while self.running:
                with self.frame_lock:
                    if self.processed_frame is not None:
                        frame_to_send = self.processed_frame.copy()
                
                if frame_to_send is None:
                    time.sleep(0.01)
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