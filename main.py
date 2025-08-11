from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from detector import VideoDetector
from threading import Thread, Lock
import uuid
import json, cv2, os, io
import mysql.connector
from passlib.context import CryptContext
import time
import schedule

app = FastAPI()
video_detectors = {}
detector_lock = Lock()

app.mount('/static', StaticFiles(directory='static'), name='static')

# -- STARTUP ENDPOINT --

@app.on_event('startup')
def on_startup():
    createDB_and_tables()
    start_all_detectors()
    
    cleanup_thread = Thread(target=run_log_cleanup_scheduler, daemon=True)
    cleanup_thread.start()

# -- Database Configuration --
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'entry_db'
}

# Setup database
def createDB_and_tables():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(191) UNIQUE NOT NULL,
                hashed_password VARCHAR(191) NOT NULL
            )
        """)

        # Create cameras table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id VARCHAR(36) PRIMARY KEY,
                camera_ip VARCHAR(50),
                user_id INT,
                camera_name VARCHAR(50) NOT NULL,
                stream_url VARCHAR(191) NOT NULL,
                total_spaces INT DEFAULT 0,
                       
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Create camera settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id VARCHAR(36) UNIQUE,
                detection_threshold FLOAT DEFAULT 0.35,
                       
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
            )
        """)

        # Create Bounding Boxes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bounding_boxes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id VARCHAR(36),
                box_index INT,
                points TEXT,
                mode VARCHAR(10) DEFAULT 'entry',
                       
                FOREIGN KEY (camera_id) REFERENCES cameras (id) ON DELETE CASCADE,
                UNIQUE KEY (camera_id, box_index)
            )
        """)

        # Create logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id VARCHAR(36),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                message VARCHAR(255),
                       
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
            )

        """)

        conn.commit()
        cursor.close()
        conn.close()

        print('Database and tables are ready.')
    except mysql.connector.Error as err:
        print(f'Error creating database and tables: {err}')

# -- Stream Endpoints --

def start_all_detectors():
    print('Starting detectors for all registered cameras...')
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, stream_url, camera_name, total_spaces FROM cameras")
        all_cameras = cursor.fetchall()
        cursor.close()
        conn.close()

        for camera in all_cameras:
            if camera['id'] not in video_detectors:
                start_single_detector(camera['id'], camera['stream_url'], camera['camera_name'], camera['total_spaces'])
        print(f'Completed startup. {len(video_detectors)} detectors are running.')

    except mysql.connector.Error as err:
        print(f'Database error during startup: {err}')

def stop_all_detectors():
    print('Stopping detectors for all registered cameras...')
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, stream_url FROM cameras")
        all_cameras = cursor.fetchall()
        cursor.close()
        conn.close()

        for camera in all_cameras:
            camera_id = camera['id']

            if camera_id in video_detectors:
                video_detectors[camera_id].stop()
                del video_detectors[camera_id]
        print('All detectors have been stopped.')

    except mysql.connector.Error as err:
        print(f'Database error during startup: {err}')

def start_single_detector(camera_id: str, rtsp_url: str, name: str, total_spaces: int):
    if camera_id not in video_detectors:
        print(f"Initializing detector for camera: {camera_id}")
        detector = VideoDetector(rtsp_url, camera_id, event_callback=handle_crossing_event)
        detector.camera_name = name
        detector.total_spaces = total_spaces
        detector.available_slots = total_spaces
        video_detectors[camera_id] = detector
        t = Thread(target=detector.run, daemon=True)
        t.start()

@app.get('/video_feed/{camera_id}')
def video_feed(camera_id: str):
    if camera_id not in video_detectors:
        return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
    detector = video_detectors[camera_id]
    return StreamingResponse(detector.generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/stop_all_streams')
def stop_stream():
    stop_all_detectors()
    return JSONResponse(content={'success': True})

# -- Serve HTML Pages --

@app.get('/')
def serve_login_page():
    return FileResponse('static/auth_user.html')

@app.get('/register_camera.html')
def serve_register_page():
    return FileResponse('static/register_camera.html')

@app.get('/picker.html')
def serve_picker_page():
    return FileResponse('static/picker.html')

@app.get('/set_spaces.html')
def serve_set_spaces_page():
    return FileResponse('static/set_spaces.html')

@app.get('/stream.html')
def serve_stream_page():
    return FileResponse('static/stream.html')


# -- Authentication Endpoints --

# Initialize password hash
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_camera_ownership(camera_id: str, user_id: int) -> bool:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM cameras WHERE id = %s AND user_id = %s", (camera_id, user_id))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except mysql.connector.Error:
        return False

@app.post('/register')
async def register_user(request: Request):
    data = await request.json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Username and password are required.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Username already exists.'})
        
        hashed_password = get_password_hash(password)
        cursor.execute('INSERT INTO users(username, hashed_password) VALUES (%s, %s)', (username, hashed_password))
        conn.commit()
        user_id = cursor.lastrowid

        cursor.close()
        conn.close()

        return JSONResponse(content={'success': True, 'user_id': user_id})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error {err}'})

@app.post('/login')
async def login_user(request: Request):
    data = await request.json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Username and password are required.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        cursor.execute('SELECT id, hashed_password FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if not user or not verify_password(password, user['hashed_password']):
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid username or password.'})
        
        start_all_detectors()
        return JSONResponse(content={'success': True, 'user_id': user['id'], 'username': username})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error {err}'})

# -- Camera Management --

@app.post('/register_camera')
async def register_camera(request: Request):
    data = await request.json()
    user_id = data.get('user_id')
    stream_url = data.get('stream_url')
    camera_name = data.get('camera_name', 'Unnamed Camera')

    camera_ip = get_camera_ip(stream_url)

    if not all([user_id, stream_url, camera_name]):
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Missing required fields.'})
    
    camera_id = str(uuid.uuid4()) # -> Give camera a unique ID
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO cameras (id, camera_ip, user_id, camera_name, stream_url) VALUES (%s, %s, %s, %s, %s)",
                       (camera_id, camera_ip, user_id, camera_name, stream_url)) 
        
        conn.commit()
        cursor.close()
        conn.close()

        start_all_detectors()
        return JSONResponse(content={'success': True, 'camera_id': camera_id})
    
    except mysql.connector.Error as err:    
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error {err}'})
    
@app.get('/get_cameras/{user_id}')
def get_cameras(user_id: int):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, camera_name, total_spaces FROM cameras WHERE user_id = %s", (user_id,))
        cameras = cursor.fetchall()
        cursor.close()
        conn.close()
        return JSONResponse(content=cameras)
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})
    
@app.post('/delete_camera/{camera_id}')
async def delete_camera(camera_id: str, request: Request):
    data = await request.json()
    user_id = data.get('user_id')

    if not user_id and not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'success': False, 'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM logs WHERE camera_id = %s", (camera_id,))
        cursor.execute("DELETE FROM camera_settings WHERE camera_id = %s", (camera_id,))
        cursor.execute("DELETE FROM bounding_boxes WHERE camera_id = %s", (camera_id,))
        cursor.execute("DELETE FROM cameras WHERE id = %s AND user_id = %s", (camera_id, user_id))

        conn.commit()

        if camera_id in video_detectors:
            video_detectors[camera_id].stop()
            del video_detectors[camera_id]

        cursor.close()
        conn.close()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
# -- Camera Settings --
    
def handle_crossing_event(camera_id: str, event_type: str):
    with detector_lock:
        detector = video_detectors.get(camera_id)
        if not detector:
            return

        timestamp = time.strftime('%H:%M:%S')
        print(f"Event from {camera_id} ({event_type})")

        # Mode 1: Increment Self (Exit Only Camera)
        if event_type == 'EXIT':
            detector.available_slots = detector.available_slots + 1
            log = f'[{timestamp}] {detector.camera_name} (EXIT)'
            detector.log_messages.append(log)
            save_log_to_db(camera_id, log)
        
        # Mode 2: Decrement Self (Entry Only Camera)
        elif event_type == 'ENTRY':
            detector.available_slots = detector.available_slots - 1
            log = f'[{timestamp}] {detector.camera_name} (ENTRY)'
            detector.log_messages.append(log)
            save_log_to_db(camera_id, log)

# -- Bounding Box Endpoints --

@app.get('/capture_frame/{camera_id}')
def capture_frame(camera_id: str, user_id: int):
    if not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT stream_url FROM cameras WHERE id = %s", (camera_id,))
        camera = cursor.fetchone()
        cursor.close()
        conn.close()

        if not camera:
            return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
        stream_url = camera['stream_url']
        cap = cv2.VideoCapture(stream_url)

        success, frame = cap.read()
        cap.release()

        if not success:
            return JSONResponse(status_code=500, content={'error': 'Failed to capture footage.'})
        
        _, img_encoded = cv2.imencode('.jpg', frame)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type='image/jpeg')
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error {err}'})

@app.post('/save_bounding_boxes/{camera_id}')
async def save_boxes(camera_id: str, request: Request):
    data = await request.json()
    user_id = data.get('user_id')
    if not user_id or not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    boxes_data = data.get('boxes', [])
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bounding_boxes WHERE camera_id = %s", (camera_id,))
        
        for index, space_data in enumerate(boxes_data):
            points = space_data.get('points') # Data from picker.html
            mode = space_data.get('mode', 'entry')

            if points:
                points_json = json.dumps(points)
                query = "INSERT INTO bounding_boxes (camera_id, box_index, points, mode) VALUES (%s, %s, %s, %s)"
                cursor.execute(query, (camera_id, index, points_json, mode))
        
        conn.commit()
        cursor.close()
        conn.close()

        if camera_id in video_detectors:
            video_detectors[camera_id].reload_configuration()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})

@app.get('/get_bounding_boxes/{camera_id}')
def get_boxes(camera_id: str, user_id: int):
    if not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT points, mode FROM bounding_boxes WHERE camera_id = %s ORDER BY box_index ASC", (camera_id,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        reconstructed_boxes = [{'points': json.loads(row['points']), 'mode': row['mode']} for row in results]
        return JSONResponse(content={'bounding_boxes': reconstructed_boxes})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})
    
# -- Detection Settings Endpoints --
    
@app.post('/save_settings/{camera_id}')
async def save_settings(camera_id: str, request: Request):
    data = await request.json()
    user_id = data.get('user_id')

    if not user_id or not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    detection_threshold = data.get('detection_threshold')

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO camera_settings (camera_id, detection_threshold) 
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE 
                detection_threshold = VALUES(detection_threshold)
        """
        cursor.execute(query, (camera_id, detection_threshold))
        conn.commit()
        cursor.close()
        conn.close()

        if camera_id in video_detectors:
            video_detectors[camera_id].reload_configuration()
        
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.get('/get_settings/{camera_id}')
def get_settings(camera_id: str, user_id: int):
    if not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM camera_settings WHERE camera_id = %s", (camera_id,))
        settings = cursor.fetchone()
        cursor.close()
        conn.close()

        if settings:
            return JSONResponse(content=settings)
        else:
            return JSONResponse(content={
                'detection_threshold': 0.35
            })
        
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})

# -- Logging and Information Display Endpoints --
def get_camera_ip(url):
    no_proto = url.split('://', 1)[1]
    if '@' in no_proto:
        no_proto = no_proto.split('@', 1)[1]

    camera_ip = no_proto.split(':')[0].split('/')[0]

    return camera_ip

# def save_to_info_db(camera_id, event_type):
#     detector = video_detectors.get(camera_id)
#     if not detector:
#         return
    
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         cursor = conn.cursor()

#         zdesc = detector.camera_name
#         current_date =  time.strftime("%Y-%m-%d")
#         current_time = time.strftime("%H:%M:%S")

#         sorted_cameras = sorted(video_detectors.items(), key=lambda x: str(x[0]))
#         zgroup_map = {cam_id: f"{i:02}" for i, (cam_id, _) in enumerate(sorted_cameras, start=1)}
        
#         zgroup = zgroup_map[camera_id]
#         cam_ip = get_camera_ip(detector.rtsp_url)

#         jml =  detector.total_spaces
#         cup = 1 if event_type == 'ENTRY' else 0
#         cdw = 1 if event_type == 'EXIT' else 0

#         query = """
#             INSERT INTO dgroup (zgroup, ip_kamera, tgl, jam, jml, cup, cdw, adj, `default`, status, isFull, min, zdesc)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, 0, 1, 0, 1, 0, %s)
#             ON DUPLICATE KEY UPDATE
#                 cup = cup + VALUES(cup),
#                 cdw = cdw + VALUES(cdw)
#         """
#         cursor.execute(query, (zgroup, cam_ip, current_date, current_time, jml, cup, cdw, zdesc))

#         conn.commit()
#         cursor.close()
#         conn.close()
#         print(f"Updated info table for camera: {detector.camera_name} ({event_type})")

#     except mysql.connector.Error as err:
#         print(f"Error saving to info table: {err}")

def save_log_to_db(camera_id, log_message):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO logs (camera_id, message) VALUES (%s, %s)"
        cursor.execute(query, (camera_id, log_message))
        conn.commit()
        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"Error saving log to DB: {err}")

@app.get('/get_logs/{camera_id}')
def get_logs(camera_id: str):
    if camera_id not in video_detectors:
        return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
    detector = video_detectors[camera_id]
    logs = detector.log_messages[-100:]
    total_spaces = detector.total_spaces
    available_slots = detector.available_slots

    aggregate_total, aggregate_available = get_aggregate_stats()
    return JSONResponse(content={'logs': logs, 'available_slots': available_slots, 'total_spaces': total_spaces, 
                                               'aggregate_available': aggregate_available, 'aggregate_total': aggregate_total})

@app.post('/set_bulk_total_spaces')
async def set_bulk_total_spaces(request: Request):
    data = await request.json()
    user_id = data.get('user_id')
    spaces_data = data.get('spaces_data', []) 

    if not user_id:
        return JSONResponse(status_code=401, content={'success': False, 'error': 'Authentication required.'})

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Security check - Verify ownership before updating
        for item in spaces_data:
            camera_id = item.get('id')
            if not verify_camera_ownership(camera_id, user_id):
                cursor.close()
                conn.close()
                return JSONResponse(status_code=403, content={'success': False, 'error': f'Permission denied for camera {camera_id}.'})

        # If all checks pass, proceed with the updates
        for item in spaces_data:
            camera_id = item.get('id')
            total_spaces = item.get('spaces')
            if camera_id is not None and isinstance(total_spaces, int) and total_spaces >= 0:
                cursor.execute("UPDATE cameras SET total_spaces = %s WHERE id = %s", (total_spaces, camera_id))
                
                if camera_id in video_detectors:
                    with detector_lock:
                        video_detectors[camera_id].total_spaces = total_spaces
                        video_detectors[camera_id].available_slots = total_spaces
        
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})

def get_aggregate_stats():
    total_spaces_all = 0
    total_available_spaces = 0

    for camera_id in video_detectors:
        total_spaces_all += video_detectors[camera_id].total_spaces
        total_available_spaces += video_detectors[camera_id].available_slots

    return total_spaces_all, total_available_spaces

def delete_old_logs():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "DELETE FROM logs WHERE timestamp > NOW() - INTERVAL 30 DAYS"

        cursor.execute(query)
        conn.commit()
        print(f"[{time.strftime('%H:%M:%S')}] Performed daily log cleanup. {cursor.rowcount} old logs deleted.")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error during log cleanup: {err}")

def run_log_cleanup_scheduler():
    schedule.every().day.at('01:00').do(delete_old_logs)
    while(True):
        schedule.run_pending()
        time.sleep(60)