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
camera_links = {}

app.mount('/static', StaticFiles(directory='static'), name='static')

# -- STARTUP ENDPOINT --

@app.on_event('startup')
def on_startup():
    createDB_and_tables()
    load_camera_links()
    start_all_detectors()
    
    cleanup_thread = Thread(target=run_log_cleanup_scheduler, daemon=True)
    cleanup_thread.start()

# -- Database Configuration --
DB_CONFIG = {
    'host':'localhost',
    'user': 'root',
    'password': '',
    'database': 'entry_db'
}

# -- Setup database --
def createDB_and_tables():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL
            )
        """)

        # Create cameras table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id VARCHAR(36) PRIMARY KEY,
                user_id INT,
                camera_name VARCHAR(255) NOT NULL,
                stream_url VARCHAR(1024) NOT NULL,
                mode VARCHAR(101) DEFAULT 'increment_self',
                       
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

        # Create camera links table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_links (
                source_camera_id VARCHAR(36) PRIMARY KEY,
                target_camera_id VARCHAR(36) NOT NULL,
                
                FOREIGN KEY (source_camera_id) REFERENCES cameras (id) ON DELETE CASCADE,
                FOREIGN KEY (target_camera_id) REFERENCES cameras (id) ON DELETE CASCADE
            )
        """)

        # Create Bounding Boxes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bounding_boxes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id VARCHAR(36),
                box_index INT,
                zone_name VARCHAR(255),
                points TEXT,
                       
                FOREIGN KEY (camera_id) REFERENCES cameras (id),
                UNIQUE KEY (camera_id, box_index)
            )
        """)

        # Create logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id VARCHAR(36),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
        cursor.execute("SELECT id, stream_url, mode FROM cameras")
        all_cameras = cursor.fetchall()
        cursor.close()
        conn.close()

        for camera in all_cameras:
            start_single_detector(camera['id'], camera['stream_url'], camera['mode'])
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

def start_single_detector(camera_id: str, rtsp_url: str, mode: str):
    if camera_id not in video_detectors:
        print(f"Initializing detector for camera: {camera_id} with mode: {mode}")
        detector = VideoDetector(rtsp_url, camera_id, event_callback=handle_crossing_event)
        detector.mode = mode
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

@app.get('/zone_picker.html')
def serve_zone_picker_page():
    return FileResponse('static/zone_picker.html')

@app.get('/stream.html')
def serve_stream_page():
    return FileResponse('static/stream.html')


# -- Authentication Endpoints --

# -- Initialize password hash --
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

    if not all([user_id, stream_url, camera_name]):
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Missing required fields.'})
    
    camera_id = str(uuid.uuid4())
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO cameras (id, user_id, camera_name, stream_url) VALUES (%s, %s, %s, %s)",
                       (camera_id, user_id, camera_name, stream_url)) 
        
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
        cursor.execute("SELECT id, camera_name FROM cameras WHERE user_id = %s", (user_id,))
        cameras = {row['id']: {'camera_name': row['camera_name']} for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return JSONResponse(content=cameras)
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})
    
@app.get('/get_camera_details/{camera_id}')
def get_camera_details(camera_id: str, user_id: int):
    if not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, camera_name, mode FROM cameras WHERE id = %s", (camera_id,))
        camera_details = cursor.fetchone()
        cursor.close()
        conn.close()

        if camera_details:
            return JSONResponse(content=camera_details)
        else:
            return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
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
    
def handle_crossing_event(source_id: str, event_type: str):
    with detector_lock:
        source_detector = video_detectors.get(source_id)
        if not source_detector:
            return

        mode = source_detector.mode
        print(f"Event from {source_id} ({event_type}) with mode: {mode}")

        # Mode 1: Increment Self (Exit Only Camera)
        if mode == 'increment_self' and event_type == 'EXIT':
            source_detector.available_slots = source_detector.available_slots + 1
        
        # Mode 2: Decrement Self (Entry Only Camera)
        elif mode == 'decrement_self' and event_type == 'ENTRY':
            source_detector.available_slots = source_detector.available_slots - 1
        
        # Mode 3: Add a space for this camera AND erase one from a linked camera
        elif mode == 'increment_self_decrement_target':
            if event_type == 'ENTRY':
                # source_detector.available_slots = source_detector.available_slots + 1
                target_id = camera_links.get(source_id)
                if target_id:
                    target_detector = video_detectors.get(target_id)
                    if target_detector:
                        print(f"Link triggered: {source_id} -> {target_id}")
                        target_detector.available_slots = target_detector.available_slots - 1
                else:
                    print('Target camera not found.')

def load_camera_links():
    global camera_links
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT source_camera_id, target_camera_id FROM camera_links")
        links = cursor.fetchall()
        camera_links = {link['source_camera_id']: link['target_camera_id'] for link in links}

        print(f'Loaded {len(camera_links)} camera links.')
        cursor.close()
        conn.close()
    
    except mysql.connector.Error as err:
        print(f"Could not load camera links: {err}")

@app.post('/link_cameras')
async def link_cameras(request: Request):
    data = await request.json()
    source_id = data.get('source_id')
    target_id = data.get('target_id')

    if not source_id and not target_id:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Source and target IDs are required.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO camera_links (source_camera_id, target_camera_id)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE target_camera_id = VALUES(target_camera_id)
        """
        cursor.execute(query, (source_id, target_id))
        conn.commit()
        cursor.close()
        conn.close()

        camera_links[source_id] = target_id
        return JSONResponse(content={'success': True, 'message': f'Camera {source_id} now linked to {target_id}.'})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.post('/set_camera_mode/{camera_id}')
async def set_camera_mode(camera_id: str, request: Request):
    data = await request.json()
    mode = data.get('mode')

    if mode not in ['increment_self', 'decrement_self', 'increment_self_decrement_target']:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid mode specified.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("UPDATE cameras SET mode = %s WHERE id = %s", (mode, camera_id))
        conn.commit()
        cursor.close()
        conn.close()

        if camera_id in video_detectors:
            video_detectors[camera_id].mode = mode
            print(f"Updated mode for camera {camera_id} to {mode}")
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})

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
            points = []
            zone_name = None
            
            if isinstance(space_data, list):
                points = space_data # Data from picker.html
            elif isinstance(space_data, dict):
                points = space_data.get('points') # Data from zone_picker.html
                zone_name = space_data.get('zone')

            if points:
                points_json = json.dumps(points)
                query = "INSERT INTO bounding_boxes (camera_id, box_index, points, zone_name) VALUES (%s, %s, %s, %s)"
                cursor.execute(query, (camera_id, index, points_json, zone_name))
        
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
        cursor.execute("SELECT points, zone_name FROM bounding_boxes WHERE camera_id = %s ORDER BY box_index ASC", (camera_id,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        reconstructed_boxes = [{"points": json.loads(row['points']), "zone": row['zone_name']} for row in results]
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

@app.get('/get_logs/{camera_id}')
def get_logs(camera_id: str):
    if camera_id not in video_detectors:
        return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
    detector = video_detectors[camera_id]
    logs = detector.log_messages[-100:]
    total_spaces, available_slots = get_aggregate_stats()
    return JSONResponse(content={'logs': logs, 'available_slots': available_slots, 'total_spaces': total_spaces})

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