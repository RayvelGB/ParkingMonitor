from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from detector import VideoDetector
from threading import Thread
import uuid
import json, cv2, os, io
import mysql.connector
from passlib.context import CryptContext
import time
import schedule

app = FastAPI()
video_detectors = {}

app.mount('/static', StaticFiles(directory='static'), name='static')

DB_CONFIG = {
    'host':'localhost',
    'user': 'root',
    'password': '',
    'database': 'parking_monitor_db'
}

# -- Initialize password hash --
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

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
                       
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Create camera settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id VARCHAR(36) UNIQUE,
                detection_threshold FLOAT DEFAULT 0.35,
                iou_threshold FLOAT DEFAULT 0.3,
                use_intersection_only BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
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

createDB_and_tables()

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

cleanup_thread = Thread(target=run_log_cleanup_scheduler, daemon=True)
cleanup_thread.start()

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

# -- Bounding Box and Stream Endpoints --
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
    
@app.post('/save_settings/{camera_id}')
async def save_settings(camera_id: str, request: Request):
    data = await request.json()
    user_id = data.get('user_id')

    if not user_id and not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    detection_threshold = data.get('detection_threshold')
    iou_threshold = data.get('iou_threshold')
    use_intersection_only = data.get('use_intersection_only')

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO camera_settings (camera_id, detection_threshold, iou_threshold, use_intersection_only) 
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                detection_threshold = VALUES(detection_threshold), 
                iou_threshold = VALUES(iou_threshold), 
                use_intersection_only = VALUES(use_intersection_only)
        """
        cursor.execute(query, (camera_id, detection_threshold, iou_threshold, use_intersection_only))
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
                'detection_threshold': 0.35,
                'iou_threshold': 0.3,
                'use_intersection_only': False
            })
        
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})

@app.post('/start_stream')
async def start_stream(request: Request):
    data = await request.json()
    camera_id = data.get('camera_id')
    user_id = data.get('user_id')
    
    if not user_id or not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'success': False, 'error': 'Permission denied.'})
    
    if camera_id in video_detectors and video_detectors[camera_id].running:
        print(f"Stream for camera {camera_id} is already running. No action needed.")
        return JSONResponse({'success': True, 'message': 'Stream already running.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT stream_url FROM cameras WHERE id = %s", (camera_id,))
        camera = cursor.fetchone()
        cursor.close()
        conn.close()
        if not camera:
            return JSONResponse(status_code=404, content={'error': 'Camera not registered.'})
        
        rtsp_url = camera['stream_url']
        detector = VideoDetector(rtsp_url, camera_id)
        video_detectors[camera_id] = detector
        t = Thread(target=detector.run, daemon=True)
        t.start()
        return JSONResponse({'success': True})
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})

@app.get('/get_logs/{camera_id}')
def get_logs(camera_id: str):
    if camera_id not in video_detectors:
        return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
    detector = video_detectors[camera_id]
    logs = detector.log_messages[-100:]
    available_slots = detector.available_slots
    total_spaces = detector.total_spaces
    return JSONResponse(content={'logs': logs, 'available_slots': available_slots, 'total_spaces': total_spaces})

@app.get('/video_feed/{camera_id}')
def video_feed(camera_id: str):
    if camera_id not in video_detectors:
        return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
    detector = video_detectors[camera_id]
    return StreamingResponse(detector.generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/stop_stream')
async def stop_stream(request: Request):
    data = await request.json()
    camera_id = data.get('camera_id')
    if camera_id in video_detectors:
        video_detectors[camera_id].stop()
        del video_detectors[camera_id]
        return JSONResponse(content={'success': True})
    return JSONResponse(content={'success': False, 'error': 'Camera not found.'})
