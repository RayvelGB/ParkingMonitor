from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from detector import TripwireDetector, ParkingDetector
from db_configs import DB_CONFIG, DGROUP_CONFIG
from threading import Thread, Lock
import uuid
import json, cv2, io
import mysql.connector
from passlib.context import CryptContext
import time
import schedule

# -- INITIALIZE APP & VARIABLES --

app = FastAPI()
video_detectors = {}
detector_lock = Lock()

app.mount('/static', StaticFiles(directory='static'), name='static')

# -- STARTUP ENDPOINT --

@app.on_event('startup')
def on_startup():
    createDB_and_tables()
    start_all_detectors()

    cleanup_thread = Thread(target=run_log_cleanup_scheduler, daemon=True) # -> Start thread of log cleanup
    cleanup_thread.start()

# -- Database Configuration --
# Setup database
def createDB_and_tables():
    try:
        conn = mysql.connector.connect(**DGROUP_CONFIG)
        cursor = conn.cursor()

        # Create dgroup table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dgroup (
                id INT AUTO_INCREMENT PRIMARY KEY,
                zgroup VARCHAR(50),
                ip_kamera VARCHAR(50),
                zdesc VARCHAR(101),
                tgl DATE,
                jam TIME,
                jml INT,
                cup INT,
                cdw INT,
                adj INT,
                `default` INT,
                status INT,
                isFull INT,
                min INT,
                zone_id INT UNIQUE,
                       
                FOREIGN KEY (zone_id) REFERENCES entry_db.zones (id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        
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
                detection_type VARCHAR(50) DEFAULT 'parking',
                       
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
                confirmation_time INT DEFAULT 10,
                       
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
            )
        """)

        # Create zones table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id VARCHAR(36) NOT NULL,
                zone_name VARCHAR(50) NOT NULL,
                       
                FOREIGN KEY (camera_id) REFERENCES cameras (id) ON DELETE CASCADE,
                UNIQUE KEY (camera_id, zone_name)
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
                zone_id INT,
                       
                FOREIGN KEY (camera_id) REFERENCES cameras (id) ON DELETE CASCADE,
                FOREIGN KEY (zone_id) REFERENCES zones (id) ON DELETE SET NULL,
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
        cursor.execute("SELECT id, stream_url, camera_name, detection_type FROM cameras") # Grab camera_id, rtsp_url, and cameraname from database
        all_cameras = cursor.fetchall()
        cursor.close()
        conn.close()

        for camera in all_cameras:
            if camera['id'] not in video_detectors: # If camera currently not running, start it up
                start_single_detector(camera['id'], camera['stream_url'], camera['camera_name'], camera['detection_type'])
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

            if camera_id in video_detectors: # If camera is currently running, stop it and remove from list
                video_detectors[camera_id].stop()
                del video_detectors[camera_id]
        print('All detectors have been stopped.')

    except mysql.connector.Error as err:
        print(f'Database error during startup: {err}')

def start_single_detector(camera_id: str, rtsp_url: str, name: str, det_type: str):
    if camera_id not in video_detectors:
        print(f"Initializing detector for camera: {camera_id}")
        # Use different classes for different detection types
        if det_type == 'tripwire':
            detector = TripwireDetector(rtsp_url, camera_id, event_callback=handle_crossing_event) # -> Send handle_crossing_event to detector.py
        elif det_type == 'parking':
            detector = ParkingDetector(rtsp_url, camera_id)
            
        detector.camera_name = name
        video_detectors[camera_id] = detector
        t = Thread(target=detector.run, daemon=True)
        t.start()

# Send video feed to frontend
@app.get('/video_feed/{camera_id}')
def video_feed(camera_id: str):
    if camera_id not in video_detectors:
        return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
    detector = video_detectors[camera_id]
    return StreamingResponse(detector.generate(), media_type='multipart/x-mixed-replace; boundary=frame') # Start the stream

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

# Check if camera was registered by another user or not
def verify_camera_ownership(camera_id: str, user_id: int) -> bool:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM cameras WHERE id = %s AND user_id = %s", (camera_id, user_id)) # Take from database where row has the same camera id and user id
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        # If no row exists, current user is not camera owner
        return result is not None
    except mysql.connector.Error:
        return False

# Register User
@app.post('/register')
async def register_user(request: Request):
    data = await request.json()
    username = data.get('username')
    password = data.get('password')

    # Check if username and password are valid
    if not username or not password:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Username and password are required.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Check if username has already been registered in database
        cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
        if cursor.fetchone(): # -> If exists throw error
            cursor.close()
            conn.close()
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Username already exists.'})
        
        # Hash password for security and privacy
        hashed_password = get_password_hash(password)
        cursor.execute('INSERT INTO users(username, hashed_password) VALUES (%s, %s)', (username, hashed_password))
        conn.commit()
        user_id = cursor.lastrowid # -> user id is always incrementing

        cursor.close()
        conn.close()

        return JSONResponse(content={'success': True, 'user_id': user_id})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error {err}'})

# Login User
@app.post('/login')
async def login_user(request: Request):
    data = await request.json()
    username = data.get('username')
    password = data.get('password')

    # Check if username and password are valid
    if not username or not password:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Username and password are required.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        cursor.execute('SELECT id, hashed_password FROM users WHERE username = %s', (username,)) # -> Take from database where username is the same
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        # If user does not exist or password is wrong, throw error
        if not user or not verify_password(password, user['hashed_password']):
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid username or password.'})
        
        # Start detector like on start up
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

    # Check for missing fields
    if not all([user_id, stream_url, camera_name]):
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Missing required fields.'})
    
    camera_id = str(uuid.uuid4()) # -> Give camera a unique ID
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO cameras (id, camera_ip, user_id, camera_name, stream_url) VALUES (%s, %s, %s, %s, %s)", # -> Insert camera details to database
                       (camera_id, camera_ip, user_id, camera_name, stream_url)) 
        
        conn.commit()
        cursor.close()
        conn.close()

        # Start cameras
        start_all_detectors()
        return JSONResponse(content={'success': True, 'camera_id': camera_id})
    
    except mysql.connector.Error as err:    
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error {err}'})
    
@app.get('/get_cameras/{user_id}')
def get_cameras(user_id: int):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, camera_name, detection_type FROM cameras WHERE user_id = %s", (user_id,)) # -> Take camera details from database
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

    # Check for user login and camera ownership
    if not user_id and not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'success': False, 'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Delete camera details from database
        cursor.execute("DELETE FROM logs WHERE camera_id = %s", (camera_id,))
        cursor.execute("DELETE FROM camera_settings WHERE camera_id = %s", (camera_id,))
        cursor.execute("DELETE FROM bounding_boxes WHERE camera_id = %s", (camera_id,))
        cursor.execute("DELETE FROM cameras WHERE id = %s AND user_id = %s", (camera_id, user_id))

        conn.commit()

        # If camera is currently running, stop and delete for dictionary
        if camera_id in video_detectors:
            video_detectors[camera_id].stop()
            del video_detectors[camera_id]

        cursor.close()
        conn.close()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
# -- Camera Settings --
# Event listener for tripwire detection
def handle_crossing_event(camera_id: str, box_index: str, event_type: str):
    with detector_lock:
        detector = video_detectors.get(camera_id)
        if not detector:
            return

        # Check for index out of bound
        if box_index < len(detector.parking_boxes_data):
            box_data = detector.parking_boxes_data[box_index]
            zone_id = box_data.get('zone_id')
            timestamp = time.strftime('%H:%M:%S')

            # Make sure zone_id is valid
            if zone_id and zone_id in detector.zone_counts:
                zone_name = detector.zone_counts[zone_id]['name']

                # Mode 1: ENTRY (Removes available space)
                if event_type == 'ENTRY':
                    detector.zone_counts[zone_id]['available'] = detector.zone_counts[zone_id]['available'] - 1
                    log = f'[{timestamp}] {zone_name} (ENTRY)'
                    save_to_info_db(camera_id, zone_id, event_type)

                # Mode 2: EXIT (Adds available space)
                elif event_type == 'EXIT':
                    detector.zone_counts[zone_id]['available'] = detector.zone_counts[zone_id]['available'] + 1
                    log = f'[{timestamp}] {zone_name} (EXIT)'
                    save_to_info_db(camera_id, zone_id, event_type)
                
                save_log_to_db(camera_id, log) # -> Save log to database
                print(f"Event: Cam {camera_id} -> Zone {zone_id} -> {event_type}. New count: {detector.zone_counts[zone_id]['available']}")

@app.post('/set_detection_type/{camera_id}')
async def set_detection_type(camera_id: str, request: Request):
    data = await request.json()
    det_type = data.get('detection_type')

    # Check if the detection type sent is valid
    if det_type not in ['tripwire', 'parking']:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid detection type.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("UPDATE cameras SET detection_type = %s WHERE id = %s", (det_type, camera_id)) # -> Change default detection type to the one picked from frontend 
        conn.commit()

        # If camera is currenly running, stop it to change detection class (Tripwire to Parking or vice versa)
        if camera_id in video_detectors:
            video_detectors[camera_id].stop()
            del video_detectors[camera_id]
    
        cursor.execute("SELECT stream_url, camera_name FROM cameras WHERE id = %s", (camera_id,))
        camera = cursor.fetchone()
        
        cursor.close()
        conn.close()

        # Start up the detector again with a different class
        if camera:
            start_single_detector(camera_id, camera['stream_url'], camera['camera_name'], det_type)
            return JSONResponse(content={'success': True})
        else:
            return JSONResponse(status_code=404, content={'success': False, 'error': 'Camera not found after update.'})

    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.get('/get_detection_type/{camera_id}')
def get_detection_type(camera_id: str, user_id: int):
    # Check if user is owner of camera
    if not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT detection_type FROM cameras WHERE id = %s", (camera_id,)) # -> Take camera detection type from database
        det_type = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if det_type:
            return JSONResponse(content=det_type)
        else:
            return JSONResponse(content={
                'detection_type': 'parking' # -> Return 'parking' detection type as default
            })
        
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})

# -- Bounding Box Endpoints --
# Take current frame of camera and send to frontend
@app.get('/capture_frame/{camera_id}')
def capture_frame(camera_id: str, user_id: int):
    # Check if user is owner of camera
    if not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT stream_url FROM cameras WHERE id = %s", (camera_id,)) # -> Take stream url from database
        camera = cursor.fetchone()
        cursor.close()
        conn.close()

        # Check if camera exists
        if not camera:
            return JSONResponse(status_code=404, content={'error': 'Camera not found.'})

        # Capture stream
        stream_url = camera['stream_url']
        cap = cv2.VideoCapture(stream_url)

        success, frame = cap.read() # -> Take 1 frame from stream (Latest Frame) 
        cap.release()
        
        # Check for succession
        if not success:
            return JSONResponse(status_code=500, content={'error': 'Failed to capture footage.'})
        
        # Encode frame so its compatible for streaming response
        _, img_encoded = cv2.imencode('.jpg', frame)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type='image/jpeg')
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error {err}'})

@app.post('/save_bounding_boxes/{camera_id}')
async def save_boxes(camera_id: str, request: Request):
    data = await request.json()
    user_id = data.get('user_id')

    # Check for user login and camera ownership
    if not user_id or not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    boxes_data = data.get('boxes', []) # -> Get boxes, but if none just be empty array
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bounding_boxes WHERE camera_id = %s", (camera_id,)) # -> Delete previous boxes
        
        # Loop throw all boxes
        for index, space_data in enumerate(boxes_data):
            points = space_data.get('points')
            mode = space_data.get('mode', 'entry')
            zone_id = space_data.get('zone_id')
            
            # Check if points are valid
            if points:
                # Make points into json array and put it into database
                points_json = json.dumps(points)
                query = "INSERT INTO bounding_boxes (camera_id, box_index, points, mode, zone_id) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(query, (camera_id, index, points_json, mode, zone_id))
        
        conn.commit()
        cursor.close()
        conn.close()

        # If camera is currently running, reload its configuration
        if camera_id in video_detectors:
            video_detectors[camera_id].reload_configuration()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})

@app.get('/get_bounding_boxes/{camera_id}')
def get_boxes(camera_id: str, user_id: int):
    # Check if user is owner of camera
    if not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT points, zone_id, mode FROM bounding_boxes WHERE camera_id = %s ORDER BY box_index ASC", (camera_id,)) # -> Grab box details from database
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        # Reconstruct details to one array
        reconstructed_boxes = [{'points': json.loads(row['points']), 'zone_id': row['zone_id'], 'mode': row['mode']} for row in results]
        return JSONResponse(content={'bounding_boxes': reconstructed_boxes})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})
    
# -- Zone Management Endpoints --

@app.post('/create_zones/{camera_id}')
async def create_zones(camera_id: str, request: Request):
    data = await request.json()
    zone_name = data.get('zone_name')

    # Check if zone name is valid
    if not zone_name:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Zone name is required.'})
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO zones (camera_id, zone_name) VALUES (%s, %s)", (camera_id, zone_name)) # -> Insert new zone to database 
        conn.commit()

        # New zone id is the last id created
        new_zone_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return JSONResponse(content={'success': True, 'zone_id': new_zone_id})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.get('/get_zones/{camera_id}')  
def get_zones(camera_id: str):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT
                z.id,
                z.zone_name,
                dg.jml AS total_spaces
            FROM
                zones z
            LEFT JOIN
                sap8di.dgroup dg on z.id = dg.zone_id
            WHERE
                camera_id = %s
            ORDER BY
                z.id ASC
        """

        cursor.execute(query, (camera_id,)) # -> Grab zone details from database
        zones = cursor.fetchall()
        cursor.close()
        conn.close()

        # Check if camera is running
        if camera_id in video_detectors:
            detector = video_detectors[camera_id]
            with detector_lock:
                for zone in zones:
                    zone_id = zone['id']

                    # Grab space details from detector.py
                    if zone_id in detector.zone_counts:
                        zone['total_spaces'] = detector.zone_counts[zone_id]['total']
                        zone['available_spaces'] = detector.zone_counts[zone_id]['available']
                    else:
                        zone['total_spaces'] = 0
                        zone['available_spaces'] = 0
        return JSONResponse(content=zones)
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})
    
@app.delete('/delete_zone/{zone_id}')
async def delete_zone(zone_id: int, request: Request):
    data = await request.json()
    user_id = data.get('user_id')

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT camera_id FROM zones WHERE id = %s", (zone_id,)) # -> Grab zone from database
        result = cursor.fetchone()

        # Check if zone is valid
        if not result:
            return JSONResponse(status_code=404, content={'success': False, 'error': 'Zone not found.'})
        camera_id  = result['camera_id']

        # Delete zone
        cursor.execute("DELETE FROM zones WHERE id = %s", (zone_id,))
        conn.commit()
        cursor.close()
        conn.close()

        # If camera is running, reload configuration
        if camera_id in video_detectors:
            video_detectors[camera_id].reload_configuration()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.post('/update_zone_spaces/{camera_id}')
async def update_zone_spaces(camera_id: str, request: Request):
    data = await request.json()
    zone_data = data.get('zone_data', [])
    try:
        conn = mysql.connector.connect(**DGROUP_CONFIG)
        cursor = conn.cursor()

        # Loop through zone datas
        for item in zone_data:
            zone_id = item.get('id')
            total_spaces = item.get('spaces')

            # Check if zone and total spaces are valid
            if zone_id is not None and isinstance(total_spaces, int) and total_spaces >= 0:
                cursor.execute("UPDATE dgroup SET jml = %s WHERE zone_id = %s", (total_spaces, zone_id)) # -> Update total space in database

        conn.commit()
        cursor.close()
        conn.close()

        # If camera is running, reload configuration
        if camera_id in video_detectors:
            video_detectors[camera_id].reload_configuration()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.post('/assign_box_to_zone')
async def assign_box_to_zone(request: Request):
    data = await request.json()
    camera_id = data.get('camera_id')
    box_index = data.get('box_index')
    zone_id = data.get('zone_id')
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("UPDATE bounding_boxes SET zone_id = %s WHERE camera_id = %s AND box_index = %s", (zone_id, camera_id, box_index)) # -> Set zone_id to box
        conn.commit()
        cursor.close()
        conn.close()
        
        # If camera is running, reload configuration
        if camera_id in video_detectors:
            video_detectors[camera_id].reload_configuration()
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.get('/get_aggregate_stats')
def get_aggregate_stats():
    total_spaces = 0
    available_spaces = 0

    with detector_lock:
        # Iterate through all running cameras
        for cam_id, detector in video_detectors.items():
            for zone_id, zone_data in detector.zone_counts.items():
                total_spaces += zone_data['total']
                space = max(0, min(zone_data['available'], zone_data['total'])) # -> Softlock it so it doesn't overflow in the frontend 
                available_spaces += space

    return JSONResponse(content={'total_spaces': total_spaces, 'available_slots': available_spaces})

# -- Detection Settings Endpoints --
    
@app.post('/save_settings/{camera_id}')
async def save_settings(camera_id: str, request: Request):
    data = await request.json()
    user_id = data.get('user_id')

    # Check for user login and camera ownership
    if not user_id or not verify_camera_ownership(camera_id, user_id):
        return JSONResponse(status_code=403, content={'error': 'Permission denied.'})
    
    # Get settings from frontend
    detection_threshold = data.get('detection_threshold')
    iou_threshold = data.get('iou_threshold')
    confirmation_time = data.get('confirmation_time')

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO camera_settings (camera_id, detection_threshold, iou_threshold, confirmation_time) 
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                detection_threshold = COALESCE(VALUES(detection_threshold), detection_threshold),
                iou_threshold = COALESCE(VALUES(iou_threshold), iou_threshold),
                confirmation_time = COALESCE(VALUES(confirmation_time), confirmation_time)
        """
        cursor.execute(query, (camera_id, detection_threshold, iou_threshold, confirmation_time)) # -> Save settings into database if not null
        conn.commit()
        cursor.close()
        conn.close()

        # If camera is running, reload configuration
        if camera_id in video_detectors:
            video_detectors[camera_id].reload_configuration()
        
        return JSONResponse(content={'success': True})
    
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'success': False, 'error': f'Database error: {err}'})
    
@app.get('/get_settings/{camera_id}')
def get_settings(camera_id: str, user_id: int):
    # Check if user is owner of camera
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
                'confirmation_time': 10
            }) # If settings are null, send these as default
        
    except mysql.connector.Error as err:
        return JSONResponse(status_code=500, content={'error': f'Database error: {err}'})

# -- Logging and Information Display Endpoints --

def get_camera_ip(url):
    # rtsp://username:password@ip_address:port/stream_url
    no_proto = url.split('://', 1)[1]
    if '@' in no_proto:
        no_proto = no_proto.split('@', 1)[1]

    camera_ip = no_proto.split(':')[0].split('/')[0]

    return camera_ip

def save_to_info_db(camera_id, zone_id, event_type):
    detector = video_detectors.get(camera_id)
    if not detector:
        return
    
    # Check if zone_id is valid
    if zone_id is None:
        print("--- ERROR: save_to_info_db called with zone_id = None. Aborting. ---")
        return

    try:
        conn = mysql.connector.connect(**DGROUP_CONFIG)
        cursor = conn.cursor(dictionary=True)

        cam_ip = get_camera_ip(detector.rtsp_url)

        # Check if a row for this zone already exists
        cursor.execute("SELECT id, zgroup, cup, cdw FROM dgroup WHERE zone_id = %s", (zone_id,))
        existing_row = cursor.fetchone()

        # Prepare details to insert in database
        current_date = time.strftime("%Y-%m-%d")
        current_time = time.strftime("%H:%M:%S")
        keterangan = detector.camera_name + ' ' + detector.zone_counts[zone_id]['name']
        jml = detector.zone_counts[zone_id]['total']
        
        # If zone_id already exists in row
        if existing_row:
            print(f"--- DEBUG: Found row for zone_id {zone_id}. Updating. ---")
            
            # Use the existing zgroup
            zgroup = existing_row['zgroup']
            
            # Calculate new cup/cdw counts by adding to the old ones
            new_cup = existing_row['cup'] + (1 if event_type == 'EXIT' else 0)
            new_cdw = existing_row['cdw'] + (1 if event_type == 'ENTRY' else 0)

            update_query = """
                UPDATE dgroup 
                SET cup = %s, cdw = %s, jml = CASE WHEN jml = 0 THEN %s ELSE jml END
                WHERE zone_id = %s
            """
            execute_cursor = conn.cursor()
            execute_cursor.execute(update_query, (new_cup, new_cdw, jml, zone_id))
            execute_cursor.close()

        # If zone_id is new entry
        else:
            print(f"--- DEBUG: No row for zone_id {zone_id}. Inserting. ---")
            
            # Calculate the next available zgroup for this camera
            cursor.execute("SELECT COUNT(*) as zone_count FROM dgroup WHERE ip_kamera = %s", (cam_ip,))
            count_result = cursor.fetchone()
            new_zgroup_number = count_result['zone_count'] + 1
            zgroup = f"{new_zgroup_number:02}"
            
            # Set initial cup/cdw counts
            cup = 1 if event_type == 'EXIT' else 0
            cdw = 1 if event_type == 'ENTRY' else 0

            insert_query = """
                INSERT INTO dgroup 
                    (zone_id, zgroup, ip_kamera, zdesc, tgl, jam, jml, cup, cdw, adj, `default`, status, isFull, min)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 1, 0, 1, 0)
            """
            execute_cursor = conn.cursor()
            execute_cursor.execute(insert_query, (zone_id, zgroup, cam_ip, keterangan, current_date, current_time, jml, cup, cdw))
            execute_cursor.close()

        conn.commit()
        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"Error saving to info table: {err}")

def save_log_to_db(camera_id, log_message):
    try:
        # Save log in detector class to show in frontend
        if camera_id in video_detectors:
            video_detectors[camera_id].log_messages.append(log_message)

            if len(video_detectors[camera_id].log_messages) > 50:
                video_detectors[camera_id].log_messages.pop(0)

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO logs (camera_id, message) VALUES (%s, %s)" # -> Also insert logs to database
        cursor.execute(query, (camera_id, log_message))
        conn.commit()
        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"Error saving log to DB: {err}")

@app.get('/get_logs/{camera_id}')
def get_logs(camera_id: str):
    # Check if camera is running
    if camera_id not in video_detectors:
        return JSONResponse(status_code=404, content={'error': 'Camera not found.'})
    
    detector = video_detectors[camera_id]
    logs = detector.log_messages[-50:] # Get latest 50 logs from detector class
    return JSONResponse(content={'logs': logs})

# Delete logs if older than 30 days
def delete_old_logs():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "DELETE FROM logs WHERE timestamp > NOW() - INTERVAL 30 DAYS" # -> Delete logs

        cursor.execute(query)
        conn.commit()
        print(f"[{time.strftime('%H:%M:%S')}] Performed daily log cleanup. {cursor.rowcount} old logs deleted.")
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as err:
        print(f"Error during log cleanup: {err}")

def run_log_cleanup_scheduler():
    # Schedule to check log everyday at 1
    schedule.every().day.at('01:00').do(delete_old_logs)
    while(True):
        schedule.run_pending()
        time.sleep(60)