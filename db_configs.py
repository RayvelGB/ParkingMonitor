import mysql.connector

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'entry_db' # -> Database name
}

DGROUP_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'sap8di'
}

# -- DB PAK ALEX --
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