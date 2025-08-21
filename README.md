-- USER --
username: admin
password: admin

FOLDER RCNN_TESTING -> VISUALISASI BOX
RUN AJA

-- RUN WEBSITE -- 
uvicorn main:app --reload --host 0.0.0.0 --port 8000
DI TERMINAL

-- WEBSITE URL --
http://127.0.0.1:8000/

-- SETTINGS -- (Buka RCNN_TESTING buat lebih ngerti)
DETECTION THRESHOLD -> Seberapa pastinya AI merasa pada prediksinya
IOU THRESHOLD -> Kalkulasi antara kotak yang dibuat oleh AI dan Kotak yang dibuat user
CONFIRMATION TIME -> Waktu delay ketika suatu object dideteksi pada lot

*iou dan confirmation hanya ada pada parking detection (lot)
*akun sebelah settings bisa dipencet