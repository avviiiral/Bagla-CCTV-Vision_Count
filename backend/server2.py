import cv2
import threading
import time
import asyncio
import csv
from datetime import datetime
from queue import Queue
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ================= CONFIG =================
CAMERAS = {
    "cam1": {"ip": "192.168.1.193", "name": "AIR WASHING", "mode": "y", "pos": 500, "dir": "tb"},
    "cam2": {"ip": "192.168.1.191", "name": "MIDDLE TESTING", "mode": "x", "pos": 1050, "dir": "lr"},
    "cam3": {"ip": "192.168.1.196", "name": "COIL SOLDRING", "mode": "x", "pos": 650, "dir": "rl"},
    "cam4": {"ip": "192.168.1.198", "name": "FRAME CRIMPING", "mode": "x", "pos": 750, "dir": "rl"},
    "cam5": {"ip": "192.168.1.194", "name": "COMMON CRIMPING", "mode": "x", "pos": 800, "dir": "rl"},
    "cam6": {"ip": "192.168.1.197", "name": "WIRE ROUTING", "mode": "x", "pos": 650, "dir": "lr"},
    "cam7": {"ip": "192.168.1.188", "name": "BASE ASSEMBLY", "mode": "x", "pos": 850, "dir": "lr"},
    "cam8": {"ip": "192.168.1.189", "name": "M SPRING RIVETTING", "mode": "x", "pos": 950, "dir": "lr"},
    "cam9": {"ip": "192.168.1.190", "name": "CORE RIVETTING", "mode": "x", "pos": 850, "dir": "lr"},
    "cam10": {"ip": "192.168.1.201", "name": "BOBBIN PRESSING", "mode": "x", "pos": 900, "dir": "lr"},
}

USERNAME = "admin"
PASSWORD = "Admin%40123"

MODEL_PATH = r"D:\cctv_ai_system\backend\model\runs\detect\train\weights\best.pt"

FRAME_W, FRAME_H = 320, 180
ORIGINAL_W, ORIGINAL_H = 1920, 1080

CSV_FILE = "counts_log.csv"

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= FASTAPI =================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ================= STORAGE =================
frames = {cam: None for cam in CAMERAS}
encoded_frames = {cam: None for cam in CAMERAS}
counts = {cam: 0 for cam in CAMERAS}
track_history = {cam: {} for cam in CAMERAS}
counted_ids = {cam: set() for cam in CAMERAS}

frame_queues = {cam: Queue(maxsize=2) for cam in CAMERAS}

# ================= CAPTURE =================
def capture_worker(cam_id, config):
    rtsp = f"rtsp://{USERNAME}:{PASSWORD}@{config['ip']}:554/video/live?channel=1&subtype=1&rtsp_transport=tcp"

    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frames[cam_id] = frame

        # encode once
        _, buffer = cv2.imencode(".jpg", frame)
        encoded_frames[cam_id] = buffer.tobytes()

        if not frame_queues[cam_id].full():
            frame_queues[cam_id].put(frame)

# ================= INFERENCE =================
def inference_worker():
    frame_skip = 0

    while True:
        batch = []
        cam_ids = []

        for cam in CAMERAS:
            if not frame_queues[cam].empty():
                batch.append(frame_queues[cam].get())
                cam_ids.append(cam)

        if not batch:
            time.sleep(0.005)
            continue

        frame_skip += 1
        if frame_skip % 3 != 0:
            continue

        results = model.track(batch, persist=True, tracker="bytetrack.yaml", verbose=False)

        for i, cam in enumerate(cam_ids):
            config = CAMERAS[cam]
            result = results[i]

            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()

            if config["mode"] == "x":
                line_pos = int(config["pos"] * FRAME_W / ORIGINAL_W)
            else:
                line_pos = int(config["pos"] * FRAME_H / ORIGINAL_H)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                prev = track_history[cam].get(track_id)

                if prev:
                    px, py = prev
                    crossed = False

                    if config["mode"] == "x":
                        if config["dir"] == "lr":
                            crossed = px < line_pos and cx >= line_pos
                        elif config["dir"] == "rl":
                            crossed = px > line_pos and cx <= line_pos
                    else:
                        if config["dir"] == "tb":
                            crossed = py < line_pos and cy >= line_pos

                    if crossed and track_id not in counted_ids[cam]:
                        counts[cam] += 1
                        counted_ids[cam].add(track_id)

                track_history[cam][track_id] = (cx, cy)

# ================= STREAM =================
def generate_stream(cam_id):
    while True:
        if cam_id not in encoded_frames:
            time.sleep(0.1)
            continue

        if encoded_frames[cam_id] is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + encoded_frames[cam_id] + b"\r\n")

        time.sleep(0.1)

@app.get("/camera/{cam_id}")
def stream(cam_id: str):
    if cam_id not in CAMERAS:
        return {"error": f"Invalid camera: {cam_id}"}

    return StreamingResponse(generate_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame")

# ================= CSV =================
def log_counts():
    while True:
        time.sleep(60)
        now = datetime.now()

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            for cam in CAMERAS:
                writer.writerow([
                    CAMERAS[cam]["name"],
                    counts[cam],
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H"),
                    now.strftime("%M")
                ])
                counts[cam] = 0

# ================= START =================
@app.on_event("startup")
def start():
    for cam_id, config in CAMERAS.items():
        threading.Thread(target=capture_worker, args=(cam_id, config), daemon=True).start()

    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=log_counts, daemon=True).start()