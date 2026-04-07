import cv2
import time
import asyncio
import csv
import threading
import queue
from datetime import datetime
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

# 🔥 CPU optimized resolution
FRAME_W, FRAME_H = 320, 180
ORIGINAL_W, ORIGINAL_H = 1920, 1080

CSV_FILE = "counts_log.csv"

# ================= FASTAPI =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= STORAGE =================
frame_queues = {}
frames = {}
counts = {cam: 0 for cam in CAMERAS}
track_history = {cam: {} for cam in CAMERAS}
counted_ids = {cam: set() for cam in CAMERAS}
last_infer_time = {cam: 0 for cam in CAMERAS}

model = None

# ================= CAPTURE =================
def capture_worker(cam_id, config):
    ip = config["ip"]
    rtsp = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/video/live?channel=1&subtype=1&rtsp_transport=tcp"

    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"[RECONNECT] {cam_id}")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # 🔥 Drop old frames (no lag)
        if frame_queues[cam_id].full():
            try:
                frame_queues[cam_id].get_nowait()
            except:
                pass

        frame_queues[cam_id].put(frame)

# ================= GLOBAL INFERENCE =================
def inference_worker():
    while True:
        for cam_id in CAMERAS:

            if frame_queues[cam_id].empty():
                continue

            # 🔥 Limit inference rate per camera
            if time.time() - last_infer_time[cam_id] < 0.15:
                continue

            last_infer_time[cam_id] = time.time()

            frame = frame_queues[cam_id].get()
            process_frame(cam_id, frame)

# ================= PROCESS FRAME =================
def process_frame(cam_id, frame):
    config = CAMERAS[cam_id]

    if config["mode"] == "x":
        line_pos = int(config["pos"] * FRAME_W / ORIGINAL_W)
    else:
        line_pos = int(config["pos"] * FRAME_H / ORIGINAL_H)

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        history = track_history[cam_id]
        counted = counted_ids[cam_id]

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            prev = history.get(track_id)

            if prev:
                prev_x, prev_y = prev
                direction = config["dir"]

                crossed = False

                if config["mode"] == "x":
                    if direction == "lr":
                        crossed = prev_x < line_pos and cx >= line_pos
                    elif direction == "rl":
                        crossed = prev_x > line_pos and cx <= line_pos
                else:
                    if direction == "tb":
                        crossed = prev_y < line_pos and cy >= line_pos
                    elif direction == "bt":
                        crossed = prev_y > line_pos and cy <= line_pos

                if crossed and track_id not in counted:
                    counts[cam_id] += 1
                    counted.add(track_id)

                    if len(counted) > 1000:
                        counted.clear()

            history[track_id] = (cx, cy)

        track_history[cam_id] = history
        counted_ids[cam_id] = counted

    frames[cam_id] = results[0].plot()

# ================= CSV LOGGER =================
def log_counts():
    while True:
        time.sleep(60)
        now = datetime.now()

        rows = []
        for cam in CAMERAS:
            rows.append([
                CAMERAS[cam]["name"],
                counts[cam],
                now.strftime("%Y-%m-%d"),
                now.strftime("%H"),
                now.strftime("%M")
            ])
            counts[cam] = 0

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print("[CSV LOGGED]", rows)

# ================= STARTUP =================
@app.on_event("startup")
def start():
    global model

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    for cam in CAMERAS:
        frame_queues[cam] = queue.Queue(maxsize=2)
        frames[cam] = None

    # camera threads
    for cam_id, config in CAMERAS.items():
        threading.Thread(target=capture_worker, args=(cam_id, config), daemon=True).start()

    # 🔥 single inference worker
    threading.Thread(target=inference_worker, daemon=True).start()

    threading.Thread(target=log_counts, daemon=True).start()

# ================= STREAM =================
def generate_stream(cam_id):
    while True:
        if frames[cam_id] is not None:
            _, buffer = cv2.imencode(".jpg", frames[cam_id])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        else:
            time.sleep(0.05)

@app.get("/camera/{cam_id}")
def stream(cam_id: str):
    return StreamingResponse(generate_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame")

# ================= APIs =================
@app.get("/counts")
def get_counts():
    return {CAMERAS[cam]["name"]: counts[cam] for cam in CAMERAS}

@app.get("/cameras")
def get_cameras():
    return {cam: {"name": CAMERAS[cam]["name"]} for cam in CAMERAS}

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    while True:
        await ws.send_json({CAMERAS[cam]["name"]: counts[cam] for cam in CAMERAS})
        await asyncio.sleep(1)