# 🏭 CCTV AI Monitoring System

An industrial-grade AI system for monitoring production lines using CCTV cameras.

---

## 🚀 Features

- 🎥 Multi-camera live streaming (RTSP)
- 🧠 YOLOv8 object detection + ByteTrack tracking
- 🔢 Line-crossing based piece counting
- 📊 Real-time dashboard (React)
- 📄 CSV-based production logging

---

## 📁 Project Structure

cctv_ai_system/
│
├── backend/
│   ├── server.py
│   ├── requirements.txt
│   └── counts_log.csv (auto-generated)
│
├── frontend/
│   └── cctv/
│       ├── src/
│       ├── package.json
│       └── vite.config.js
│
└── README.md

---

## ⚙️ Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --reload
```

Backend runs at: http://localhost:8000

---

## 🌐 Frontend Setup

```bash
cd frontend/cctv
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

---

## 🔗 System Architecture

CCTV Cameras (RTSP)
        ↓
FastAPI Backend (YOLO + Tracking + Counting)
        ↓
WebSocket + REST APIs
        ↓
React Dashboard (Live Monitoring)

---

## 📡 API Endpoints

- GET /camera/{cam_id}
- GET /counts
- GET /cameras
- WebSocket: ws://localhost:8000/ws

---

## 📄 CSV Logging

- File: counts_log.csv
- Logs every 1 minute

Format:
Camera,Count,Date,Hour,Minute

---

## ⚙️ Camera Configuration

Edit in backend/server.py:

CAMERAS = {
    "cam1": {
        "ip": "192.168.x.x",
        "name": "AIR WASHING",
        "mode": "x",
        "pos": 500,
        "dir": "lr"
    }
}

---

## ⚠️ Requirements

- Python 3.10+
- Node.js 18+
- Stable CCTV network
- GPU recommended (RTX 3060+)

---

## 🔧 Troubleshooting

- No stream → check RTSP & network
- Yellow imports → select correct venv
- High CPU → reduce resolution & skip frames

---

## 🚀 Future Improvements

- Power BI integration
- Shift-wise analytics
- Alerts & defect detection

---

## 👨‍💻 Author

Industrial AI CCTV monitoring system
