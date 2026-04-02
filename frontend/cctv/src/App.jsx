import { useEffect, useState } from "react";

const API = "http://localhost:8000";
const WS = "ws://localhost:8000/ws";

function App() {
  const [counts, setCounts] = useState({});
  const [cameras, setCameras] = useState({});

  useEffect(() => {
    fetch(`${API}/cameras`)
      .then(res => res.json())
      .then(setCameras);
  }, []);

  useEffect(() => {
    const ws = new WebSocket(WS);
    ws.onmessage = (e) => setCounts(JSON.parse(e.data));
    return () => ws.close();
  }, []);

  return (
    <div style={{ background: "#111", color: "white", minHeight: "100vh", padding: 20 }}>
      <h1 style={{ textAlign: "center" }}>🏭 CCTV Dashboard</h1>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(5, 1fr)",
        gap: "10px"
      }}>
        {Object.entries(cameras).map(([camId, cam]) => (
          <div
            key={camId}
            style={{ background: "#222", padding: 10, cursor: "pointer" }}
            onClick={() => window.location.href = `/camera/${camId}`}
          >
            <h4>{cam.name}</h4>

            <img
              src={`${API}/camera/${camId}`}
              style={{ width: "100%" }}
            />

            <p style={{ color: "lime" }}>
              Count: {counts[cam.name] || 0}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;