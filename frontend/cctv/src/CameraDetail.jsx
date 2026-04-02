import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, Cell
} from "recharts";

const API = "http://localhost:8000";

export default function CameraDetail() {
  const { id } = useParams();
  const [data, setData] = useState([]);
  const [target, setTarget] = useState(10);

  useEffect(() => {
    fetch(`${API}/counts/hourly/${id}`)
      .then(res => res.json())
      .then(setData);
  }, [id]);

  // ================= SHIFT LOGIC =================
  const getShift = (hour) => {
    if (hour >= 8 && hour < 16) return "Shift 1 (8-16)";
    if (hour >= 16 && hour < 24) return "Shift 2 (16-24)";
    return "Shift 3 (0-8)";
  };

  // ================= PROCESS DATA =================
  let shiftTotals = {};
  let dailyTotal = 0;

  const chartData = data.map(d => {
    const hour = Number(d.hour);
    const efficiency = target > 0 ? ((d.count / target) * 100).toFixed(1) : 0;

    const shift = getShift(hour);

    shiftTotals[shift] = (shiftTotals[shift] || 0) + d.count;
    dailyTotal += d.count;

    return {
      ...d,
      target,
      efficiency,
      color: d.count >= target ? "#00ff00" : "#ff4444"
    };
  });

  return (
    <div style={{ background: "#111", color: "white", minHeight: "100vh", padding: 20 }}>
      
      <h1>📊 Camera Analytics: {id}</h1>

      {/* 🎥 Live Feed */}
      <img
        src={`${API}/camera/${id}`}
        style={{ width: "40%", marginBottom: 20 }}
      />

      {/* 🎯 Target */}
      <div style={{ marginBottom: 20 }}>
        <label>Target per hour: </label>
        <input
          type="number"
          value={target}
          onChange={(e) => setTarget(Number(e.target.value))}
        />
      </div>

      {/* 📊 SHIFT SUMMARY */}
      <div style={{ display: "flex", gap: 20, marginBottom: 20 }}>
        {Object.entries(shiftTotals).map(([shift, val]) => (
          <div key={shift} style={{ background: "#222", padding: 10 }}>
            <h4>{shift}</h4>
            <p>{val}</p>
          </div>
        ))}
      </div>

      {/* 📅 DAILY TOTAL */}
      <h3>Daily Total: {dailyTotal}</h3>

      {/* 📊 BAR CHART */}
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="hour" tickFormatter={(h) => `${h}:00`} />
          <YAxis />
          <Tooltip />

          <Bar dataKey="count">
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.color} />
            ))}
          </Bar>

          <Bar dataKey="target" />
        </BarChart>
      </ResponsiveContainer>

      {/* 📊 TABLE */}
      <table border="1" style={{ marginTop: 20, width: "100%", textAlign: "center" }}>
        <thead>
          <tr>
            <th>Hour</th>
            <th>Count</th>
            <th>Target</th>
            <th>Efficiency %</th>
            <th>Shift</th>
          </tr>
        </thead>
        <tbody>
          {chartData.map((row, i) => (
            <tr key={i}>
              <td>{row.hour}:00</td>
              <td>{row.count}</td>
              <td>{row.target}</td>
              <td style={{ color: row.count >= row.target ? "lime" : "red" }}>
                {row.efficiency}%
              </td>
              <td>{getShift(Number(row.hour))}</td>
            </tr>
          ))}
        </tbody>
      </table>

    </div>
  );
}