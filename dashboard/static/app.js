const wsStatusEl = document.getElementById('ws-status');
const rateEl = document.getElementById('rate');
const socEl = document.getElementById('soc');
const currentEl = document.getElementById('current');
const voltageEl = document.getElementById('voltage');
const gyroEl = document.getElementById('gyro');
const accelEl = document.getElementById('accel');
const rpyEl = document.getElementById('rpy');
const yawspeedEl = document.getElementById('yawspeed');
const posEl = document.getElementById('pos');
const velEl = document.getElementById('vel');
const scoreEl = document.getElementById('score');
const penaltiesEl = document.getElementById('penalties');

const canvas = document.getElementById('pos-canvas');
const ctx = canvas.getContext('2d');

let points = [];
let lastTs = performance.now();
let frames = 0;

function setWsStatus(connected) {
  wsStatusEl.textContent = connected ? 'CONNECTED' : 'DISCONNECTED';
  wsStatusEl.className = 'badge ' + (connected ? 'green' : 'red');
}

function updateRate() {
  const now = performance.now();
  frames += 1;
  if (now - lastTs >= 1000) {
    rateEl.textContent = `${frames} Hz`;
    frames = 0;
    lastTs = now;
  }
}

function drawPath() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  // compute bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (!isFinite(minX)) {
    ctx.restore();
    return;
  }
  // padding and scale
  const pad = 20;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  const dx = maxX - minX || 1;
  const dy = maxY - minY || 1;
  const sx = w / dx;
  const sy = h / dy;
  const s = Math.min(sx, sy);
  const cx = pad + w / 2;
  const cy = pad + h / 2;
  const ox = (minX + maxX) / 2;
  const oy = (minY + maxY) / 2;

  // draw axes
  ctx.strokeStyle = '#ddd';
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(canvas.width, cy);
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, canvas.height);
  ctx.stroke();

  // draw path
  ctx.strokeStyle = '#2b7fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const px = cx + (points[i].x - ox) * s;
    const py = cy - (points[i].y - oy) * s;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // draw head
  const last = points[points.length - 1];
  const hx = cx + (last.x - ox) * s;
  const hy = cy - (last.y - oy) * s;
  ctx.fillStyle = '#ff6b2b';
  ctx.beginPath();
  ctx.arc(hx, hy, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function connect() {
  const loc = window.location;
  const wsProto = loc.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${wsProto}://${loc.host}/ws`;
  const ws = new WebSocket(url);

  ws.onopen = () => setWsStatus(true);
  ws.onclose = () => setWsStatus(false);
  ws.onerror = () => setWsStatus(false);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      updateRate();

      // Battery
      socEl.textContent = msg?.bms?.soc ?? '-';
      currentEl.textContent = msg?.bms?.current ?? '-';
      voltageEl.textContent = msg?.bms?.overall_voltage_mv ?? '-';

      // IMU
      const g = msg?.imu?.gyroscope || [0, 0, 0];
      const a = msg?.imu?.accelerometer || [0, 0, 0];
      const r = msg?.imu?.rpy || [0, 0, 0];
      gyroEl.textContent = `${g[0].toFixed(1)}, ${g[1].toFixed(1)}, ${g[2].toFixed(1)}`;
      accelEl.textContent = `${a[0].toFixed(2)}, ${a[1].toFixed(2)}, ${a[2].toFixed(2)}`;
      rpyEl.textContent = `${r[0].toFixed(1)}, ${r[1].toFixed(1)}, ${r[2].toFixed(1)}`;
      yawspeedEl.textContent = (msg?.yawspeed ?? 0).toFixed(2);

      // Pose/Velocity
      const x = msg?.pose?.x ?? 0;
      const y = msg?.pose?.y ?? 0;
      const vx = msg?.velocity?.vx ?? 0;
      const vy = msg?.velocity?.vy ?? 0;
      posEl.textContent = `${x.toFixed(3)}, ${y.toFixed(3)}`;
      velEl.textContent = `${vx.toFixed(3)}, ${vy.toFixed(3)}`;

      // score / penalties
      scoreEl.textContent = (msg?.stability?.score ?? 0).toFixed(1);
      penaltiesEl.innerHTML = '';
      const pens = msg?.stability?.penalties || [];
      if (pens.length === 0) {
        penaltiesEl.textContent = 'No penalties';
      } else {
        for (const p of pens) {
          const div = document.createElement('div');
          div.className = 'penalty';
          div.textContent = `${p.type}: -${p.value} (detail=${p.detail?.toFixed ? p.detail.toFixed(2) : p.detail})`;
          penaltiesEl.appendChild(div);
        }
      }

      // path
      points.push({ x, y });
      if (points.length > 2000) points.shift();
      drawPath();
    } catch (e) {
      // noop
    }
  };

  // send periodic keepalive to keep server loop happy
  setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping');
  }, 1000);
}

connect();


