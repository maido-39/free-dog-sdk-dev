import asyncio
import json
import math
import os
import csv
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import sys

# Ensure parent directory (project root) is on sys.path for absolute imports
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("dashboard")

from ucl.highCmd import highCmd
from ucl.highState import highState
from ucl.unitreeConnection import unitreeConnection, HIGH_WIFI_DEFAULTS
from ucl.common import getVoltage


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected. total=%d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass
        logger.info("WebSocket disconnected. total=%d", len(self.active_connections))

    async def broadcast(self, message: str):
        stale: List[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning("Broadcast to a client failed: %s", e)
                stale.append(connection)
        for s in stale:
            self.disconnect(s)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="dashboard/static", html=True), name="static")

manager = ConnectionManager()


class DataStreamer:
    def __init__(self, target_hz: float = 50.0):
        self.target_period_s: float = 1.0 / target_hz
        self.conn: Optional[unitreeConnection] = None
        self.hcmd: Optional[highCmd] = None
        self.hstate: Optional[highState] = None
        self.running: bool = False
        self.last_packet_ts: Optional[float] = None
        # CSV logging
        self.log_dir: str = os.path.join("dashboard", "logs")
        self.csv_file: Optional[Any] = None
        self.csv_writer: Optional[csv.writer] = None
        self.csv_header_written: bool = False

    def start(self):
        if self.running:
            return
        # Initialize connection
        logger.info("Starting Unitree connection with defaults: %s", HIGH_WIFI_DEFAULTS)
        self.conn = unitreeConnection(HIGH_WIFI_DEFAULTS)
        try:
            self.conn.startRecv()
        except Exception as e:
            logger.error("Failed to start receiver: %s", e)
            raise
        self.hcmd = highCmd()
        self.hstate = highState()
        # Send empty command once to register receive port
        try:
            cmd_bytes = self.hcmd.buildCmd(debug=False)
            self.conn.send(cmd_bytes)
            logger.info("Sent initial registration command")
        except Exception as e:
            logger.error("Failed to send initial command: %s", e)
        # Prepare CSV logger
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.log_dir, f"telemetry_{ts}.csv")
            self.csv_file = open(file_path, mode="w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "timestamp_iso",
                "soc",
                "current_mA",
                "voltage_mV",
                "gyro_x","gyro_y","gyro_z",
                "acc_x","acc_y","acc_z",
                "rpy_r","rpy_p","rpy_y",
                "pos_x","pos_y",
                "vel_x","vel_y",
                "yawspeed",
                "stability_score"
            ])
            self.csv_header_written = True
        except Exception as e:
            # logging is optional; continue even if it fails
            self.csv_file = None
            self.csv_writer = None
            logger.warning("CSV logging disabled due to error: %s", e)
        self.running = True
        logger.info("DataStreamer started")

    def stop(self):
        self.running = False
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception as e:
            logger.warning("Error while closing CSV file: %s", e)

    def _compute_stability(self, imu_data: Dict[str, Any], yawspeed: float) -> Dict[str, Any]:
        # Score 100 -> penalize based on thresholds
        score = 100.0
        penalties: List[Dict[str, Any]] = []

        # rpy thresholds (degrees)
        roll = imu_data.get("rpy", [0.0, 0.0, 0.0])[0]
        pitch = imu_data.get("rpy", [0.0, 0.0, 0.0])[1]
        abs_roll = abs(roll)
        abs_pitch = abs(pitch)
        def rpy_penalty(val: float) -> float:
            if val > 20:  # severe tilt
                return 10.0
            if val > 10:
                return 5.0
            if val > 5:
                return 2.0
            return 0.0
        p_roll = rpy_penalty(abs_roll)
        p_pitch = rpy_penalty(abs_pitch)
        if p_roll:
            penalties.append({"type": "rpy_roll", "value": p_roll, "detail": abs_roll})
            score -= p_roll
        if p_pitch:
            penalties.append({"type": "rpy_pitch", "value": p_pitch, "detail": abs_pitch})
            score -= p_pitch

        # gyroscope thresholds (deg/s)
        gx, gy, gz = imu_data.get("gyroscope", [0.0, 0.0, 0.0])
        g_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        if g_mag > 300:
            penalties.append({"type": "gyro", "value": 8.0, "detail": g_mag})
            score -= 8.0
        elif g_mag > 200:
            penalties.append({"type": "gyro", "value": 5.0, "detail": g_mag})
            score -= 5.0
        elif g_mag > 120:
            penalties.append({"type": "gyro", "value": 2.0, "detail": g_mag})
            score -= 2.0

        # accelerometer thresholds (m/s^2 or unit in feed) - use magnitude deviation from 9.81
        ax, ay, az = imu_data.get("accelerometer", [0.0, 0.0, 0.0])
        a_mag = math.sqrt(ax * ax + ay * ay + az * az)
        # use 9.81 as nominal, but be robust if units differ
        deviation = abs(a_mag - 9.81)
        if deviation > 6.0:
            penalties.append({"type": "accel", "value": 6.0, "detail": deviation})
            score -= 6.0
        elif deviation > 3.0:
            penalties.append({"type": "accel", "value": 3.0, "detail": deviation})
            score -= 3.0

        # yaw speed (rad/s) if provided; additional safeguard
        if abs(yawspeed) > 2.5:
            penalties.append({"type": "yawspeed", "value": 4.0, "detail": yawspeed})
            score -= 4.0
        elif abs(yawspeed) > 1.5:
            penalties.append({"type": "yawspeed", "value": 2.0, "detail": yawspeed})
            score -= 2.0

        score = max(0.0, min(100.0, score))
        return {"score": score, "penalties": penalties}

    def _state_to_payload(self, hs: highState) -> Dict[str, Any]:
        # BMS
        soc = getattr(hs.bms, "SOC", None) if hs.bms is not None else None
        current = getattr(hs.bms, "current", None) if hs.bms is not None else None
        overall_voltage_mv: Optional[int] = None
        if hs.bms is not None and getattr(hs.bms, "cell_vol", None):
            try:
                overall_voltage_mv = getVoltage(hs.bms.cell_vol)
            except Exception:
                overall_voltage_mv = None

        imu_data = {
            "quaternion": getattr(hs.imu, "quaternion", [0.0, 0.0, 0.0, 0.0]),
            "gyroscope": getattr(hs.imu, "gyroscope", [0.0, 0.0, 0.0]),
            "accelerometer": getattr(hs.imu, "accelerometer", [0.0, 0.0, 0.0]),
            "rpy": getattr(hs.imu, "rpy", [0.0, 0.0, 0.0]),
            "temperature": getattr(hs.imu, "temperature", 0),
        }

        position = getattr(hs, "position", [0.0, 0.0])
        # some firmwares expose 3D; keep 2D for dashboard
        x = position[0] if len(position) > 0 else 0.0
        y = position[1] if len(position) > 1 else 0.0

        velocity = getattr(hs, "velocity", [0.0, 0.0])
        vx = velocity[0] if len(velocity) > 0 else 0.0
        vy = velocity[1] if len(velocity) > 1 else 0.0

        yawspeed = getattr(hs, "yawSpeed", 0.0)

        stability = self._compute_stability(imu_data, yawspeed)

        return {
            "bms": {"soc": soc, "current": current, "overall_voltage_mv": overall_voltage_mv},
            "imu": imu_data,
            "pose": {"x": x, "y": y},
            "velocity": {"vx": vx, "vy": vy},
            "yawspeed": yawspeed,
            "stability": stability,
        }

    async def loop(self):
        while self.running:
            loop_started = asyncio.get_event_loop().time()
            try:
                if self.conn is None:
                    logger.debug("Connection not initialized yet; sleeping")
                    await asyncio.sleep(self.target_period_s)
                    continue
                data_packets: List[bytes] = self.conn.getData()
                # Only process the latest packet to maintain ~50Hz cadence
                if data_packets:
                    pkt = data_packets[-1]
                    try:
                        if self.hstate is None:
                            logger.debug("HighState not initialized; skipping packet")
                        else:
                            self.hstate.parseData(pkt)
                        payload = self._state_to_payload(self.hstate)
                        await manager.broadcast(json.dumps(payload))
                        self.last_packet_ts = loop_started
                    except Exception as e:
                        # skip malformed latest packet
                        logger.debug("Malformed/latest packet parse error: %s", e)
                    # CSV logging
                    try:
                        if self.csv_writer:
                            ts_iso = datetime.now().isoformat()
                            b = payload.get("bms", {})
                            imu = payload.get("imu", {})
                            pose = payload.get("pose", {})
                            vel = payload.get("velocity", {})
                            stab = payload.get("stability", {})
                            gx, gy, gz = imu.get("gyroscope", [None, None, None])
                            ax, ay, az = imu.get("accelerometer", [None, None, None])
                            rr, rp, ry = imu.get("rpy", [None, None, None])
                            row = [
                                ts_iso,
                                b.get("soc"),
                                b.get("current"),
                                b.get("overall_voltage_mv"),
                                gx, gy, gz,
                                ax, ay, az,
                                rr, rp, ry,
                                pose.get("x"), pose.get("y"),
                                vel.get("vx"), vel.get("vy"),
                                payload.get("yawspeed"),
                                stab.get("score"),
                            ]
                            self.csv_writer.writerow(row)
                            if self.csv_file:
                                self.csv_file.flush()
                    except Exception as e:
                        # ignore logging errors
                        logger.debug("CSV write error: %s", e)
                else:
                    logger.debug("No data packets received this cycle")
            except Exception as e:
                # keep running despite transient errors
                logger.warning("Data loop error: %s", e)
            elapsed = asyncio.get_event_loop().time() - loop_started
            sleep_time = max(0.0, self.target_period_s - elapsed)
            await asyncio.sleep(sleep_time)


streamer = DataStreamer(target_hz=50.0)


@app.on_event("startup")
async def on_startup():
    try:
        streamer.start()
    except Exception as e:
        logger.error("Streamer failed to start: %s", e)
        # keep app up; allow retries via future controls
    asyncio.create_task(streamer.loop())
    logger.info("FastAPI startup complete; streamer loop scheduled (running=%s)", streamer.running)


@app.on_event("shutdown")
async def on_shutdown():
    streamer.stop()
    logger.info("FastAPI shutdown complete; streamer stopped")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; we don't expect messages from client
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket endpoint error: %s", e)
        manager.disconnect(websocket)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> Dict[str, Any]:
    return {
        "clients": len(manager.active_connections),
        "running": streamer.running,
        "last_packet_ts": streamer.last_packet_ts,
        "target_hz": 1.0 / streamer.target_period_s if streamer.target_period_s else None,
    }


def main():
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws_max_size=8 * 1024 * 1024,
        log_level="info",
    )


if __name__ == "__main__":
    main()


