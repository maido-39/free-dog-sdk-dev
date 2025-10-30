"""
High-level command interface for Unitree Go1 quadruped robot
with velocity ramping (accel/decay) control, ijkl/u-o keyboard mapping,
and 500Hz logging (.log)

Author: Jiyang Lee (original)
Modified by: ChatGPT (adaptive speed, logging, ijkl/uo mapping)
License: MIT
"""

from ucl.common import byte_print, decode_version, decode_sn, getVoltage, lib_version
from ucl.highCmd import highCmd
from ucl.highState import highState
from ucl.unitreeConnection import unitreeConnection, HIGH_WIRED_DEFAULTS, HIGH_WIFI_DEFAULTS
from ucl.enums import MotorModeHigh, GaitType, SpeedLevel
import time
import pygame
import threading
from collections import deque
import csv
import os
import logging
from datetime import datetime
import json


def main():
    print(f'Running lib version: {lib_version()}')

    # ===================== Logging Setup =====================
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    start_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"go1_{start_str}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("=== Go1 Controller Started (ijkl: x,y | u/o: yaw) ===")

    # ===================== Keyboard Setup =====================
    pygame.init()
    _ = pygame.display.set_mode((420, 320))
    pygame.display.set_caption("Go1 Control (i/j/k/l for x,y; u/o for yaw; ESC quits)")
    clock = pygame.time.Clock()

    # 키 상태 공유
    shared_state = {
        'x_pos': False,   # i : +x (forward)
        'x_neg': False,   # k : -x (backward)
        'y_pos': False,   # j : +y (strafe left)
        'y_neg': False,   # l : -y (strafe right)
        'yaw_pos': False, # u : +yaw (turn left)
        'yaw_neg': False, # o : -yaw (turn right)
        'quit': False
    }

    # 녹화 상태
    shared_state['recording'] = False

    # 화면 표시에 사용할 최근 상태
    state_queue = deque(maxlen=10)
    state_lock = threading.Lock()
    # 이동평균 윈도우 (metrics 전용)
    stab_window = deque(maxlen=5)
    speed_window = deque(maxlen=5)
    time_window = deque(maxlen=5)
    # CSV 파일 핸들
    csv_fp = None
    csv_writer = None

    # ===================== Motion Model Params =====================
    DT = 0.02                 # 50Hz control loop (20ms)
    # 속도 제한
    MAX_VX = 1.2              # m/s (forward/back)
    MAX_VY = 1.2              # m/s (lateral)
    MAX_YAW = 1.5            # rad/s (turn)

    # 가속도 (키 유지 시 가속)
    ACC_VX = 1.0              # m/s^2
    ACC_VY = 1.0              # m/s^2 
    ACC_YAW = 7.0             # rad/s^2

    # 입력 해제 시 감쇠(급정지 느낌). 0~1, 작을수록 급하게 멈춤
    DECAY_VX = 0.6
    DECAY_VY = 0.6
    DECAY_YAW = 0.5

    # 누적 명령 상태
    vx_cmd = 0.0
    vy_cmd = 0.0
    yaw_cmd = 0.0
    program_start_time = time.time()

    # ===================== Thread: Keyboard =====================
    def check_keyboard():
        """Keyboard input loop (i/j/k/l for x,y, u/o for yaw, ESC to quit)"""
        nonlocal csv_fp, csv_writer, program_start_time

        while not shared_state['quit']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    shared_state['quit'] = True

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_i:
                        shared_state['x_pos'] = True
                        print("Forward (+x)")
                    elif event.key == pygame.K_k:
                        shared_state['x_neg'] = True
                        print("Backward (-x)")
                    elif event.key == pygame.K_j:
                        shared_state['y_pos'] = True
                        print("Strafe Left (+y)")
                    elif event.key == pygame.K_l:
                        shared_state['y_neg'] = True
                        print("Strafe Right (-y)")
                    elif event.key == pygame.K_u:
                        shared_state['yaw_pos'] = True
                        print("Turn Left (+yaw)")
                    elif event.key == pygame.K_o:
                        shared_state['yaw_neg'] = True
                        print("Turn Right (-yaw)")
                    elif event.key == pygame.K_ESCAPE:
                        shared_state['quit'] = True
                    elif event.key == pygame.K_n:
                        # Start recording
                        with state_lock:
                            if not shared_state.get('recording', False):
                                # reset metric windows
                                stab_window.clear()
                                speed_window.clear()
                                time_window.clear()
                                # update program start time for time_score
                                program_start_time = time.time()

                                # open csv
                                os.makedirs('expr_data', exist_ok=True)
                                fname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '-go1expr.csv'
                                fpath = os.path.join('expr_data', fname)
                                csv_fp = open(fpath, 'w', newline='')
                                csv_writer = csv.writer(csv_fp)
                                # header
                                header = [
                                    'timestamp', 'elapsed', 'battery', 'mode',
                                    'pos_x', 'pos_y', 'pos_z',
                                    'vel_x', 'vel_y', 'vel_z', 'yaw_speed',
                                    'roll', 'pitch', 'yaw',
                                    'footf_0', 'footf_1', 'footf_2', 'footf_3',
                                    'stability_smooth', 'speed_score_smooth', 'time_score_smooth', 'total_score'
                                ]
                                csv_writer.writerow(header)
                                csv_fp.flush()
                                shared_state['recording'] = True
                                print(f"Recording started -> {fpath}")
                    elif event.key == pygame.K_m:
                        # Stop recording
                        with state_lock:
                            if shared_state.get('recording', False):
                                shared_state['recording'] = False
                                try:
                                    if csv_fp:
                                        csv_fp.close()
                                except Exception:
                                    pass
                                csv_fp = None
                                csv_writer = None
                                # reset metric windows
                                stab_window.clear()
                                speed_window.clear()
                                time_window.clear()
                                print("Recording stopped")

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_i:
                        shared_state['x_pos'] = False
                    elif event.key == pygame.K_k:
                        shared_state['x_neg'] = False
                    elif event.key == pygame.K_j:
                        shared_state['y_pos'] = False
                    elif event.key == pygame.K_l:
                        shared_state['y_neg'] = False
                    elif event.key == pygame.K_u:
                        shared_state['yaw_pos'] = False
                    elif event.key == pygame.K_o:
                        shared_state['yaw_neg'] = False

            clock.tick(120)  # 충분히 높은 키 폴링
        pygame.quit()

    # ===================== Thread: Receive State =====================
    def receive_state():
        """Receive robot state at 500Hz and log"""
        while not shared_state['quit']:
            try:
                data = conn.getData()
                if data:
                    for packet in data:
                        hstate.parseData(packet)

                        # 받은 상태를 안전하게 추출
                        pos = list(getattr(hstate, 'position', [0, 0, 0]))
                        vel = list(getattr(hstate, 'velocity', [0, 0, 0]))
                        yaw_speed = getattr(hstate, 'yawSpeed', 0.0)
                        battery = getattr(hstate.bms, 'SOC', 0.0)
                        foot_force = getattr(hstate, 'footForce', [0, 0, 0, 0])
                        mode = getattr(hstate, 'mode', 0)

                        # 화면 표시용 최신 상태
                        with state_lock:
                            state_queue.append({
                                'battery': battery,
                                'pos': pos,
                                'vel': vel,
                                'rpy': list(getattr(hstate.imu, 'rpy', [0.0, 0.0, 0.0])),
                                'yaw_speed': yaw_speed,
                                'foot_force': foot_force,
                                'mode': mode,
                                'time': time.time()
                            })

                        # 로그 기록
                        log_line = (
                            f"Battery={battery:.1f}% | "
                            f"Pos={json.dumps(pos)} | "
                            f"Vel={json.dumps(vel)} | "
                            f"YawSpeed={yaw_speed:.3f} | "
                            f"FootForce={json.dumps(foot_force)} | "
                            f"Mode={mode}"
                        )
                        logging.info(log_line)

            except Exception as e:
                logging.error(f"Error in receive_state: {e}")
                time.sleep(0.01)

            time.sleep(0.002)  # 500Hz 주기

    # ===================== Thread: Display =====================
    def display_state():
        """Print recent state to console"""
        while not shared_state['quit']:
            try:
                with state_lock:
                    if state_queue:
                        state = state_queue[-1]
                        # compute metrics
                        roll, pitch, yaw = state.get('rpy', [0.0, 0.0, 0.0])
                        # Stability: 1 / (roll^2 + pitch^2) with small eps to avoid div0
                        eps = 1e-6
                        stability = 1.0 / (roll * roll + pitch * pitch + eps)

                        # Speed score: every 0.1 m/s -> +3 points. clamp speed to [0, 3.4]
                        vx, vy, vz = state.get('vel', [0.0, 0.0, 0.0])
                        speed = (vx * vx + vy * vy) ** 0.5
                        speed_clamped = max(0.0, min(3.4, speed))
                        speed_score = int(round((speed_clamped / 0.1))) * 3

                        # Time score: -1 point per 0.1s elapsed since program start (penalty)
                        elapsed = state.get('time', time.time()) - program_start_time
                        time_score = -int(round(elapsed / 0.1))

                        # --- apply moving average filter to metrics only ---
                        stab_window.append(stability)
                        speed_window.append(float(speed_score))
                        time_window.append(float(time_score))

                        def mean(window):
                            return (sum(window) / len(window)) if window else 0.0

                        smooth_stability = mean(stab_window)
                        smooth_speed_score = mean(speed_window)
                        smooth_time_score = mean(time_window)

                        # Total (using smoothed metrics)
                        total_score = smooth_stability + smooth_speed_score + smooth_time_score

                        # Pretty print, rounded and aligned
                        pos = state.get('pos', [0.0, 0.0, 0.0])
                        foot_force = state.get('foot_force', [0, 0, 0, 0])

                        print("\n" + "=" * 80)
                        print(f"{ 'Battery':>10}: {state['battery']:6.1f}%   { 'Mode':>6}: {state['mode']:>2}   { 'Time':>6}: {elapsed:6.2f}s")
                        print(f"{ 'Pos':>10}: {pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}")
                        print(f"{ 'Vel':>10}: {vx:7.3f}, {vy:7.3f}, {vz:7.3f}   YawSp: {state['yaw_speed']:7.3f}")
                        print(f"{ 'Roll':>10}: {roll:7.3f}   { 'Pitch':>7}: {pitch:7.3f}   { 'Yaw':>6}: {yaw:7.3f}")
                        print(f"{ 'FootF':>10}: {foot_force}   { 'Stability':>9}: {smooth_stability:8.3f}")
                        print(f"{ 'SpeedScore':>10}: {int(round(smooth_speed_score)):6d}   { 'TimeScore':>10}: {int(round(smooth_time_score)):6d}   { 'Total':>6}: {total_score:8.2f}")
                        print("=" * 80)

                        # CSV 기록 (recording 중일 때)
                        if shared_state.get('recording', False) and csv_writer:
                            try:
                                row = [
                                    datetime.now().isoformat(),
                                    f"{elapsed:.3f}",
                                    f"{state['battery']:.1f}",
                                    state['mode'],
                                    f"{pos[0]:.3f}", f"{pos[1]:.3f}", f"{pos[2]:.3f}",
                                    f"{vx:.3f}", f"{vy:.3f}", f"{vz:.3f}", f"{state['yaw_speed']:.3f}",
                                    f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}",
                                    foot_force[0], foot_force[1], foot_force[2], foot_force[3],
                                    f"{smooth_stability:.6f}", f"{smooth_speed_score:.1f}", f"{smooth_time_score:.1f}", f"{total_score:.3f}"
                                ]
                                csv_writer.writerow(row)
                                csv_fp.flush()
                            except Exception as e:
                                logging.error(f"Error writing CSV: {e}")
            except Exception as e:
                logging.error(f"display_state error: {e}")
            time.sleep(0.2)

    # ===================== Connection =====================
    conn = unitreeConnection(HIGH_WIFI_DEFAULTS)
    conn.startRecv()
    hcmd = highCmd()
    hstate = highState()

    # 초기 패킷 송신 후 상태 파싱
    cmd_bytes = hcmd.buildCmd(debug=False)
    conn.send(cmd_bytes)
    time.sleep(0.5)

    data = conn.getData()
    for packet in data:
        hstate.parseData(packet)
        print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        print(f'SN [{byte_print(hstate.SN)}]:\t{decode_sn(hstate.SN)}')
        print(f'Ver [{byte_print(hstate.version)}]:\t{decode_version(hstate.version)}')
        print(f'SOC:\t{hstate.bms.SOC} %')
        print(f'Voltage:\t{getVoltage(hstate.bms.cell_vol)} mV')
        print(f'FootForce:\t{hstate.footForce}')
        print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')

    # ===================== Start Threads =====================
    threading.Thread(target=check_keyboard, daemon=True).start()
    threading.Thread(target=receive_state, daemon=True).start()
    threading.Thread(target=display_state, daemon=True).start()

    # ===================== Main Control Loop =====================
    print("\nStarting control loop...")
    print("Use i/j/k/l for x,y; u/o for yaw; ESC to quit.\n")

    try:
        while not shared_state['quit']:
            time.sleep(DT)

            # 디지털 입력 → -1/0/+1 강도
            in_x = float(shared_state['x_pos']) - float(shared_state['x_neg'])   # +x: i, -x: k
            in_y = float(shared_state['y_pos']) - float(shared_state['y_neg'])   # +y: j, -y: l
            in_yaw = float(shared_state['yaw_pos']) - float(shared_state['yaw_neg'])  # +yaw: u, -yaw: o

            # === Integrate acceleration & apply decay ===
            if in_x != 0.0:
                vx_cmd += in_x * ACC_VX * DT
            else:
                vx_cmd *= DECAY_VX

            if in_y != 0.0:
                vy_cmd += in_y * ACC_VY * DT
            else:
                vy_cmd *= DECAY_VY

            if in_yaw != 0.0:
                yaw_cmd += in_yaw * ACC_YAW * DT
            else:
                yaw_cmd *= DECAY_YAW

            # Clamp speeds
            vx_cmd = max(-MAX_VX, min(MAX_VX, vx_cmd))
            vy_cmd = max(-MAX_VY, min(MAX_VY, vy_cmd))
            yaw_cmd = max(-MAX_YAW, min(MAX_YAW, yaw_cmd))

            # 작은 값은 0으로(지터 방지)
            EPS_V = 0.02
            EPS_W = 0.05
            if abs(vx_cmd) < EPS_V: vx_cmd = 0.0
            if abs(vy_cmd) < EPS_V: vy_cmd = 0.0
            if abs(yaw_cmd) < EPS_W: yaw_cmd = 0.0

            moving = (vx_cmd != 0.0) or (vy_cmd != 0.0) or (yaw_cmd != 0.0)

            if moving:
                hcmd.mode = MotorModeHigh.VEL_WALK
                hcmd.gaitType = GaitType.TROT
                hcmd.velocity = [vx_cmd, vy_cmd]  # x, y 둘 다 사용
                hcmd.yawSpeed = yaw_cmd
                hcmd.footRaiseHeight = 0.1
                hcmd.speedLevel = SpeedLevel.MEDIUM_SPEED
            else:
                hcmd.mode = MotorModeHigh.IDLE
                hcmd.velocity = [0.0, 0.0]
                hcmd.yawSpeed = 0.0

            try:
                cmd_bytes = hcmd.buildCmd(debug=False)
                conn.send(cmd_bytes)
            except Exception as e:
                logging.error(f"Error sending command: {e}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        shared_state['quit'] = True
        time.sleep(0.1)
        try:
            hcmd.mode = MotorModeHigh.IDLE
            hcmd.velocity = [0.0, 0.0]
            hcmd.yawSpeed = 0.0
            conn.send(hcmd.buildCmd(debug=False))
            conn.stopRecv()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        pygame.quit()
        logging.info("Connection closed.")
        print("Connection closed.")


if __name__ == "__main__":
    main()
