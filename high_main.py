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
from ucl.unitreeConnection import unitreeConnection, HIGH_WIRED_DEFAULTS
from ucl.enums import MotorModeHigh, GaitType, SpeedLevel
import time
import pygame
import threading
from collections import deque
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

    # 화면 표시에 사용할 최근 상태
    state_queue = deque(maxlen=10)
    state_lock = threading.Lock()

    # ===================== Motion Model Params =====================
    DT = 0.02                 # 50Hz control loop (20ms)
    # 속도 제한
    MAX_VX = 0.3              # m/s (forward/back)
    MAX_VY = 0.3              # m/s (lateral)
    MAX_YAW = 0.8             # rad/s (turn)

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

    # ===================== Thread: Keyboard =====================
    def check_keyboard():
        """Keyboard input loop (i/j/k/l for x,y, u/o for yaw, ESC to quit)"""
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
                    elif event.key == pygame.K_q:
                        shared_state['yaw_pos'] = True
                        print("Turn Left (+yaw)")
                    elif event.key == pygame.K_e:
                        shared_state['yaw_neg'] = True
                        print("Turn Right (-yaw)")
                    elif event.key == pygame.K_ESCAPE:
                        shared_state['quit'] = True

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_i:
                        shared_state['x_pos'] = False
                    elif event.key == pygame.K_k:
                        shared_state['x_neg'] = False
                    elif event.key == pygame.K_j:
                        shared_state['y_pos'] = False
                    elif event.key == pygame.K_l:
                        shared_state['y_neg'] = False
                    elif event.key == pygame.K_q:
                        shared_state['yaw_pos'] = False
                    elif event.key == pygame.K_e:
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
                        print("\n" + "=" * 80)
                        print(f"Battery: {state['battery']}%")
                        print(f"Pos: {state['pos']}")
                        print(f"Vel: {state['vel']} | YawSpeed: {state['yaw_speed']:.3f}")
                        print(f"Mode: {state['mode']} | FootForce: {state['foot_force']}")
                        print("=" * 80)
            except Exception as e:
                logging.error(f"display_state error: {e}")
            time.sleep(0.5)

    # ===================== Connection =====================
    conn = unitreeConnection(HIGH_WIRED_DEFAULTS)
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
