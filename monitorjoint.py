from ucl.common import byte_print, decode_version, decode_sn, getVoltage, pretty_print_obj, lib_version
from ucl.lowState import lowState
from ucl.lowCmd import lowCmd
from ucl.unitreeConnection import unitreeConnection, LOW_WIFI_DEFAULTS, LOW_WIRED_DEFAULTS
from ucl.enums import GaitType, SpeedLevel, MotorModeLow
from ucl.complex import motorCmd, motorCmdArray
import time
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint


def jointLinearInterpolation(initPos, targetPos, rate):

    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

# You can use one of the 3 Presets WIFI_DEFAULTS, LOW_CMD_DEFAULTS or HIGH_CMD_DEFAULTS.
# IF NONE OF THEM ARE WORKING YOU CAN DEFINE A CUSTOM ONE LIKE THIS:
#
# MY_CONNECTION_SETTINGS = (listenPort, addr_wifi, sendPort_high, local_ip_wifi)
# conn = unitreeConnection(MY_CONNECTION_SETTINGS)
d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5,
     'RR_0':6, 'RR_1':7, 'RR_2':8,
     'RL_0':9, 'RL_1':10, 'RL_2':11 }

def printRobotStatus(conn,lstate):
    data = conn.getData()
    for paket in data:
        lstate.parseData(paket)
        print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        print(f'SN [{byte_print(lstate.SN)}]:\t{decode_sn(lstate.SN)}')
        print(f'Ver [{byte_print(lstate.version)}]:\t{decode_version(lstate.version)}')
        print(f'SOC:\t\t\t{lstate.bms.SOC} %')
        print(f'Overall Voltage:\t{getVoltage(lstate.bms.cell_vol)} mv') #something is still wrong here ?!
        print(f'Current:\t\t{lstate.bms.current} mA')
        print(f'Cycles:\t\t\t{lstate.bms.cycle}')
        print(f'Temps BQ:\t\t{lstate.bms.BQ_NTC[0]} °C, {lstate.bms.BQ_NTC[1]}°C')
        print(f'Temps MCU:\t\t{lstate.bms.MCU_NTC[0]} °C, {lstate.bms.MCU_NTC[1]}°C')
        print(f'FootForce:\t\t{lstate.footForce}')
        print(f'FootForceEst:\t\t{lstate.footForceEst}')
        print(f'IMU Temp:\t\t{lstate.imu.temperature}')
        print(f'MotorState FL_2 MODE:\t\t{lstate.motorState[d["FL_2"]].mode}')
        print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')

## Initialization ##
print(f'Running lib version: {lib_version()}')
conn = unitreeConnection(LOW_WIFI_DEFAULTS)
conn.startRecv()
lcmd = lowCmd()
# lcmd.encrypt = True
lstate = lowState()
mCmdArr = motorCmdArray()
# Send empty command to tell the dog the receive port and initialize the connection
cmd_bytes = lcmd.buildCmd(debug=False)
conn.send(cmd_bytes)
printRobotStatus(conn,lstate)


motiontime = 0
joint_list = []

try:
    while True:
        time.sleep(0.002)
        motiontime +=1
        data = conn.getData()
        for paket in data:
            lstate.parseData(paket)
            if motiontime % 100 == 0: #Print every 100 cycles
                print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
                joint = [math.degrees(lstate.motorState[d["FL_0"]].q),math.degrees(lstate.motorState[d["FL_1"]].q),math.degrees(lstate.motorState[d["FL_2"]].q)]
                print(f'MotorState FL_0 q:\t\t{joint[0]}')
                print(f'MotorState FL_1 q:\t\t{joint[1]}')
                print(f'MotorState FL_2 q:\t\t{joint[2]}')
                joint_list.append(joint)
                
                print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')

except KeyboardInterrupt:
    print("\n프로그램이 중단되었습니다. 데이터를 저장하고 분석합니다...")
    
    # CSV 저장
    if joint_list:
        df = pd.DataFrame(joint_list, columns=['FL_0', 'FL_1', 'FL_2'])
        df.to_csv('joint_data.csv', index=False)
        print(f"joint_list가 'joint_data.csv'로 저장되었습니다. (총 {len(joint_list)}개 데이터)")
        
        # 각 joint의 통계 계산 및 출력
        print("\n=== Joint 각도 통계 ===")
        joint_names = ['FL_0', 'FL_1', 'FL_2']
        for i, name in enumerate(joint_names):
            joint_data = [joint[i] for joint in joint_list]
            min_val = min(joint_data)
            max_val = max(joint_data)
            diff = max_val - min_val
            print(f"{name}: 최소={min_val:.1f}°, 최대={max_val:.1f}°, 차이={diff:.1f}°")
        
        # 플롯 생성
        fig, axes = plt.subplots(3, 1, figsize=(16, 9))
        fig.suptitle('Joint Angle Range Analysis', fontsize=16)
        
        colors = ['red', 'green', 'blue']
        joint_names = ['FL_0', 'FL_1', 'FL_2']
        
        for i, (ax, name, color) in enumerate(zip(axes, joint_names, colors)):
            joint_data = [joint[i] for joint in joint_list]
            time_points = list(range(len(joint_data)))
            
            ax.plot(time_points, joint_data, color=color, linewidth=1)
            ax.set_title(f'{name} Angle Variation', fontsize=12)
            ax.set_ylabel('Angle (degrees)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(min(joint_data) - 5, max(joint_data) + 5)
            
            # 최대/최소값 표시
            min_val = min(joint_data)
            max_val = max(joint_data)
            ax.axhline(y=min_val, color=color, linestyle='--', alpha=0.7, label=f'Min: {min_val:.1f}°')
            ax.axhline(y=max_val, color=color, linestyle='--', alpha=0.7, label=f'Max: {max_val:.1f}°')
            ax.legend()
        
        axes[-1].set_xlabel('Time (100 cycle units)', fontsize=10)
        plt.tight_layout()
        plt.savefig('joint_analysis.png', dpi=300, bbox_inches='tight')
        print("플롯이 'joint_analysis.png'로 저장되었습니다.")
        plt.show()
        
    else:
        print("저장된 joint 데이터가 없습니다.")

