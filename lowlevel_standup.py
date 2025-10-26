from curses.ascii import ctrl
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
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import glob


## Linear interpolation between two joint positions
# Input : initPos(rate=0) ~~~~~~ targetPos(rate=1)
# Output : interpolated_position =(p)
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
# jointd = list(d.keys()) # 이렇게도 joint 리스트 만들 수 있음
joint = ['FR_0', 'FR_1', 'FR_2', 'FL_0', 'FL_1', 'FL_2', 'RL_0', 'RL_1', 'RL_2', 'RR_0', 'RR_1', 'RR_2']
jointnum = len(joint)

##   Define constants     ##
PosStopF  = math.pow(10,9)
VelStopF  = 16000.0
LOWLEVEL  = 0xff
dt = 0.002 # 2ms control loop, While 문 도는 Hz를 의미함, 모터 제어 Hz 다름 = 50Hz(ctrl_dt)
## == Define constants == ##

# p / d gains
# [FR_0, FR_1, FR_2]
kp = [15] * jointnum # [8,8,...,8]
kd = [4] * jointnum # [1,1,...,1]
## This one Should be Modified when changed to 4legs ##
tau = [-0.65,2,2,-0.65,2,2,-0.65,2,2,-0.65,2,2] # [FR_0, FR_1, FR_2, FL_0, FL_1, FL_2, RR_0, RR_1, RR_2, RL_0, RL_1, RL_2]


## Initialize connection ##
print(f'Running lib version: {lib_version()}')
conn = unitreeConnection(LOW_WIFI_DEFAULTS)
conn.startRecv()
## == Initialize connection == ##


## instantiate lowlevel command and state objects ##
lcmd = lowCmd()
# lcmd.encrypt = True
lstate = lowState()
mCmdArr = motorCmdArray()


## get initial state & print Log ##
# Send empty command to tell the dog the receive port and initialize the connection
cmd_bytes = lcmd.buildCmd(debug=False)
conn.send(cmd_bytes)

# define a function to print log data
def printLog(conn,lstate):
    data = conn.getData()
    try:
        paket = data[-1]
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
        print(f'MotorState FR_0 MODE:\t\t{lstate.motorState[d["FR_0"]].mode}')
        print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
    except Exception as e:
        print(f"데이터 처리 중 오류 발생: {e}")
        print(data)
        return None

printLog(conn,lstate) # print initial state

## 자주쓰는 함수들 정의

# 12개 관절의 각도(q)를 radian 단위의 리스트로 리턴하는 함수
def get_RPY(conn,lstate,deg = False):
    """
    conn에서 최신 패킷을 받아 로봇의 imu RPY 각도를 리턴
    
        rpy (list): 로봇의 imu RPY 각도 리스트 (roll, pitch, yaw)
    """
    data = conn.getData()
    try:
        lstate.parseData(data[-1])# 최신 패킷만 처리
        # data 에서 rpy 파싱
        rpy = lstate.imu.rpy
        ## Debug Print
        # print("RPY : ",np.round(np.rad2deg(rpy),2))

        if deg: # deg 플래그가 True 이면 라디안을 도로 변환
            return np.rad2deg(rpy)
        else: # deg 플래그가 False 이면 라디안 그대로 리턴
            return rpy

    except Exception as e:
        print(f"imu 읽기 오류: {e}")
        # 만약 'list index out of range' 에러면 너무 빨리 요청한거라 잠시 대기 후 재시도
        if 'list index out of range' in str(e) and len(data) < 1:
            time.sleep(0.002)
            return get_RPY(conn, lstate)
        else:
            return None

# 12개 관절의 각도(q)를 radian 단위의 리스트로 리턴하는 함수
def get_joint_angles_rad(conn,lstate):
    """
    conn에서 최신 패킷을 받아 12개 관절의 각도(q, radian) 리스트를 반환

    Returns:
        q_list (list): 12개 관절의 각도 리스트 (radian, joint 순서는 global 'joint' 리스트와 동일)
                       오류시 None 반환
    """
    joint = ['FR_0', 'FR_1', 'FR_2', 'FL_0', 'FL_1', 'FL_2', 'RL_0', 'RL_1', 'RL_2', 'RR_0', 'RR_1', 'RR_2']

    data = conn.getData()
    try:
        lstate.parseData(data[-1])# 최신 패킷만 처리
        # 관절 이름이 'joint' 리스트에 있다고 가정
        q_list = [lstate.motorState[d[jname]].q for jname in joint]
        return q_list

    except Exception as e:
        print(f"관절 각도 읽기 오류: {e}")
        # 만약 'list index out of range' 에러면 너무 빨리 요청한거라 잠시 대기 후 재시도
        if 'list index out of range' in str(e) and len(data) < 1:
            time.sleep(0.002)
            return get_joint_angles_rad(conn, lstate)
        else:
            return None

def doMotion(qDes,joint = joint, kp = kp, kd = kd, tau = tau):
    ## qList 에 조인트 절대값 넣으면 움직임
    # Hardcode Vars for Test
    if type(joint) is not list:
        joint = ['FR_0', 'FR_1', 'FR_2','FL_0', 'FL_1', 'FL_2','RR_0', 'RR_1', 'RR_2','RL_0', 'RL_1', 'RL_2']
    jointnum = len(joint)
    kp = [20] * jointnum
    kd = [4] * jointnum
    tau = [2] * jointnum
    vel = [0] * jointnum

    for i in range(len(joint)):
        mCmdArr.setMotorCmd(joint[i],  motorCmd(mode=MotorModeLow.Servo, q=qDes[i], dq = vel[i], Kp = kp[i], Kd = kd[i], tau = tau[i]))

    lcmd.motorCmd = mCmdArr
    # build command bytes
    cmd_bytes = lcmd.buildCmd(debug=False)
    # send command bytes to robot
    conn.send(cmd_bytes)

def doAction(qStart,qEnd,duration):
    step = int(duration*hz)
    for i in range(step):
        qDes = jointLinearInterpolationList(qStart,qEnd,i/step)
        doMotion(qDes,tau = tau)
        time.sleep(1/hz)
    print("Action done")

## Linear interpolation between two joint positions
# Input : initPos(rate=0) ~~~~~~ targetPos(rate=1)
# Output : interpolated_position =(p)
def jointLinearInterpolationList(initPos, targetPos, rate):
    # list to np.array for vectorized operation
    np_initPos = np.array(initPos)
    np_targetPos = np.array(targetPos)
    # Clamp rate between 0 and 1
    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    # Do Interpolation
    np_p = np_initPos*(1-rate) + np_targetPos*rate
    p = np_p.tolist()
    return p

def set_hip_zero(q):
    # 3의 배수 인덱스(즉, 각 관절 그룹의 첫 번째)에 대해 0으로 설정하고 나머지는 기존 값을 유지
    return [0 if i % 3 == 0 else q[i] for i in range(len(q))]



## 1. get curr angle
# Curr Angle
qInit = get_joint_angles_rad(conn,lstate)
# Set Hip Zero
qInit_cute = set_hip_zero(qInit)
print("Curr Angle : ",[round(math.degrees(q),2) for q in qInit])
print("1stage Target Angle : ",[round(math.degrees(q),2) for q in qInit_cute])


## 1. move leg to standup pos - effect to joint 1,2
def set_joint12_standup(q):
    # 1번쨰 관절 일괄 10도감소
    # 2번쨰 관절 일괄 60도증가
    q = [q[i] - math.radians(20)  if i % 3 == 1 else q[i] for i in range(len(q))] 
    q = [q[i] + math.radians(75)  if i % 3 == 2 else q[i] for i in range(len(q))] 
    return q
qStandup = set_joint12_standup(qInit_cute)

print("1stage Target Angle : ",[round(math.degrees(q),2) for q in qInit_cute])
print("2stage Target Angle : ",[round(math.degrees(q),2) for q in qStandup])



## Do action 1+2stage (move Joint to Target Angle)
hz = 50 #[Hz]
duration_st1 = 0.5 #[Sec]
duration_st2 = 2 #[Sec]
duration_st3 = 2 #[Sec]

step_st1 = int(duration_st1*hz)
print("step_st1 : ",step_st1)
step_st2 = int(duration_st2*hz)
print("step_st2 : ",step_st2)
step_st3 = int(duration_st3*hz)
print("step_st3 : ",step_st3)
torque = 20
tau = [torque]*12

time.sleep(8)
# Do action 1stage (move Joint to Target Angle)
doAction(qInit,qInit_cute,duration_st1)
print("step_st1 done")

# Do action 2stage (move Joint to Target Angle)
doAction(qInit_cute,qStandup,duration_st2)
print("step_st2 done")

start_time = time.time()
holdtime = 5 # [Sec]
while True:
    if time.time() - start_time > holdtime:
        break
    doMotion(qStandup,tau = tau)
    time.sleep(1/hz)
print("step_st2 hold done")

# Do action Standdown (move Joint to Initial Angle)
doAction(qStandup,qInit,duration_st3)
print("Standdown done")




