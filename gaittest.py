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

## Utility functions for file management
def ensure_logs_directory():
    """Create logs directory if it doesn't exist"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("Created 'logs' directory")

def get_timestamp_filename(base_name, extension):
    """Generate filename with timestamp prefix"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{base_name}.{extension}"

def get_latest_csv_file():
    """Get the most recent CSV file from logs directory"""
    ensure_logs_directory()
    csv_pattern = os.path.join('logs', '*_joint_tracking_data.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("No CSV files found in logs directory")
        return None
    
    # Sort by filename (timestamp) in descending order
    csv_files.sort(reverse=True)
    latest_file = csv_files[0]
    print(f"Latest CSV file: {latest_file}")
    return latest_file

## Modularized plotting function for joint tracking analysis
def print_timing_drift_statistics(timing_drifts, actual_loop_times, ctrldt):
    """
    타이밍 드리프트 분석 통계를 출력하는 함수
    
    Args:
        timing_drifts (list): 타이밍 드리프트 데이터 (ms 단위)
        actual_loop_times (list): 실제 루프 시간 데이터 (ms 단위)
        ctrldt (float): 제어 주기 (초)
    """
    if len(timing_drifts) == 0:
        print("타이밍 드리프트 데이터가 없습니다.")
        return
    
    print("\n=== 타이밍 Drift 분석 통계 ===")
    avg_drift = np.mean(timing_drifts)
    max_drift = np.max(timing_drifts)
    min_drift = np.min(timing_drifts)
    std_drift = np.std(timing_drifts)
    
    avg_loop_time = np.mean(actual_loop_times)
    actual_freq = 1000 / avg_loop_time  # Hz
    
    print(f"평균 Drift: {avg_drift:.3f} ms")
    print(f"최대 Drift: {max_drift:.3f} ms")
    print(f"최소 Drift: {min_drift:.3f} ms")
    print(f"Drift 표준편차: {std_drift:.3f} ms")
    print(f"평균 실제 루프 주기: {avg_loop_time:.3f} ms")
    print(f"실제 제어 주파수: {actual_freq:.2f} Hz (목표: {1/ctrldt:.2f} Hz)")
    print(f"주파수 오차: {abs(actual_freq - 1/ctrldt):.2f} Hz")
    
    # Drift 분포 분석
    positive_drifts = [d for d in timing_drifts if d > 0]
    negative_drifts = [d for d in timing_drifts if d < 0]
    print(f"양수 Drift 비율: {len(positive_drifts)/len(timing_drifts)*100:.1f}%")
    print(f"음수 Drift 비율: {len(negative_drifts)/len(timing_drifts)*100:.1f}%")

def plot_joint_tracking_from_csv(csv_file_path, output_file=None):
    """
    CSV 파일에서 joint tracking 데이터를 읽어서 플롯 생성
    
    Args:
        csv_file_path (str): CSV 파일 경로
        output_file (str, optional): 출력 이미지 파일 경로. None이면 화면에 표시만 함.
    """
    if not os.path.exists(csv_file_path):
        print(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}")
        return
    
    # CSV 데이터 읽기
    df = pd.read_csv(csv_file_path)
    
    # Joint 이름 추출 (target 컬럼에서)
    target_cols = [col for col in df.columns if col.endswith('_target')]
    joint_names = [col.replace('_target', '') for col in target_cols]
    jointnum = len(joint_names)
    
    if jointnum == 0:
        print("CSV 파일에서 joint 데이터를 찾을 수 없습니다.")
        return
    
    # 시간 데이터 추출 (실제 시간 사용)
    if 'real_time' in df.columns:
        time_array = df['real_time'].values
        time_label = 'Real Time (s)'
    elif 'timestamp' in df.columns:
        time_array = df['timestamp'].values
        time_label = 'Time (s)'
    else:
        print("시간 데이터를 찾을 수 없습니다.")
        return
    
    # Create 3x4 subplot grid with doubled width
    fig, axes = plt.subplots(3, 4, figsize=(40, 15))  # 2배 폭으로 증가
    fig.suptitle('Joint Tracking Performance Analysis', fontsize=20)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each joint
    for i in range(jointnum):
        ax = axes_flat[i]
        
        # Get target and actual data
        target_col = f'{joint_names[i]}_target'
        actual_col = f'{joint_names[i]}_actual'
        
        if target_col in df.columns and actual_col in df.columns:
            target_data = df[target_col].values
            actual_data = df[actual_col].values
            
            # Plot target and actual positions
            ax.plot(time_array, target_data, 'b-', label='Target', linewidth=2)
            ax.plot(time_array, actual_data, 'r-', label='Actual', linewidth=1.5)
            
            # Calculate tracking error
            error = actual_data - target_data
            ax2 = ax.twinx()
            ax2.plot(time_array, error, 'g--', label='Error', alpha=0.7)
            ax2.set_ylabel('Error (deg)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Set labels and title
            ax.set_title(f'{joint_names[i]} Tracking Performance', fontsize=14)
            ax.set_xlabel(time_label)
            ax.set_ylabel('Position (deg)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Calculate and display statistics
            rmse = np.sqrt(np.mean(error**2))
            max_error = np.max(np.abs(error))
            ax.text(0.02, 0.98, f'RMSE: {rmse:.2f}°\nMax Error: {max_error:.2f}°', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No data for {joint_names[i]}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    # Hide unused subplots
    for i in range(jointnum, 12):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"플롯이 '{output_file}'로 저장되었습니다.")
    
    plt.show()
    
    # Print overall statistics
    print("\n=== 전체 추종 성능 통계 ===")
    for i in range(jointnum):
        target_col = f'{joint_names[i]}_target'
        actual_col = f'{joint_names[i]}_actual'
        if target_col in df.columns and actual_col in df.columns:
            error = df[actual_col].values - df[target_col].values
            rmse = np.sqrt(np.mean(error**2))
            max_error = np.max(np.abs(error))
            print(f"{joint_names[i]}: RMSE={rmse:.2f}°, Max Error={max_error:.2f}°")


# You can use one of the 3 Presets WIFI_DEFAULTS, LOW_CMD_DEFAULTS or HIGH_CMD_DEFAULTS.
# IF NONE OF THEM ARE WORKING YOU CAN DEFINE A CUSTOM ONE LIKE THIS:
#
# MY_CONNECTION_SETTINGS = (listenPort, addr_wifi, sendPort_high, local_ip_wifi)
# conn = unitreeConnection(MY_CONNECTION_SETTINGS)
d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5,
     'RR_0':6, 'RR_1':7, 'RR_2':8,
     'RL_0':9, 'RL_1':10, 'RL_2':11 }


##   Define constants     ##
PosStopF  = math.pow(10,9)
VelStopF  = 16000.0
LOWLEVEL  = 0xff
# dt 제거됨 - 메인 루프는 ctrldt(50Hz)로 실행
## == Define constants == ##


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

## PRINT LOG PROCESSING - OLD CODE ##
# # define a function to print log data
# def printLog(conn):
#     data = conn.getData()
#     for paket in data:
#         print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
#         lstate.parseData(paket)
#         print(f'SN [{byte_print(lstate.SN)}]:\t{decode_sn(lstate.SN)}')
#         print(f'Ver [{byte_print(lstate.version)}]:\t{decode_version(lstate.version)}')
#         print(f'SOC:\t\t\t{lstate.bms.SOC} %')
#         print(f'Overall Voltage:\t{getVoltage(lstate.bms.cell_vol)} mv') #something is still wrong here ?!
#         print(f'Current:\t\t{lstate.bms.current} mA')
#         print(f'Cycles:\t\t\t{lstate.bms.cycle}')
#         print(f'Temps BQ:\t\t{lstate.bms.BQ_NTC[0]} °C, {lstate.bms.BQ_NTC[1]}°C')
#         print(f'Temps MCU:\t\t{lstate.bms.MCU_NTC[0]} °C, {lstate.bms.MCU_NTC[1]}°C')
#         print(f'FootForce:\t\t{lstate.footForce}')
#         print(f'FootForceEst:\t\t{lstate.footForceEst}')
#         print(f'IMU Temp:\t\t{lstate.imu.temperature}')
#         print(f'MotorState FR_0 MODE:\t\t{lstate.motorState[d["FR_0"]].mode}')
#         print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')

# define a function to print log data
def printLog(conn):
    data = conn.getData()
    try:
        paket = data[-1]  # Try to process only the latest packet
        print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        lstate.parseData(paket)
        print(f'SN [{byte_print(lstate.SN)}]:\t{decode_sn(lstate.SN)}')
        print(f'Ver [{byte_print(lstate.version)}]:\t{decode_version(lstate.version)}')
        print(f'SOC:\t\t\t{lstate.bms.SOC} %')
        print(f'Overall Voltage:\t{getVoltage(lstate.bms.cell_vol)} mv')  # something is still wrong here ?!
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

# return the initial data 
printLog(conn)


## Main control loop ##
motiontime = 0 # <-- control timing counter
print('control Freq : {} Hz'.format(1/ctrldt))


## Robot Control Parameters
# control HZ : 50Hz
ctrldt = 1/50 # Ctrl loop HZ : 50Hz

# joints for control 
joint = ['FR_0', 'FR_1', 'FR_2','FL_0', 'FL_1', 'FL_2','RR_0', 'RR_1', 'RR_2','RL_0', 'RL_1', 'RL_2']
jointnum = len(joint)
print("num of controlled joints :", jointnum)

## Joint Configuration for Sine Wave Motion
# 각 joint마다 다른 주파수와 위상 지연 설정
joint_config = {
    'FR_0': {'freq_hz': 0.0, 'phase_delay': 0.0},    # Hip joint - 정지
    'FR_1': {'freq_hz': 1, 'phase_delay': 0.0},     # Thigh joint - 기본 주파수
    'FR_2': {'freq_hz': 1, 'phase_delay': 0.0},     # Calf joint - 위상 지연

    'FL_0': {'freq_hz': 0.0, 'phase_delay': 0.0},     # Hip joint - 정지
    'FL_1': {'freq_hz': 1, 'phase_delay': 0.25},     # Thigh joint - 기본 주파수
    'FL_2': {'freq_hz': 1, 'phase_delay': 0.25},     # Calf joint - 위상 지연

    'RR_0': {'freq_hz': 0.0, 'phase_delay': 0.0},     # Hip joint - 정지
    'RR_1': {'freq_hz': 1, 'phase_delay': 0.5},     # Thigh joint - 기본 주파수
    'RR_2': {'freq_hz': 1, 'phase_delay': 0.5},     # Calf joint - 위상 지연

    'RL_0': {'freq_hz': 0.0, 'phase_delay': 0.0},     # Hip joint - 정지
    'RL_1': {'freq_hz': 1, 'phase_delay': .75},     # Thigh joint - 기본 주파수
    'RL_2': {'freq_hz': 1, 'phase_delay': .75},     # Calf joint - 위상 지연
}

# Config를 리스트로 변환 (joint 순서에 맞게)
freq_hz_list = [joint_config[joint[i]]['freq_hz'] for i in range(jointnum)]
phase_delay_list = [joint_config[joint[i]]['phase_delay'] for i in range(jointnum)]

print("Joint Frequency Config (Hz):", freq_hz_list)
print("Joint Phase Delay Config:", phase_delay_list)

# p / d gains
# [FR_0, FR_1, FR_2]
kp = [15] * jointnum # [8,8,...,8]
kd = [4] * jointnum # [1,1,...,1]
## This one Should be Modified when changed to 4legs ##
tau = [-0.65,2,2,-0.65,2,2,-0.65,2,2,-0.65,2,2] # [FR_0, FR_1, FR_2, FL_0, FL_1, FL_2, RR_0, RR_1, RR_2, RL_0, RL_1, RL_2]

## Set Min/Max value of joints
# sine wave amplitude, median value comes from here
motionscale = [0.6,0.8]

minmaxjoint = [ [0  ,0   ], #XX_0_min,XX_0_max
                [-10 * motionscale[0],80 * motionscale[0]  ], #XX_1_min,XX_1_max
                [-110 * motionscale[1],-60 * motionscale[1]]] #XX_2_min,XX_2_max
diffjoint = [0] * jointnum
medjoint = [0] * jointnum

for i in range(jointnum):
    # Use modulo to cycle through minmaxjoint indices (0,1,2) for 6 joints
    joint_idx = i % 3  # 0,1,2,0,1,2 for 6 joints
    diffjoint[i] = minmaxjoint[joint_idx][1] - minmaxjoint[joint_idx][0]
    medjoint[i] = (minmaxjoint[joint_idx][1] + minmaxjoint[joint_idx][0]) / 2
deg_sineamplitude = [x/2 for x in diffjoint]
deg_sinemedian = medjoint
print("Sine Amplitude (deg) :", deg_sineamplitude)
print("Sine Median (deg)    :", deg_sinemedian)

##   Inintialize variables container     ##
qInit = [0] * jointnum
qDes = [0] * jointnum

# Data logging for tracking performance
target_positions = []  # Store target positions
actual_positions = []  # Store actual motor positions
timestamps = []        # Store dt-based timestamps
real_times = []        # Store real time.time() timestamps
timing_drifts = []     # Store timing drift in ms
actual_loop_times = [] # Store actual loop times in ms
start_time = None      # Store start time for real time calculation
## == Inintialize variables container == ##

## functionGenerator for sine wave motion
def functionGenerator(t,freq_hz, deg_amplitude, deg_median,range_phase_delay=0.0):
    # 2pi = 360deg = 1 loop
    # Thus, for 5Hz -> 5 loop in 1 sec -> (2pi*5) rad in 1 sec
    freq_rad = (2* math.pi)/freq_hz #sine frequency in rad/s

    # range_phase_delay : phase delay in range [0(No delay), 1(Full Phase delay)]
    if range_phase_delay < 0.0 or range_phase_delay > 1.0:
        raise ValueError("range_phase_delay must be in range [0, 1]")

    phase_delay = range_phase_delay * freq_rad * 2 * math.pi

    rad_amplitude, rad_median = math.radians(deg_amplitude), math.radians(deg_median)
    rad = rad_amplitude * math.sin(t*freq_rad + phase_delay) + rad_median
    return rad


## Initial Position of Sine Wave Motion
# Calculate initial positions using functionGenerator for all joints
rad_init_pos = []
for i in range(jointnum):
    if i % 3 == 0:  # XX_0 joints (hip)
        rad_init_pos.append(0)  # Hip joints stay at 0
    else:  # XX_1, XX_2 joints (thigh, calf)
        rad_init_pos.append(functionGenerator(0, freq_hz_list[i], deg_sineamplitude[i], deg_sinemedian[i], phase_delay_list[i]))





## Send motion to robot
def motionArr(mCmdArr, joint, pos, vel, kp, kd, tau):
    """
    joint | pos : q | vel : dq | kp : Kp | kd : Kd | tau : ff torque
    each parameter is list for each joint [joint1, joint2, joint3] 
        -> in this code : [FR_0, FR_1, FR_2]

    Returns:
        mCmdArr: array of motorCmd to be set in lowCmd
    """
    for i in range(len(joint)):
        mCmdArr.setMotorCmd(joint[i],  motorCmd(mode=MotorModeLow.Servo, q=pos[i], dq = vel[i], Kp = kp[i], Kd = kd[i], tau = tau[i]))
    return mCmdArr



## Main Control Loop ##
try:
    loop_start_time = time.time()
    prev_loop_time = loop_start_time
    
    while True:
        motiontime += 1
        
        # Get Data from robot at every cycle
        data = conn.getData()
        
        # # Process All data at once
        # for paket in data:
        #     lstate.parseData(paket)
        
        # safely Process Last Packet ONLY!
        try:
            lastpacket = data[-1]
            lstate.parseData(lastpacket)
        except Exception as e:
            print(f"Error processing last packet: {e}")
            continue

        # Do loop
        if( motiontime >= 0):

            ## First Phase 
            # Move to the origin point of a sine movement with Kp Kd
            if( motiontime >= 0 and motiontime < 10):
                
                # first, get record initial position of robot
                for i in range(len(joint)):
                    qInit[i] = lstate.motorState[d[joint[i]]].q

            # second, move to the origin point of a sine movement with Kp Kd
            if( motiontime >= 10 and motiontime < 400):
                rate = (motiontime - 10) / (400-10) # Slowly increase "`rate` 0 -> 1" over "`motiontime` 10 -> 400" for smooth transition to INITIAL POSE
                print(rate)
                for joint_idx in range(len(joint)):
                    qDes[joint_idx] = jointLinearInterpolation(qInit[joint_idx], rad_init_pos[joint_idx], rate)


            ## Third Phase
            # Real Robot Motion

            # Do sine wave motion at every control loop time ctrldt
            if( motiontime >= 400):
                # get current Time from motiontime, now t = 0 at motiontime = 400
                t = ctrldt*(motiontime - 400) # time variable for sine function control (ctrldt 기반)
                
                ## Should be Replaced by Function Generator!!!
                # sin_joint1 = 0.6 * math.sin(t*freq_rad)
                # sin_joint2 = -0.9 * math.sin(t*freq_rad/2)

                # Calculate Joint Positions by Function Generator for all joints
                for i in range(jointnum):
                    if i % 3 == 0:  # XX_0 joints (hip) - keep at 0
                        qDes[i] = 0
                    else:  # XX_1, XX_2 joints (thigh, calf) - sine wave motion
                        qDes[i] = functionGenerator(t, freq_hz_list[i], deg_sineamplitude[i], deg_sinemedian[i], phase_delay_list[i])
                
                ## Verbose Target Joint Position Print
                print('')
                print('>>> Target Joint Positions (deg)')
                for deschan in qDes:
                    print(round(math.degrees(deschan),2))
                print('>>> Current Time : {:.2f} sec'.format(t))
                print('--------------------------------')
                # ====================================
                
            ## Build and send command
            # orgnize motorCmdArray
            """커맨드 배열 생성 함수!!"""
            lcmd.motorCmd = motionArr(mCmdArr, joint, qDes, [0]*jointnum, kp, kd, tau)
            # build command bytes
            cmd_bytes = lcmd.buildCmd(debug=False)
            # send command bytes to robot
            conn.send(cmd_bytes)
            
            # Data logging for tracking performance
            if motiontime >= 400:  # Start logging after initial positioning
                # Initialize start time on first logging
                if start_time is None:
                    start_time = time.time()
                
                # Store target positions (in degrees)
                target_positions.append([math.degrees(qDes[i]) for i in range(jointnum)])
                
                # Store actual motor positions (in degrees)
                actual_positions.append([math.degrees(lstate.motorState[d[joint[i]]].q) for i in range(jointnum)])
                
                # Store both timestamp types
                timestamps.append(motiontime * ctrldt)  # ctrldt-based time
                real_times.append(time.time() - start_time)  # real time elapsed
            
            # -- End of Robot Control -- #
        
        # [루프 끝에서] 타이밍 계산
        current_time = time.time()
        elapsed = current_time - loop_start_time
        sleep_time = max(0, ctrldt - elapsed)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Drift 계산 및 로깅
        actual_loop_time = time.time() - prev_loop_time
        drift = loop_start_time + ctrldt - time.time()
        
        # Drift 데이터 수집 (모든 루프에서)
        timing_drifts.append(drift * 1000)  # ms 단위
        actual_loop_times.append(actual_loop_time * 1000)  # ms 단위
        
        prev_loop_time = time.time()
        loop_start_time += ctrldt  # 누적 오차 방지

except KeyboardInterrupt:
    print("\n프로그램이 중단되었습니다. 추종 성능을 분석합니다...")
    
    if len(target_positions) > 0 and len(actual_positions) > 0:
        # Convert to numpy arrays for easier handling
        target_array = np.array(target_positions)
        actual_array = np.array(actual_positions)
        time_array = np.array(timestamps)
        real_time_array = np.array(real_times)
        
        # Save data to CSV first
        data_dict = {
            'timestamp': time_array,  # dt-based time
            'real_time': real_time_array  # real time
        }
        for i in range(jointnum):
            data_dict[f'{joint[i]}_target'] = target_array[:, i]
            data_dict[f'{joint[i]}_actual'] = actual_array[:, i]
            data_dict[f'{joint[i]}_error'] = actual_array[:, i] - target_array[:, i]
        
        # Add timing data (only for logged data points)
        if len(timing_drifts) > 0 and len(actual_loop_times) > 0:
            # Match timing data length with logged data
            timing_data_length = min(len(timing_drifts), len(target_positions))
            data_dict['timing_drift_ms'] = timing_drifts[-timing_data_length:]
            data_dict['actual_loop_time_ms'] = actual_loop_times[-timing_data_length:]
            data_dict['target_loop_time_ms'] = [ctrldt * 1000] * timing_data_length
        
        # Ensure logs directory exists and generate timestamped filenames
        ensure_logs_directory()
        csv_filename = get_timestamp_filename('joint_tracking_data', 'csv')
        plot_filename = get_timestamp_filename('joint_tracking_analysis', 'png')
        
        csv_path = os.path.join('logs', csv_filename)
        plot_path = os.path.join('logs', plot_filename)
        
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, index=False)
        print(f"추종 데이터가 '{csv_path}'로 저장되었습니다.")
        
        # Then use the modularized plotting function
        plot_joint_tracking_from_csv(csv_path, plot_path)
        
        # Print timing drift statistics
        print_timing_drift_statistics(timing_drifts, actual_loop_times, ctrldt)
            
    else:
        print("저장된 추종 데이터가 없습니다.")

# 사용 예시:
# plot_joint_tracking_from_csv('logs/2024-01-15_14-30-25_joint_tracking_data.csv', 'logs/analysis_plot.png')
# 
# 최신 CSV 파일 자동 분석:
# latest_csv = get_latest_csv_file()
# if latest_csv:
#     plot_joint_tracking_from_csv(latest_csv)
