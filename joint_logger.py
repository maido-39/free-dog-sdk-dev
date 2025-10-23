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

def plot_joint_positions_from_csv(csv_file_path, output_file=None):
    """
    CSV 파일에서 joint position 데이터를 읽어서 플롯 생성
    
    Args:
        csv_file_path (str): CSV 파일 경로
        output_file (str, optional): 출력 이미지 파일 경로. None이면 화면에 표시만 함.
    """
    if not os.path.exists(csv_file_path):
        print(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}")
        return
    
    # CSV 데이터 읽기
    df = pd.read_csv(csv_file_path)
    
    # Joint 이름 추출 (position 컬럼에서)
    position_cols = [col for col in df.columns if col.endswith('_position')]
    joint_names = [col.replace('_position', '') for col in position_cols]
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
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(40, 15))
    fig.suptitle('Joint Position Analysis', fontsize=20)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each joint
    for i in range(jointnum):
        ax = axes_flat[i]
        
        # Get position data
        position_col = f'{joint_names[i]}_position'
        
        if position_col in df.columns:
            position_data = df[position_col].values
            
            # Plot joint positions
            ax.plot(time_array, position_data, 'b-', label='Position', linewidth=2)
            
            # Set labels and title
            ax.set_title(f'{joint_names[i]} Position', fontsize=14)
            ax.set_xlabel(time_label)
            ax.set_ylabel('Position (deg)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Calculate and display statistics
            mean_pos = np.mean(position_data)
            std_pos = np.std(position_data)
            min_pos = np.min(position_data)
            max_pos = np.max(position_data)
            ax.text(0.02, 0.98, f'Mean: {mean_pos:.2f}°\nStd: {std_pos:.2f}°\nRange: [{min_pos:.2f}, {max_pos:.2f}]°', 
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
    print("\n=== 전체 관절 위치 통계 ===")
    for i in range(jointnum):
        position_col = f'{joint_names[i]}_position'
        if position_col in df.columns:
            position_data = df[position_col].values
            mean_pos = np.mean(position_data)
            std_pos = np.std(position_data)
            min_pos = np.min(position_data)
            max_pos = np.max(position_data)
            print(f"{joint_names[i]}: Mean={mean_pos:.2f}°, Std={std_pos:.2f}°, Range=[{min_pos:.2f}, {max_pos:.2f}]°")

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

# Joint mapping dictionary
d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5,
     'RR_0':6, 'RR_1':7, 'RR_2':8,
     'RL_0':9, 'RL_1':10, 'RL_2':11 }

## Define constants
PosStopF  = math.pow(10,9)
VelStopF  = 16000.0
LOWLEVEL  = 0xff

## Initialize connection
print(f'Running lib version: {lib_version()}')
conn = unitreeConnection(LOW_WIFI_DEFAULTS)
conn.startRecv()

## Instantiate lowlevel command and state objects
lcmd = lowCmd()
lstate = lowState()
mCmdArr = motorCmdArray()

## Get initial state & print Log
# Send empty command to tell the dog the receive port and initialize the connection
cmd_bytes = lcmd.buildCmd(debug=False)
conn.send(cmd_bytes)

def printLog(conn):
    """Print robot state information"""
    data = conn.getData()
    try:
        paket = data[-1]  # Try to process only the latest packet
        print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        lstate.parseData(paket)
        print(f'SN [{byte_print(lstate.SN)}]:\t{decode_sn(lstate.SN)}')
        print(f'Ver [{byte_print(lstate.version)}]:\t{decode_version(lstate.version)}')
        print(f'SOC:\t\t\t{lstate.bms.SOC} %')
        print(f'Overall Voltage:\t{getVoltage(lstate.bms.cell_vol)} mv')
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

def get_joint_angles_rad(conn):
    """
    conn에서 최신 패킷을 받아 12개 관절의 각도(q, radian) 리스트를 반환

    Returns:
        q_list (list): 12개 관절의 각도 리스트 (radian, joint 순서는 global 'joint' 리스트와 동일)
                       오류시 None 반환
    """
    data = conn.getData()
    try:
        paket = data[-1]  # 최신 패킷만 처리
        lstate.parseData(paket)
        # 관절 이름이 'joint' 리스트에 있다고 가정
        q_list = [lstate.motorState[d[jname]].q for jname in joint]
        return q_list
    except Exception as e:
        print(f"관절 각도 읽기 오류: {e}")
        return None

# Return the initial data
printLog(conn)

## Logging Parameters
# control HZ : 50Hz
ctrldt = 1/50  # Ctrl loop HZ : 50Hz

## Main logging loop
motiontime = 0  # <-- timing counter
print('Logging Freq : {} Hz'.format(1/ctrldt))

# Joints for logging 
joint = ['FR_0', 'FR_1', 'FR_2','FL_0', 'FL_1', 'FL_2','RR_0', 'RR_1', 'RR_2','RL_0', 'RL_1', 'RL_2']
jointnum = len(joint)
print("num of logged joints :", jointnum)

# Data logging for joint positions
joint_positions = []  # Store actual joint positions
timestamps = []        # Store dt-based timestamps
real_times = []        # Store real time.time() timestamps
timing_drifts = []     # Store timing drift in ms
actual_loop_times = [] # Store actual loop times in ms
start_time = None      # Store start time for real time calculation

# No motion control needed - only logging

## Main Logging Loop
try:
    loop_start_time = time.time()
    prev_loop_time = loop_start_time
    
    print("\n=== Joint Logging Started ===")
    print("Press Ctrl+C to stop logging and generate analysis...")
    
    while True:
        motiontime += 1
        
        # Get Data from robot at every cycle
        data = conn.getData()
        
        # Safely Process Last Packet ONLY!
        try:
            lastpacket = data[-1]
            lstate.parseData(lastpacket)
        except Exception as e:
            print(f"Error processing last packet: {e}")
            continue

        # Data logging using get_joint_angles_rad function
        if motiontime >= 10:  # Start logging after initial stabilization
            # Initialize start time on first logging
            if start_time is None:
                start_time = time.time()
            
            # Get joint angles using the function
            q_list = get_joint_angles_rad(conn)
            
            if q_list is not None:
                # Store joint positions (in degrees)
                joint_positions.append([math.degrees(q) for q in q_list])
                
                # Store both timestamp types
                timestamps.append(motiontime * ctrldt)  # ctrldt-based time
                real_times.append(time.time() - start_time)  # real time elapsed
                
                # Print progress every 50 cycles (1 second at 50Hz)
                if motiontime % 50 == 0:
                    print(f"Logged {len(joint_positions)} data points...")
        
        # Timing control
        current_time = time.time()
        elapsed = current_time - loop_start_time
        sleep_time = max(0, ctrldt - elapsed)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Drift calculation and logging
        actual_loop_time = time.time() - prev_loop_time
        drift = loop_start_time + ctrldt - time.time()
        
        # Drift data collection (for all loops)
        timing_drifts.append(drift * 1000)  # ms unit
        actual_loop_times.append(actual_loop_time * 1000)  # ms unit
        
        prev_loop_time = time.time()
        loop_start_time += ctrldt  # Prevent cumulative error

except KeyboardInterrupt:
    print("\n프로그램이 중단되었습니다. 관절 데이터를 분석합니다...")
    
    if len(joint_positions) > 0:
        # Convert to numpy arrays for easier handling
        joint_array = np.array(joint_positions)
        time_array = np.array(timestamps)
        real_time_array = np.array(real_times)
        
        # Save data to CSV first
        data_dict = {
            'timestamp': time_array,  # dt-based time
            'real_time': real_time_array  # real time
        }
        for i in range(jointnum):
            data_dict[f'{joint[i]}_position'] = joint_array[:, i]
        
        # Add timing data (only for logged data points)
        if len(timing_drifts) > 0 and len(actual_loop_times) > 0:
            # Match timing data length with logged data
            timing_data_length = min(len(timing_drifts), len(joint_positions))
            data_dict['timing_drift_ms'] = timing_drifts[-timing_data_length:]
            data_dict['actual_loop_time_ms'] = actual_loop_times[-timing_data_length:]
            data_dict['target_loop_time_ms'] = [ctrldt * 1000] * timing_data_length
        
        # Ensure logs directory exists and generate timestamped filenames
        ensure_logs_directory()
        csv_filename = get_timestamp_filename('joint_position_data', 'csv')
        plot_filename = get_timestamp_filename('joint_position_analysis', 'png')
        
        csv_path = os.path.join('logs', csv_filename)
        plot_path = os.path.join('logs', plot_filename)
        
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, index=False)
        print(f"관절 데이터가 '{csv_path}'로 저장되었습니다.")
        
        # Generate simple position plots
        plot_joint_positions_from_csv(csv_path, plot_path)
        
        # Print timing drift statistics
        print_timing_drift_statistics(timing_drifts, actual_loop_times, ctrldt)
            
    else:
        print("저장된 관절 데이터가 없습니다.")

def analyze_latest_csv():
    """최신 CSV 파일을 자동으로 찾아서 분석하는 함수"""
    ensure_logs_directory()
    latest_csv = get_latest_csv_file()
    if latest_csv:
        print(f"Analyzing latest CSV file: {latest_csv}")
        plot_joint_positions_from_csv(latest_csv)
        return latest_csv
    else:
        print("No CSV files found in logs directory.")
        print("Run joint logging first to generate some data.")
        return None

def analyze_csv_file(csv_file_path):
    """지정된 CSV 파일을 분석하는 함수"""
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        return False
    
    print(f"Analyzing CSV file: {csv_file_path}")
    plot_joint_positions_from_csv(csv_file_path)
    return True

# 사용 예시:
# analyze_latest_csv()  # 최신 CSV 파일 자동 분석
# analyze_csv_file('logs/2024-01-15_14-30-25_joint_position_data.csv')  # 특정 파일 분석
