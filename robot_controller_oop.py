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

def printLog(conn, lstate, d):
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


class DataLogger:
    """Handles data logging, analysis, and visualization for robot control"""
    
    def __init__(self):
        self.target_positions = []
        self.actual_positions = []
        self.timestamps = []
        self.real_times = []
        self.timing_drifts = []
        self.actual_loop_times = []
        self.start_time = None
        self.joint_names = ['FR_0', 'FR_1', 'FR_2','FL_0', 'FL_1', 'FL_2','RR_0', 'RR_1', 'RR_2','RL_0', 'RL_1', 'RL_2']
    
    def log_data(self, target_positions, actual_positions, motiontime, ctrldt, timing_drift=None, actual_loop_time=None):
        """Log data for tracking performance analysis"""
        if self.start_time is None:
            self.start_time = time.time()
        
        # Store target and actual positions (in degrees)
        self.target_positions.append(target_positions)
        self.actual_positions.append(actual_positions)
        
        # Store timestamps
        self.timestamps.append(motiontime * ctrldt)
        self.real_times.append(time.time() - self.start_time)
        
        # Store timing data if provided
        if timing_drift is not None:
            self.timing_drifts.append(timing_drift * 1000)  # ms
        if actual_loop_time is not None:
            self.actual_loop_times.append(actual_loop_time * 1000)  # ms
    
    def save_to_csv(self, ctrldt):
        """Save logged data to CSV file"""
        if len(self.target_positions) == 0:
            print("저장된 추종 데이터가 없습니다.")
            return None
        
        # Convert to numpy arrays
        target_array = np.array(self.target_positions)
        actual_array = np.array(self.actual_positions)
        time_array = np.array(self.timestamps)
        real_time_array = np.array(self.real_times)
        
        # Create data dictionary
        data_dict = {
            'timestamp': time_array,
            'real_time': real_time_array
        }
        
        # Add joint data
        for i, joint_name in enumerate(self.joint_names):
            data_dict[f'{joint_name}_target'] = target_array[:, i]
            data_dict[f'{joint_name}_actual'] = actual_array[:, i]
            data_dict[f'{joint_name}_error'] = actual_array[:, i] - target_array[:, i]
        
        # Add timing data
        if len(self.timing_drifts) > 0 and len(self.actual_loop_times) > 0:
            timing_data_length = min(len(self.timing_drifts), len(self.target_positions))
            data_dict['timing_drift_ms'] = self.timing_drifts[-timing_data_length:]
            data_dict['actual_loop_time_ms'] = self.actual_loop_times[-timing_data_length:]
            data_dict['target_loop_time_ms'] = [ctrldt * 1000] * timing_data_length
        
        # Save to CSV
        ensure_logs_directory()
        csv_filename = get_timestamp_filename('joint_tracking_data', 'csv')
        csv_path = os.path.join('logs', csv_filename)
        
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, index=False)
        print(f"추종 데이터가 '{csv_path}'로 저장되었습니다.")
        
        return csv_path
    
    def print_statistics(self, ctrldt):
        """Print timing drift statistics"""
        print_timing_drift_statistics(self.timing_drifts, self.actual_loop_times, ctrldt)
    
    def plot_tracking(self, csv_path):
        """Generate tracking performance plots"""
        plot_filename = get_timestamp_filename('joint_tracking_analysis', 'png')
        plot_path = os.path.join('logs', plot_filename)
        plot_joint_tracking_from_csv(csv_path, plot_path)


class MotionGenerator:
    """Generates motion trajectories and handles interpolation"""
    
    def __init__(self, joint_config=None, motion_scale=None):
        # Default joint configuration
        self.joint_config = joint_config or {
            'FR_0': {'freq_hz': 0.0, 'phase_delay': 0.0},
            'FR_1': {'freq_hz': 1, 'phase_delay': 0.0},
            'FR_2': {'freq_hz': 1, 'phase_delay': 0.0},
            'FL_0': {'freq_hz': 0.0, 'phase_delay': 0.0},
            'FL_1': {'freq_hz': 1, 'phase_delay': 0.25},
            'FL_2': {'freq_hz': 1, 'phase_delay': 0.25},
            'RR_0': {'freq_hz': 0.0, 'phase_delay': 0.0},
            'RR_1': {'freq_hz': 1, 'phase_delay': 0.5},
            'RR_2': {'freq_hz': 1, 'phase_delay': 0.5},
            'RL_0': {'freq_hz': 0.0, 'phase_delay': 0.0},
            'RL_1': {'freq_hz': 1, 'phase_delay': .75},
            'RL_2': {'freq_hz': 1, 'phase_delay': .75},
        }
        
        self.joint_names = ['FR_0', 'FR_1', 'FR_2','FL_0', 'FL_1', 'FL_2','RR_0', 'RR_1', 'RR_2','RL_0', 'RL_1', 'RL_2']
        self.jointnum = len(self.joint_names)
        
        # Motion scale parameters
        self.motion_scale = motion_scale or [0.6, 0.8]
        
        # Calculate motion parameters
        self._calculate_motion_parameters()
    
    def _calculate_motion_parameters(self):
        """Calculate motion parameters from configuration"""
        # Convert config to lists
        self.freq_hz_list = [self.joint_config[joint]['freq_hz'] for joint in self.joint_names]
        self.phase_delay_list = [self.joint_config[joint]['phase_delay'] for joint in self.joint_names]
        
        # Calculate joint limits and amplitudes
        minmaxjoint = [
            [0, 0],  # XX_0_min,XX_0_max
            [-10 * self.motion_scale[0], 80 * self.motion_scale[0]],  # XX_1_min,XX_1_max
            [-110 * self.motion_scale[1], -60 * self.motion_scale[1]]  # XX_2_min,XX_2_max
        ]
        
        diffjoint = [0] * self.jointnum
        medjoint = [0] * self.jointnum
        
        for i in range(self.jointnum):
            joint_idx = i % 3
            diffjoint[i] = minmaxjoint[joint_idx][1] - minmaxjoint[joint_idx][0]
            medjoint[i] = (minmaxjoint[joint_idx][1] + minmaxjoint[joint_idx][0]) / 2
        
        self.deg_sineamplitude = [x/2 for x in diffjoint]
        self.deg_sinemedian = medjoint
        
        # Calculate initial positions
        self.rad_init_pos = []
        for i in range(self.jointnum):
            if i % 3 == 0:  # XX_0 joints (hip)
                self.rad_init_pos.append(0)
            else:  # XX_1, XX_2 joints (thigh, calf)
                self.rad_init_pos.append(self.generate_sine_wave(
                    0, self.freq_hz_list[i], self.deg_sineamplitude[i], 
                    self.deg_sinemedian[i], self.phase_delay_list[i]
                ))
    
    def linear_interpolation(self, initPos, targetPos, rate):
        """Linear interpolation between two joint positions"""
        rate = np.fmin(np.fmax(rate, 0.0), 1.0)
        p = initPos*(1-rate) + targetPos*rate
        return p
    
    def generate_sine_wave(self, t, freq_hz, deg_amplitude, deg_median, range_phase_delay=0.0):
        """Generate sine wave motion for a joint"""
        if freq_hz == 0:
            return math.radians(deg_median)
        
        freq_rad = (2 * math.pi) / freq_hz
        
        if range_phase_delay < 0.0 or range_phase_delay > 1.0:
            raise ValueError("range_phase_delay must be in range [0, 1]")
        
        phase_delay = range_phase_delay * freq_rad * 2 * math.pi
        rad_amplitude, rad_median = math.radians(deg_amplitude), math.radians(deg_median)
        rad = rad_amplitude * math.sin(t*freq_rad + phase_delay) + rad_median
        return rad
    
    def calculate_initial_positions(self, qInit):
        """Calculate initial positions using interpolation"""
        qDes = [0] * self.jointnum
        for joint_idx in range(self.jointnum):
            qDes[joint_idx] = self.linear_interpolation(qInit[joint_idx], self.rad_init_pos[joint_idx], 1.0)
        return qDes
    
    def generate_motion(self, t):
        """Generate motion for all joints at time t"""
        qDes = [0] * self.jointnum
        for i in range(self.jointnum):
            if i % 3 == 0:  # XX_0 joints (hip) - keep at 0
                qDes[i] = 0
            else:  # XX_1, XX_2 joints (thigh, calf) - sine wave motion
                qDes[i] = self.generate_sine_wave(
                    t, self.freq_hz_list[i], self.deg_sineamplitude[i], 
                    self.deg_sinemedian[i], self.phase_delay_list[i]
                )
        return qDes


class RobotController:
    """Main robot control and communication"""
    
    def __init__(self, connection_settings=LOW_WIFI_DEFAULTS, motion_generator=None, data_logger=None):
        self.connection_settings = connection_settings
        self.motion_generator = motion_generator or MotionGenerator()
        self.data_logger = data_logger or DataLogger()
        
        # Robot state
        self.conn = None
        self.lcmd = None
        self.lstate = None
        self.mCmdArr = None
        
        # Control parameters
        self.ctrldt = 1/50  # 50Hz control loop
        self.joint_names = self.motion_generator.joint_names
        self.jointnum = self.motion_generator.jointnum
        
        # Joint mapping
        self.d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
                 'FL_0':3, 'FL_1':4, 'FL_2':5,
                 'RR_0':6, 'RR_1':7, 'RR_2':8,
                 'RL_0':9, 'RL_1':10, 'RL_2':11}
        
        # Control gains
        self.kp = [15] * self.jointnum
        self.kd = [4] * self.jointnum
        self.tau = [-0.65,2,2,-0.65,2,2,-0.65,2,2,-0.65,2,2]
        
        # Control state
        self.motiontime = 0
        self.qInit = [0] * self.jointnum
        self.qDes = [0] * self.jointnum
        
        # Timing
        self.loop_start_time = None
        self.prev_loop_time = None
    
    def initialize(self):
        """Initialize robot connection and state"""
        print(f'Running lib version: {lib_version()}')
        self.conn = unitreeConnection(self.connection_settings)
        self.conn.startRecv()
        
        # Initialize command and state objects
        self.lcmd = lowCmd()
        self.lstate = lowState()
        self.mCmdArr = motorCmdArray()
        
        # Send initial command
        cmd_bytes = self.lcmd.buildCmd(debug=False)
        self.conn.send(cmd_bytes)
        
        # Print initial state
        printLog(self.conn, self.lstate, self.d)
        
        print('control Freq : {} Hz'.format(1/self.ctrldt))
        print("num of controlled joints :", self.jointnum)
        print("Joint Frequency Config (Hz):", self.motion_generator.freq_hz_list)
        print("Joint Phase Delay Config:", self.motion_generator.phase_delay_list)
        print("Sine Amplitude (deg) :", self.motion_generator.deg_sineamplitude)
        print("Sine Median (deg)    :", self.motion_generator.deg_sinemedian)
    
    def _process_state(self):
        """Process robot state data"""
        data = self.conn.getData()
        try:
            lastpacket = data[-1]
            self.lstate.parseData(lastpacket)
        except Exception as e:
            print(f"Error processing last packet: {e}")
            return False
        return True
    
    def _send_command(self):
        """Send command to robot"""
        # Build motor command array
        self.lcmd.motorCmd = self._motionArr(self.mCmdArr, self.joint_names, self.qDes, 
                                           [0]*self.jointnum, self.kp, self.kd, self.tau)
        # Build and send command
        cmd_bytes = self.lcmd.buildCmd(debug=False)
        self.conn.send(cmd_bytes)
    
    def _motionArr(self, mCmdArr, joint, pos, vel, kp, kd, tau):
        """Create motor command array"""
        for i in range(len(joint)):
            mCmdArr.setMotorCmd(joint[i], motorCmd(mode=MotorModeLow.Servo, q=pos[i], 
                                                 dq=vel[i], Kp=kp[i], Kd=kd[i], tau=tau[i]))
        return mCmdArr
    
    def _control_loop(self):
        """Main control loop logic"""
        self.motiontime += 1
        
        if self.motiontime >= 0:
            # Phase 1: Record initial position
            if 0 <= self.motiontime < 10:
                for i in range(len(self.joint_names)):
                    self.qInit[i] = self.lstate.motorState[self.d[self.joint_names[i]]].q
            
            # Phase 2: Move to initial pose
            elif 10 <= self.motiontime < 400:
                rate = (self.motiontime - 10) / (400 - 10)
                print(rate)
                for joint_idx in range(len(self.joint_names)):
                    self.qDes[joint_idx] = self.motion_generator.linear_interpolation(
                        self.qInit[joint_idx], self.motion_generator.rad_init_pos[joint_idx], rate
                    )
            
            # Phase 3: Sine wave motion
            elif self.motiontime >= 400:
                t = self.ctrdt * (self.motiontime - 400)
                self.qDes = self.motion_generator.generate_motion(t)
                
                # Print target positions
                print('')
                print('>>> Target Joint Positions (deg)')
                for deschan in self.qDes:
                    print(round(math.degrees(deschan), 2))
                print('>>> Current Time : {:.2f} sec'.format(t))
                print('--------------------------------')
        
        # Send command
        self._send_command()
        
        # Log data if in motion phase
        if self.motiontime >= 400:
            target_positions = [math.degrees(q) for q in self.qDes]
            actual_positions = [math.degrees(self.lstate.motorState[self.d[joint]].q) 
                              for joint in self.joint_names]
            self.data_logger.log_data(target_positions, actual_positions, 
                                    self.motiontime, self.ctrdt)
    
    def run(self):
        """Run the main control loop"""
        self.initialize()
        
        try:
            self.loop_start_time = time.time()
            self.prev_loop_time = self.loop_start_time
            
            while True:
                # Process robot state
                if not self._process_state():
                    continue
                
                # Execute control logic
                self._control_loop()
                
                # Timing control
                current_time = time.time()
                elapsed = current_time - self.loop_start_time
                sleep_time = max(0, self.ctrdt - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Calculate timing drift
                actual_loop_time = time.time() - self.prev_loop_time
                drift = self.loop_start_time + self.ctrdt - time.time()
                
                # Log timing data
                self.data_logger.timing_drifts.append(drift * 1000)
                self.data_logger.actual_loop_times.append(actual_loop_time * 1000)
                
                self.prev_loop_time = time.time()
                self.loop_start_time += self.ctrdt
                
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다. 추종 성능을 분석합니다...")
            
            # Save data and generate analysis
            csv_path = self.data_logger.save_to_csv(self.ctrdt)
            if csv_path:
                self.data_logger.plot_tracking(csv_path)
                self.data_logger.print_statistics(self.ctrdt)


if __name__ == "__main__":
    # Create instances
    motion_gen = MotionGenerator()
    data_logger = DataLogger()
    controller = RobotController(
        connection_settings=LOW_WIFI_DEFAULTS,
        motion_generator=motion_gen,
        data_logger=data_logger
    )
    
    # Run controller
    controller.run()
