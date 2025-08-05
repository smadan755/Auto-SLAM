"""
Main coordination file for the drone visual odometry system with GPS plotting.
This file orchestrates all the modular components including GPS visualization.
"""

import airsim
import asyncio
import threading
import queue
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

from config import AIRSIM_IP
from input_handler import pygame_thread_func
from gui_display_3d import gui_thread_func
from drone_controller import run_drone_control
from visual_odometry_processor_improved import run_video_and_vo_improved as run_video_and_vo
from utils import save_path_to_csv

GUI_QUEUE_SIZE = 1

class IntegratedGPSPlotter:
    """GPS data monitor integrated with Visual Odometry."""
    
    def __init__(self):
        self.gps_data = deque(maxlen=2000)
        self.pose_data = deque(maxlen=2000)
        self.vo_data = deque(maxlen=1000)
        self.running = False
        self.client = None
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle('AirSim GPS & Visual Odometry Data Monitor')
        
        print("🗺️  GPS+VO Data Monitor initialized")
    
    def set_client(self, client):
        """Set the AirSim client safely."""
        self.client = client
    
    def collect_data_loop(self):
        """Data collection loop (runs in background thread)."""
        print("📡 Starting high-frequency GPS+Pose data collection...")
        
        while self.running and self.client:
            try:
                current_time = time.time()
                
                # Create a new client instance to avoid IOLoop conflicts with asyncio
                import airsim
                temp_client = airsim.MultirotorClient(ip=AIRSIM_IP)
                temp_client.confirmConnection()
                
                gps_data = temp_client.getGpsData()
                self.gps_data.append({
                    'lat': gps_data.gnss.geo_point.latitude,
                    'lon': gps_data.gnss.geo_point.longitude,
                    'alt': gps_data.gnss.geo_point.altitude,
                    'timestamp': current_time
                })
                
                pose = temp_client.simGetVehiclePose()
                self.pose_data.append({
                    'x': pose.position.x_val,
                    'y': pose.position.y_val,
                    'z': pose.position.z_val,
                    'timestamp': current_time
                })
                
                if len(self.gps_data) % 100 == 0:
                    print(f"📊 High-Freq GPS Logger: GPS={len(self.gps_data)}, Pose={len(self.pose_data)}, VO={len(self.vo_data)} samples")
            
            except Exception as e:
                print(f"❌ GPS+Pose collection error: {e}")
            
            time.sleep(0.02)  # 50Hz collection rate
    
    def add_vo_data(self, vo_position):
        """Add Visual Odometry trajectory data from the main VO thread."""
        try:
            # Handle different VO data formats safely
            if vo_position is not None:
                if hasattr(vo_position, '__len__') and len(vo_position) >= 3:
                    pos_list = list(vo_position) if not isinstance(vo_position, list) else vo_position
                    self.vo_data.append({
                        'x': float(pos_list[0]),
                        'y': float(pos_list[1]),
                        'z': float(pos_list[2]),
                        'timestamp': time.time()
                    })
                elif hasattr(vo_position, 'x') and hasattr(vo_position, 'y'):
                    self.vo_data.append({
                        'x': float(vo_position.x),
                        'y': float(vo_position.y),
                        'z': float(getattr(vo_position, 'z', 0.0)),
                        'timestamp': time.time()
                    })
        except Exception:
            # Silently ignore VO data errors to prevent spam
            pass
    
    def animate(self, frame):
        """Animation function for matplotlib (runs in main thread)."""
        if len(self.gps_data) < 2:
            return
        
        try:
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.clear()
            
            gps_list = list(self.gps_data)
            pose_list = list(self.pose_data)
            vo_list = list(self.vo_data)
            
            if not gps_list or not pose_list:
                return
            
            # GPS Status and Coordinates Panel
            current_gps = gps_list[-1]
            self.ax1.axis('off')
            self.ax1.set_title('📡 GPS Status & Coordinates', fontsize=14, pad=20)
            
            gps_info = f"""🌍 Current GPS Position:
Latitude:  {current_gps['lat']:.9f}°
Longitude: {current_gps['lon']:.9f}°
Altitude:  {current_gps['alt']:.3f} m

📊 Data Collection:
GPS Samples: {len(gps_list)}
Pose Samples: {len(pose_list)}
VO Points: {len(vo_list)}

⏱️ Session Stats:
Duration: {(gps_list[-1]['timestamp'] - gps_list[0]['timestamp']):.1f}s
GPS Rate: {len(gps_list)/(gps_list[-1]['timestamp'] - gps_list[0]['timestamp']):.1f} Hz
Status: {'🟢 High-Freq Tracking' if len(gps_list) > 10 else '🟡 Starting...'}

🔄 Recent Movement:
Lat Change: {(gps_list[-1]['lat'] - gps_list[-min(10, len(gps_list))]['lat'])*1000000:.1f} µ°
Lon Change: {(gps_list[-1]['lon'] - gps_list[-min(10, len(gps_list))]['lon'])*1000000:.1f} µ°
Alt Change: {(gps_list[-1]['alt'] - gps_list[-min(10, len(gps_list))]['alt']):.3f} m"""
            
            self.ax1.text(0.05, 0.95, gps_info, transform=self.ax1.transAxes,
                          fontsize=12, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Altitude and Time Series Panel
            if len(gps_list) > 1:
                start_time = gps_list[0]['timestamp']
                gps_times = [(g['timestamp'] - start_time) for g in gps_list]
                pose_times = [(p['timestamp'] - start_time) for p in pose_list]
                
                gps_alts = [g['alt'] for g in gps_list]
                pose_alts = [-p['z'] for p in pose_list]  # Invert Z for altitude
                
                self.ax2.plot(gps_times, gps_alts, 'b-', linewidth=3, label='GPS Altitude', alpha=0.8)
                self.ax2.plot(pose_times, pose_alts, 'g-', linewidth=2, label='AirSim Z (inverted)', alpha=0.7)
                
                if vo_list:
                    vo_times = [(s['timestamp'] - start_time) for s in vo_list]
                    vo_alts = [-s['z'] for s in vo_list]
                    self.ax2.plot(vo_times, vo_alts, 'r--', linewidth=2, label='VO Z (inverted)', alpha=0.9)
                
                self.ax2.set_title('📈 Altitude & Height Data', fontsize=14)
                self.ax2.set_xlabel('Time (seconds)')
                self.ax2.set_ylabel('Altitude (meters)')
                self.ax2.grid(True, alpha=0.3)
                self.ax2.legend()
                
                current_alt_text = f"Current: GPS={current_gps['alt']:.3f}m | Last 10s Δ: {(gps_alts[-1] - gps_alts[-min(50, len(gps_alts))]):.3f}m"
                self.ax2.text(0.02, 0.98, current_alt_text, transform=self.ax2.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # VO Integration Panel
            self.ax3.axis('off')
            self.ax3.set_title('🗺️ Visual Odometry Integration', fontsize=14, pad=20)
            
            pose_xs = [p['x'] for p in pose_list]
            pose_ys = [p['y'] for p in pose_list]
            
            if pose_xs and pose_ys:
                travel_distance = sum([
                    ((pose_xs[i] - pose_xs[i-1])**2 + (pose_ys[i] - pose_ys[i-1])**2)**0.5
                    for i in range(1, len(pose_xs))
                ])
                
                pose_range_x = max(pose_xs) - min(pose_xs)
                pose_range_y = max(pose_ys) - min(pose_ys)
                current_pose = pose_list[-1]
                
                slam_info = f"""🎯 AirSim Pose Data:
Current Position:
  X: {current_pose['x']:.2f} m
  Y: {current_pose['y']:.2f} m 
  Z: {current_pose['z']:.2f} m

Movement Analysis:
  Travel Distance: {travel_distance:.2f} m
  Coverage Area: {pose_range_x:.1f} × {pose_range_y:.1f} m

🗺️ Visual Odometry Status:
  Active Points: {len(vo_list)}
  Status: {'🟢 VO Active' if vo_list else '🟡 Initializing'}
  Integration: {'✅ GPS+VO Linked' if vo_list and gps_list else '⏳ Syncing...'}

🔄 System Status:
  GPS-Pose Sync: {'✅ Synchronized' if len(gps_list) > 0 and len(pose_list) > 0 else '❌ No Data'}
  Real-time Update: {'Active' if self.running else '🔴 Stopped'}"""
                
                self.ax3.text(0.05, 0.95, slam_info, transform=self.ax3.transAxes,
                              fontsize=11, verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def start(self):
        """Start GPS plotting with animation."""
        self.running = True
        
        self.data_thread = threading.Thread(target=self.collect_data_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        print("GPS+VO data monitoring started with high-frequency tracking.")
        
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, interval=500, blit=False, cache_frame_data=False # Update every 500ms
        )
        return self.anim
    
    def stop(self):
        """Stop GPS plotting and save a summary."""
        self.running = False
        if hasattr(self, 'data_thread'):
            self.data_thread.join(timeout=2.0)
        
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()
        
        try:
            if len(self.gps_data) > 0:
                plt.figure(figsize=(12, 4))
                
                gps_list = list(self.gps_data)
                pose_list = list(self.pose_data)
                vo_list = list(self.vo_data)
                
                if gps_list:
                    start_time = gps_list[0]['timestamp']
                    gps_times = [(g['timestamp'] - start_time) for g in gps_list]
                    gps_alts = [g['alt'] for g in gps_list]
                    plt.plot(gps_times, gps_alts, 'b-', linewidth=3, label='GPS Altitude')
                    
                    if pose_list:
                        pose_times = [(p['timestamp'] - start_time) for p in pose_list]
                        pose_alts = [-p['z'] for p in pose_list]
                        plt.plot(pose_times, pose_alts, 'g-', linewidth=2, label='AirSim Z')
                    
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Altitude (meters)')
                    plt.title(f'GPS+VO Session Summary - {len(gps_list)} GPS samples, {len(vo_list)} VO points')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.savefig('airsim_integrated_gps_vo_plot.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    
                print(f"Final GPS+VO summary saved as 'airsim_integrated_gps_vo_plot.png'")
                print(f"Session: GPS={len(self.gps_data)}, Pose={len(self.pose_data)}, VO={len(self.vo_data)} samples")
        except Exception as e:
            print(f" Could not save final plot: {e}")

gps_plotter = None
est_path_data = []
stop_event = threading.Event()

def update_gps_with_vo_data():
    """Periodically update GPS plotter with VO data."""
    global gps_plotter, est_path_data, stop_event
    
    last_vo_count = 0
    
    while not stop_event.is_set():
        try:
            if gps_plotter and est_path_data:
                if len(est_path_data) > last_vo_count:
                    for i in range(last_vo_count, len(est_path_data)):
                        try:
                            vo_point = est_path_data[i]
                            gps_plotter.add_vo_data(vo_point)
                        except Exception:
                            continue
                    last_vo_count = len(est_path_data)
        except Exception as e:
            # Reduce error spam by only printing occasionally
            if not hasattr(update_gps_with_vo_data, 'error_count'):
                update_gps_with_vo_data.error_count = 0
            update_gps_with_vo_data.error_count += 1
            if update_gps_with_vo_data.error_count % 100 == 1:
                print(f"VO-GPS sync error: {e}")
        
        time.sleep(0.1)  # 10Hz update rate

async def main():
    """Main coordination function with integrated GPS+VO plotting."""
    global gps_plotter
    
    print("Enhanced AirSim Visual Odometry with Integrated GPS Plotting")
    print("="*60)
    
    velocity_cmd = [0.0, 0.0, 0.0, 0.0] 
    gui_queue = queue.Queue(maxsize=GUI_QUEUE_SIZE)
    
    try:
        airsim_client = airsim.MultirotorClient(ip=AIRSIM_IP)
        airsim_client.confirmConnection()
        airsim_client.enableApiControl(True)
        print(f"Connected to AirSim at {AIRSIM_IP}")
    except Exception as e:
        print(f"Failed to connect to AirSim: {e}")
        return

    try:
        gps_data = airsim_client.getGpsData()
        print(f"GPS available: {gps_data.gnss.geo_point.latitude:.6f}, {gps_data.gnss.geo_point.longitude:.6f}")
        
        gps_plotter = IntegratedGPSPlotter()
        gps_plotter.set_client(airsim_client)
        gps_plotter.start()
        
    except Exception as e:
        print(f" GPS error: {e}. Continuing without GPS plotting.")
        gps_plotter = None

    pygame_t = threading.Thread(target=pygame_thread_func, args=(velocity_cmd, stop_event))
    gui_t = threading.Thread(target=gui_thread_func, args=(gui_queue, stop_event))
    pygame_t.daemon = True
    gui_t.daemon = True
    pygame_t.start()
    gui_t.start()

    vo_sync_thread = None
    if gps_plotter:
        vo_sync_thread = threading.Thread(target=update_gps_with_vo_data)
        vo_sync_thread.daemon = True
        vo_sync_thread.start()

    try:
        print("🚀 Starting integrated VO+GPS system...")
        print("\n Controls: Use pygame window to control drone. Press 'L' to land and quit.")
        print(" Monitor high-frequency GPS+VO data in the plot window.")
        print(" View 3D trajectories in the separate OpenCV window.")
        
        slam_task = run_video_and_vo(airsim_client, gui_queue, est_path_data, stop_event)
        control_task = run_drone_control(velocity_cmd, stop_event)
        
        await asyncio.gather(slam_task, control_task)
        
    except Exception as e:
        print(f"Error in integrated tasks: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(" Stopping integrated system...")
        stop_event.set()
        
        if gps_plotter:
            gps_plotter.stop()
        
        if est_path_data:
            save_path_to_csv(est_path_data, "vo_path_with_gps_integration.csv")
            print("Integrated trajectory data saved.")
        else:
            print("No trajectory data to save.")
        
        pygame_t.join()
        gui_t.join()
        if vo_sync_thread:
            vo_sync_thread.join(timeout=2.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        print("Enhanced program exited.")