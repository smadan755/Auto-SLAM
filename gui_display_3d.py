import cv2
import numpy as np
import queue
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import io

# Try relative imports first (when run as module), fall back to absolute imports
try:
    from .config import FIGURE_SIZE, GUI_TIMEOUT, ENABLE_3D_VISUALIZATION, ENABLE_3D_MAPPING
except ImportError:
    # Fallback for direct execution
    from config import FIGURE_SIZE, GUI_TIMEOUT, ENABLE_3D_VISUALIZATION, ENABLE_3D_MAPPING


class Enhanced3DVisualizer:
    """Enhanced 3D visualization for SLAM data."""
    
    def __init__(self):
        self.fig_3d = None
        self.ax_3d = None
        self.setup_3d_plot()
        
    def setup_3d_plot(self):
        """Initialize 3D matplotlib plot."""
        if ENABLE_3D_VISUALIZATION:
            self.fig_3d = plt.figure(figsize=(10, 8))
            self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
            self.ax_3d.set_xlabel('X (m)')
            self.ax_3d.set_ylabel('Y (m)')
            self.ax_3d.set_zlabel('Z (m)')
            self.ax_3d.set_title('3D SLAM Map & Trajectory')
            
    def update_3d_visualization(self, trajectory_3d, map_points, current_pose):
        """Update the 3D visualization with new data."""
        if not ENABLE_3D_VISUALIZATION or self.ax_3d is None:
            return None
            
        self.ax_3d.clear()
        
        # Plot 3D trajectory
        if trajectory_3d:
            positions = np.array([t['position'] for t in trajectory_3d])
            if len(positions) > 1:
                self.ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                               'g-', linewidth=2, label='Trajectory')
                # Mark start and end
                self.ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                                 c='blue', s=100, marker='o', label='Start')
                self.ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                                 c='red', s=100, marker='o', label='Current')
        
        # Plot 3D map points
        if len(map_points) > 0:
            valid_points = map_points[np.all(np.isfinite(map_points), axis=1)]
            if len(valid_points) > 0:
                # Color points by height (Z coordinate)
                colors = valid_points[:, 2] if valid_points.shape[1] > 2 else 'blue'
                self.ax_3d.scatter(valid_points[:, 0], valid_points[:, 1], 
                                 valid_points[:, 2] if valid_points.shape[1] > 2 else np.zeros(len(valid_points)),
                                 c=colors, s=1, alpha=0.6, cmap='viridis', label='Map Points')
        
        # Draw current camera pose
        if current_pose is not None:
            pos = current_pose[:3, 3]
            # Draw camera orientation using rotation matrix
            rot = current_pose[:3, :3]
            scale = 0.5
            
            # Camera axes
            x_axis = pos + rot[:, 0] * scale
            y_axis = pos + rot[:, 1] * scale  
            z_axis = pos + rot[:, 2] * scale
            
            self.ax_3d.plot([pos[0], x_axis[0]], [pos[1], x_axis[1]], [pos[2], x_axis[2]], 'r-', linewidth=3)
            self.ax_3d.plot([pos[0], y_axis[0]], [pos[1], y_axis[1]], [pos[2], y_axis[2]], 'g-', linewidth=3)
            self.ax_3d.plot([pos[0], z_axis[0]], [pos[1], z_axis[1]], [pos[2], z_axis[2]], 'b-', linewidth=3)
        
        # Set equal aspect ratio and limits
        if trajectory_3d or len(map_points) > 0:
            all_points = []
            if trajectory_3d:
                all_points.extend([t['position'] for t in trajectory_3d])
            if len(map_points) > 0:
                all_points.extend(map_points[:, :3].tolist())
            
            if all_points:
                all_points = np.array(all_points)
                margin = 2.0
                self.ax_3d.set_xlim(np.min(all_points[:, 0]) - margin, np.max(all_points[:, 0]) + margin)
                self.ax_3d.set_ylim(np.min(all_points[:, 1]) - margin, np.max(all_points[:, 1]) + margin)
                self.ax_3d.set_zlim(np.min(all_points[:, 2]) - margin, np.max(all_points[:, 2]) + margin)
        
        self.ax_3d.legend()
        self.ax_3d.grid(True)
        
        # Convert to image for display
        buffer = io.BytesIO()
        self.fig_3d.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to OpenCV image
        img_array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        buffer.close()
        
        return img


def gui_thread_func(gui_queue, stop_event):
    """Enhanced GUI function with 3D visualization only."""
    
    # Initialize 3D visualizer
    visualizer_3d = Enhanced3DVisualizer() if ENABLE_3D_VISUALIZATION else None
    
    print("GUI: Starting Enhanced 3D GUI thread - 3D visualization only.")
    
    while not stop_event.is_set():
        try:
            # Get the latest data packet from the queue
            data_packet = gui_queue.get(timeout=GUI_TIMEOUT)
            annotated_frame = data_packet['frame']
            est_path = data_packet['path']
            map_points = data_packet.get('map_points', np.array([]))
            trajectory_3d = data_packet.get('trajectory_3d', [])
            current_pose = data_packet.get('current_pose', None)
            is_3d_enabled = data_packet.get('is_3d_enabled', False)

            # === Display Windows ===
            cv2.imshow("Drone Feed & Features", annotated_frame)
            
            # === 3D Visualization Only ===
            if ENABLE_3D_VISUALIZATION and visualizer_3d and is_3d_enabled:
                img_3d = visualizer_3d.update_3d_visualization(trajectory_3d, map_points, current_pose)
                if img_3d is not None:
                    cv2.imshow("3D SLAM Visualization", img_3d)
            else:
                # Show basic info if 3D is not available
                info_img = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(info_img, "3D Visualization", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_img, f"Path Points: {len(est_path)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(info_img, f"Map Points: {len(map_points)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(info_img, f"3D Enabled: {is_3d_enabled}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(info_img, f"Trajectory: {len(trajectory_3d)}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                cv2.imshow("3D SLAM Info", info_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        except queue.Empty:
            continue
        except Exception as e:
            print(f"GUI: Error in display thread: {e}")
            continue
    
    cv2.destroyAllWindows()
    if visualizer_3d and visualizer_3d.fig_3d:
        plt.close(visualizer_3d.fig_3d)
    print("GUI: Enhanced 3D windows closed.")
