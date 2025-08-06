import cv2
import numpy as np
import asyncio
import airsim
import time

# Simple imports for basic configuration
try:
    from .config import CALIBRATION_FILE, ASYNC_SLEEP_INTERVAL, FRAME_SKIP_COUNT
except ImportError:
    from config import CALIBRATION_FILE, ASYNC_SLEEP_INTERVAL, FRAME_SKIP_COUNT

# ===============================================================================
# --- 1. VISUAL ODOMETRY CLASS 
# ===============================================================================
class VisualOdometry():
    def __init__(self, calib_filepath):
        self.K, self.P = self._load_calib(calib_filepath)
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.current_pose = np.eye(4)
        self.previous_frame = None
        self.map_points = []  # Store 3D map points for visualization
        self.trajectory_3d = []  # Store 3D trajectory

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_stereo_matches(self, left_frame, right_frame):
        """Get feature matches between left and right stereo cameras."""
        # Detect features in both cameras
        keypoints_left, descriptors_left = self.orb.detectAndCompute(left_frame, None)
        keypoints_right, descriptors_right = self.orb.detectAndCompute(right_frame, None)
        
        matches = []
        if descriptors_left is not None and descriptors_right is not None:
            # Use BFMatcher for stereo matching (more reliable than FLANN for stereo)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors_left, descriptors_right)
            
            # Sort matches by distance (best matches first)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Apply epipolar constraint - features should be on roughly the same horizontal line
            good_matches = []
            for match in matches:
                left_pt = keypoints_left[match.queryIdx].pt
                right_pt = keypoints_right[match.trainIdx].pt
                
                # Check if points are on roughly the same horizontal line (epipolar constraint)
                y_diff = abs(left_pt[1] - right_pt[1])
                if y_diff < 2.0:  # Allow 2 pixel vertical difference
                    good_matches.append(match)
            
            # Keep only the best matches
            good_matches = good_matches[:100]  # Limit to best 100 matches
        
        return keypoints_left, keypoints_right, good_matches

    def get_matches(self, prev_frame, current_frame):
        keypoints1, descriptors1 = self.orb.detectAndCompute(prev_frame, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(current_frame, None)
        matches = []
        if descriptors1 is not None and descriptors2 is not None:
            matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        return q1, q2, keypoints2

    def get_pose(self, q1, q2):
        if len(q1) < 8:
            return None
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if Essential is None:
            return None
        _, R, t, _ = cv2.recoverPose(Essential, q1, q2, self.K, mask=mask)
        return self._form_transf(R, np.squeeze(t))

    def triangulate_points(self, pose1, pose2, pts1, pts2):
        """Simple triangulation to create 3D map points."""
        if len(pts1) < 4:
            return np.array([])
        
        # Create projection matrices
        P1 = self.K @ pose1[:3, :]
        P2 = self.K @ pose2[:3, :]
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert to 3D (homogeneous to euclidean)
        points_3d = points_4d[:3] / points_4d[3]
        
        # Filter out points that are too far or behind camera
        valid_mask = (points_4d[3] > 0.1) & (points_3d[2] > 0) & (points_3d[2] < 100)
        
        return points_3d[:, valid_mask].T

    def process_frame(self, frame):
        """Process a frame and return keypoints and updated pose."""
        if self.previous_frame is None:
            self.previous_frame = frame
            self.previous_pose = self.current_pose.copy()
            return [], self.current_pose
        
        # Get matches between previous and current frame
        q1, q2, keypoints = self.get_matches(self.previous_frame, frame)
        
        if len(q1) < 8:
            # Not enough matches, return empty keypoints
            self.previous_frame = frame
            return [], self.current_pose
        
        # Get relative transformation
        transformation = self.get_pose(q1, q2)
        
        if transformation is not None:
            # Store previous pose before updating
            prev_pose = self.current_pose.copy()
            
            # Update current pose
            self.current_pose = self.current_pose @ np.linalg.inv(transformation)
            
            # Add current position to trajectory
            self.trajectory_3d.append({
                'position': self.current_pose[:3, 3].copy(),
                'rotation': self.current_pose[:3, :3].copy(),
                'timestamp': time.time()
            })
            
            # Keep trajectory manageable
            if len(self.trajectory_3d) > 1000:
                self.trajectory_3d.pop(0)
            
            # Triangulate new map points
            new_points = self.triangulate_points(prev_pose, self.current_pose, q1, q2)
            if len(new_points) > 0:
                self.map_points.extend(new_points)
                
                # Keep map points manageable
                if len(self.map_points) > 5000:
                    # Keep the most recent 3000 points
                    self.map_points = self.map_points[-3000:]
        
        # Update previous frame
        self.previous_frame = frame
        
        return keypoints, self.current_pose

# ===============================================================================
# --- 2. Main Execution Loop
# ===============================================================================

async def run_video_and_vo_improved(airsim_client, gui_queue, est_path, stop_event):
    """
    Simplified main async function for video processing and visual odometry.
    """
    print("VO/VIDEO: Starting Simple Visual Odometry.")
    
    # Initialize Visual Odometry
    vo = VisualOdometry(CALIBRATION_FILE)
    
    # Statistics tracking
    frames_processed = 0
    frames_skipped = 0
    frame_counter = 0
    
    try:
        while not stop_event.is_set():
            # Frame skipping for performance
            frame_counter += 1
            if frame_counter % (FRAME_SKIP_COUNT + 1) != 0:
                frames_skipped += 1
                await asyncio.sleep(ASYNC_SLEEP_INTERVAL)
                continue
                
            responses = airsim_client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # Left camera
                airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)   # Right camera
            ])
            
            # Debug: Print response information
            print(f"Debug: Received {len(responses)} camera responses")
            if len(responses) >= 1:
                print(f"Debug: Camera 0 - Size: {len(responses[0].image_data_uint8) if responses[0].image_data_uint8 else 0}")
            if len(responses) >= 2:
                print(f"Debug: Camera 1 - Size: {len(responses[1].image_data_uint8) if responses[1].image_data_uint8 else 0}")
            
            if not responses or len(responses) < 2 or not responses[0].image_data_uint8 or not responses[1].image_data_uint8:
                print("Debug: Missing camera data - skipping frame")
                await asyncio.sleep(ASYNC_SLEEP_INTERVAL)
                continue
                
            # Process left camera (camera "0")
            resp_left = responses[0]
            img1d_left = np.frombuffer(resp_left.image_data_uint8, dtype=np.uint8)
            left_frame_color = img1d_left.reshape(resp_left.height, resp_left.width, 3)
            left_frame_gray = cv2.cvtColor(left_frame_color, cv2.COLOR_BGR2GRAY)
            
            # Process right camera (camera "1")
            resp_right = responses[1]
            img1d_right = np.frombuffer(resp_right.image_data_uint8, dtype=np.uint8)
            right_frame_color = img1d_right.reshape(resp_right.height, resp_right.width, 3)
            right_frame_gray = cv2.cvtColor(right_frame_color, cv2.COLOR_BGR2GRAY)
            
            # Use left camera for visual odometry processing
            current_frame_gray = left_frame_gray
            
            # Create a combined stereo display
            stereo_display = np.hstack((left_frame_color, right_frame_color))
            annotated_frame = stereo_display.copy()

            # Process the frame with the visual odometry system
            try:
                keypoints, current_pose = vo.process_frame(current_frame_gray)
                    
            except Exception as e:
                print(f"VO processing error: {e}")
                keypoints, current_pose = [], vo.current_pose
            
            if keypoints and len(keypoints) > 0:
                # Get stereo matches between left and right cameras
                left_keypoints, right_keypoints, stereo_matches = vo.get_stereo_matches(left_frame_gray, right_frame_gray)
                
                # Draw VO features on left camera (green)
                left_with_features = cv2.drawKeypoints(left_frame_color, keypoints, None, color=(0, 255, 0))
                
                # Draw stereo matched features on right camera (cyan)
                right_with_features = cv2.drawKeypoints(right_frame_color, right_keypoints, None, color=(0, 255, 255))
                
                # Draw stereo matching lines between cameras
                if stereo_matches and len(stereo_matches) > 0:
                    # Create a combined image for drawing matches
                    stereo_combined = np.hstack((left_with_features, right_with_features))
                    
                    # Draw lines connecting matched features
                    for match in stereo_matches[:20]:  # Show only first 20 matches for clarity
                        left_pt = left_keypoints[match.queryIdx].pt
                        right_pt = right_keypoints[match.trainIdx].pt
                        
                        # Adjust right point coordinates for combined image
                        right_pt_adjusted = (int(right_pt[0] + left_frame_color.shape[1]), int(right_pt[1]))
                        left_pt_int = (int(left_pt[0]), int(left_pt[1]))
                        
                        # Draw line connecting matched features
                        cv2.line(stereo_combined, left_pt_int, right_pt_adjusted, (255, 0, 255), 1)
                    
                    annotated_frame = stereo_combined
                else:
                    # No stereo matches, just show features on both cameras
                    annotated_frame = np.hstack((left_with_features, right_with_features))
                
                # Update text overlay
                stereo_match_count = len(stereo_matches) if stereo_matches else 0
                cv2.putText(annotated_frame, f"VO:{len(keypoints)} Stereo:{stereo_match_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Left (VO+Stereo)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"Right (Stereo Matches)", (left_frame_color.shape[1] + 10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(annotated_frame, f"Pink lines = Stereo matches", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                cv2.putText(annotated_frame, f"Frames: {frames_processed}/{frame_counter}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Store trajectory data
                x, y, z = current_pose[:3, 3]
                est_path.append((x, z))  # Store as 2D for compatibility
                
                frames_processed += 1
            else:
                cv2.putText(annotated_frame, "Stereo VO INIT / TRACKING LOST", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Left Camera", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Right Camera", (left_frame_color.shape[1] + 10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Frames: {frames_processed}/{frame_counter}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Put results into the queue for the GUI
            try:
                # Get map points from VO
                map_points = np.array(vo.map_points[-500:]) if vo.map_points else np.array([])  # Show last 500 points
                trajectory_3d = vo.trajectory_3d[-100:] if vo.trajectory_3d else []  # Show last 100 trajectory points
                
                gui_data = {
                    'frame': annotated_frame, 
                    'path': est_path, 
                    'map_points': map_points,  # Now we have map points from triangulation
                    'trajectory_3d': trajectory_3d,  # And 3D trajectory
                    'current_pose': current_pose,
                    'is_3d_enabled': True,  # Enable 3D visualization
                    'gps_enabled': False
                }
                
                gui_queue.put_nowait(gui_data)
                
            except:  # Catch any queue exception
                pass # Don't block if GUI is slow

            await asyncio.sleep(ASYNC_SLEEP_INTERVAL)
    
    finally:
        # Cleanup and statistics
        print(f"\nVO PERFORMANCE STATISTICS:")
        print(f"  Total Frames Received: {frame_counter}")
        print(f"  Frames Processed: {frames_processed}")
        print(f"  Frames Skipped: {frames_skipped}")
        print(f"  Processing Rate: {frames_processed/frame_counter*100:.1f}%")
        print(f"  Map Points Generated: {len(vo.map_points)}")
        print(f"  Trajectory Points: {len(vo.trajectory_3d)}")

# Keep the original function for backward compatibility
async def run_video_and_vo(airsim_client, gui_queue, est_path, stop_event):
    """Original VO function - now points to the simplified implementation."""
    return await run_video_and_vo_improved(airsim_client, gui_queue, est_path, stop_event)