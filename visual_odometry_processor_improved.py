import cv2
import numpy as np
import asyncio
import airsim
from VO.Vision import VisualOdometry
from .config import (
    CALIBRATION_FILE, ASYNC_SLEEP_INTERVAL,
    MOTION_DETECTION_THRESHOLD, MIN_FEATURES_FOR_VO, TEMPORAL_SMOOTHING_ALPHA,
    MAX_POSE_HISTORY, SIFT_FEATURES, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD,
    RATIO_TEST_THRESHOLD, RANSAC_THRESHOLD, RANSAC_CONFIDENCE
)


def detect_motion(prev_frame, curr_frame, threshold=MOTION_DETECTION_THRESHOLD):
    """
    Detect if there's sufficient motion between frames to warrant VO processing.
    This prevents drift accumulation during stationary periods in simulation.
    """
    if prev_frame is None:
        return True
    
    # Calculate frame difference
    diff = cv2.absdiff(prev_frame, curr_frame)
    motion_level = np.mean(diff)
    
    return motion_level > threshold


def apply_temporal_smoothing(new_pose, prev_poses, alpha=TEMPORAL_SMOOTHING_ALPHA):
    """
    Apply exponential smoothing to pose estimates to reduce noise.
    
    Args:
        new_pose: Current pose estimate (4x4 matrix)
        prev_poses: List of previous poses
        alpha: Smoothing factor (0.7 = 70% new, 30% previous)
    """
    if not prev_poses:
        return new_pose
    
    # Simple exponential smoothing on position
    prev_pose = prev_poses[-1]
    smoothed_pose = new_pose.copy()
    
    # Smooth translation components
    smoothed_pose[:3, 3] = alpha * new_pose[:3, 3] + (1 - alpha) * prev_pose[:3, 3]
    
    return smoothed_pose


class ImprovedVisualOdometry:
    """
    Improved VO class with motion detection and better feature matching for simulation.
    """
    
    def __init__(self, calib_file):
        # Load original VO
        self.base_vo = VisualOdometry(calib_file)
        
        # Enhanced feature detector for simulation
        self.detector = cv2.SIFT_create(
            nfeatures=SIFT_FEATURES,
            contrastThreshold=SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_EDGE_THRESHOLD,
            sigma=1.6
        )
        
        # Feature matcher with ratio test
        self.matcher = cv2.BFMatcher()
        
        # Motion detection parameters
        self.motion_threshold = MOTION_DETECTION_THRESHOLD
        self.min_features = MIN_FEATURES_FOR_VO
        
        # Temporal smoothing
        self.pose_history = []
        self.max_history = MAX_POSE_HISTORY
    
    def get_enhanced_matches(self, img1, img2):
        """
        Enhanced feature matching with SIFT and RANSAC for simulation.
        """
        # Detect features
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < self.min_features or len(des2) < self.min_features:
            return None, None, []
        
        # Match features with ratio test
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_features:
            return None, None, []
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Convert to the format expected by original VO
        q1 = pts1.reshape(-1, 2)
        q2 = pts2.reshape(-1, 2)
        
        # Create keypoints for visualization
        matched_kp = [kp2[m.trainIdx] for m in good_matches]
        
        return q1, q2, matched_kp
    
    def get_robust_pose(self, q1, q2):
        """
        Get pose estimate with RANSAC for outlier rejection.
        """
        if q1 is None or q2 is None or len(q1) < 8:
            return None
        
        # Use the original VO's camera matrix
        K = self.base_vo.K
        
        # Find essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            q1, q2, K,
            method=cv2.RANSAC,
            prob=RANSAC_CONFIDENCE,
            threshold=RANSAC_THRESHOLD
        )
        
        if E is None:
            return None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, q1, q2, K)
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        return T


async def run_video_and_vo_improved(airsim_client, gui_queue, est_path, stop_event):
    """
    Improved async function for video processing and visual odometry with:
    - Motion detection to skip stationary periods
    - Enhanced feature matching for simulation
    - Temporal smoothing to reduce noise
    """
    print("VO/VIDEO: Starting improved VO tasks.")
    
    # Initialize improved VO
    improved_vo = ImprovedVisualOdometry(CALIBRATION_FILE)
    current_pose = np.eye(4)
    prev_frame_gray = None
    
    # Statistics tracking
    frames_processed = 0
    frames_skipped_motion = 0
    frames_skipped_features = 0
    
    while not stop_event.is_set():
        responses = airsim_client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        if not responses or not responses[0].image_data_uint8:
            await asyncio.sleep(ASYNC_SLEEP_INTERVAL)
            continue
            
        resp = responses[0]
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        current_frame_color = img1d.reshape(resp.height, resp.width, 3)
        current_frame_gray = cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2GRAY)
        annotated_frame = current_frame_color.copy()

        if prev_frame_gray is not None:
            # 1. Motion Detection - Skip VO if drone is stationary
            if not detect_motion(prev_frame_gray, current_frame_gray, improved_vo.motion_threshold):
                frames_skipped_motion += 1
                # Add text overlay to show motion detection
                cv2.putText(annotated_frame, "STATIONARY - VO SKIPPED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # 2. Enhanced Feature Matching
                q1, q2, keypoints = improved_vo.get_enhanced_matches(prev_frame_gray, current_frame_gray)
                
                if q1 is not None and q2 is not None:
                    # Draw enhanced features
                    annotated_frame = cv2.drawKeypoints(annotated_frame, keypoints, None, color=(0, 255, 0))
                    
                    # 3. Robust Pose Estimation with RANSAC
                    transf = improved_vo.get_robust_pose(q1, q2)
                    
                    if transf is not None:
                        # 4. Temporal Smoothing
                        raw_pose = current_pose @ np.linalg.inv(transf)
                        smoothed_pose = apply_temporal_smoothing(raw_pose, improved_vo.pose_history)
                        
                        # Update pose history
                        improved_vo.pose_history.append(smoothed_pose)
                        if len(improved_vo.pose_history) > improved_vo.max_history:
                            improved_vo.pose_history.pop(0)
                        
                        current_pose = smoothed_pose
                        x, y, z = current_pose[:3, 3]
                        est_path.append((x, z))
                        
                        frames_processed += 1
                        
                        # Add success indicator
                        cv2.putText(annotated_frame, f"VO OK - Features: {len(keypoints)}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        frames_skipped_features += 1
                        cv2.putText(annotated_frame, "POSE ESTIMATION FAILED", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    frames_skipped_features += 1
                    cv2.putText(annotated_frame, "INSUFFICIENT FEATURES", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add statistics overlay
        stats_text = f"Processed: {frames_processed} | Skipped(Motion): {frames_skipped_motion} | Skipped(Features): {frames_skipped_features}"
        cv2.putText(annotated_frame, stats_text, (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        prev_frame_gray = current_frame_gray
        
        # Put the results into the queue for the GUI thread to display
        try:
            gui_queue.put_nowait({'frame': annotated_frame, 'path': est_path})
        except:
            pass  # Don't block if the GUI is slow
            
        await asyncio.sleep(ASYNC_SLEEP_INTERVAL)
    
    # Print final statistics
    total_frames = frames_processed + frames_skipped_motion + frames_skipped_features
    print(f"\nVO STATISTICS:")
    print(f"  Total frames: {total_frames}")
    print(f"  Processed: {frames_processed} ({100*frames_processed/total_frames:.1f}%)")
    print(f"  Skipped (Motion): {frames_skipped_motion} ({100*frames_skipped_motion/total_frames:.1f}%)")
    print(f"  Skipped (Features): {frames_skipped_features} ({100*frames_skipped_features/total_frames:.1f}%)")


# Keep the original function for backward compatibility
async def run_video_and_vo(airsim_client, gui_queue, est_path, stop_event):
    """Original VO function - use run_video_and_vo_improved for better results."""
    return await run_video_and_vo_improved(airsim_client, gui_queue, est_path, stop_event)
