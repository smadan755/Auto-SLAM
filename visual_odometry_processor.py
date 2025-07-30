import cv2
import numpy as np
import asyncio
import airsim
from VO.Vision import VisualOdometry
from .config import CALIBRATION_FILE, ASYNC_SLEEP_INTERVAL


async def run_video_and_vo(airsim_client, gui_queue, est_path, stop_event):
    """Async function to handle video processing and visual odometry."""
    print("VO/VIDEO: Starting tasks.")
    vo = VisualOdometry(CALIBRATION_FILE)
    current_pose = np.eye(4)
    prev_frame_gray = None
    
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
            q1, q2, keypoints = vo.get_matches(prev_frame_gray, current_frame_gray)
            annotated_frame = cv2.drawKeypoints(annotated_frame, keypoints, None, color=(0, 255, 0))
            transf = vo.get_pose(q1, q2)
            if transf is not None:
                current_pose = current_pose @ np.linalg.inv(transf)
                x, y, z = current_pose[:3, 3]
                est_path.append((x, z))
        
        prev_frame_gray = current_frame_gray
        
        # Put the results into the queue for the GUI thread to display
        try:
            gui_queue.put_nowait({'frame': annotated_frame, 'path': est_path})
        except:
            pass  # Don't block if the GUI is slow
            
        await asyncio.sleep(ASYNC_SLEEP_INTERVAL)
