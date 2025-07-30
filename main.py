"""
Main coordination file for the drone visual odometry system.
This file orchestrates all the modular components.
"""

import airsim
import asyncio
import threading
import queue


from config import AIRSIM_IP, GUI_QUEUE_SIZE
from input_handler import pygame_thread_func
from gui_display import gui_thread_func
from drone_controller import run_drone_control
from visual_odometry_processor_improved import run_video_and_vo_improved as run_video_and_vo
from utils import save_path_to_csv


async def main():
    """Main coordination function that orchestrates all components."""
    # --- Shared state between threads ---
    velocity_cmd = [0.0, 0.0, 0.0, 0.0] 
    stop_event = threading.Event()
    gui_queue = queue.Queue(maxsize=GUI_QUEUE_SIZE)
    
    # --- Initialize path data list in the main scope ---
    est_path_data = []

    # --- Initialize AirSim Client ---
    airsim_client = airsim.MultirotorClient(ip=AIRSIM_IP)
    airsim_client.confirmConnection()

    # --- Start blocking threads ---
    pygame_t = threading.Thread(target=pygame_thread_func, args=(velocity_cmd, stop_event))
    gui_t = threading.Thread(target=gui_thread_func, args=(gui_queue, stop_event))
    pygame_t.daemon = True
    gui_t.daemon = True
    pygame_t.start()
    gui_t.start()

    # --- Run async tasks ---
    try:
        await asyncio.gather(
            run_drone_control(velocity_cmd, stop_event),
            run_video_and_vo(airsim_client, gui_queue, est_path_data, stop_event)
        )
    except Exception as e:
        print(f"An error occurred in async tasks: {e}")
    finally:
        print("MAIN: Async tasks finished, setting stop event for threads.")
        stop_event.set()
        
        # Save the path data to CSV before joining threads
        save_path_to_csv(est_path_data)
        
        pygame_t.join()
        gui_t.join()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("--- Program interrupted by user.")
    finally:
        print("--- Program exited.")
