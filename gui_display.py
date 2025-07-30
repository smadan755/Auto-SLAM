import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import queue
from .config import FIGURE_SIZE, GUI_TIMEOUT


def gui_thread_func(gui_queue, stop_event):
    """Function to run in a separate thread for all GUI updates."""
    # --- Matplotlib Plot Setup ---
    plt.ion()  # Turn on interactive mode
    fig, ax_path = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    line_est, = ax_path.plot([], [], 'g-', label='VO Estimate')
    ax_path.set_title('Live VO Path Estimate')
    ax_path.legend()
    ax_path.set_xlabel('X')
    ax_path.set_ylabel('Z')

    while not stop_event.is_set():
        try:
            # Get the latest data packet from the queue
            data_packet = gui_queue.get(timeout=GUI_TIMEOUT)
            annotated_frame = data_packet['frame']
            est_path = data_packet['path']

            # --- Update Path Plot Data ---
            if est_path:
                est_x_data, est_y_data = zip(*est_path)
                line_est.set_data(est_x_data, est_y_data)
                ax_path.relim()
                ax_path.autoscale_view()
            
            # --- Render Plot and Display Windows ---
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            cv2.imshow("Drone Feed & Features", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        except queue.Empty:
            continue
    
    cv2.destroyAllWindows()
    plt.close(fig)
    print("GUI: Windows closed.")
