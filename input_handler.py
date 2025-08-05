import pygame
import threading

# Try relative imports first (when run as module), fall back to absolute imports
try:
    from .config import DEFAULT_SPEED, YAW_SPEED_RATE, PYGAME_WAIT_TIME
except ImportError:
    # Fallback for direct execution
    from config import DEFAULT_SPEED, YAW_SPEED_RATE, PYGAME_WAIT_TIME


def pygame_thread_func(velocity_cmd, stop_event):
    """Function to run in a separate thread for handling Pygame input."""
    pygame.init()
    pygame.display.set_mode((400, 400))
    print("PYGAME: Pygame thread initialized. Press 'L' to land and quit.")

    def getKey(keyName):
        ans = False
        # This event pump is critical for Pygame to process OS messages
        for event in pygame.event.get(): 
            pass
        keyInput = pygame.key.get_pressed()
        myKey = getattr(pygame, f'K_{keyName}')
        if keyInput[myKey]:
            ans = True
        pygame.display.update()
        return ans

    while not stop_event.is_set():
        forward, right, down, yaw_speed = 0.0, 0.0, 0.0, 0.0

        if getKey("a"): 
            right = -DEFAULT_SPEED
        elif getKey("d"): 
            right = DEFAULT_SPEED
        if getKey("UP"): 
            down = -DEFAULT_SPEED
        elif getKey("DOWN"): 
            down = DEFAULT_SPEED
        if getKey("w"): 
            forward = DEFAULT_SPEED
        elif getKey("s"): 
            forward = -DEFAULT_SPEED
        if getKey("q"): 
            yaw_speed = -YAW_SPEED_RATE
        elif getKey("e"): 
            yaw_speed = YAW_SPEED_RATE
        
        # Update the shared velocity command
        velocity_cmd[:] = [forward, right, down, yaw_speed]

        if getKey("l"):
            print("PYGAME: Landing key pressed. Shutting down.")
            stop_event.set()

        # Give the CPU a break
        pygame.time.wait(PYGAME_WAIT_TIME)
