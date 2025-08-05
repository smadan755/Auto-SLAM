import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed

# Try relative imports first (when run as module), fall back to absolute imports
try:
    from .config import DRONE_CONNECTION, TAKEOFF_WAIT_TIME, CONTROL_LOOP_INTERVAL
except ImportError:
    # Fallback for direct execution
    from config import DRONE_CONNECTION, TAKEOFF_WAIT_TIME, CONTROL_LOOP_INTERVAL


async def run_drone_control(velocity_cmd, stop_event):
    """Async function to handle drone control via MAVSDK."""
    drone = System()
    await drone.connect(system_address=DRONE_CONNECTION)

    print("CONTROL: Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("CONTROL: Drone discovered!")
            break

    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(TAKEOFF_WAIT_TIME)
    
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()
    print("CONTROL: Offboard mode started. Ready for input.")

    while not stop_event.is_set():
        vel = velocity_cmd
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vel[0], vel[1], vel[2], vel[3]))
        await asyncio.sleep(CONTROL_LOOP_INTERVAL)

    print("CONTROL: Landing drone.")
    await drone.offboard.stop()
    await drone.action.land()
