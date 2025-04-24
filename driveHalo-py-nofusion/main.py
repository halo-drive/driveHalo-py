import asyncio
import logging
import cv2
import numpy as np
import argparse
import time
from typing import Tuple, Optional
import os

from lane_detector import LaneDetector
from steering_controller import SteeringController
from vehicle_interface import MCMController
from gearshifter import GearController, GearPositions  # Import from gearshifter, not vehicle_interface


def configure_camera(source=0, width=640, height=480, fps=30):
    """Configure camera with explicit format selection for optimal performance on Tegra"""
    if isinstance(source, int):
        # Explicitly request MJPG format to achieve 30fps
        pipeline = (
            f"v4l2src device=/dev/video{source} io-mode=2 ! "
            f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            "jpegdec ! "  # Hardware-accelerated JPEG decoder
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=2 sync=false"  # Disable sync for real-time processing
        )

        logging.info(f"Initializing camera with pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            logging.warning("Failed to open with MJPG pipeline, trying fallback...")
            # Fallback pipeline with explicit format-specific caps
            pipeline = (
                f"v4l2src device=/dev/video{source} ! "
                f"video/x-raw, width={width}, height={height}, framerate={fps}/1, format=YUYV ! "
                "videoconvert ! video/x-raw, format=BGR ! "
                "appsink drop=1"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        # For file inputs
        cap = cv2.VideoCapture(source)

    # Verify actual configuration
    if cap.isOpened():
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
    else:
        logging.error("Failed to open camera source")

    return cap


class AutonomousSystem:
    """Main autonomous driving system with concurrent task architecture"""

    def __init__(self, engine_path: str, can_channel: str = 'can0'):
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        self.logger = logging.getLogger("AutonomousSystem")

        try:
            # Initialize components
            self.lane_detector = LaneDetector(engine_path)
            self.steering_controller = SteeringController()
            self.mcm = MCMController(can_channel)
            self.gear_controller = GearController(can_channel)

            # System state
            self.running = False
            self.frame_count = 0
            self.start_time = None

            self.logger.info("Autonomous system initialized successfully")
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    async def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
        """Process frame and calculate control signals"""
        try:
            # Detect lane and calculate curvature
            annotated_frame, radius, offset  = self.lane_detector.detect_lane(frame)

            # Calculate control signals
            steering, throttle, brake = await self.steering_controller.coordinate_controls(radius, offset)

            # Update display with control information
            height = frame.shape[0]
            info_text = [
                f'Curve Radius: {radius:.1f}m',
                f'Lateral Offset: {offset:.2f}m',
                f'Steering: {steering:.3f}',
                f'Throttle: {throttle:.3f}',
                f'Brake: {brake:.3f}',
                f'Temp Release: {"Active" if self.steering_controller.is_temp_release_active else "Inactive"}'
            ]

            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text,
                            (20, height - 30 * (len(info_text) - i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            return annotated_frame, (steering, throttle, brake)

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame, None

    async def initialize_drive_mode(self):
        """Initialize the vehicle in drive mode with proper state verification"""
        try:
            self.logger.info("Initializing drive mode")

            # Clear states for both subsystems before gear change
            for subsystem_id in [0, 1]:
                self.logger.info(f"Clearing initial state for subsystem {subsystem_id}")
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.2)  # Ensure complete processing

            # Apply initial brake for safety during gear change
            await self.mcm.control_request('brake', True)
            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)
            await asyncio.sleep(0.2)

            # Execute gear change to Drive
            self.logger.info("Shifting to Drive mode")
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.DRIVE,
                brake_percentage=0.5,
                brake_duration=1
            )

            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)
            self.steering_controller.is_braking = True

            # Verify system state after gear change
            for subsystem_id in [0, 1]:
                self.logger.info(f"Verifying post-gear state for subsystem {subsystem_id}")
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.1)

            self.logger.info("Drive mode initialization completed")
            await asyncio.sleep(0.5)  # Allow time for final gear engagement

        except Exception as e:
            self.logger.error(f"Drive mode initialization failed: {e}")
            raise

    async def safe_stop_sequence(self):
        """Execute safe stop sequence with proper state clearing"""
        try:
            self.logger.info("Executing safe stop sequence")

            # First shift to Neutral with moderate braking
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.NEUTRAL,
                brake_percentage=0.4,
                brake_duration=1.5
            )
            await asyncio.sleep(0.5)  # Allow for gear disengagement

            # Then shift to Park with firm braking
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.PARK,
                brake_percentage=0.5,
                brake_duration=2.0
            )

            # Clear states for both subsystems with proper seed-key sequence
            for subsystem_id in [0, 1]:
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.2)  # Ensure complete processing between subsystems

            self.logger.info("Safe stop sequence completed")
        except Exception as e:
            self.logger.error(f"Error during safe stop sequence: {e}")
            raise

    async def run(self, source=0, width=640, height=480, fps=30):
        """Main control loop with concurrent task architecture"""
        self.logger.info(f"Opening video source: {source}")

        # Use custom camera configuration function
        cap = configure_camera(source, width=width, height=height, fps=fps)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            return

        try:
            # Initialize queues for producer-consumer pattern
            frame_queue = asyncio.Queue(maxsize=2)  # Limit frame buffer to prevent memory issues
            control_queue = asyncio.Queue(maxsize=4)  # Buffer for control commands

            # Initialize state
            self.running = True
            self.frame_count = 0
            self.start_time = time.time()

            # Initialize drive mode
            await self.initialize_drive_mode()

            # Enable all controls
            for control in ['steer', 'throttle', 'brake']:
                await self.mcm.control_request(control, True)
                self.logger.info(f"{control.capitalize()} control enabled")

            # Apply initial brake force
            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)

            # Start concurrent tasks
            self.logger.info("Starting concurrent processing tasks")
            tasks = [
                asyncio.create_task(self._capture_frames(cap, frame_queue)),
                asyncio.create_task(self._process_frames(frame_queue, control_queue)),
                asyncio.create_task(self._send_control_commands(control_queue))
            ]

            # Wait for tasks to complete (or exception)
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION
            )

            # Check for exceptions
            for task in done:
                if task.exception():
                    raise task.exception()

            # Cancel pending tasks
            for task in pending:
                task.cancel()

        except asyncio.CancelledError:
            self.logger.info("Tasks were cancelled")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            # Mark system as not running
            self.running = False

            # Clean up resources
            self.logger.info("Initiating shutdown sequence")
            try:
                # Sequence control shutdown
                await self.mcm.update_setpoint('throttle', 0)
                await asyncio.sleep(0.1)
                await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)
                await asyncio.sleep(0.1)
                await self.mcm.update_setpoint('steer', 0)
                await asyncio.sleep(0.1)

                await self.safe_stop_sequence()

                # Disable all controls in reverse order
                for control in ['throttle', 'brake', 'steer']:
                    await self.mcm.control_request(control, False)
                    await asyncio.sleep(0.05)

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Shutdown complete")

    async def _capture_frames(self, cap, frame_queue):
        """Dedicated task for frame acquisition"""
        self.logger.info("Frame capture task started")
        last_fps_log_time = time.time()

        try:
            while self.running:
                success, frame = cap.read()
                if not success:
                    self.logger.error("Failed to read frame")
                    await asyncio.sleep(0.01)
                    continue

                self.frame_count += 1
                current_time = time.time()

                # Log FPS at regular intervals
                if current_time - last_fps_log_time >= 1.0:
                    fps = self.frame_count / (current_time - self.start_time)
                    self.logger.info(f"Current FPS: {fps:.2f}")
                    last_fps_log_time = current_time

                # Non-blocking queue put with short timeout
                try:
                    await asyncio.wait_for(frame_queue.put(frame), 0.01)
                except asyncio.TimeoutError:
                    # Skip frame if queue is full (maintains real-time processing)
                    pass

                # Small yield to prevent CPU overutilization
                await asyncio.sleep(0.001)

        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            self.running = False
            raise

    async def _process_frames(self, frame_queue, control_queue):
        """Dedicated task for frame processing with proper GPU context management"""
        self.logger.info("Frame processing task started")

        # CRITICAL: Create a local synchronization lock for GPU operations
        gpu_lock = asyncio.Lock()

        try:
            while self.running:
                try:
                    frame = await asyncio.wait_for(frame_queue.get(), 0.1)

                    # Ensure exclusive access to GPU resources during inference and post-processing
                    async with gpu_lock:
                        # Process frame with exclusive GPU access
                        annotated_frame, radius, lateral_offset = self.lane_detector.detect_lane(frame)

                    # Calculate control signals with verified radius
                    steering, throttle, brake = await self.steering_controller.coordinate_controls(radius, lateral_offset)

                    # Queue control commands (outside the GPU lock)
                    await control_queue.put((steering, throttle, brake))

                    # Handle display (can be done outside the lock)
                    if annotated_frame is not None:
                        fps = self.frame_count / (time.time() - self.start_time)
                        cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow('Autonomous Driving Window', annotated_frame)

                    # Check for exit request
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("User requested exit")
                        self.running = False
                        break

                    frame_queue.task_done()

                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            self.running = False
            raise

    async def _send_control_commands(self, control_queue):
        """Dedicated task for control command distribution"""
        self.logger.info("Control command task started")

        try:
            while self.running:
                # Get control values from queue (with timeout)
                try:
                    steering, throttle, brake = await asyncio.wait_for(control_queue.get(), 0.1)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue

                # Create tasks for parallel command distribution
                command_tasks = [
                    asyncio.create_task(self.mcm.update_setpoint('brake', brake)),
                    asyncio.create_task(self.mcm.update_setpoint('steer', steering)),
                    asyncio.create_task(self.mcm.update_setpoint('throttle', throttle))
                ]

                # Wait for all commands to complete
                await asyncio.gather(*command_tasks)

                # Mark task as complete
                control_queue.task_done()

        except Exception as e:
            self.logger.error(f"Control command error: {e}")
            self.running = False
            raise

def main():
    """Entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Autonomous Driving System')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to TensorRT engine file')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--can-channel', type=str, default='can0',
                        help='CAN bus channel for vehicle control')
    # Add camera configuration arguments
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width resolution')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height resolution')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera frames per second')
    args = parser.parse_args()

    try:
        system = AutonomousSystem(args.engine, args.can_channel)

        # Parse source properly
        source = 0 if args.source == '0' else args.source

        # Modify line 207 in run() to use these parameters directly:
        # Change:
        # cap = configure_camera(source, width=640, height=480, fps=30)
        # To:
        # cap = configure_camera(source, width=args.width, height=args.height, fps=args.fps)

        asyncio.run(system.run(
            source=source,
            width=args.width,
            height=args.height,
            fps=args.fps
        ))
    except Exception as e:
        logging.error(f"System failed to start: {e}", exc_info=True)
        raise
if __name__ == '__main__':
    main()