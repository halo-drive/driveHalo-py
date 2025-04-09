import asyncio
import logging
import cv2
import numpy as np
import argparse
import time
import threading
from typing import Tuple, Optional
import os

# Original components
from lane_detector import LaneDetector
from steering_controller import SteeringController
from vehicle_interface import MCMController
from gearshifter import GearController, GearPositions

# New components for sensor fusion
from ros_interface import ROSInterface
from imu_processor import IMUProcessor
from pose_estimator import PoseEstimator
from sensor_fusion import SensorFusion
from vehicle_state import VehicleStateManager


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

class MultiSensorAutonomousSystem:
    """Enhanced autonomous driving system with multi-sensor fusion"""

    def __init__(self, engine_path: str, can_channel: str = 'can0', use_ros: bool = True):
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        self.logger = logging.getLogger("MultiSensorAutonomousSystem")

        try:
            # Initialize original components
            self.lane_detector = LaneDetector(engine_path)
            self.steering_controller = SteeringController()
            self.mcm = MCMController(can_channel)
            self.gear_controller = GearController(can_channel)

            # Initialize new multi-sensor components
            self.vehicle_state = VehicleStateManager()
            self.sensor_fusion = SensorFusion()
            self.pose_estimator = PoseEstimator()
            self.imu_processor = IMUProcessor()

            # ROS interface for Livox LiDAR IMU
            self.use_ros = use_ros
            if use_ros:
                self.ros_interface = ROSInterface("autonomous_vehicle")
                # Register IMU callback
                self.ros_interface.register_imu_callback(self._imu_callback)
            else:
                self.ros_interface = None

            # Threading for sensor processing
            self.should_run = True
            self.imu_thread = None
            self.fusion_thread = None

            self.logger.info("Multi-sensor autonomous system initialized successfully")
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    def _imu_callback(self, imu_msg):
        """Callback for IMU data from ROS"""
        try:
            # Process IMU data
            imu_data = self.imu_processor.process_imu_data(imu_msg)

            # Update pose estimator with IMU data
            self.pose_estimator.update_from_imu(imu_data)

            # Update sensor fusion
            self.sensor_fusion.update_from_imu(imu_data)

            # Update vehicle state
            self.vehicle_state.update_from_imu(imu_data)
        except Exception as e:
            self.logger.error(f"Error processing IMU data: {e}")

    async def _process_frame_with_fusion(self, frame: np.ndarray) -> Tuple[
        np.ndarray, Optional[Tuple[float, float, float]]]:
        """Process frame with sensor fusion integration"""
        try:
            # Detect lane and calculate curvature
            annotated_frame, radius = self.lane_detector.detect_lane(frame)

            # Create lane data dictionary
            lane_data = {
                "curvature_radius": radius,
                "lateral_offset": 0.0,  # Would need dedicated lane position calculation
                "heading_error": 0.0,  # Would need lane orientation calculation
                "confidence": 0.8 if radius != float('inf') else 0.0,
                "timestamp": time.time()
            }

            # Update pose estimator with lane data
            self.pose_estimator.update_from_lane_detection(lane_data)

            # Update sensor fusion
            self.sensor_fusion.update_from_lane_detection(lane_data)

            # Update vehicle state
            self.vehicle_state.update_from_camera(lane_data)

            # Get fused state for control decisions
            fused_state = self.sensor_fusion.get_fused_state()

            # Update vehicle state with fused data
            self.vehicle_state.update_from_sensor_fusion(fused_state)

            # Calculate control signals using sensor fusion
            heading, lateral_offset, curvature = self.sensor_fusion.calculate_control_inputs()

            # Calculate final control signals
            steering, throttle, brake = await self.steering_controller.coordinate_controls(curvature)

            # Update vehicle state with control signals
            self.vehicle_state.update_control_signals(steering, throttle, brake)

            # Update display with control and state information
            height = frame.shape[0]
            info_text = [
                f'Curve Radius: {curvature:.1f}m',
                f'Steering: {steering:.3f}',
                f'Throttle: {throttle:.3f}',
                f'Brake: {brake:.3f}',
                f'Lateral Offset: {lateral_offset:.2f}m',
                f'Heading: {np.degrees(heading):.1f}°',
                f'Temp Release: {"Active" if self.steering_controller.is_temp_release_active else "Inactive"}'
            ]

            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text,
                            (20, height - 30 * (len(info_text) - i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return annotated_frame, (steering, throttle, brake)

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame, None

    async def initialize_drive_mode(self):
        """Initialize the vehicle in drive mode"""
        try:
            self.logger.info("Initializing drive mode")
            self.vehicle_state.update_system_status("INITIALIZING")

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
                brake_percentage=0.8,
                brake_duration=1.0
            )

            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)
            self.steering_controller.is_braking = True

            # Verify system state after gear change
            for subsystem_id in [0, 1]:
                self.logger.info(f"Verifying post-gear state for subsystem {subsystem_id}")
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.1)

            self.logger.info("Drive mode initialization completed")
            self.vehicle_state.update_system_status("ACTIVE", "")
            await asyncio.sleep(0.5)  # Allow time for final gear engagement

        except Exception as e:
            self.logger.error(f"Drive mode initialization failed: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Drive mode initialization failed: {e}")
            raise

    async def safe_stop_sequence(self):
        """Execute safe stop sequence"""
        try:
            self.logger.info("Executing safe stop sequence")
            self.vehicle_state.update_system_status("SHUTDOWN", "Safe stop initiated")

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
            self.vehicle_state.update_system_status("SHUTDOWN", "Safe stop completed")
        except Exception as e:
            self.logger.error(f"Error during safe stop sequence: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Safe stop failed: {e}")
            raise

    def _sensor_fusion_thread(self):
        """Thread for continuous sensor fusion updates"""
        try:
            rate = 0.01  # 100Hz update rate
            while self.should_run:
                # Predict forward at regular intervals
                self.sensor_fusion.predict(rate)
                time.sleep(rate)
        except Exception as e:
            self.logger.error(f"Sensor fusion thread error: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Sensor fusion error: {e}")

    def start_sensor_threads(self):
        """Start threads for sensor processing"""
        try:
            # Start ROS interface if enabled
            if self.use_ros:
                self.ros_interface.start()

            # Start IMU calibration
            self.imu_processor.start_calibration()

            # Start sensor fusion thread
            self.fusion_thread = threading.Thread(target=self._sensor_fusion_thread)
            self.fusion_thread.daemon = True
            self.fusion_thread.start()

            self.logger.info("Sensor threads started")
        except Exception as e:
            self.logger.error(f"Failed to start sensor threads: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Sensor thread start failed: {e}")
            raise

    def stop_sensor_threads(self):
        """Stop all sensor processing threads"""
        self.should_run = False

        # Stop ROS interface
        if self.use_ros and self.ros_interface:
            self.ros_interface.stop()

        # Wait for threads to finish
        if self.fusion_thread and self.fusion_thread.is_alive():
            self.fusion_thread.join(timeout=1.0)

        self.logger.info("Sensor threads stopped")

    async def run(self, source=0, width=640, height=480, fps=30):
        """Main control loop with integrated sensor fusion"""
        self.logger.info(f"Opening video source: {source}")
        self.vehicle_state.update_system_status("INITIALIZING", "Starting sensor systems")

        # Start sensor threads
        self.start_sensor_threads()

        # Use custom camera configuration function
        cap = configure_camera(source, width=width, height=height, fps=fps)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            self.vehicle_state.update_system_status("ERROR", f"Failed to open video source: {source}")
            return

        try:
            # Allow time for sensor calibration
            self.logger.info("Waiting for sensor calibration...")
            for i in range(10, 0, -1):
                self.logger.info(f"Starting in {i} seconds...")
                await asyncio.sleep(1.0)

            # Initialize drive mode
            await self.initialize_drive_mode()

            # Enable all controls while maintaining brake state
            for control in ['steer', 'throttle', 'brake']:
                await self.mcm.control_request(control, True)
                self.logger.info(f"{control.capitalize()} control enabled")

            # Ensure permanent brake force is applied initially
            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)

            self.logger.info("Starting main control loop")
            self.vehicle_state.update_system_status("ACTIVE", "")

            frame_count = 0
            start_time = time.time()
            last_fps_log_time = start_time

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.logger.error("Failed to read frame")
                    # Try to recover rather than breaking
                    await asyncio.sleep(0.1)
                    continue

                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Log FPS every second rather than every 30 frames
                if current_time - last_fps_log_time >= 1.0:
                    current_fps = frame_count / elapsed_time
                    self.logger.info(f"Current FPS: {current_fps:.2f}")
                    last_fps_log_time = current_time

                # Process frame with sensor fusion
                annotated_frame, controls = await self._process_frame_with_fusion(frame)

                if controls:
                    steering, throttle, brake = controls
                    try:
                        # Apply controls while respecting brake states
                        await self.mcm.update_setpoint('brake', brake)
                        await self.mcm.update_setpoint('steer', steering)
                        await asyncio.sleep(0.02)  # Brief delay between control updates
                        await self.mcm.update_setpoint('throttle', throttle)
                    except Exception as e:
                        self.logger.error(f"Control update failed: {e}")
                        self.vehicle_state.update_system_status("ERROR", f"Control update failed: {e}")
                        # Ensure brakes are applied on error
                        await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)

                # Update display with control information
                if annotated_frame is not None:
                    fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
                    cv2.putText(annotated_frame, fps_text, (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Add system status
                    status = self.vehicle_state.get_state().system_status
                    cv2.putText(annotated_frame, f"Status: {status}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0) if status == "ACTIVE" else (0, 0, 255), 2)

                    cv2.imshow('Multi-Sensor Autonomous System', annotated_frame)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    self.logger.info("User requested exit")
                    break

                # Add a small yield to prevent CPU overutilization
                await asyncio.sleep(0.001)

        except Exception as e:
            self.logger.error(f"Runtime error: {e}", exc_info=True)
            self.vehicle_state.update_system_status("ERROR", f"Runtime error: {e}")
        finally:
            # Stop sensor threads
            self.stop_sensor_threads()

            # Always ensure brakes are applied during shutdown
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
                self.vehicle_state.update_system_status("ERROR", f"Cleanup error: {e}")

            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Shutdown complete")
            self.vehicle_state.update_system_status("SHUTDOWN", "System shutdown complete")


def main():
    """Entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Multi-Sensor Autonomous Driving System')
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
    # Add argument for ROS usage
    parser.add_argument('--use-ros', action='store_true',
                        help='Enable ROS integration for Livox LiDAR')
    parser.add_argument('--imu-topic', type=str, default='/livox/imu',
                        help='ROS topic for Livox IMU data')

    args = parser.parse_args()

    try:
        system = MultiSensorAutonomousSystem(
            args.engine,
            args.can_channel,
            use_ros=args.use_ros
        )

        # Parse source properly
        source = 0 if args.source == '0' else args.source

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