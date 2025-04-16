import numpy as np
import logging
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial.transform import Rotation

from utils import measure_execution_time


class PoseEstimator:
    """Fuse camera lane detection with IMU data for robust vehicle pose estimation"""

    def __init__(self):
        self.logger = logging.getLogger("PoseEstimator")
        self.logger.info("Initializing pose estimator")

        # State variables
        self.position = np.zeros(3)  # x, y, z in world frame
        self.orientation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.velocity = np.zeros(3)  # vx, vy, vz in world frame
        self.angular_velocity = np.zeros(3)  # wx, wy, wz

        # Lane-related state
        self.lane_offset = 0.0  # Lateral offset from lane center (m)
        self.lane_heading_error = 0.0  # Heading error relative to lane (rad)
        self.lane_curvature = float('inf')  # Radius of curvature (m)

        # Confidence values
        self.imu_confidence = 0.0
        self.lane_confidence = 0.0

        # Filter parameters
        self.imu_weight = 0.6  # Weight for IMU in fusion (0-1)
        self.lane_detection_weight = 0.4  # Weight for lane detection in fusion (0-1)

        # Timing
        self.last_update_time = time.time()
        self.last_imu_update = 0.0
        self.last_lane_update = 0.0

        # Thread safety
        self.lock = threading.RLock()

        self.logger.info("Pose estimator initialized")

    @measure_execution_time
    def update_from_imu(self, imu_data: Dict[str, Any]) -> None:
        """Update state estimation using IMU data"""
        with self.lock:
            current_time = time.time()
            dt = current_time - self.last_update_time if self.last_update_time else 0.0

            # Skip if IMU is being calibrated
            if imu_data.get("calibrating", False):
                return

            # Extract data
            orientation_q = imu_data["orientation_quaternion"]
            angular_vel = imu_data["angular_velocity"]
            linear_accel = imu_data["linear_acceleration"]

            # IMU provides absolute orientation
            # We use interpolation to blend with existing estimate
            alpha = min(1.0, dt * 5.0)  # Smooth transition

            # Simple linear interpolation for quaternions
            # (Could be replaced with proper SLERP)
            blended_q = alpha * orientation_q + (1.0 - alpha) * self.orientation_quaternion
            self.orientation_quaternion = blended_q / np.linalg.norm(blended_q)

            # Update angular velocity directly from IMU
            self.angular_velocity = angular_vel

            # Update position and velocity using IMU acceleration
            # First transform acceleration to world frame
            rotation = Rotation.from_quat([
                orientation_q[1], orientation_q[2], orientation_q[3], orientation_q[0]
            ])
            world_accel = rotation.apply(linear_accel)

            # Simple integration (could be improved with more sophisticated methods)
            if dt > 0 and dt < 0.1:  # Sanity check on dt
                # First update velocity using acceleration
                self.velocity += world_accel * dt

                # Then update position using velocity
                self.position += self.velocity * dt

            self.last_update_time = current_time
            self.last_imu_update = current_time
            self.imu_confidence = 0.8  # Example confidence value

    def update_from_lane_detection(self, lane_data: Dict[str, Any]) -> None:
        """Update state estimation using lane detection data"""
        with self.lock:
            current_time = time.time()

            # Extract lane data
            curvature_radius = lane_data.get("curvature_radius", float('inf'))
            lateral_offset = lane_data.get("lateral_offset", 0.0)
            heading_error = lane_data.get("heading_error", 0.0)
            detection_confidence = lane_data.get("confidence", 0.0)

            # Update lane-related state
            self.lane_curvature = curvature_radius
            self.lane_offset = lateral_offset
            self.lane_heading_error = heading_error

            # Adjust heading based on lane detection
            # This is a simplified approach - in reality you'd want to
            # integrate this with IMU heading in a more sophisticated way
            if detection_confidence > 0.5:
                # Convert current orientation to Euler angles
                rotation = Rotation.from_quat([
                    self.orientation_quaternion[1],
                    self.orientation_quaternion[2],
                    self.orientation_quaternion[3],
                    self.orientation_quaternion[0]
                ])
                euler = rotation.as_euler('xyz')

                # Adjust yaw based on lane heading error
                adjustment_strength = min(0.3, detection_confidence)
                euler[2] -= heading_error * adjustment_strength

                # Convert back to quaternion
                new_rotation = Rotation.from_euler('xyz', euler)
                quat = new_rotation.as_quat()
                self.orientation_quaternion = np.array([quat[3], quat[0], quat[1], quat[2]])

            self.last_lane_update = current_time
            self.lane_confidence = detection_confidence

    def get_pose_state(self) -> Dict[str, Any]:
        """Get the current estimated pose state"""
        with self.lock:
            # Convert quaternion to roll, pitch, yaw
            rotation = Rotation.from_quat([
                self.orientation_quaternion[1],
                self.orientation_quaternion[2],
                self.orientation_quaternion[3],
                self.orientation_quaternion[0]
            ])
            euler = rotation.as_euler('xyz')

            # Calculate control-relevant metrics
            time_since_imu = time.time() - self.last_imu_update
            time_since_lane = time.time() - self.last_lane_update

            # Decay confidence over time
            imu_conf = self.imu_confidence * np.exp(-time_since_imu / 0.5)
            lane_conf = self.lane_confidence * np.exp(-time_since_lane / 1.0)

            # Total system confidence
            total_conf = (imu_conf + lane_conf) / 2.0

            return {
                "position": self.position.copy(),
                "orientation_quaternion": self.orientation_quaternion.copy(),
                "velocity": self.velocity.copy(),
                "angular_velocity": self.angular_velocity.copy(),
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
                "lane_offset": self.lane_offset,
                "lane_heading_error": self.lane_heading_error,
                "lane_curvature": self.lane_curvature,
                "imu_confidence": imu_conf,
                "lane_confidence": lane_conf,
                "total_confidence": total_conf
            }

    def calculate_steering_inputs(self) -> Tuple[float, float, float]:
        """Calculate steering angle, lane offset correction, and curvature factors"""
        with self.lock:
            state = self.get_pose_state()

            # Calculate steering angle from orientation
            yaw = state['yaw']

            # Get lane offset correction
            lane_offset = state['lane_offset']

            # Use lane curvature radius
            curvature = state['lane_curvature']

            return yaw, lane_offset, curvature