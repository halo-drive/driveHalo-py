import numpy as np
import logging
import time
from sensor_msgs.msg import Imu
from typing import Tuple, Optional, List, Dict, Any
from scipy.spatial.transform import Rotation


class IMUProcessor:
    """Process IMU data from Livox LiDAR for motion estimation"""

    def __init__(self, filter_strength: float = 0.85):
        self.logger = logging.getLogger("IMUProcessor")
        self.logger.info("Initializing IMU processor")

        # Filtering parameters
        self.filter_strength = filter_strength  # Higher values = more filtering

        # State variables
        self.orientation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.angular_velocity = np.zeros(3)  # x, y, z
        self.linear_acceleration = np.zeros(3)  # x, y, z

        # Filtered state variables
        self.filtered_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.filtered_angular_velocity = np.zeros(3)
        self.filtered_linear_acceleration = np.zeros(3)

        # Derived values
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Bias estimation
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.bias_samples = 0
        self.bias_max_samples = 100
        self.is_calibrating = False

        # Timing
        self.last_timestamp = None
        self.dt = 0.0

        self.logger.info("IMU processor initialized")

    def start_calibration(self):
        """Start calibration procedure to estimate sensor biases"""
        self.logger.info("Starting IMU calibration...")
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.bias_samples = 0
        self.is_calibrating = True

    def process_imu_data(self, imu_msg: Imu) -> Dict[str, Any]:
        """Process IMU message and update state"""
        # Extract data from IMU message
        orientation_q = np.array([
            imu_msg.orientation.w,
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z
        ])

        angular_vel = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])

        linear_accel = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        # Update timing
        current_time = imu_msg.header.stamp.to_sec()
        if self.last_timestamp is not None:
            self.dt = current_time - self.last_timestamp
        self.last_timestamp = current_time

        # Handle calibration if active
        if self.is_calibrating:
            self._update_calibration(angular_vel, linear_accel)
            if self.bias_samples >= self.bias_max_samples:
                self.is_calibrating = False
                self.logger.info(
                    f"IMU calibration complete. Gyro bias: {self.gyro_bias}, Accel bias: {self.accel_bias}")

            # During calibration, return minimal processing
            return {
                "calibrating": True,
                "progress": self.bias_samples / self.bias_max_samples,
                "timestamp": current_time
            }

        # Apply bias correction
        angular_vel = angular_vel - self.gyro_bias
        linear_accel = linear_accel - self.accel_bias

        # Ensure quaternion is normalized
        quat_norm = np.linalg.norm(orientation_q)
        if quat_norm > 0:
            orientation_q = orientation_q / quat_norm

        # Apply filtering
        if np.all(self.orientation_quaternion == np.array([1.0, 0.0, 0.0, 0.0])):
            # First valid measurement
            self.orientation_quaternion = orientation_q
            self.angular_velocity = angular_vel
            self.linear_acceleration = linear_accel

            self.filtered_orientation = orientation_q
            self.filtered_angular_velocity = angular_vel
            self.filtered_linear_acceleration = linear_accel
        else:
            # Apply complementary filter to orientation (SLERP)
            a = self.filter_strength
            inv_a = 1.0 - a

            # Simple low-pass filtering for velocity and acceleration
            self.filtered_angular_velocity = a * self.filtered_angular_velocity + inv_a * angular_vel
            self.filtered_linear_acceleration = a * self.filtered_linear_acceleration + inv_a * linear_accel

            # SLERP for quaternion filtering
            # For simplicity, just doing linear interpolation, but could be replaced with proper SLERP
            interp_quat = a * self.filtered_orientation + inv_a * orientation_q
            norm = np.linalg.norm(interp_quat)
            if norm > 1e-10:
                self.filtered_orientation = interp_quat / norm
            else:
                self.logger.warning("Invalid quaternion with zero norm detected")
                self.filtered_orientation = np.array([1.0,0.0,0.0,0.0])

        # Update state variables
        self.orientation_quaternion = orientation_q
        self.angular_velocity = angular_vel
        self.linear_acceleration = linear_accel

        # Calculate roll, pitch, yaw from filtered quaternion
        self._update_euler_angles()

        # Return processed data
        return {
            "orientation_quaternion": self.filtered_orientation,
            "angular_velocity": self.filtered_angular_velocity,
            "linear_acceleration": self.filtered_linear_acceleration,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "dt": self.dt,
            "timestamp": current_time,
            "calibrating": False
        }

    def _update_euler_angles(self):
        """Update roll, pitch, yaw from quaternion"""
        # Convert quaternion to euler angles
        try:
            rot = Rotation.from_quat([
                self.filtered_orientation[1],  # x
                self.filtered_orientation[2],  # y
                self.filtered_orientation[3],  # z
                self.filtered_orientation[0]  # w
            ])
            euler = rot.as_euler('xyz', degrees=False)
            self.roll = euler[0]
            self.pitch = euler[1]
            self.yaw = euler[2]
        except Exception as e:
            self.logger.error(f"Error calculating euler angles: {e}")

    def _update_calibration(self, angular_vel: np.ndarray, linear_accel: np.ndarray):
        """Update bias estimation during calibration"""
        # Accumulate samples
        self.gyro_bias += angular_vel

        # For accelerometer, we know gravity should be ~9.81 m/s² in the z-direction
        # So we subtract the expected gravity vector from the measured acceleration
        gravity_corrected = linear_accel.copy()
        gravity_corrected[2] -= 9.81  # Assuming z points down in sensor frame
        self.accel_bias += gravity_corrected

        self.bias_samples += 1

        # Finalize calibration if enough samples gathered
        if self.bias_samples >= self.bias_max_samples:
            self.gyro_bias /= self.bias_samples
            self.accel_bias /= self.bias_samples

    def get_motion_state(self) -> Dict[str, Any]:
        """Get the current motion state"""
        return {
            "orientation_quaternion": self.filtered_orientation.copy(),
            "angular_velocity": self.filtered_angular_velocity.copy(),
            "linear_acceleration": self.filtered_linear_acceleration.copy(),
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "dt": self.dt,
            "timestamp": self.last_timestamp
        }