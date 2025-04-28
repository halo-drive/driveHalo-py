import numpy as np
import open3d as o3d
import logging
import threading
import time
import cv2
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation

from fast_slam import FastSLAM
from map_manager import MapManager


class SLAMIntegration:
    """
    Integration module to connect the SLAM system with the vehicle control system

    This class provides an interface between the SLAM components and the
    existing sensor fusion and control systems.
    """

    def __init__(self,
                 use_lidar: bool = True,
                 use_imu: bool = True,
                 map_resolution: float = 0.1,
                 map_size: Tuple[int, int] = (2000, 2000),
                 num_particles: int = 30):
        """
        Initialize the SLAM integration module

        Args:
            use_lidar: Whether to use LiDAR data
            use_imu: Whether to use IMU data
            map_resolution: Resolution of the map in meters per cell
            map_size: Size of the map in cells (width, height)
            num_particles: Number of particles for FastSLAM
        """
        self.logger = logging.getLogger("SLAMIntegration")
        self.logger.info("Initializing SLAM integration")

        # Configuration
        self.use_lidar = use_lidar
        self.use_imu = use_imu

        # Initialize SLAM system
        self.slam = FastSLAM(
            num_particles=num_particles,
            map_resolution=map_resolution,
            map_size=map_size
        )

        # Thread safety
        self.lock = threading.RLock()

        # State variables
        self.is_active = False
        self.lidar_data_buffer = []
        self.imu_data_buffer = []
        self.max_buffer_size = 100

        # Visualization
        self.viz_image = None
        self.viz_timestamp = 0
        self.viz_update_interval = 0.5  # seconds

        # Performance metrics
        self.processing_times = []
        self.max_processing_times = 100

        # Coordinate transformation between SLAM and vehicle frames
        # In most cases, the LiDAR frame would be different from the vehicle frame
        # Here we assume the LiDAR is mounted at the center of the vehicle with same orientation
        self.lidar_to_vehicle_transform = np.eye(4)

        self.logger.info("SLAM integration initialized")

    def start(self):
        """Start the SLAM system"""
        with self.lock:
            if not self.is_active:
                self.is_active = True
                self.logger.info("SLAM system started")
                return True
            return False

    def stop(self):
        """Stop the SLAM system"""
        with self.lock:
            if self.is_active:
                self.is_active = False
                self.logger.info("SLAM system stopped")
                return True
            return False

    def reset(self):
        """Reset the SLAM system"""
        with self.lock:
            self.slam.reset()
            self.lidar_data_buffer = []
            self.imu_data_buffer = []
            self.processing_times = []
            self.logger.info("SLAM system reset")
            return True

    def process_lidar_data(self, points: np.ndarray,
                           intensities: Optional[np.ndarray] = None,
                           timestamp: float = None) -> Dict[str, Any]:
        """
        Process LiDAR data for SLAM

        Args:
            points: Nx3 array of point coordinates (x, y, z)
            intensities: N array of point intensities (optional)
            timestamp: Data timestamp (seconds, optional)

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = start_time

        with self.lock:
            # Skip if SLAM is not active
            if not self.is_active:
                return {
                    "success": False,
                    "message": "SLAM system is not active",
                    "processing_time": 0.0
                }

            # Skip if not using LiDAR
            if not self.use_lidar:
                return {
                    "success": False,
                    "message": "LiDAR data processing is disabled",
                    "processing_time": 0.0
                }

            # Store in buffer for potential future use
            self.lidar_data_buffer.append({
                "points": points,
                "intensities": intensities,
                "timestamp": timestamp
            })

            # Limit buffer size
            if len(self.lidar_data_buffer) > self.max_buffer_size:
                self.lidar_data_buffer.pop(0)

            # Process with SLAM
            try:
                result = self.slam.process_scan(points, intensities, timestamp)

                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                # Limit processing times list
                if len(self.processing_times) > self.max_processing_times:
                    self.processing_times.pop(0)

                # Update visualization if needed
                if time.time() - self.viz_timestamp > self.viz_update_interval:
                    self.viz_image = self.slam.get_visualization_image()
                    self.viz_timestamp = time.time()

                return {
                    "success": result["success"],
                    "message": result.get("message", ""),
                    "slam_pose": result.get("current_pose", np.eye(4)),
                    "vehicle_pose": self._slam_to_vehicle_pose(result.get("current_pose", np.eye(4))),
                    "processing_time": processing_time
                }
            except Exception as e:
                self.logger.error(f"Error processing LiDAR data: {e}")
                return {
                    "success": False,
                    "message": f"Error processing LiDAR data: {e}",
                    "processing_time": time.time() - start_time
                }

    def process_imu_data(self, imu_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process IMU data for SLAM

        Args:
            imu_data: IMU data dictionary with orientation, angular velocity, linear acceleration

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        with self.lock:
            # Skip if SLAM is not active
            if not self.is_active:
                return {
                    "success": False,
                    "message": "SLAM system is not active",
                    "processing_time": 0.0
                }

            # Skip if not using IMU
            if not self.use_imu:
                return {
                    "success": False,
                    "message": "IMU data processing is disabled",
                    "processing_time": 0.0
                }

            # Store in buffer for potential future use
            self.imu_data_buffer.append({
                "data": imu_data,
                "timestamp": imu_data.get("timestamp", time.time())
            })

            # Limit buffer size
            if len(self.imu_data_buffer) > self.max_buffer_size:
                self.imu_data_buffer.pop(0)

            # Currently, our FastSLAM implementation doesn't use IMU directly
            # In a more sophisticated system, we would use IMU for motion prediction
            # or integrate with a more advanced sensor fusion system

            return {
                "success": True,
                "message": "IMU data stored (not used directly in current SLAM implementation)",
                "processing_time": time.time() - start_time
            }

    def get_current_pose(self) -> Tuple[np.ndarray, bool]:
        """
        Get the current estimated vehicle pose

        Returns:
            Tuple of (4x4 transformation matrix, valid flag)
        """
        with self.lock:
            if not self.is_active:
                return np.eye(4), False

            slam_pose = self.slam.get_pose()
            vehicle_pose = self._slam_to_vehicle_pose(slam_pose)

            return vehicle_pose, True

    def get_current_map(self) -> Tuple[MapManager, bool]:
        """
        Get the current map

        Returns:
            Tuple of (map manager, valid flag)
        """
        with self.lock:
            if not self.is_active:
                return None, False

            return self.slam.get_map(), True

    def get_pose_for_sensor_fusion(self) -> Dict[str, Any]:
        """
        Get pose information formatted for sensor fusion

        Returns:
            Dictionary with pose information for sensor fusion
        """
        with self.lock:
            if not self.is_active:
                return {
                    "valid": False,
                    "confidence": 0.0
                }

            # Get current pose
            slam_pose = self.slam.get_pose()
            vehicle_pose = self._slam_to_vehicle_pose(slam_pose)

            # Extract position, orientation, and velocity
            position = vehicle_pose[:3, 3]

            # Convert rotation matrix to quaternion
            rotation = vehicle_pose[:3, :3]
            r = Rotation.from_matrix(rotation)
            quat = r.as_quat()  # [qx, qy, qz, qw]

            # For velocity, we can estimate it from the last two poses
            # if we have a trajectory with more than one pose
            trajectory = self.slam.get_trajectory()
            velocity = np.zeros(3)

            if len(trajectory) > 1:
                prev_pose = trajectory[-2]
                curr_pose = trajectory[-1]

                # Position difference
                pos_diff = curr_pose[:3, 3] - prev_pose[:3, 3]

                # Get timestamp difference
                time_diff = 0.1  # Assume 10Hz if no timestamps available

                # Get timestamps from buffer if available
                if len(self.lidar_data_buffer) > 1:
                    curr_time = self.lidar_data_buffer[-1]["timestamp"]
                    prev_time = self.lidar_data_buffer[-2]["timestamp"]
                    time_diff = max(curr_time - prev_time, 0.001)

                # Calculate velocity
                velocity = pos_diff / time_diff

            # Calculate confidence based on SLAM stats
            stats = self.slam.get_stats()
            particle_ess = stats.get("effective_sample_ratio", 0.0)

            # Higher ESS ratio means more certain pose estimate
            confidence = min(0.9, particle_ess * 2.0)  # Scale to max 0.9

            # Create result for sensor fusion
            return {
                "valid": True,
                "timestamp": time.time(),
                "position": position,
                "orientation_quaternion": np.array([quat[3], quat[0], quat[1], quat[2]]),  # [w, x, y, z]
                "velocity": velocity,
                "confidence": confidence,
                "source": "slam"
            }

    def get_visualization_image(self, width: int = 800, height: int = 800) -> Optional[np.ndarray]:
        """
        Get visualization image of the SLAM system

        Args:
            width: Desired width of the output image
            height: Desired height of the output image

        Returns:
            BGR visualization image or None if SLAM is not active
        """
        with self.lock:
            if not self.is_active or self.viz_image is None:
                return None

            # Resize if needed
            if self.viz_image.shape[0] != height or self.viz_image.shape[1] != width:
                resized = cv2.resize(self.viz_image, (width, height))
                return resized

            return self.viz_image.copy()

    def get_cropped_map(self, center_x: float = 0.0, center_y: float = 0.0,
                        size: float = 20.0) -> Optional[np.ndarray]:
        """
        Get a cropped map visualization around a center point

        Args:
            center_x: Center X in world coordinates
            center_y: Center Y in world coordinates
            size: Size of the region in meters

        Returns:
            BGR visualization image or None if SLAM is not active
        """
        with self.lock:
            if not self.is_active:
                return None

            map_manager = self.slam.get_map()
            return map_manager.crop_to_region(center_x, center_y, size)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the SLAM system

        Returns:
            Dictionary with SLAM statistics
        """
        with self.lock:
            if not self.is_active:
                return {
                    "is_active": False,
                    "lidar_buffer_size": len(self.lidar_data_buffer),
                    "imu_buffer_size": len(self.imu_data_buffer)
                }

            # Get SLAM stats
            slam_stats = self.slam.get_stats()

            # Calculate processing time statistics
            processing_time_stats = {}
            if self.processing_times:
                processing_time_stats = {
                    "mean_time": np.mean(self.processing_times),
                    "max_time": np.max(self.processing_times),
                    "min_time": np.min(self.processing_times),
                    "std_time": np.std(self.processing_times),
                    "fps": 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0.0
                }

            return {
                "is_active": True,
                "use_lidar": self.use_lidar,
                "use_imu": self.use_imu,
                "lidar_buffer_size": len(self.lidar_data_buffer),
                "imu_buffer_size": len(self.imu_data_buffer),
                "processing_time_stats": processing_time_stats,
                "slam_stats": slam_stats
            }

    def _slam_to_vehicle_pose(self, slam_pose: np.ndarray) -> np.ndarray:
        """
        Convert SLAM pose to vehicle pose

        Args:
            slam_pose: 4x4 transformation matrix in SLAM frame

        Returns:
            4x4 transformation matrix in vehicle frame
        """
        # SLAM pose is in LiDAR frame, convert to vehicle frame
        # vehicle_pose = slam_pose @ self.lidar_to_vehicle_transform
        vehicle_pose = slam_pose
        return vehicle_pose

    def _vehicle_to_slam_pose(self, vehicle_pose: np.ndarray) -> np.ndarray:
        """
        Convert vehicle pose to SLAM pose

        Args:
            vehicle_pose: 4x4 transformation matrix in vehicle frame

        Returns:
            4x4 transformation matrix in SLAM frame
        """
        # Vehicle pose is in vehicle frame, convert to LiDAR frame
        # slam_pose = vehicle_pose @ np.linalg.inv(self.lidar_to_vehicle_transform)
        slam_pose = vehicle_pose
        return slam_pose