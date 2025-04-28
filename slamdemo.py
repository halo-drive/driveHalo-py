#!/usr/bin/env python3
"""
Example usage of the SLAM system integrated with the vehicle's sensor systems
"""

import numpy as np
import cv2
import time
import argparse
import logging
import threading
import os
from typing import Dict, Any

# Import SLAM components
from fast_slam import FastSLAM
from map_manager import MapManager
from point_cloud_processor import PointCloudProcessor
from pose_graph import PoseGraph
from slam_integration import SLAMIntegration

# Import original codebase components
from ros_interface import ROSInterface
from imu_processor import IMUProcessor
from sensor_fusion import SensorFusion
from vehicle_state import VehicleStateManager


class SLAMDemo:
    """
    Demo class showing how to use the SLAM system with the vehicle's sensors
    """

    def __init__(self, use_ros: bool = True, log_level: int = logging.INFO):
        """
        Initialize the SLAM demo

        Args:
            use_ros: Whether to use ROS for data input
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        self.logger = logging.getLogger("SLAMDemo")
        self.logger.info("Initializing SLAM demo")

        # Initialize SLAM integration
        self.slam_integration = SLAMIntegration(
            use_lidar=True,
            use_imu=True,
            map_resolution=0.1,
            map_size=(2000, 2000),
            num_particles=30
        )

        # Initialize original codebase components
        self.vehicle_state = VehicleStateManager()
        self.sensor_fusion = SensorFusion()
        self.imu_processor = IMUProcessor()

        # ROS interface for Livox LiDAR
        self.use_ros = use_ros
        if use_ros:
            import rospy
            from sensor_msgs.msg import PointCloud2
            import sensor_msgs.point_cloud2 as pc2
            self.lidar_sub = rospy.Subscriber(
                '/livox/lidar',
                PointCloud2,
                self._lidar_callback,
                queue_size=5
            )
            self.logger.info(f"Subscribing to /livox/lidar")
            self.ros_interface = ROSInterface("slam_demo")
            # Register ROS callbacks
            self.ros_interface.register_imu_callback(self._imu_callback)
            # For LiDAR point cloud, we need to set up the ROS subscriber separately
            # since it's not part of the original codebase
        else:
            self.ros_interface = None

        # State flags
        self.is_running = False
        self.thread = None

        # Create output directory
        os.makedirs("slam_output", exist_ok=True)

        self.logger.info("SLAM demo initialized")

    def _lidar_callback(self, pointcloud_msg):
        """
        Callback for LiDAR data from ROS

        Args:
            pointcloud_msg: ROS PointCloud2 message
        """
        try:
            import sensor_msgs.point_cloud2 as pc2

            # Convert PointCloud2 to numpy array
            points_list = []
            intensities_list = []

            # Extract x,y,z points and intensity if available
            for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
                if len(point) > 3:  # If intensity is available
                    intensities_list.append(point[3])

            if not points_list:
                self.logger.warning("Received empty point cloud")
                return

            points = np.array(points_list)
            intensities = np.array(intensities_list) if intensities_list else None

            # Process with SLAM
            self.logger.debug(f"Processing point cloud with {len(points)} points")
            result = self._process_lidar_data(points, intensities, pointcloud_msg.header.stamp.to_sec())

            if result["success"]:
                self.logger.debug("Successfully processed LiDAR scan")

        except Exception as e:
            self.logger.error(f"Error in LiDAR callback: {e}")

    def _imu_callback(self, imu_msg):
        """
        Callback for IMU data from ROS

        Args:
            imu_msg: ROS IMU message
        """
        try:
            # Process IMU data with original IMU processor
            imu_data = self.imu_processor.process_imu_data(imu_msg)

            # Update vehicle state
            self.vehicle_state.update_from_imu(imu_data)

            # Update sensor fusion
            self.sensor_fusion.update_from_imu(imu_data)

            # Process with SLAM integration
            self.slam_integration.process_imu_data(imu_data)

        except Exception as e:
            self.logger.error(f"Error in IMU callback: {e}")

    def _process_lidar_data(self, points, intensities=None, timestamp=None):
        """
        Process LiDAR data with SLAM integration

        Args:
            points: Nx3 array of point coordinates
            intensities: N array of intensities (optional)
            timestamp: Data timestamp (optional)

        Returns:
            Dictionary with processing results
        """
        try:
            # Process with SLAM integration
            result = self.slam_integration.process_lidar_data(points, intensities, timestamp)

            if result["success"]:
                # Get pose from SLAM
                slam_pose_data = self.slam_integration.get_pose_for_sensor_fusion()

                # Update sensor fusion if pose is valid
                if slam_pose_data["valid"]:
                    # Create synthetic LiDAR odometry data for sensor fusion
                    odom_data = {
                        "position": slam_pose_data["position"],
                        "orientation_quaternion": slam_pose_data["orientation_quaternion"],
                        "velocity": slam_pose_data["velocity"],
                        "timestamp": slam_pose_data["timestamp"],
                        "confidence": slam_pose_data["confidence"]
                    }

                    # Update sensor fusion (assuming a method for LiDAR odometry exists)
                    # In practice, you would integrate this into the sensor fusion pipeline
                    if hasattr(self.sensor_fusion, 'update_from_lidar_odom'):
                        self.sensor_fusion.update_from_lidar_odom(odom_data)

                    # Update vehicle state
                    self.vehicle_state.update_from_sensor_fusion({
                        "position": slam_pose_data["position"],
                        "velocity": slam_pose_data["velocity"],
                        "heading": np.arctan2(
                            slam_pose_data["orientation_quaternion"][2],
                            slam_pose_data["orientation_quaternion"][0]
                        )
                    })

            return result

        except Exception as e:
            self.logger.error(f"Error processing LiDAR data: {e}")
            return {
                "success": False,
                "message": f"Error processing LiDAR data: {e}"
            }

    def start(self):
        """Start the SLAM demo"""
        if self.is_running:
            self.logger.warning("SLAM demo is already running")
            return False

        self.logger.info("Starting SLAM demo")

        # Start SLAM integration
        self.slam_integration.start()

        # Start ROS interface if using ROS
        if self.use_ros:
            self.ros_interface.start(imu_topic="/livox/imu")

        # Start processing thread
        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        self.logger.info("SLAM demo started")
        return True

    def stop(self):
        """Stop the SLAM demo"""
        if not self.is_running:
            self.logger.warning("SLAM demo is not running")
            return False

        self.logger.info("Stopping SLAM demo")

        # Stop processing thread
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        # Stop SLAM integration
        self.slam_integration.stop()

        # Stop ROS interface if using ROS
        if self.use_ros and self.ros_interface:
            if hasattr(self, 'lidar_sub'):
                self.lidar_sub.unregister()
            self.ros_interface.stop()

        self.logger.info("SLAM demo stopped")
        return True

    def _run(self):
        """Main processing loop"""
        # For the demo, we'll simulate some LiDAR data if not using ROS
        if not self.use_ros:
            self._simulate_data()
        else:
            # In a real system, LiDAR data would come from ROS callbacks
            # For now, we'll just wait for termination
            while self.is_running:
                time.sleep(0.1)

        self.logger.info("Processing thread terminated")

    def _simulate_data(self):
        """Simulate LiDAR data for demo purposes"""
        self.logger.info("Simulating LiDAR data")

        # Simple circular trajectory
        radius = 5.0
        total_steps = 400
        points_per_scan = 1000
        noise_level = 0.05

        for step in range(total_steps):
            if not self.is_running:
                break

            # Generate position on a circular path
            angle = step * 2 * np.pi / total_steps
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Generate a simulated point cloud around that position
            scan_points = []
            for i in range(points_per_scan):
                # Random angle and distance for point
                pt_angle = np.random.uniform(0, 2 * np.pi)
                pt_dist = np.random.uniform(0.5, 15.0)

                # Point coordinates
                pt_x = x + pt_dist * np.cos(pt_angle) + np.random.normal(0, noise_level)
                pt_y = y + pt_dist * np.sin(pt_angle) + np.random.normal(0, noise_level)
                pt_z = np.random.normal(0, noise_level)

                scan_points.append([pt_x, pt_y, pt_z])

            # Process simulated point cloud
            points_array = np.array(scan_points)
            intensities = np.random.uniform(0, 1, len(scan_points))

            result = self._process_lidar_data(points_array, intensities, time.time())

            if result["success"]:
                self.logger.debug(f"Processed simulated scan {step}/{total_steps}")
            else:
                self.logger.warning(f"Failed to process simulated scan: {result['message']}")

            # Save map visualization every 50 steps
            if step % 50 == 0:
                viz_img = self.slam_integration.get_visualization_image()
                if viz_img is not None:
                    cv2.imwrite(f"slam_output/slam_map_{step:04d}.png", viz_img)
                    self.logger.info(f"Saved map visualization at step {step}")

            # Optimization and visualization runs at a lower frequency
            if step % 20 == 0:
                # Record SLAM statistics
                stats = self.slam_integration.get_stats()
                self.logger.info(f"SLAM stats: "
                                 f"Particle count: {stats['slam_stats']['particle_count']}, "
                                 f"Distance traveled: {stats['slam_stats']['distance_traveled']:.2f}m, "
                                 f"Trajectory length: {stats['slam_stats']['trajectory_length']}")

            # Sleep to simulate sensor rate
            time.sleep(0.1)

        self.logger.info("Simulation completed")

    def save_map(self, filename: str = "slam_output/final_map.png"):
        """
        Save the final map visualization

        Args:
            filename: Output filename
        """
        viz_img = self.slam_integration.get_visualization_image()
        if viz_img is not None:
            cv2.imwrite(filename, viz_img)
            self.logger.info(f"Saved final map to {filename}")
            return True
        else:
            self.logger.warning("Failed to save map: No visualization available")
            return False

    def save_stats(self, filename: str = "slam_output/slam_stats.txt"):
        """
        Save SLAM statistics to a file

        Args:
            filename: Output filename
        """
        try:
            stats = self.slam_integration.get_stats()

            with open(filename, 'w') as f:
                f.write("SLAM Statistics:\n")
                f.write("=" * 50 + "\n")

                f.write(f"Active: {stats['is_active']}\n")
                f.write(f"Using LiDAR: {stats['use_lidar']}\n")
                f.write(f"Using IMU: {stats['use_imu']}\n")
                f.write(f"LiDAR buffer size: {stats['lidar_buffer_size']}\n")
                f.write(f"IMU buffer size: {stats['imu_buffer_size']}\n")

                if 'processing_time_stats' in stats:
                    f.write("\nProcessing Time Statistics:\n")
                    for key, value in stats['processing_time_stats'].items():
                        f.write(f"  {key}: {value}\n")

                if 'slam_stats' in stats:
                    f.write("\nSLAM System Statistics:\n")
                    for key, value in stats['slam_stats'].items():
                        if key != 'map_stats' and key != 'pose_graph_stats':
                            f.write(f"  {key}: {value}\n")

                    if 'map_stats' in stats['slam_stats']:
                        f.write("\nMap Statistics:\n")
                        for key, value in stats['slam_stats']['map_stats'].items():
                            f.write(f"  {key}: {value}\n")

                    if 'pose_graph_stats' in stats['slam_stats']:
                        f.write("\nPose Graph Statistics:\n")
                        for key, value in stats['slam_stats']['pose_graph_stats'].items():
                            f.write(f"  {key}: {value}\n")

            self.logger.info(f"Saved statistics to {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
            return False


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='SLAM Demo')
    parser.add_argument('--use-ros', action='store_true', help='Use ROS for data input')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--output-dir', type=str, default='slam_output',
                        help='Output directory for maps and statistics')

    args = parser.parse_args()

    # Configure log level
    log_level = getattr(logging, args.log_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize and run the demo
    demo = SLAMDemo(use_ros=args.use_ros, log_level=log_level)

    try:
        # Start the demo
        demo.start()

        # In a real application, we would wait for user input to stop
        # For this demo, we'll just run for a fixed amount of time
        print("Running SLAM demo for 60 seconds...")
        for i in range(60):
            time.sleep(1)
            print(f"Time elapsed: {i + 1}s/60s", end='\r')
        print("\nDemo completed")

        # Save final outputs
        demo.save_map(os.path.join(args.output_dir, "final_map.png"))
        demo.save_stats(os.path.join(args.output_dir, "slam_stats.txt"))

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        # Stop the demo
        demo.stop()
        print("Demo stopped, exiting")


if __name__ == "__main__":
    main()