#!/usr/bin/env python3
import rospy
import numpy as np
import threading
import logging
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
from typing import Callable, Dict, Any, Optional


class ROSInterface:
    """Interface for ROS communication with Livox LiDAR and other sensors"""

    def __init__(self, node_name: str = "autonomous_system"):
        self.logger = logging.getLogger("ROSInterface")

        # Initialize ROS node (with anonymous=True to avoid conflicts)
        try:
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
            self.logger.info(f"ROS node '{node_name}' initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS node: {e}")
            raise

        # Latest data storage
        self.latest_imu_data = None
        self.latest_odom_data = None
        self.latest_twist_data = None

        # Callback registration storage
        self.imu_callbacks = []
        self.odom_callbacks = []
        self.twist_callbacks = []

        # Threading lock for thread-safe data access
        self.lock = threading.RLock()

        # ROS Subscribers
        self.imu_sub = None
        self.odom_sub = None
        self.twist_sub = None

        self.is_running = True
        self.logger.info("ROSInterface ready")

    def start(self, imu_topic: str = "/livox/imu",
              odom_topic: Optional[str] = None,
              twist_topic: Optional[str] = None):
        """Start ROS subscribers for specified topics"""
        try:
            # IMU data from Livox
            self.imu_sub = rospy.Subscriber(
                imu_topic,
                Imu,
                self._imu_callback,
                queue_size=10
            )
            self.logger.info(f"Subscribed to IMU topic: {imu_topic}")

            # Optional odometry data
            if odom_topic:
                self.odom_sub = rospy.Subscriber(
                    odom_topic,
                    Odometry,
                    self._odom_callback,
                    queue_size=10
                )
                self.logger.info(f"Subscribed to Odometry topic: {odom_topic}")

            # Optional twist (velocity) data
            if twist_topic:
                self.twist_sub = rospy.Subscriber(
                    twist_topic,
                    TwistWithCovarianceStamped,
                    self._twist_callback,
                    queue_size=10
                )
                self.logger.info(f"Subscribed to Twist topic: {twist_topic}")

            # Start a thread for ROS spin
            self.spin_thread = threading.Thread(target=self._ros_spin)
            self.spin_thread.daemon = True
            self.spin_thread.start()

            self.logger.info("ROS interface started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start ROS interface: {e}")
            self.stop()
            raise

    def _ros_spin(self):
        """Run ROS spin in a separate thread"""
        rate = rospy.Rate(100)  # 100Hz spin rate
        while self.is_running and not rospy.is_shutdown():
            rate.sleep()

    def _imu_callback(self, msg: Imu):
        """Process incoming IMU messages"""
        with self.lock:
            self.latest_imu_data = msg

            # Process registered callbacks
            for callback in self.imu_callbacks:
                try:
                    callback(msg)
                except Exception as e:
                    self.logger.error(f"Error in IMU callback: {e}")

    def _odom_callback(self, msg: Odometry):
        """Process incoming Odometry messages"""
        with self.lock:
            self.latest_odom_data = msg

            # Process registered callbacks
            for callback in self.odom_callbacks:
                try:
                    callback(msg)
                except Exception as e:
                    self.logger.error(f"Error in Odometry callback: {e}")

    def _twist_callback(self, msg: TwistWithCovarianceStamped):
        """Process incoming Twist messages"""
        with self.lock:
            self.latest_twist_data = msg

            # Process registered callbacks
            for callback in self.twist_callbacks:
                try:
                    callback(msg)
                except Exception as e:
                    self.logger.error(f"Error in Twist callback: {e}")

    def register_imu_callback(self, callback: Callable[[Imu], None]):
        """Register a callback for IMU data"""
        with self.lock:
            self.imu_callbacks.append(callback)
        return len(self.imu_callbacks) - 1  # Return index for potential deregistration

    def register_odom_callback(self, callback: Callable[[Odometry], None]):
        """Register a callback for Odometry data"""
        with self.lock:
            self.odom_callbacks.append(callback)
        return len(self.odom_callbacks) - 1

    def register_twist_callback(self, callback: Callable[[TwistWithCovarianceStamped], None]):
        """Register a callback for Twist data"""
        with self.lock:
            self.twist_callbacks.append(callback)
        return len(self.twist_callbacks) - 1

    def get_latest_imu_data(self) -> Optional[Imu]:
        """Get the latest IMU data thread-safely"""
        with self.lock:
            return self.latest_imu_data

    def get_latest_odom_data(self) -> Optional[Odometry]:
        """Get the latest Odometry data thread-safely"""
        with self.lock:
            return self.latest_odom_data

    def get_latest_twist_data(self) -> Optional[TwistWithCovarianceStamped]:
        """Get the latest Twist data thread-safely"""
        with self.lock:
            return self.latest_twist_data

    def stop(self):
        """Stop all ROS subscribers and clean up"""
        self.is_running = False

        # Unregister subscribers
        if self.imu_sub:
            self.imu_sub.unregister()
        if self.odom_sub:
            self.odom_sub.unregister()
        if self.twist_sub:
            self.twist_sub.unregister()

        self.logger.info("ROS interface stopped")

    def __del__(self):
        """Destructor to ensure clean shutdown"""
        self.stop()