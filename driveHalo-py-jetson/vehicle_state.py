import logging
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class VehicleState:
    """Centralized representation of vehicle state"""
    # Timestamps
    timestamp: float = 0.0
    camera_timestamp: float = 0.0
    imu_timestamp: float = 0.0
    control_timestamp: float = 0.0

    # Position and orientation
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Motion
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    angular_velocity_x: float = 0.0
    angular_velocity_y: float = 0.0
    angular_velocity_z: float = 0.0

    # Lane properties
    lane_curvature: float = float('inf')
    lane_width: float = 3.5
    lane_offset: float = 0.0  # Lateral offset from lane center
    lane_heading_error: float = 0.0  # Heading error relative to lane

    # Control signals
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    # Status flags
    control_active: bool = False
    emergency_stop: bool = False
    temporary_brake_release: bool = False

    # Sensor confidence
    camera_confidence: float = 0.0
    imu_confidence: float = 0.0

    # System status
    system_status: str = "INITIALIZING"  # INITIALIZING, ACTIVE, ERROR, SHUTDOWN
    error_message: str = ""


class VehicleStateManager:
    """Thread-safe manager for vehicle state"""

    def __init__(self):
        self.logger = logging.getLogger("VehicleStateManager")
        self.logger.info("Initializing vehicle state manager")

        # Initialize state
        self.state = VehicleState()

        # Thread safety
        self.lock = threading.RLock()

        # State history for debugging and analysis
        self.history_size = 1000
        self.state_history = []

        # Callbacks for state changes
        self.state_update_callbacks = []

        self.logger.info("Vehicle state manager initialized")

    def update_from_camera(self, camera_data: Dict[str, Any]) -> None:
        """Update state with camera-derived data"""
        with self.lock:
            # Extract camera data
            self.state.camera_timestamp = camera_data.get("timestamp", time.time())
            self.state.lane_curvature = camera_data.get("curvature_radius", float('inf'))
            self.state.lane_offset = camera_data.get("lateral_offset", 0.0)
            self.state.lane_heading_error = camera_data.get("heading_error", 0.0)
            self.state.camera_confidence = camera_data.get("confidence", 0.0)

            # Update global timestamp
            self.state.timestamp = time.time()

            # Record state change and notify callbacks
            self._record_state_change()

    def update_from_imu(self, imu_data: Dict[str, Any]) -> None:
        """Update state with IMU data"""
        with self.lock:
            # Extract IMU data
            self.state.imu_timestamp = imu_data.get("timestamp", time.time())

            # Orientation
            self.state.roll = imu_data.get("roll", self.state.roll)
            self.state.pitch = imu_data.get("pitch", self.state.pitch)
            self.state.yaw = imu_data.get("yaw", self.state.yaw)

            # Angular velocity
            angular_velocity = imu_data.get("angular_velocity", [0, 0, 0])
            self.state.angular_velocity_x = angular_velocity[0]
            self.state.angular_velocity_y = angular_velocity[1]
            self.state.angular_velocity_z = angular_velocity[2]

            # Linear acceleration is not stored in state directly
            # but could be used to update velocity with proper integration

            self.state.imu_confidence = imu_data.get("confidence", 0.0)

            # Update global timestamp
            self.state.timestamp = time.time()

            # Record state change and notify callbacks
            self._record_state_change()

    def update_from_sensor_fusion(self, fusion_data: Dict[str, Any]) -> None:
        """Update state with sensor fusion data"""
        with self.lock:
            # Extract fused state
            position = fusion_data.get("position", [0, 0, 0])
            self.state.position_x = position[0]
            self.state.position_y = position[1]
            self.state.position_z = position[2]

            velocity = fusion_data.get("velocity", [0, 0, 0])
            self.state.velocity_x = velocity[0]
            self.state.velocity_y = velocity[1]
            self.state.velocity_z = velocity[2]

            self.state.yaw = fusion_data.get("heading", self.state.yaw)
            self.state.lane_curvature = fusion_data.get("lane_curvature", self.state.lane_curvature)

            # Update global timestamp
            self.state.timestamp = time.time()

            # Record state change and notify callbacks
            self._record_state_change()

    def update_control_signals(self, steering: float, throttle: float, brake: float) -> None:
        """Update control signals"""
        with self.lock:
            self.state.steering = steering
            self.state.throttle = throttle
            self.state.brake = brake
            self.state.control_timestamp = time.time()
            self.state.timestamp = time.time()

            # Record state change and notify callbacks
            self._record_state_change()

    def update_system_status(self, status: str, error_message: str = "") -> None:
        """Update system status"""
        with self.lock:
            self.state.system_status = status
            self.state.error_message = error_message
            self.state.timestamp = time.time()

            # Log status changes
            self.logger.info(f"System status changed to {status}")
            if error_message:
                self.logger.error(f"Error: {error_message}")

            # Record state change and notify callbacks
            self._record_state_change()

    def get_state(self) -> VehicleState:
        """Get current vehicle state"""
        with self.lock:
            # Return a copy to prevent external modification
            return VehicleState(**vars(self.state))

    def register_state_update_callback(self, callback) -> int:
        """Register a callback for state updates"""
        with self.lock:
            self.state_update_callbacks.append(callback)
            return len(self.state_update_callbacks) - 1

    def unregister_state_update_callback(self, callback_id: int) -> bool:
        """Unregister a state update callback"""
        with self.lock:
            if 0 <= callback_id < len(self.state_update_callbacks):
                self.state_update_callbacks.pop(callback_id)
                return True
            return False

    def _record_state_change(self) -> None:
        """Record state in history and trigger callbacks"""
        # Create a copy of the current state
        state_copy = VehicleState(**vars(self.state))

        # Add to history
        self.state_history.append(state_copy)

        # Limit history size
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)

        # Call registered callbacks
        for callback in self.state_update_callbacks:
            try:
                callback(state_copy)
            except Exception as e:
                self.logger.error(f"Error in state update callback: {e}")

    def get_historical_states(self, count: int = None) -> List[VehicleState]:
        """Get historical states"""
        with self.lock:
            if count is None:
                return self.state_history.copy()
            else:
                return self.state_history[-count:].copy()

    def get_state_for_control(self) -> Tuple[float, float, float]:
        """Get relevant state variables for control decisions"""
        with self.lock:
            return (
                self.state.lane_curvature,
                self.state.lane_offset,
                self.state.lane_heading_error
            )