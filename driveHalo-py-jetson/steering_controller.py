import numpy as np
import logging
import time
import asyncio
from typing import Tuple, Optional, Dict, Any

from utils import measure_execution_time


class SteeringController:
    """Steering control logic with optimized calculations"""

    def __init__(self):
        self.logger = logging.getLogger("SteeringController")

        # Steering control parameters
        self.max_torque = 0.1
        self.min_curvature_radius = 1.0
        self.torque_smoothing = 0.3
        self.prev_torque = 0.0

        #Lateral control parameters
        self.lateral_gain = 0.7
        self.lateral_deadband = 0.1

        # Throttle control parameters
        self.max_throttle = 0.000
        self.min_throttle = 0.000
        self.throttle_smoothing = 0.00
        self.prev_throttle = 0.0

        # Brake control parameters
        self.permanent_brake_force = 0.1

        self.brake_threshold_radius = 11.0
        self.max_brake_force = 0.5
        self.is_braking = True

        # Temporary brake release parameters
        self.tight_turn_threshold = 9.0  # Radius threshold for temp release
        self.temp_brake_release_duration = 2.0
        self.temp_release_start_time = 0.0
        self.is_temp_release_active = False
        self.min_time_between_releases = 4.0
        self.last_temp_release_time = 0.0

        self.logger.info("Steering controller initialized with max torque: %.2f", self.max_torque)

    def calculate_torque(self, radius: float) -> float:
        """Calculate steering torque based on curve radius"""
        if abs(radius) > 1000.0 or radius == float('inf'):
            return 0.0

        direction = 1.0 if radius > 0 else -1.0
        radius = abs(radius)

        if radius < self.min_curvature_radius:
            torque = self.max_torque
        else:
            # Exponential decay of torque with increasing radius
            torque = self.max_torque * np.exp(-(radius - self.min_curvature_radius) / 90.0)

        # Apply smoothing
        torque = (self.torque_smoothing * self.prev_torque +
                  (1.0 - self.torque_smoothing) * torque)
        self.prev_torque = torque

        return direction * min(abs(torque), self.max_torque)

    async def calculate_throttle(self, curve_radius: float) -> float:
        """Calculate throttle based on curve radius"""
        if abs(curve_radius) < self.brake_threshold_radius:
            throttle = self.min_throttle
        else:
            # Linear increase in throttle as radius increases above threshold
            throttle = min(
                self.max_throttle,
                self.min_throttle + (self.max_throttle - self.min_throttle) *
                (abs(curve_radius) - self.brake_threshold_radius) / 20.0
            )

        # Apply smoothing
        throttle = (self.throttle_smoothing * self.prev_throttle +
                    (1.0 - self.throttle_smoothing) * throttle)
        self.prev_throttle = throttle

        return throttle

    @measure_execution_time
    async def coordinate_controls(self, curve_radius: float, lateral_offset: float = 0.0) -> Tuple[float, float, float]:
        """Coordinate steering, throttle, and brake controls"""
        current_time = time.time()

        # Check for tight turn condition
        if (abs(curve_radius) < self.tight_turn_threshold and
                current_time - self.last_temp_release_time > self.min_time_between_releases and
                not self.is_temp_release_active):
            # Initiate temporary brake release
            self.is_temp_release_active = True
            self.temp_release_start_time = current_time
            self.last_temp_release_time = current_time
            self.logger.info(f"Temporary brake release for tight turn (radius: {curve_radius:.1f}m)")

            # Calculate strong steering input and lateral correction for tight turn
            curve_steering = self.calculate_torque(curve_radius)
            lateral_correction = 0.0
            if abs(lateral_offset) > self.lateral_deadband:
                lateral_correction = self.lateral_gain * lateral_offset

            steering = np.clip(curve_steering + lateral_correction, -self.max_torque, self.max_torque)
            throttle = await self.calculate_throttle(curve_radius)
            self.logger.debug(
                f"Steering: curve={curve_steering:.3f}, lateral={lateral_correction:.3f}, final={steering:.3f}")
            return (steering, throttle, self.permanent_brake_force)  # Release brakes completely

        # Check if we're in an active temporary release
        if self.is_temp_release_active:
            release_duration = current_time - self.temp_release_start_time

            if release_duration >= self.temp_brake_release_duration:
                self.is_temp_release_active = False
                self.logger.info("Ending temporary brake release")
                return (0.0, 0.0, self.permanent_brake_force)
            else:
                # During temporary release, allow steering but no throttle
                steering = self.calculate_torque(curve_radius)
                return (steering, 0.0, 0.0)

        # Normal operation
        steering = self.calculate_torque(curve_radius)
        throttle = await self.calculate_throttle(curve_radius)
        return (steering, throttle, self.permanent_brake_force)