import numpy as np
import open3d as o3d
import logging
import threading
import time
import math
import cv2
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation
import random

from point_cloud_processor import PointCloudProcessor
from map_manager import MapManager
from pose_graph import PoseGraph


class Particle:
    """
    Particle for FastSLAM algorithm with pose and weight
    """

    def __init__(self, pose: np.ndarray = None, weight: float = 1.0):
        """
        Initialize a particle

        Args:
            pose: 4x4 transformation matrix (default: identity)
            weight: Particle weight (default: 1.0)
        """
        self.pose = np.eye(4) if pose is None else pose.copy()
        self.weight = weight
        self.trajectory = [self.pose.copy()]
        self.local_map = None  # For particle-specific map (optional)
        self.last_visualization_time = 0.0

    def get_position(self) -> np.ndarray:
        """Get the position part of the pose"""
        return self.pose[:3, 3]

    def get_rotation(self) -> np.ndarray:
        """Get the rotation part of the pose"""
        return self.pose[:3, :3]

    def get_yaw(self) -> float:
        """Get the yaw angle in radians"""
        r = Rotation.from_matrix(self.get_rotation())
        euler = r.as_euler('xyz')
        return euler[2]

    def set_pose(self, pose: np.ndarray):
        """Set the particle pose"""
        self.pose = pose.copy()
        self.trajectory.append(self.pose.copy())

        # Limit trajectory size
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-1000:]

    def update_pose(self, delta_pose: np.ndarray, noise_scale: float = 0.1):
        """
        Update the particle pose with added noise

        Args:
            delta_pose: 4x4 transformation matrix representing the pose change
            noise_scale: Scale factor for motion noise
        """
        # Add noise to delta_pose
        noisy_delta = self._add_noise_to_transform(delta_pose, noise_scale)

        # Apply the noisy motion
        self.pose = self.pose @ noisy_delta
        self.trajectory.append(self.pose.copy())

        # Limit trajectory size
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-1000:]

    def _add_noise_to_transform(self, transform: np.ndarray, scale: float) -> np.ndarray:
        """
        Add Gaussian noise to a transformation matrix

        Args:
            transform: 4x4 transformation matrix
            scale: Noise scale factor

        Returns:
            Noisy transformation matrix
        """
        # Extract translation and rotation
        translation = transform[:3, 3]
        rotation = transform[:3, :3]

        # Add noise to translation
        translation_noise = np.random.normal(0, scale, 3)
        noisy_translation = translation + translation_noise

        # Add noise to rotation (convert to euler, add noise, convert back)
        r = Rotation.from_matrix(rotation)
        euler = r.as_euler('xyz')
        euler_noise = np.random.normal(0, scale * 0.1, 3)  # Less noise for rotation
        noisy_euler = euler + euler_noise
        noisy_rotation = Rotation.from_euler('xyz', noisy_euler).as_matrix()

        # Reconstruct transformation matrix
        noisy_transform = np.eye(4)
        noisy_transform[:3, :3] = noisy_rotation
        noisy_transform[:3, 3] = noisy_translation

        return noisy_transform


class FastSLAM:
    """
    FastSLAM implementation for simultaneous localization and mapping

    This implements FastSLAM 2.0 algorithm with a particle filter for pose estimation
    and a global map for mapping.
    """

    def __init__(self,
                 num_particles: int = 30,
                 resample_threshold: float = 0.5,
                 motion_noise: float = 0.1,
                 map_resolution: float = 0.1,
                 map_size: Tuple[int, int] = (2000, 2000)):
        """
        Initialize the FastSLAM system

        Args:
            num_particles: Number of particles in the filter
            resample_threshold: Threshold for effective sample size ratio to trigger resampling
            motion_noise: Scale factor for motion noise
            map_resolution: Resolution of the map in meters per cell
            map_size: Size of the map in cells (width, height)
        """
        self.logger = logging.getLogger("FastSLAM")
        self.logger.info("Initializing FastSLAM")

        # Algorithm parameters
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.motion_noise = motion_noise

        # Initialize particles
        self.particles = [Particle() for _ in range(num_particles)]
        self.best_particle_idx = 0

        # Thread safety
        self.lock = threading.RLock()

        # Initialize components
        self.point_cloud_processor = PointCloudProcessor()
        self.map = MapManager(
            resolution=map_resolution,
            width=map_size[0],
            height=map_size[1],
            initial_position=(map_size[0] // 2, map_size[1] // 2)
        )
        self.pose_graph = PoseGraph()
        self.last_visualization_time = 0.0
        # Global trajectory
        self.global_trajectory = [np.eye(4)]
        self.last_pose = np.eye(4)
        self.current_pose = np.eye(4)

        # Loop closure detection
        self.keyframes = []
        self.keyframe_features = []
        self.last_keyframe_time = 0
        self.keyframe_interval = 1.0  # seconds
        self.min_loop_closure_distance = 3.0  # meters

        # ICP parameters for scan matching
        self.icp_max_distance = 0.5
        self.icp_max_iterations = 30
        self.use_icp_fine_tuning = True

        # Edge cases
        self.last_scan_time = 0
        self.min_scan_interval = 0.05  # seconds
        self.min_translation = 0.05  # meters
        self.min_rotation = 0.05  # radians
        self.processing_first_scan = True

        # Add the first node to the pose graph
        self.pose_graph.add_node(np.eye(4), is_fixed=True)

        self.logger.info(f"FastSLAM initialized with {num_particles} particles")

    def reset(self):
        """Reset the SLAM system to initial state"""
        with self.lock:
            # Reset particles
            self.particles = [Particle() for _ in range(self.num_particles)]
            self.best_particle_idx = 0

            # Reset trajectory
            self.global_trajectory = [np.eye(4)]
            self.last_pose = np.eye(4)
            self.current_pose = np.eye(4)

            # Reset components
            self.map = MapManager(
                resolution=self.map.resolution,
                width=self.map.width,
                height=self.map.height,
                initial_position=(self.map.width // 2, self.map.height // 2)
            )
            self.pose_graph = PoseGraph()
            self.pose_graph.add_node(np.eye(4), is_fixed=True)

            # Reset loop closure detection
            self.keyframes = []
            self.keyframe_features = []
            self.last_keyframe_time = 0

            # Reset other state variables
            self.last_scan_time = 0
            self.processing_first_scan = True

            self.logger.info("FastSLAM system reset")

    def process_scan(self, points: np.ndarray, intensities: Optional[np.ndarray] = None,
                     timestamp: float = None) -> Dict[str, Any]:
        """
        Process a LiDAR scan for SLAM

        Args:
            points: Nx3 array of point coordinates (x, y, z)
            intensities: N array of point intensities (optional)
            timestamp: Scan timestamp (seconds, optional)

        Returns:
            Dictionary with processed data
        """
        start_time = time.time()
        if not isinstance(points, np.ndarray) or not points.flags.writeable:
            self.logger.debug("Converting points to writeable array in FastSLAM")
            points = np.array(points, copy=True)

        if intensities is not None:
            if not isinstance(intensities, np.ndarray) or not intensities.flags.writeable:
                intensities = np.array(intensities, copy=True)

        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = start_time

        with self.lock:
            # Check if enough time has passed since last scan
            if timestamp - self.last_scan_time < self.min_scan_interval and not self.processing_first_scan:
                self.logger.debug(f"Skipping scan: too soon after last scan ({timestamp - self.last_scan_time:.3f}s)")
                return {
                    "success": False,
                    "message": "Too soon after last scan",
                    "current_pose": self.current_pose,
                    "processing_time": 0.0
                }
            # Add static optimization: detect if scanner hasnt moved using IMU
            is_static = self._detect_static_state()
            if is_static and not self.processing_first_scan:
                # if static, we update visualization occasionaly
                if (timestamp - self.last_visualization_time) < 5.0:  # Update viz every 5s when static
                    return {
                        "success": True,
                        "message": "Static state, skipping processing",
                        "current_pose": self.current_pose,
                        "processing_time": time.time() - start_time
                    }
                self.last_visualization_time = timestamp

            # Process the point cloud
            scan_result = self.point_cloud_processor.process_scan(points, intensities)

            if not scan_result["success"]:
                self.logger.warning(f"Point cloud processing failed: {scan_result['message']}")
                return {
                    "success": False,
                    "message": scan_result["message"],
                    "current_pose": self.current_pose,
                    "processing_time": time.time() - start_time
                }

            # First scan is just for initialization
            if self.processing_first_scan:
                self.processing_first_scan = False
                self.last_scan_time = timestamp
                self.logger.debug(f"Fist scan processing, please wait, getting ready for mapping")
                return {
                    "success": True,
                    "message": "First scan processed",
                    "current_pose": self.current_pose,
                    "processing_time": time.time() - start_time
                }
                self.logger.debug(f"First scan completed, system ready for mapping")

            # Get the processed point cloud
            pcd = scan_result["pcd"]

            # Get transformation from scan matching
            scan_transform = scan_result["transformation"]
            scan_fitness = scan_result["fitness"]

            # Skip update if scan matching failed
            if scan_fitness < 0.1:
                self.logger.warning(f"Scan matching failed with low fitness: {scan_fitness:.3f}")
                return {
                    "success": False,
                    "message": "Scan matching failed with low fitness",
                    "current_pose": self.current_pose,
                    "processing_time": time.time() - start_time
                }

            # Fine-tune with ICP if requested
            if self.use_icp_fine_tuning and pcd is not None and len(pcd.points) > 10:
                try:
                    # Create a point cloud from the map around the current position
                    # This would require extracting points from the occupancy grid
                    # For simplicity, we'll assume we have a point cloud from previous scans
                    if self.point_cloud_processor.previous_cloud is not None:
                        # Perform ICP
                        result_icp = o3d.pipelines.registration.registration_icp(
                            pcd,
                            self.point_cloud_processor.previous_cloud,
                            self.icp_max_distance,
                            scan_transform,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint())

                        if result_icp.fitness > scan_fitness:
                            scan_transform = np.array(result_icp.transformation, copy=True)
                            scan_fitness = result_icp.fitness
                            self.logger.debug(f"ICP fine-tuning improved fitness to {scan_fitness:.3f}")
                except Exception as e:
                    self.logger.warning(f"ICP fine-tuning failed: {e}")

            # Check motion magnitude
            delta_translation = np.linalg.norm(scan_transform[:3, 3])
            r = Rotation.from_matrix(scan_transform[:3, :3])
            euler = r.as_euler('xyz')
            delta_rotation = np.linalg.norm(euler)

            if delta_translation < self.min_translation and delta_rotation < self.min_rotation:
                self.logger.debug(f"minimal motion detected, skipping updates")
                self.last_scan_time = timestamp
                return {
                    "success": True,
                    "message": "Insufficient motion",
                    "current_pose": self.current_pose,
                    "processing_time": time.time() - start_time
                }

            # Update particles with the motion model
            for i, particle in enumerate(self.particles):
                particle.update_pose(scan_transform, self.motion_noise)

            # Extract point positions (ignoring z)
            scan_points_xy = np.asarray(pcd.points)[:, :2]

            # Compute importance weights
            total_weight = self._compute_particle_weights(scan_points_xy)

            # Normalize weights
            for i in range(self.num_particles):
                self.particles[i].weight /= total_weight

            # Find best particle
            self.best_particle_idx = np.argmax([p.weight for p in self.particles])
            best_particle = self.particles[self.best_particle_idx]

            # Update current pose
            self.last_pose = self.current_pose.copy()
            self.current_pose = best_particle.pose.copy()
            self.global_trajectory.append(self.current_pose.copy())

            # Calculate world coordinates for each scan point and update map
            best_position = best_particle.get_position()
            world_points = self._transform_points_to_world(scan_points_xy, best_particle.pose)
            if not world_points.flags.writeable:
                world_points = np.array(world_points, copy=True)
            self.map.update_from_scan(best_position[0], best_position[1], world_points)

            # Add node to pose graph
            current_node_id = self.pose_graph.add_node(self.current_pose)

            # Add edge to previous node
            prev_node_id = current_node_id - 1
            if prev_node_id >= 0:
                # Compute relative transformation between nodes
                prev_node = self.pose_graph.get_node(prev_node_id)
                if prev_node:
                    # T_curr = T_prev * T_delta
                    # T_delta = inv(T_prev) * T_curr
                    prev_pose_inv = np.linalg.inv(prev_node.pose)
                    relative_transform = prev_pose_inv @ self.current_pose

                    # Add edge with identity information matrix
                    # In a more sophisticated system, we would compute this based on scan matching uncertainty
                    information = np.identity(6)
                    self.pose_graph.add_edge(prev_node_id, current_node_id, relative_transform, information)

            # ENHANCEMENT: Periodically optimize pose graph even without loop closures
            if len(self.pose_graph.nodes) % 10 == 0 and len(self.pose_graph.nodes) > 5:
                self.logger.info(f"Performing periodic pose graph optimization with {len(self.pose_graph.nodes)} nodes")
                optimization_success = self.pose_graph.optimize(max_iterations=30)
                if optimization_success:
                    self.logger.info(f"Optimization succeeded: error={self.pose_graph.last_optimization_error:.6f}")
                    # Update trajectory based on optimized poses
                    self._update_trajectory_from_pose_graph()

                    # Also update map with improved trajectory - visualization will reflect the optimized path
                    self.logger.info("Trajectory updated based on optimized pose graph")
                else:
                    self.logger.warning("Periodic pose graph optimization failed")

            # Check for loop closures
            if timestamp - self.last_keyframe_time > self.keyframe_interval:
                # Store keyframe
                self._add_keyframe(pcd, self.current_pose)

                # Detect loop closures
                loop_closure = self._detect_loop_closures(pcd, current_node_id)

                if loop_closure["found"]:
                    self.logger.info(
                        f"Loop closure detected between nodes {current_node_id} and {loop_closure['node_id']}")

                    # Optimize pose graph
                    self.pose_graph.optimize()

                    # Update trajectory based on optimized poses
                    self._update_trajectory_from_pose_graph()

            # Check if resampling is needed
            effective_sample_size = 1.0 / sum(p.weight ** 2 for p in self.particles)
            effective_sample_ratio = effective_sample_size / self.num_particles

            if effective_sample_ratio < self.resample_threshold:
                self._resample_particles()

            # Update timing
            self.last_scan_time = timestamp
            processing_time = time.time() - start_time

            return {
                "success": True,
                "current_pose": self.current_pose,
                "best_particle_idx": self.best_particle_idx,
                "best_particle_weight": best_particle.weight,
                "scan_fitness": scan_fitness,
                "processing_time": processing_time
            }

    def _detect_static_state(self) -> bool:
        """
        Use IMU data to determine if system is static
        """
        # Check if we have enough IMU history
        if not hasattr(self, 'imu_history') or len(self.imu_history) < 10:
            return False

        # Analyze recent IMU data variance
        accel_samples = np.array([imu['data'].get('linear_acceleration', [0, 0, 0])
                                  for imu in self.imu_history[-10:]])
        gyro_samples = np.array([imu['data'].get('angular_velocity', [0, 0, 0])
                                 for imu in self.imu_history[-10:]])

        accel_variance = np.var(accel_samples, axis=0).sum()
        gyro_variance = np.var(gyro_samples, axis=0).sum()

        # Consider static if variances are below thresholds
        return accel_variance < 0.01 and gyro_variance < 0.001

    def _compute_particle_weights(self, scan_points: np.ndarray) -> float:
        """
        Compute importance weights for particles based on scan matching

        Args:
            scan_points: Nx2 array of scan points (x, y)

        Returns:
            Sum of all weights
        """
        total_weight = 0.0

        for i, particle in enumerate(self.particles):
            # Transform scan points to world coordinates using particle pose
            world_points = self._transform_points_to_world(scan_points, particle.pose)

            # Count matched points
            matched_count = 0
            miss_count = 0
            total_count = len(world_points)

            if total_count == 0:
                particle.weight = 1e-10
                total_weight += particle.weight
                continue

            # Check if each point matches the map
            for j in range(total_count):
                x, y = world_points[j, 0], world_points[j, 1]

                # Check if point is in a mapped cell
                if self.map.is_explored(x, y):
                    # Match with score based on certainity
                    if self.map.is_occupied(x, y):
                        # point matched an occupied cell
                        matched_count += 1
                    else:
                        # point contradicts with map
                        miss_count += 1

            # Compute weight based on match quality
            if total_count > 0:
                # Include both matches and misses in the score
                match_ratio = matched_count / total_count
                mismatch_ratio = miss_count / total_count

                # Weight calculation with penalty for contradictions
                score = 0.7 * match_ratio - 0.3 * mismatch_ratio
                weight = max(particle.weight * (0.1 + 0.9 * (max(score, 0) ** 2)), 1e-10)

                # Add small random noise to prevent identical weights
                weight *= (1.0 + random.uniform(-0.01, 0.01))

                particle.weight = weight
                total_weight += weight

                # Log weights periodically
                if i % 10 == 0:
                    self.logger.debug(f"Particle {i} weight: {weight:.6f}, match ratio: {match_ratio:.3f}")

        return total_weight

    def _transform_points_to_world(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Transform scan points from LiDAR frame to world frame

        Args:
            points: Nx2 array of points (x, y)
            pose: 4x4 transformation matrix

        Returns:
            Nx3 array of transformed points (x, y, z)
        """
        # Convert 2D points to 3D (z=0)
        points_3d = np.zeros((len(points), 3))
        points_3d[:, :2] = points

        # Convert to homogeneous coordinates
        points_hom = np.ones((len(points_3d), 4))
        points_hom[:, :3] = points_3d

        # Transform
        transformed_hom = (pose @ points_hom.T).T

        # Convert back to 3D
        return transformed_hom[:, :3]

    def _resample_particles(self):
        """Resample particles using low variance sampling"""
        # Get weights
        weights = np.array([p.weight for p in self.particles])

        # Calculate effective sample size
        effective_sample_size = 1.0 / np.sum(weights ** 2)
        effective_sample_ratio = effective_sample_size / self.num_particles

        self.logger.info(f"Resampling particales. Effective sample size: {effective_sample_ratio:.4f}")

        # Create new particle set
        new_particles = []

        # Low variance resampling
        M = self.num_particles
        r = random.uniform(0, 1.0 / M)
        c = weights[0]
        i = 0

        for m in range(M):
            u = r + m / M
            while u > c:
                i += 1
                if i >= M:
                    i = M - 1
                c += weights[i]

            new_particle = Particle(
                pose=self._add_noise_to_transform(self.particles[i].pose, self.motion_noise * 0.5),
                weight=1.0 / M  # Equal weights after resampling
            )

            # Copy trajectory with noise
            new_particle.trajectory = self.particles[i].trajectory.copy()
            new_particles.append(new_particle)

            # Ensure best particle is preserved (elitist approach)
        best_idx = np.argmax(weights)
        new_particles[0] = Particle(
            pose=self.particles[best_idx].pose.copy(),
            weight=1.0 / M
        )
        new_particles[0].trajectory = self.particles[best_idx].trajectory.copy()

        # Replace particles
        self.particles = new_particles
        self.logger.debug("Particles resampled")


    def _add_keyframe(self, pcd: o3d.geometry.PointCloud, pose: np.ndarray):
        """
        Add a keyframe for loop closure detection

        Args:
            pcd: Point cloud at the keyframe
            pose: Robot pose at the keyframe
        """
        # Extract features
        keypoints, _, descriptors = self.point_cloud_processor.extract_features(pcd)

        if len(keypoints) > 0:
            self.keyframes.append({
                "id": len(self.keyframes),
                "pose": pose.copy(),
                "timestamp": time.time()
            })
            self.keyframe_features.append(descriptors)
            self.last_keyframe_time = time.time()

    def _detect_loop_closures(self, pcd: o3d.geometry.PointCloud,
                              current_node_id: int) -> Dict[str, Any]:
        """
        Detect loop closures between current scan and previous keyframes

        Args:
            pcd: Current point cloud
            current_node_id: ID of current node in pose graph

        Returns:
            Dictionary with loop closure detection results
        """
        # Extract features from current scan
        _, _, current_descriptors = self.point_cloud_processor.extract_features(pcd)

        if len(current_descriptors) == 0 or len(self.keyframes) < 5:
            return {"found": False, "node_id": -1, "score": 0.0}

        best_match_id = -1
        best_match_score = 0.0

        # Check only keyframes that are far enough in the past
        for i, keyframe in enumerate(self.keyframes[:-4]):  # Skip the most recent keyframes
            # Check if far enough in the past
            if i >= len(self.keyframes) - 5:
                continue

            # Check spatial distance (avoid nearby frames)
            dist = np.linalg.norm(self.current_pose[:3, 3] - keyframe["pose"][:3, 3])
            if dist < self.min_loop_closure_distance:
                continue

            # Match feature descriptors
            keyframe_descriptors = self.keyframe_features[i]

            try:
                # Compute feature matches
                matches = []
                for j in range(len(current_descriptors)):
                    distances = np.linalg.norm(keyframe_descriptors - current_descriptors[j], axis=1)
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]
                    second_best_idx = np.argpartition(distances, 1)[1]
                    second_best_dist = distances[second_best_idx]

                    # Lowe's ratio test
                    if best_dist < 0.75 * second_best_dist:
                        matches.append((j, best_idx, best_dist))

                # Compute match score
                if len(matches) > 10:
                    # Simple score based on number of matches and mean distance
                    match_score = len(matches) / (1.0 + np.mean([m[2] for m in matches]))

                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_id = i
            except Exception as e:
                self.logger.warning(f"Error in feature matching: {e}")
                continue

        # Threshold for considering a loop closure
        if best_match_score > 5.0 and best_match_id >= 0:
            # Get the node ID corresponding to the keyframe
            keyframe_node_id = best_match_id

            # Add loop closure edge to pose graph
            if keyframe_node_id < current_node_id - 2:  # Avoid recent nodes
                try:
                    # Get poses
                    current_pose = self.current_pose
                    keyframe_pose = self.keyframes[best_match_id]["pose"]

                    # Compute relative transformation
                    keyframe_pose_inv = np.linalg.inv(keyframe_pose)
                    relative_transform = keyframe_pose_inv @ current_pose

                    # Add edge with identity information matrix
                    # In practice, we would compute this based on matching uncertainty
                    information = np.identity(6)
                    self.pose_graph.add_edge(keyframe_node_id, current_node_id, relative_transform, information)

                    return {
                        "found": True,
                        "node_id": keyframe_node_id,
                        "score": best_match_score
                    }
                except Exception as e:
                    self.logger.warning(f"Error adding loop closure edge: {e}")

        return {"found": False, "node_id": -1, "score": best_match_score}

    def _update_trajectory_from_pose_graph(self):
        """Update global trajectory from optimized pose graph"""
        # Get optimized trajectory
        optimized_poses = self.pose_graph.get_trajectory()

        if len(optimized_poses) > 0:
            # Update current pose
            self.current_pose = optimized_poses[-1].copy()

            # Update global trajectory
            self.global_trajectory = [pose.copy() for pose in optimized_poses]

            # Update particles (best particle gets optimized pose, others are dispersed)
            best_particle = self.particles[self.best_particle_idx]
            best_particle.set_pose(self.current_pose)

            for i, particle in enumerate(self.particles):
                if i != self.best_particle_idx:
                    # Add noise around optimized pose
                    noise_scale = self.motion_noise * 2.0
                    noisy_pose = best_particle._add_noise_to_transform(self.current_pose, noise_scale)
                    particle.set_pose(noisy_pose)
                    particle.weight = 1.0 / self.num_particles

            # Reset weights for best particle
            best_particle.weight = 1.0 / self.num_particles

    def get_map(self) -> MapManager:
        """Get the current map"""
        with self.lock:
            return self.map

    def get_pose(self) -> np.ndarray:
        """Get the current estimated pose"""
        with self.lock:
            return self.current_pose.copy()

    def get_trajectory(self) -> List[np.ndarray]:
        """Get the estimated trajectory"""
        with self.lock:
            return [pose.copy() for pose in self.global_trajectory]

    def get_visualization_image(self, include_particles: bool = True) -> np.ndarray:
        """
        Get a visualization image of the map with robot pose and trajectory

        Args:
            include_particles: Whether to include particle visualization

        Returns:
            BGR visualization image
        """
        with self.lock:
            # Get position from best particle
            position = self.current_pose[:3, 3]

            # Get yaw from best particle
            r = Rotation.from_matrix(self.current_pose[:3, :3])
            euler = r.as_euler('xyz')
            yaw = euler[2]

            # Get the map image with robot pose
            map_img = self.map.get_visualization_image(position[0], position[1], yaw)
            self.logger.debug(f"Generating visualisation with map size : {self.map.width}x{self.map.height}")

            # Draw trajectory
            for i in range(1, len(self.global_trajectory)):
                prev_pos = self.global_trajectory[i - 1][:3, 3]
                curr_pos = self.global_trajectory[i][:3, 3]

                prev_cell_x, prev_cell_y = self.map.world_to_map(prev_pos[0], prev_pos[1])
                curr_cell_x, curr_cell_y = self.map.world_to_map(curr_pos[0], curr_pos[1])

                if (self.map.is_in_bounds(prev_cell_x, prev_cell_y) and
                        self.map.is_in_bounds(curr_cell_x, curr_cell_y)):
                    cv2.line(map_img, (prev_cell_x, prev_cell_y), (curr_cell_x, curr_cell_y),
                             (0, 0, 255), 1)

            # Draw particles if requested
            if include_particles:
                for particle in self.particles:
                    pos = particle.get_position()
                    cell_x, cell_y = self.map.world_to_map(pos[0], pos[1])

                    if self.map.is_in_bounds(cell_x, cell_y):
                        # Color based on weight (blue to red)
                        weight_norm = min(particle.weight * self.num_particles * 5.0, 1.0)
                        r = int(255 * weight_norm)
                        b = int(255 * (1.0 - weight_norm))

                        # Draw particle as a dot
                        cv2.circle(map_img, (cell_x, cell_y), 1, (b, 0, r), -1)

            return map_img

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the SLAM system

        Returns:
            Dictionary with SLAM statistics
        """
        with self.lock:
            # Get information from subsystems
            map_stats = self.map.get_stats()
            pose_graph_stats = self.pose_graph.get_stats()

            # Compute effective sample size
            weights = np.array([p.weight for p in self.particles])
            ess = 1.0 / sum(weights ** 2)
            ess_ratio = ess / self.num_particles

            # Get distance traveled
            distance = 0.0
            for i in range(1, len(self.global_trajectory)):
                prev_pos = self.global_trajectory[i - 1][:3, 3]
                curr_pos = self.global_trajectory[i][:3, 3]
                distance += np.linalg.norm(curr_pos - prev_pos)

            return {
                "particle_count": self.num_particles,
                "effective_sample_size": ess,
                "effective_sample_ratio": ess_ratio,
                "distance_traveled": distance,
                "trajectory_length": len(self.global_trajectory),
                "keyframe_count": len(self.keyframes),
                "map_stats": map_stats,
                "pose_graph_stats": pose_graph_stats
            }