import numpy as np
import open3d as o3d
import logging
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation


class PointCloudProcessor:
    """
    Process LiDAR point cloud data for SLAM and obstacle detection

    This class handles preprocessing, downsampling, feature extraction, and
    registration of point clouds from the Livox LiDAR.
    """

    def __init__(self,
                 voxel_size: float = 0.05,
                 normal_radius: float = 0.1,
                 feature_radius: float = 0.2,
                 icp_max_distance: float = 0.05):
        """
        Initialize the point cloud processor

        Args:
            voxel_size: Size of voxels for downsampling (meters)
            normal_radius: Radius for normal estimation
            feature_radius: Radius for feature extraction
            icp_threshold: ICP convergence threshold
        """
        self.logger = logging.getLogger("PointCloudProcessor")
        self.logger.info("Initializing point cloud processor")

        # Parameters
        self.voxel_size = voxel_size
        self.normal_radius = normal_radius
        self.feature_radius = feature_radius
        self.icp_max_distance = icp_max_distance

        # Thread safety
        self.lock = threading.RLock()

        # Point cloud history for scan matching
        self.previous_cloud = None
        self.current_cloud = None

        # Feature history
        self.feature_history = []
        self.max_history_size = 10

        # Transformation history
        self.transformation_history = []

        self.logger.info(f"Point cloud processor initialized with voxel size: {voxel_size}m")

    def process_point_cloud(self, points: np.ndarray,
                            intensities: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Process raw point cloud data

        Args:
            points: Nx3 array of point coordinates (x, y, z)
            intensities: N array of point intensities (optional)

        Returns:
            Processed Open3D point cloud
        """
        with self.lock:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)


            if intensities is not None:
                # Normalize intensities to [0,1] for coloring
                norm_intensities = np.zeros((len(intensities), 3))
                min_i, max_i = np.min(intensities), np.max(intensities)
                if max_i > min_i:
                    norm_value = (intensities - min_i) / (max_i - min_i)
                    # Use grayscale coloring based on intensity
                    norm_intensities[:, 0] = norm_value
                    norm_intensities[:, 1] = norm_value
                    norm_intensities[:, 2] = norm_value
                pcd.colors = o3d.utility.Vector3dVector(norm_intensities)

            # Remove invalid points (NaN or Inf)
            valid_indices = []
            points_np = np.array(np.asarray(pcd.points), copy=True)
            for i in range(len(points_np)):
                if not np.any(np.isnan(points_np[i])) and not np.any(np.isinf(points_np[i])):
                    valid_indices.append(i)
            pcd = pcd.select_by_index(valid_indices)

            # Remove outliers (statistical)
            if len(pcd.points) > 100:  # Need enough points for statistical analysis
                try:
                    pcd, _ = pcd.remove_statistical_outlier(
                        nb_neighbors=20, std_ratio=2.0)
                except Exception as e:
                    self.logger.warning(f"Failed to remove outliers: {e}")

            # Downsample point cloud
            try:
                downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                if len(downsampled_pcd.points) > 0:
                    pcd = downsampled_pcd
            except Exception as e:
                self.logger.warning(f"Failed to downsample point cloud: {e}")

            # Update point cloud history
            self.previous_cloud = self.current_cloud
            self.current_cloud = pcd

            return pcd

    def estimate_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Estimate normals for the point cloud

        Args:
            pcd: Input point cloud

        Returns:
            Point cloud with normals
        """
        if len(pcd.points) == 0:
            return pcd

        try:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius, max_nn=30))

            # Orient normals consistently
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            return pcd
        except Exception as e:
            self.logger.warning(f"Failed to estimate normals: {e}")
            return pcd

    def _is_static_scenario(self, source, target):
        """Quickly determine if point clouds represent a static scenario"""
        # Quick check: compare point count
        if abs(len(source.points) - len(target.points)) < 10:
            # Sample a subset of points (10 points) for fast comparison
            try:
                source_sample = np.asarray(source.points)[::len(source.points) // 10][:10]
                target_sample = np.asarray(target.points)[::len(target.points) // 10][:10]

                # Calculate average displacement
                if len(source_sample) > 0 and len(target_sample) > 0:
                    displacements = []
                    for i in range(min(len(source_sample), len(target_sample))):
                        dist = np.linalg.norm(source_sample[i] - target_sample[i])
                        displacements.append(dist)

                    avg_displacement = np.mean(displacements) if displacements else 0

                    # If average displacement is small, consider it static
                    return avg_displacement < 0.01  # 1cm threshold
            except:
                pass
        return False

    def extract_features(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract keypoints and features from point cloud

        Args:
            pcd: Input point cloud with normals

        Returns:
            Tuple of (keypoints, features, descriptors)
        """
        if len(pcd.points) < 10:
            return np.array([]), np.array([]), np.array([])

        if hasattr(pcd, 'points') and len(pcd.points) > 0:
            pcd_hash = hash((len(pcd.points), *pcd.get_min_bound(), *pcd.get_max_bound()))
            if hasattr(self, '_feature_cache') and pcd_hash in self._feature_cache:
                return self._feature_cache[pcd_hash]
        try:
            # Ensure normals are computed
            if not pcd.has_normals():
                pcd = self.estimate_normals(pcd)

            # FPFH feature extraction
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius, max_nn=100))

            # Get keypoints (we're using all points as keypoints for simplicity)
            # In a more sophisticated implementation, we could use ISS or other keypoint detectors
            keypoints = np.asarray(pcd.points)
            descriptors = np.asarray(fpfh.data).T

            if not hasattr(self, '_feature_cache'):
                self._feature_cache = {}
            if len(self._feature_cache) > 10:
                self._feature_cache.clear()
            self._feature_cache[pcd_hash] = (keypoints, keypoints, descriptors)

            return keypoints, keypoints, descriptors

        except Exception as e:
            self.logger.warning(f"Failed to extract features: {e}")
            return np.array([]), np.array([]), np.array([])

    def register_point_clouds(self, source: o3d.geometry.PointCloud,
                              target: o3d.geometry.PointCloud) -> Tuple[np.ndarray, float]:
        """
        Register two point clouds to find transformation

        Args:
            source: Source point cloud
            target: Target point cloud

        Returns:
            Tuple of (4x4 transformation matrix, fitness score)
        """
        if source is None or target is None:
            return np.eye(4), 0.0

        # check for if the scans are identical (static lidar)
        if self._is_static_scenario(source, target):
            return np.eye(4), 0.95

        if not isinstance(source, o3d.geometry.PointCloud):
            source = o3d.geometry.PointCloud()
            if isinstance(source, np.ndarray):
                # Ensure writeable copy source pointcloud
                source_points = np.array(source, copy=True) if not source.flags.writeable else source
                source.points = o3d.utility.Vector3dVector(source_points)

        if not isinstance(target, o3d.geometry.PointCloud):
            target = o3d.geometry.PointCloud()
            if isinstance(target, np.ndarray):
                # Ensure writeable copy for target
                target_points = np.array(target, copy=True) if not target.flags.writeable else target
                target.points = o3d.utility.Vector3dVector(target_points)

        if len(source.points) < 10 or len(target.points) < 10:
            return np.eye(4), 0.0

        try:
            # Preprocess point clouds for better results
            source = self._preprocess_point_cloud(source)
            target = self._preprocess_point_cloud(target)

            # Initial global registration with better parameters
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source, target,
                self._compute_fpfh_features(source),
                self._compute_fpfh_features(target),
                True,
                self.icp_max_distance * 1.5,  # More generous initial distance threshold
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4,  # Increased minimum correspondence points
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.icp_max_distance * 1.5)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999)  # More iterations
            )

            # Refine with ICP using point-to-plane for better accuracy
            result_icp = o3d.pipelines.registration.registration_icp(
                source, target,
                self.icp_max_distance,
                result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )

            # Add confidence check
            if result_icp.fitness < 0.3:  # Minimum threshold for acceptance
                self.logger.warning(f"Low ICP fitness: {result_icp.fitness}, using fallback estimation")
                return self._estimate_motion_from_points(source, target), 0.3

            return result_icp.transformation, result_icp.fitness
        except Exception as e:
            self.logger.warning(f"Registration failed: {e}")
            return np.eye(4), 0.0

    def _preprocess_point_cloud(self, pcd):
        """New helper method for point cloud preprocessing"""
        # Remove statistical outliers
        if len(pcd.points) > 20:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Voxel downsample
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Estimate normals if not already present
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=30))

        return pcd

    def _compute_fpfh_features(self, pcd):
        """Helper method to compute FPFH features"""
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius, max_nn=100))

    def _estimate_motion_from_points(self, source, target):
        """Fallback motion estimation when ICP fails"""
        # Simple centroid-based estimation as fallback
        source_centroid = np.mean(np.asarray(source.points), axis=0)
        target_centroid = np.mean(np.asarray(target.points), axis=0)

        # Estimate translation only
        translation = target_centroid - source_centroid

        # Create transformation matrix
        result = np.eye(4)
        result[:3, 3] = translation

        return result

    def process_scan(self, points: np.ndarray,
                     intensities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a LiDAR scan and compute odometry

        Args:
            points: Nx3 array of point coordinates (x, y, z)
            intensities: N array of point intensities (optional)

        Returns:
            Dictionary with processed data including transformation
        """
        start_time = time.time()
        if isinstance(points, np.ndarray) and not points.flags.writeable:
            points = np.array(points, copy=True)

        if intensities is not None and isinstance(intensities, np.ndarray) and not intensities.flags.writeable:
            intensities = np.array(intensities, copy=True)

        # Process the point cloud
        pcd = self.process_point_cloud(points, intensities)

        # Skip if not enough points
        if len(pcd.points) < 10:
            self.logger.warning("Not enough valid points in scan")
            return {
                "success": False,
                "message": "Not enough valid points",
                "points_processed": len(pcd.points),
                "processing_time": time.time() - start_time
            }

        # Estimate transformation if we have a previous scan
        transformation = np.eye(4)
        fitness = 0.0

        if self.previous_cloud is not None and len(self.previous_cloud.points) >= 10:
            # Register point clouds to get transformation
            transformation, fitness = self.register_point_clouds(pcd, self.previous_cloud)

            # Store transformation
            self.transformation_history.append(transformation)
            if len(self.transformation_history) > self.max_history_size:
                self.transformation_history.pop(0)

        # Extract features for future matching
        keypoints, features, descriptors = self.extract_features(pcd)

        # Time measurements
        processing_time = time.time() - start_time

        result = {
            "success": True,
            "pcd": pcd,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "transformation": transformation,
            "fitness": fitness,
            "points_processed": len(pcd.points),
            "processing_time": processing_time
        }

        return result

    def get_accumulated_transformation(self) -> np.ndarray:
        """
        Get the accumulated transformation over the history

        Returns:
            4x4 transformation matrix
        """
        if not self.transformation_history:
            return np.eye(4)

        # Accumulate transformations
        accumulated = np.eye(4)
        for transform in self.transformation_history:
            accumulated = accumulated @ transform

        return accumulated

    def get_point_cloud_segments(self, pcd: o3d.geometry.PointCloud,
                                 min_points_per_segment: int = 20) -> List[o3d.geometry.PointCloud]:
        """
        Segment point cloud into clusters

        Args:
            pcd: Input point cloud
            min_points_per_segment: Minimum points per segment

        Returns:
            List of segmented point clouds
        """
        if len(pcd.points) < min_points_per_segment:
            return []

        try:
            # Perform segmentation using DBSCAN
            labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=min_points_per_segment))

            max_label = labels.max()
            if max_label < 0:  # No segments found
                return []

            segments = []
            for i in range(max_label + 1):
                segment = pcd.select_by_index(np.where(labels == i)[0])
                if len(segment.points) >= min_points_per_segment:
                    segments.append(segment)

            return segments

        except Exception as e:
            self.logger.warning(f"Failed to segment point cloud: {e}")
            return []

    def extract_ground_plane(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    np.ndarray]:
        """
        Extract ground plane from point cloud

        Args:
            pcd: Input point cloud

        Returns:
            Tuple of (ground point cloud, non-ground point cloud, plane parameters)
        """
        if len(pcd.points) < 10:
            return pcd, o3d.geometry.PointCloud(), np.zeros(4)

        try:
            # Use RANSAC to extract dominant plane (ground)
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.05, ransac_n=3, num_iterations=1000)

            # Create ground and non-ground point clouds
            ground = pcd.select_by_index(inliers)
            non_ground = pcd.select_by_index(inliers, invert=True)

            return ground, non_ground, np.array(plane_model)

        except Exception as e:
            self.logger.warning(f"Failed to extract ground plane: {e}")
            return pcd, o3d.geometry.PointCloud(), np.zeros(4)