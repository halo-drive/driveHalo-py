import numpy as np
import open3d as o3d
import logging
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation



class PoseGraphNode:
    """
    Node in the pose graph representing a robot pose
    """

    def __init__(self, id: int, pose: np.ndarray, is_fixed: bool = False):
        """
        Initialize a pose graph node

        Args:
            id: Unique node ID
            pose: 4x4 transformation matrix
            is_fixed: Whether the node is fixed in optimization
        """
        self.id = id
        self.pose = pose.copy()
        self.is_fixed = is_fixed

    def get_translation(self) -> np.ndarray:
        """Get the translation part of the pose"""
        return self.pose[:3, 3]

    def get_rotation(self) -> np.ndarray:
        """Get the rotation part of the pose"""
        return self.pose[:3, :3]

    def get_quaternion(self) -> np.ndarray:
        """Get the rotation as a quaternion [qw, qx, qy, qz]"""
        r = Rotation.from_matrix(self.get_rotation())
        quat = r.as_quat()  # [qx, qy, qz, qw]
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # [qw, qx, qy, qz]


class PoseGraphEdge:
    """
    Edge in the pose graph representing a constraint between poses
    """

    def __init__(self, id: int, from_node_id: int, to_node_id: int,
                 transformation: np.ndarray, information: np.ndarray):
        """
        Initialize a pose graph edge

        Args:
            id: Unique edge ID
            from_node_id: Source node ID
            to_node_id: Target node ID
            transformation: 4x4 transformation matrix from source to target
            information: 6x6 information matrix (inverse of covariance)
        """
        self.id = id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.transformation = transformation.copy()
        self.information = information.copy()


class PoseGraph:
    """
    Pose graph for trajectory optimization

    This class manages a graph of robot poses and constraints between them,
    and provides methods for optimization using g2o.
    """

    def __init__(self):
        """Initialize the pose graph"""
        self.logger = logging.getLogger("PoseGraph")
        self.logger.info("Initializing pose graph")

        # Node and edge storage
        self.nodes = {}  # id -> PoseGraphNode
        self.edges = []  # List of PoseGraphEdge

        # Thread safety
        self.lock = threading.RLock()

        # Next IDs for nodes and edges
        self.next_node_id = 0
        self.next_edge_id = 0

        # Statistics
        self.last_optimization_time = 0.0
        self.last_optimization_iterations = 0
        self.last_optimization_error = 0.0

        self.logger.info("Pose graph initialized")

    def add_node(self, pose: np.ndarray, is_fixed: bool = False) -> int:
        """
        Add a node to the pose graph

        Args:
            pose: 4x4 transformation matrix
            is_fixed: Whether the node is fixed in optimization

        Returns:
            ID of the added node
        """
        with self.lock:
            node_id = self.next_node_id
            self.next_node_id += 1

            node = PoseGraphNode(node_id, pose, is_fixed)
            self.nodes[node_id] = node

            return node_id

    def add_edge(self, from_node_id: int, to_node_id: int,
                 transformation: np.ndarray, information: Optional[np.ndarray] = None) -> int:
        """
        Add an edge (constraint) to the pose graph

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            transformation: 4x4 transformation matrix from source to target
            information: 6x6 information matrix (optional, identity if not provided)

        Returns:
            ID of the added edge
        """
        with self.lock:
            # Check if nodes exist
            if from_node_id not in self.nodes or to_node_id not in self.nodes:
                self.logger.warning(f"Cannot add edge: node {from_node_id} or {to_node_id} does not exist")
                return -1

            # Use identity information matrix if not provided
            if information is None:
                information = np.identity(6)

            edge_id = self.next_edge_id
            self.next_edge_id += 1

            edge = PoseGraphEdge(edge_id, from_node_id, to_node_id, transformation, information)
            self.edges.append(edge)

            return edge_id

    def get_node(self, node_id: int) -> Optional[PoseGraphNode]:
        """
        Get a node by ID

        Args:
            node_id: Node ID

        Returns:
            Node object or None if not found
        """
        with self.lock:
            return self.nodes.get(node_id, None)

    def optimize(self, max_iterations: int = 20) -> bool:
        """
        Optimize the pose graph using SciPy instead of g2o

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            True if optimization succeeded, False otherwise
        """
        start_time = time.time()

        with self.lock:
            if len(self.nodes) < 2 or len(self.edges) < 1:
                self.logger.info(f"Not enough nodes ({len(self.nodes)}) or edges ({len(self.edges)}) to optimize")
                return False

            try:
                self.logger.info(f"Optimizing pose graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
                import numpy as np
                from scipy.optimize import minimize
                from scipy.spatial.transform import Rotation

                # Get fixed node indices and their poses
                fixed_nodes = {}
                variable_nodes = {}

                for node_id, node in self.nodes.items():
                    if node.is_fixed:
                        fixed_nodes[node_id] = node
                    else:
                        variable_nodes[node_id] = node

                if not variable_nodes:
                    self.logger.info("No nodes to optimize (all are fixed)")
                    return True

                # Prepare initial parameter vector (flatten all poses)
                # For each variable node, we need 6 parameters: 3 for position, 3 for euler angles
                initial_params = []
                node_indices = {}  # Map from node_id to index in parameter vector
                param_idx = 0

                for node_id, node in sorted(variable_nodes.items()):
                    # Store position
                    initial_params.extend(node.get_translation())

                    # Convert rotation matrix to euler angles
                    r = Rotation.from_matrix(node.get_rotation())
                    euler = r.as_euler('xyz')
                    initial_params.extend(euler)

                    # Store index mapping
                    node_indices[node_id] = param_idx
                    param_idx += 6

                initial_params = np.array(initial_params)

                # Define error function for optimization
                def pose_graph_error(params):
                    # Reconstruct node poses from parameter vector
                    node_poses = {}

                    for node_id, idx in node_indices.items():
                        pos = params[idx:idx + 3]
                        euler = params[idx + 3:idx + 6]

                        # Convert to transformation matrix
                        r = Rotation.from_euler('xyz', euler)
                        rot_matrix = r.as_matrix()

                        pose = np.eye(4)
                        pose[:3, :3] = rot_matrix
                        pose[:3, 3] = pos

                        node_poses[node_id] = pose

                    # Add fixed nodes
                    for node_id, node in fixed_nodes.items():
                        node_poses[node_id] = node.pose

                    # Compute total error (sum of squared errors across all edges)
                    total_error = 0.0

                    for edge in self.edges:
                        if edge.from_node_id not in node_poses or edge.to_node_id not in node_poses:
                            continue

                        # Get actual poses
                        from_pose = node_poses[edge.from_node_id]
                        to_pose = node_poses[edge.to_node_id]

                        # Expected: to_pose = from_pose @ edge.transformation
                        # Calculate error between expected and actual
                        expected_to_pose = from_pose @ edge.transformation
                        error_pose = np.linalg.inv(expected_to_pose) @ to_pose

                        # Extract error components (translation and rotation)
                        trans_error = error_pose[:3, 3]

                        # For rotation, convert to angle-axis representation
                        rot_matrix = error_pose[:3, :3]
                        r = Rotation.from_matrix(rot_matrix)
                        rot_vec = r.as_rotvec()

                        # Combine errors with information matrix weighting
                        error_vector = np.concatenate([trans_error, rot_vec])
                        weighted_error = 0

                        # Apply Hubber loss for robustness to outliers
                        delta = 1.0
                        for i, err in enumerate(error_vector):
                            weight = edge.information[i, i]
                            abs_err = abs(err * weight)
                            if abs_err < delta:
                                weighted_error += 0.5 * abs_err**2
                            else:
                                weighted_error += delta * (abs_err - 0.5 * delta)

                        total_error += weighted_error

                    return total_error

                # Run optimization
                result = minimize(
                    pose_graph_error,
                    initial_params,
                    method='L-BFGS-B',
                    options={'maxiter': max_iterations, 'ftol': 1e-5, 'gtol': 1e-5, 'disp': True}
                )

                # Update node poses with optimized values
                for node_id, idx in node_indices.items():
                    pos = result.x[idx:idx + 3]
                    euler = result.x[idx + 3:idx + 6]

                    # Convert to transformation matrix
                    r = Rotation.from_euler('xyz', euler)
                    rot_matrix = r.as_matrix()

                    pose = np.eye(4)
                    pose[:3, :3] = rot_matrix
                    pose[:3, 3] = pos

                    # Update node
                    self.nodes[node_id].pose = pose

                # Record statistics
                self.last_optimization_time = time.time() - start_time
                self.last_optimization_iterations = result.nit
                self.last_optimization_error = result.fun

                self.logger.info(f"Optimization completed in {self.last_optimization_time:.3f} seconds, "
                                 f"{self.last_optimization_iterations} iterations, "
                                 f"error: {self.last_optimization_error:.6f}")

                return True

            except Exception as e:
                self.logger.error(f"Optimization failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return False

    def get_trajectory(self) -> List[np.ndarray]:
        """
        Get the optimized trajectory as a list of poses

        Returns:
            List of 4x4 transformation matrices sorted by node ID
        """
        with self.lock:
            # Sort nodes by ID
            sorted_nodes = sorted(self.nodes.items())
            return [node.pose for _, node in sorted_nodes]

    def get_trajectory_points(self) -> np.ndarray:
        """
        Get the trajectory as a list of 3D points

        Returns:
            Nx3 array of positions
        """
        with self.lock:
            # Sort nodes by ID
            sorted_nodes = sorted(self.nodes.items())
            return np.array([node.get_translation() for _, node in sorted_nodes])

    def export_to_open3d(self) -> o3d.pipelines.registration.PoseGraph:
        """
        Export the pose graph to Open3D format

        Returns:
            Open3D pose graph
        """
        with self.lock:
            o3d_posegraph = o3d.pipelines.registration.PoseGraph()

            # Add nodes
            for node_id, node in sorted(self.nodes.items()):
                o3d_node = o3d.pipelines.registration.PoseGraphNode()
                o3d_node.pose = node.pose
                o3d_posegraph.nodes.append(o3d_node)

            # Add edges
            for edge in self.edges:
                o3d_edge = o3d.pipelines.registration.PoseGraphEdge()
                o3d_edge.source_node_id = edge.from_node_id
                o3d_edge.target_node_id = edge.to_node_id
                o3d_edge.transformation = edge.transformation
                o3d_edge.information = edge.information
                o3d_posegraph.edges.append(o3d_edge)

            return o3d_posegraph

    def import_from_open3d(self, o3d_posegraph: o3d.pipelines.registration.PoseGraph) -> bool:
        """
        Import a pose graph from Open3D format

        Args:
            o3d_posegraph: Open3D pose graph

        Returns:
            True if import succeeded, False otherwise
        """
        with self.lock:
            try:
                # Clear existing data
                self.nodes.clear()
                self.edges.clear()
                self.next_node_id = 0
                self.next_edge_id = 0

                # Import nodes
                for node_id, o3d_node in enumerate(o3d_posegraph.nodes):
                    node = PoseGraphNode(node_id, o3d_node.pose)
                    self.nodes[node_id] = node
                    self.next_node_id = max(self.next_node_id, node_id + 1)

                # Import edges
                for edge_id, o3d_edge in enumerate(o3d_posegraph.edges):
                    edge = PoseGraphEdge(
                        edge_id,
                        o3d_edge.source_node_id,
                        o3d_edge.target_node_id,
                        o3d_edge.transformation,
                        o3d_edge.information
                    )
                    self.edges.append(edge)
                    self.next_edge_id = max(self.next_edge_id, edge_id + 1)

                self.logger.info(f"Imported pose graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
                return True

            except Exception as e:
                self.logger.error(f"Import failed: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pose graph

        Returns:
            Dictionary with pose graph statistics
        """
        with self.lock:
            return {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "last_optimization_time": self.last_optimization_time,
                "last_optimization_iterations": self.last_optimization_iterations,
                "last_optimization_error": self.last_optimization_error
            }