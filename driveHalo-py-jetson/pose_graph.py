import numpy as np
import open3d as o3d
import logging
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation
import g2o


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
        Optimize the pose graph using g2o

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            True if optimization succeeded, False otherwise
        """
        start_time = time.time()

        with self.lock:
            if len(self.nodes) < 2 or len(self.edges) < 1:
                self.logger.warning("Cannot optimize: not enough nodes or edges")
                return False

            try:
                # Create optimizer
                optimizer = g2o.SparseOptimizer()
                solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
                algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
                optimizer.set_algorithm(algorithm)

                # Add nodes to optimizer
                for node_id, node in self.nodes.items():
                    g2o_vertex = g2o.VertexSE3()
                    g2o_vertex.set_id(node_id)
                    g2o_vertex.set_fixed(node.is_fixed)

                    # Convert to g2o format
                    translation = node.get_translation()
                    quat = node.get_quaternion()  # [qw, qx, qy, qz]

                    isometry = g2o.Isometry3d(quat, translation)
                    g2o_vertex.set_estimate(isometry)

                    optimizer.add_vertex(g2o_vertex)

                # Add edges to optimizer
                for edge in self.edges:
                    g2o_edge = g2o.EdgeSE3()
                    g2o_edge.set_vertex(0, optimizer.vertex(edge.from_node_id))
                    g2o_edge.set_vertex(1, optimizer.vertex(edge.to_node_id))

                    # Convert to g2o format
                    translation = edge.transformation[:3, 3]
                    rotation = edge.transformation[:3, :3]
                    r = Rotation.from_matrix(rotation)
                    quat = r.as_quat()  # [qx, qy, qz, qw]
                    quat_g2o = np.array([quat[3], quat[0], quat[1], quat[2]])  # [qw, qx, qy, qz]

                    measurement = g2o.Isometry3d(quat_g2o, translation)
                    g2o_edge.set_measurement(measurement)

                    # Set information matrix (inverse covariance)
                    g2o_edge.set_information(edge.information)

                    optimizer.add_edge(g2o_edge)

                # Perform optimization
                optimizer.initialize_optimization()
                self.last_optimization_iterations = optimizer.optimize(max_iterations)

                # Update node poses after optimization
                for node_id, node in self.nodes.items():
                    g2o_vertex = optimizer.vertex(node_id)
                    if g2o_vertex is None:
                        continue

                    # Get optimized estimate
                    optimized_pose = g2o_vertex.estimate()

                    # Convert back to our format
                    t = optimized_pose.translation()
                    q = optimized_pose.rotation()  # [qw, qx, qy, qz]

                    # Convert quaternion to rotation matrix
                    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # [qx, qy, qz, qw]
                    rot_matrix = r.as_matrix()

                    # Build 4x4 transformation matrix
                    new_pose = np.eye(4)
                    new_pose[:3, :3] = rot_matrix
                    new_pose[:3, 3] = t

                    # Update node
                    node.pose = new_pose

                # Record statistics
                self.last_optimization_time = time.time() - start_time
                self.last_optimization_error = optimizer.chi2()

                self.logger.info(f"Optimization completed in {self.last_optimization_time:.3f} seconds, "
                                 f"{self.last_optimization_iterations} iterations, "
                                 f"error: {self.last_optimization_error:.6f}")

                return True

            except Exception as e:
                self.logger.error(f"Optimization failed: {e}")
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