import numpy as np
import open3d as o3d
import logging
import threading
import time
import os
from typing import List, Dict, Any, Tuple, Optional
import cv2


class MapManager:
    """
    Manages a 2D occupancy grid map for SLAM and navigation

    This class handles map creation, updates, and serialization of the occupancy grid.
    It also provides methods for map queries and visualization.
    """

    def __init__(self,
                 resolution: float = 0.1,  # meters per cell
                 width: int = 1000,  # cells
                 height: int = 1000,  # cells
                 initial_position: Tuple[float, float] = (500, 500)):  # cell coordinates
        """
        Initialize the map manager

        Args:
            resolution: Map resolution in meters per cell
            width: Width of the map in cells
            height: Height of the map in cells
            initial_position: Initial position in the map (cell coordinates)
        """
        self.logger = logging.getLogger("MapManager")
        self.logger.info("Initializing map manager")

        # Map parameters
        self.resolution = resolution  # meters per cell
        self.width = width
        self.height = height
        self.origin_x, self.origin_y = initial_position

        # Occupancy values
        self.UNKNOWN = 0.5
        self.FREE = 0.0
        self.OCCUPIED = 1.0

        # Update parameters
        self.hit_increment = 0.2
        self.miss_decrement = 0.05
        self.occupancy_threshold = 0.6
        self.free_threshold = 0.3

        # Initialize the occupancy grid map with unknown values
        self.occupancy_map = np.ones((height, width), dtype=np.float32) * self.UNKNOWN

        # Thread safety
        self.lock = threading.RLock()

        # Map update count for versioning
        self.update_count = 0

        # Map visualization image
        self.map_image = None
        self.map_image_timestamp = 0

        self.logger.info(f"Map initialized with resolution: {resolution}m and size: {width}x{height} cells")

    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to map cell coordinates

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame

        Returns:
            Tuple of (cell_x, cell_y)
        """
        cell_x = int(np.round(x / self.resolution + self.origin_x))
        cell_y = int(np.round(y / self.resolution + self.origin_y))
        return cell_x, cell_y

    def map_to_world(self, cell_x: int, cell_y: int) -> Tuple[float, float]:
        """
        Convert map cell coordinates to world coordinates

        Args:
            cell_x: X coordinate in map frame
            cell_y: Y coordinate in map frame

        Returns:
            Tuple of (world_x, world_y)
        """
        world_x = (cell_x - self.origin_x) * self.resolution
        world_y = (cell_y - self.origin_y) * self.resolution
        return world_x, world_y

    def is_in_bounds(self, cell_x: int, cell_y: int) -> bool:
        """
        Check if cell coordinates are within the map bounds

        Args:
            cell_x: X coordinate in map frame
            cell_y: Y coordinate in map frame

        Returns:
            True if in bounds, False otherwise
        """
        return (0 <= cell_x < self.width) and (0 <= cell_y < self.height)

    def get_occupancy(self, cell_x: int, cell_y: int) -> float:
        """
        Get the occupancy value at a specific cell

        Args:
            cell_x: X coordinate in map frame
            cell_y: Y coordinate in map frame

        Returns:
            Occupancy value (0.0 = free, 0.5 = unknown, 1.0 = occupied)
        """
        with self.lock:
            if not self.is_in_bounds(cell_x, cell_y):
                return self.UNKNOWN
            return self.occupancy_map[cell_y, cell_x]

    def set_occupancy(self, cell_x: int, cell_y: int, value: float):
        """
        Set the occupancy value at a specific cell

        Args:
            cell_x: X coordinate in map frame
            cell_y: Y coordinate in map frame
            value: Occupancy value to set
        """
        with self.lock:
            if not self.is_in_bounds(cell_x, cell_y):
                return
            self.occupancy_map[cell_y, cell_x] = value

    def update_occupancy(self, cell_x: int, cell_y: int, is_occupied: bool):
        """
        Update the occupancy value at a specific cell

        Args:
            cell_x: X coordinate in map frame
            cell_y: Y coordinate in map frame
            is_occupied: True if cell is observed as occupied, False if free
        """
        with self.lock:
            if not self.is_in_bounds(cell_x, cell_y):
                return

            current = self.occupancy_map[cell_y, cell_x]

            if is_occupied:
                # Update with hit evidence
                new_value = min(current + self.hit_increment, self.OCCUPIED)
            else:
                # Update with miss evidence
                new_value = max(current - self.miss_decrement, self.FREE)

            self.occupancy_map[cell_y, cell_x] = new_value
            self.update_count += 1

    def trace_ray(self, start_x: float, start_y: float, end_x: float, end_y: float) -> List[Tuple[int, int]]:
        """
        Perform ray tracing from start to end point

        Args:
            start_x: Start X in world coordinates
            start_y: Start Y in world coordinates
            end_x: End X in world coordinates
            end_y: End Y in world coordinates

        Returns:
            List of cells (cell_x, cell_y) along the ray
        """
        # Convert to map coordinates
        start_cell_x, start_cell_y = self.world_to_map(start_x, start_y)
        end_cell_x, end_cell_y = self.world_to_map(end_x, end_y)

        # Bresenham line algorithm
        cells = []
        dx = abs(end_cell_x - start_cell_x)
        dy = abs(end_cell_y - start_cell_y)
        sx = 1 if start_cell_x < end_cell_x else -1
        sy = 1 if start_cell_y < end_cell_y else -1
        err = dx - dy

        x, y = start_cell_x, start_cell_y

        while True:
            if self.is_in_bounds(x, y):
                cells.append((x, y))

            if x == end_cell_x and y == end_cell_y:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

    def update_from_scan(self, sensor_x: float, sensor_y: float, points: np.ndarray):
        """
        Update map using a LiDAR scan

        Args:
            sensor_x: Sensor X position in world coordinates
            sensor_y: Sensor Y position in world coordinates
            points: Nx3 array of point coordinates in world frame
        """
        start_time = time.time()
        update_count = 0

        with self.lock:
            # Start position in map coordinates
            start_cell_x, start_cell_y = self.world_to_map(sensor_x, sensor_y)

            if not self.is_in_bounds(start_cell_x, start_cell_y):
                self.logger.warning(f"Sensor position ({sensor_x}, {sensor_y}) is outside map bounds")
                return

            # Process each point in the scan
            for i in range(len(points)):
                # Get endpoint in world coordinates (ignoring z)
                end_x, end_y = points[i, 0], points[i, 1]

                # Convert endpoint to map coordinates
                end_cell_x, end_cell_y = self.world_to_map(end_x, end_y)

                # Skip if endpoint is outside map bounds
                if not self.is_in_bounds(end_cell_x, end_cell_y):
                    continue

                # Get cells along the ray
                cells = self.trace_ray(sensor_x, sensor_y, end_x, end_y)

                if not cells:
                    continue

                # Mark cells along the ray as free except the endpoint
                for j in range(len(cells) - 1):
                    cell_x, cell_y = cells[j]
                    self.update_occupancy(cell_x, cell_y, False)
                    update_count += 1

                # Mark the endpoint as occupied
                self.update_occupancy(end_cell_x, end_cell_y, True)
                update_count += 1

        process_time = time.time() - start_time
        self.logger.debug(f"Updated map with {len(points)} points in {process_time:.3f} seconds")
        self.logger.debug(f"Updated {update_count} cells")

    def save_map(self, filename: str) -> bool:
        """
        Save the occupancy grid map to a file

        Args:
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(filename)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save the map as a NumPy array
                np.save(filename, self.occupancy_map)

                # Also save as a PNG for visualization
                img_filename = os.path.splitext(filename)[0] + ".png"
                map_image = self.get_map_image()
                cv2.imwrite(img_filename, map_image)

                self.logger.info(f"Map saved to {filename} and {img_filename}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to save map: {e}")
            return False

    def load_map(self, filename: str) -> bool:
        """
        Load an occupancy grid map from a file

        Args:
            filename: Input filename

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filename):
                self.logger.error(f"Map file {filename} does not exist")
                return False

            with self.lock:
                # Load the map from a NumPy array
                loaded_map = np.load(filename)

                # Check if dimensions match
                if loaded_map.shape != self.occupancy_map.shape:
                    self.logger.warning(
                        f"Loaded map dimensions {loaded_map.shape} do not match current map {self.occupancy_map.shape}")

                    # Resize the current map if needed
                    self.height, self.width = loaded_map.shape

                # Copy the loaded map
                self.occupancy_map = loaded_map

                # Reset update count
                self.update_count = 0

                self.logger.info(f"Map loaded from {filename}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to load map: {e}")
            return False

    def get_map_image(self) -> np.ndarray:
        """
        Get visualization image of the map

        Returns:
            BGR image of the occupancy map
        """
        # Only regenerate image if map has been updated
        if self.map_image is not None and self.map_image_timestamp == self.update_count:
            return self.map_image

        with self.lock:
            # Create a grayscale image from the occupancy map
            # Occupied cells = black, Free cells = white, Unknown = gray
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Unknown cells (gray)
            unknown_mask = (self.occupancy_map >= self.free_threshold) & (
                        self.occupancy_map <= self.occupancy_threshold)
            img[unknown_mask] = [128, 128, 128]

            # Free cells (white)
            free_mask = (self.occupancy_map < self.free_threshold)
            img[free_mask] = [255, 255, 255]

            # Occupied cells (black)
            occupied_mask = (self.occupancy_map > self.occupancy_threshold)
            img[occupied_mask] = [0, 0, 0]

            # Update image cache
            self.map_image = img
            self.map_image_timestamp = self.update_count

            return img

    def get_visualization_image(self, robot_x: float, robot_y: float,
                                heading: float, point_cloud: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get enhanced visualization image of the map with robot pose and LiDAR overlay

        Args:
            robot_x: Robot X position in world coordinates
            robot_y: Robot Y position in world coordinates
            heading: Robot heading in radians
            point_cloud: Optional Nx3 array of point coordinates in world frame for overlay

        Returns:
            BGR image with robot pose and LiDAR overlay
        """
        # Get the base map image
        map_img = self.get_map_image().copy()

        # Convert robot position to map coordinates
        robot_cell_x, robot_cell_y = self.world_to_map(robot_x, robot_y)

        # Draw robot position
        if self.is_in_bounds(robot_cell_x, robot_cell_y):
            # Draw robot as a circle
            cv2.circle(map_img, (robot_cell_x, robot_cell_y), 5, (0, 0, 255), -1)

            # Draw heading indicator
            heading_len = 15
            end_x = int(robot_cell_x + heading_len * np.cos(heading))
            end_y = int(robot_cell_y + heading_len * np.sin(heading))
            cv2.line(map_img, (robot_cell_x, robot_cell_y), (end_x, end_y), (0, 0, 255), 2)

        # Add point cloud overlay if provided
        if point_cloud is not None and len(point_cloud) > 0:
            # Draw points as small dots with gradient color based on distance from robot
            for i in range(len(point_cloud)):
                # Get world coordinates
                px, py = point_cloud[i, 0], point_cloud[i, 1]

                # Calculate distance to robot
                dist = np.sqrt((px - robot_x) ** 2 + (py - robot_y) ** 2)

                # Convert to map coordinates
                cell_x, cell_y = self.world_to_map(px, py)

                if self.is_in_bounds(cell_x, cell_y):
                    # Color based on distance (close=red, far=blue)
                    max_dist = 10.0  # Maximum distance for color scaling
                    normalized_dist = min(dist / max_dist, 1.0)

                    # RGB color (from red to blue with distance)
                    b = int(255 * normalized_dist)
                    r = int(255 * (1.0 - normalized_dist))
                    g = 0

                    # Draw point
                    cv2.circle(map_img, (cell_x, cell_y), 1, (b, g, r), -1)

        return map_img

    def crop_to_region(self, center_x: float, center_y: float, size: float = 20.0) -> np.ndarray:
        """
        Crop map to a region around the center point

        Args:
            center_x: Center X position in world coordinates
            center_y: Center Y position in world coordinates
            size: Size of the region in meters

        Returns:
            Cropped map image
        """
        with self.lock:
            # Convert center to map coordinates
            center_cell_x, center_cell_y = self.world_to_map(center_x, center_y)

            # Calculate region bounds in cells
            size_cells = int(size / self.resolution)
            min_x = max(0, center_cell_x - size_cells // 2)
            min_y = max(0, center_cell_y - size_cells // 2)
            max_x = min(self.width, center_cell_x + size_cells // 2)
            max_y = min(self.height, center_cell_y + size_cells // 2)

            # Get map image
            map_img = self.get_map_image()

            # Crop to region
            cropped = map_img[min_y:max_y, min_x:max_x]

            return cropped

    def is_occupied(self, x: float, y: float) -> bool:
        """
        Check if a world position is occupied

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame

        Returns:
            True if occupied, False otherwise
        """
        cell_x, cell_y = self.world_to_map(x, y)
        if not self.is_in_bounds(cell_x, cell_y):
            return False
        occupancy = self.get_occupancy(cell_x, cell_y)
        return occupancy > self.occupancy_threshold

    def is_free(self, x: float, y: float) -> bool:
        """
        Check if a world position is free

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame

        Returns:
            True if free, False otherwise
        """
        cell_x, cell_y = self.world_to_map(x, y)
        if not self.is_in_bounds(cell_x, cell_y):
            return False
        occupancy = self.get_occupancy(cell_x, cell_y)
        return occupancy < self.free_threshold

    def is_explored(self, x: float, y: float) -> bool:
        """
        Check if a world position has been explored

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame

        Returns:
            True if explored, False otherwise
        """
        cell_x, cell_y = self.world_to_map(x, y)
        if not self.is_in_bounds(cell_x, cell_y):
            return False
        occupancy = self.get_occupancy(cell_x, cell_y)
        return not (self.free_threshold <= occupancy <= self.occupancy_threshold)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the map

        Returns:
            Dictionary with map statistics
        """
        with self.lock:
            # Count different cell types
            free_count = np.sum(self.occupancy_map < self.free_threshold)
            occupied_count = np.sum(self.occupancy_map > self.occupancy_threshold)
            unknown_count = self.width * self.height - free_count - occupied_count

            # Calculate percentages
            total_cells = self.width * self.height
            free_percent = free_count / total_cells * 100
            occupied_percent = occupied_count / total_cells * 100
            unknown_percent = unknown_count / total_cells * 100

            return {
                "width": self.width,
                "height": self.height,
                "resolution": self.resolution,
                "size_meters": (self.width * self.resolution, self.height * self.resolution),
                "free_cells": free_count,
                "occupied_cells": occupied_count,
                "unknown_cells": unknown_count,
                "free_percent": free_percent,
                "occupied_percent": occupied_percent,
                "unknown_percent": unknown_percent,
                "update_count": self.update_count
            }