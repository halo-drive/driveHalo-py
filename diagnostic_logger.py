import time
import logging
import numpy as np
from collections import deque

class DiagnosticLogger:
    def __init__(self, log_interval=5.0):
        self.logger = logging.getLogger("Diagnostics")
        self.imu_timestamps = deque(maxlen=1000)
        self.camera_timestamps = deque(maxlen=100)
        self.sync_results = deque(maxlen=100)
        self.log_interval = log_interval
        self.last_log_time = time.time()
        
    def add_imu_timestamp(self, timestamp):
        self.imu_timestamps.append(timestamp)
        
    def add_camera_timestamp(self, timestamp):
        self.camera_timestamps.append(timestamp)
        
    def add_sync_result(self, success, time_diff):
        self.sync_results.append((success, time_diff))
        
    def log_if_needed(self):
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self._log_diagnostics()
            self.last_log_time = current_time
            
    def _log_diagnostics(self):
        # IMU rate calculation
        if len(self.imu_timestamps) > 1:
            imu_diffs = np.diff(list(self.imu_timestamps))
            imu_rate = 1.0 / np.mean(imu_diffs) if len(imu_diffs) > 0 else 0
            self.logger.info(f"IMU Rate: {imu_rate:.2f} Hz, Min interval: {np.min(imu_diffs)*1000:.1f}ms, Max interval: {np.max(imu_diffs)*1000:.1f}ms")
            
        # Camera rate calculation
        if len(self.camera_timestamps) > 1:
            cam_diffs = np.diff(list(self.camera_timestamps))
            cam_rate = 1.0 / np.mean(cam_diffs) if len(cam_diffs) > 0 else 0
            self.logger.info(f"Camera Rate: {cam_rate:.2f} Hz, Min interval: {np.min(cam_diffs)*1000:.1f}ms, Max interval: {np.max(cam_diffs)*1000:.1f}ms")
            
        # Sync statistics
        if self.sync_results:
            success_count = sum(1 for s, _ in self.sync_results if s)
            if success_count > 0:
                time_diffs = [d*1000 for s, d in self.sync_results if s]  # Convert to ms
                self.logger.info(f"Sync Success Rate: {success_count/len(self.sync_results)*100:.1f}%, Avg Diff: {np.mean(time_diffs):.1f}ms, Max Diff: {np.max(time_diffs):.1f}ms")
            else:
                self.logger.warning("No successful sensor synchronizations in last period")