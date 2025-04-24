import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Tuple, Any
import threading
import psutil

from root_logger import get_diagnostic_logger, get_sensor_logger, get_sync_logger, LogWindow

@dataclass
class SensorStats:
    """Statistics for sensor data"""
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    rates: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    intervals: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    jitter: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    dropped_count: int = 0
    last_timestamp: float = 0.0
    expected_interval: float = 0.0  # in seconds
    
    def update(self, timestamp: float):
        """Update statistics with a new timestamp"""
        self.timestamps.append(timestamp)
        
        if self.last_timestamp > 0:
            interval = timestamp - self.last_timestamp
            self.intervals.append(interval)
            
            # Calculate rate over last 5 samples
            if len(self.timestamps) >= 5:
                recent = list(self.timestamps)[-5:]
                if recent[-1] - recent[0] > 0:
                    rate = (len(recent) - 1) / (recent[-1] - recent[0])
                    self.rates.append(rate)
            
            # Calculate jitter if we have an expected interval
            if self.expected_interval > 0 and len(self.intervals) >= 2:
                # Jitter is the variance in interval from expected
                jitter = abs(interval - self.expected_interval) / self.expected_interval
                self.jitter.append(jitter)
                
            # Check for potential dropped frames
            if self.expected_interval > 0 and interval > (self.expected_interval * 1.9):
                missed = int(interval / self.expected_interval) - 1
                self.dropped_count += missed
        
        self.last_timestamp = timestamp
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        stats = {
            "sample_count": len(self.timestamps),
            "dropped_count": self.dropped_count,
        }
        
        if self.intervals:
            stats.update({
                "min_interval_ms": min(self.intervals) * 1000,
                "max_interval_ms": max(self.intervals) * 1000,
                "avg_interval_ms": np.mean(self.intervals) * 1000,
            })
        
        if self.rates:
            stats.update({
                "current_rate_hz": self.rates[-1] if self.rates else 0,
                "avg_rate_hz": np.mean(self.rates) if self.rates else 0,
            })
            
        if self.jitter:
            stats.update({
                "avg_jitter_pct": np.mean(self.jitter) * 100,
                "max_jitter_pct": max(self.jitter) * 100,
            })
            
        return stats


class DiagnosticLogger:
    """
    Enhanced diagnostic logger for autonomous system monitoring with multiple specialized
    logging categories and runtime toggling capability.
    """
    
    def __init__(self, auto_log_interval: float = 5.0):
        # Get specialized loggers
        self.logger = get_diagnostic_logger()
        self.sensor_logger = get_sensor_logger()
        self.sync_logger = get_sync_logger() 
        self.log_callback = None
        
        # Sensor statistics tracking
        self.sensor_stats = {
            "imu": SensorStats(expected_interval=0.01),  # 100Hz expected
            "camera": SensorStats(expected_interval=0.033),  # 30Hz expected
            "lane": SensorStats(expected_interval=0.033),
            "control": SensorStats(expected_interval=0.02),  # 50Hz expected
        }
        
        # Synchronization results
        self.sync_results: Deque[Tuple[bool, float]] = deque(maxlen=100)
        
        # System resource monitoring
        self.cpu_usage: Deque[float] = deque(maxlen=120)  # 2 minutes at 1Hz
        self.memory_usage: Deque[float] = deque(maxlen=120)
        self.disk_usage: Dict[str, Deque[float]] = {}
        
        # Logging control
        self.auto_log_interval = auto_log_interval
        self.last_log_time = time.time()
        self.auto_logging_enabled = True
        
        # Enable diagnostic categories based on configuration
        self.enabled_categories = {
            "sensors": False,
            "sync": False,
            "control": False,
            "system": True,
            "performance": False,
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start system monitoring thread if enabled
        if self.enabled_categories["system"]:
            self._start_system_monitoring()
        
        self.logger.info("Diagnostic logger initialized")
        
    
    def set_log_callback(self, callback):
        """set a callback function that will be called with category and message when logging"""
        self.log_callback = callback

    def set_log_window(self, log_window):
        """set a log window for visuals of log"""
        self.set_log_callback(log_window.add_log)
        #add initial message 
        if self.log_callback:
            self.log_callback("diagnostic", "Connected to Diagnostic Logger")

    def _start_system_monitoring(self):
        """Start a background thread to monitor system resources"""
        def monitor_system():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    self.cpu_usage.append(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.append(memory.percent)
                    
                    # Disk usage for root filesystem
                    root_usage = psutil.disk_usage('/').percent
                    if '/' not in self.disk_usage:
                        self.disk_usage['/'] = deque(maxlen=120)
                    self.disk_usage['/'].append(root_usage)
                    
                    # Log if system resources are critical
                    if cpu_percent > 90:
                        self.logger.warning(f"CPU usage critical: {cpu_percent}%")
                    if memory.percent > 90:
                        self.logger.warning(f"Memory usage critical: {memory.percent}%")
                    if root_usage > 90:
                        self.logger.warning(f"Disk usage critical: {root_usage}%")
                    
                    # Sleep for 1 second
                    time.time_ns()  # Force a time update
                    time.sleep(1.0)
                    
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                    time.sleep(5.0)  # Longer sleep on error
        
        # Start the monitoring thread
        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.name = "SystemMonitor"
        thread.start()
        self.logger.info("System monitoring thread started")
    
    def add_imu_timestamp(self, timestamp: float):
        """Record an IMU data timestamp"""
        with self.lock:
            self.sensor_stats["imu"].update(timestamp)
            
            # If sensor logging is enabled, log at a reasonable rate
            if self.enabled_categories["sensors"]:
                if len(self.sensor_stats["imu"].timestamps) % 100 == 0:  # Log every 100 samples
                    stats = self.sensor_stats["imu"].get_stats()
                    self.sensor_logger.info(f"IMU stats: rate={stats['avg_rate_hz']:.2f}Hz, "
                                          f"interval={stats['avg_interval_ms']:.1f}ms, "
                                          f"dropped={stats['dropped_count']}")
    
    def add_camera_timestamp(self, timestamp: float):
        """Record a camera frame timestamp"""
        with self.lock:
            self.sensor_stats["camera"].update(timestamp)
            
            # If sensor logging is enabled, log at a reasonable rate
            if self.enabled_categories["sensors"]:
                if len(self.sensor_stats["camera"].timestamps) % 30 == 0:  # Log every 30 frames
                    stats = self.sensor_stats["camera"].get_stats()
                    self.sensor_logger.info(f"Camera stats: rate={stats['avg_rate_hz']:.2f}Hz, "
                                          f"interval={stats['avg_interval_ms']:.1f}ms, "
                                          f"dropped={stats['dropped_count']}")
    
    def add_lane_detection_timestamp(self, timestamp: float):
        """Record a lane detection timestamp"""
        with self.lock:
            self.sensor_stats["lane"].update(timestamp)
            
            # If sensor logging is enabled, log at a reasonable rate
            if self.enabled_categories["sensors"]:
                if len(self.sensor_stats["lane"].timestamps) % 10 == 0:  # Log every 10 detections
                    stats = self.sensor_stats["lane"].get_stats()
                    self.sensor_logger.info(f"Lane detection stats: rate={stats['avg_rate_hz']:.2f}Hz, "
                                          f"interval={stats['avg_interval_ms']:.1f}ms")
    
    def add_control_timestamp(self, timestamp: float):
        """Record a control update timestamp"""
        with self.lock:
            self.sensor_stats["control"].update(timestamp)
            
            # If control logging is enabled, log at a reasonable rate
            if self.enabled_categories["control"]:
                if len(self.sensor_stats["control"].timestamps) % 50 == 0:  # Log every 50 updates
                    stats = self.sensor_stats["control"].get_stats()
                    self.sensor_logger.info(f"Control stats: rate={stats['avg_rate_hz']:.2f}Hz, "
                                          f"interval={stats['avg_interval_ms']:.1f}ms")
    
    def add_sync_result(self, success: bool, time_diff: float):
        """Record a sensor synchronization result"""
        with self.lock:
            self.sync_results.append((success, time_diff))
            
            # If sync logging is enabled, log individual results
            if self.enabled_categories["sync"]:
                if success:
                    self.sync_logger.debug(f"Sync successful: time_diff={time_diff*1000:.1f}ms")
                else:
                    self.sync_logger.warning(f"Sync failed: time_diff={time_diff*1000:.1f}ms")
    

    def toggle_category(self, category: str, enable: Optional[bool] = None):
        """Toggle a diagnostic category on/off"""
        with self.lock:
            if category in self.enabled_categories:
                # Toggle if no state provided
                if enable is None:
                    enable = not self.enabled_categories[category]
                    
                self.enabled_categories[category] = enable
                self.logger.info(f"{'Enabled' if enable else 'Disabled'} {category} diagnostics")

                if hasattr(self, 'log_window'):
                    self.log_window.add_log("system", f"{'Enabled' if enable else 'Disabled'} {category} diagnostics")
                
                # Special handling for system monitoring
                if category == "system" and enable and not any(t.name == "SystemMonitor" for t in threading.enumerate()):
                    self._start_system_monitoring()
                
                # Immediately log current state if enabling
                if enable:
                    self._log_category(category)
                
                return True
            else:
                self.logger.warning(f"Unknown diagnostic category: {category}")
                return False
    
    def toggle_auto_logging(self, enable: Optional[bool] = None):
        """Toggle automatic periodic logging"""
        with self.lock:
            if enable is None:
                self.auto_logging_enabled = not self.auto_logging_enabled
            else:
                self.auto_logging_enabled = enable
                
            self.logger.info(f"{'Enabled' if self.auto_logging_enabled else 'Disabled'} automatic logging")
    
    def log_if_needed(self):
        """Log diagnostics if the auto-log interval has elapsed"""
        current_time = time.time()
        
        with self.lock:
            if self.auto_logging_enabled and (current_time - self.last_log_time > self.auto_log_interval):
                self._log_diagnostics()
                self.last_log_time = current_time
    
    def force_log_all(self):
        """Force logging of all diagnostic categories"""
        with self.lock:
            self.logger.info("--- DIAGNOSTIC SUMMARY ---")
            for category in self.enabled_categories:
                if self.enabled_categories[category]:
                    self._log_category(category)
            self.logger.info("--- END DIAGNOSTIC SUMMARY ---")
    
    def _log_category(self, category: str):
        """Log a specific diagnostic category"""
        if category == "sensors":
            self._log_sensor_stats()
        elif category == "sync":
            self._log_sync_stats()
        elif category == "system":
            self._log_system_stats()
        elif category == "control":
            self._log_control_stats()
        elif category == "performance":
            self._log_performance_stats()
    
    def _log_diagnostics(self):
        """Log all enabled diagnostic categories"""
        self.logger.info("--- PERIODIC DIAGNOSTICS ---")
        
        for category, enabled in self.enabled_categories.items():
            if enabled:
                self._log_category(category)
                
        self.logger.info("--- END PERIODIC DIAGNOSTICS ---")
    
    def _log_sensor_stats(self):
        """Log sensor statistics"""
        for sensor_name, stats in self.sensor_stats.items():
            if len(stats.timestamps) > 1:
                data = stats.get_stats()
                message = (
                    f"{sensor_name.upper()} Stats: "
                    f"Rate={data.get('avg_rate_hz', 0):.2f}Hz, "
                    f"Interval={data.get('avg_interval_ms', 0):.1f}ms, "
                    f"Min={data.get('min_interval_ms', 0):.1f}ms, "
                    f"Max={data.get('max_interval_ms', 0):.1f}ms, "
                    f"Dropped={data.get('dropped_count', 0)}"
                )
                self.sensor_logger.info(message)

                if self.log_callback:
                    self.log_callback("sensors", message)
                
        
    def _log_sync_stats(self):
        """Log synchronization statistics"""
        if self.sync_results:
            success_count = sum(1 for s, _ in self.sync_results if s)
            if success_count > 0:
                time_diffs = [d*1000 for s, d in self.sync_results if s]  # Convert to ms
                message = (
                    f"Sync Success Rate: {success_count/len(self.sync_results)*100:.1f}%, "
                    f"Avg Diff: {np.mean(time_diffs):.1f}ms, "
                    f"Max Diff: {np.max(time_diffs):.1f}ms, "
                    f"Min Diff: {np.min(time_diffs):.1f}ms"
                )
                self.sync_logger.info(message)
                
                # Forward to log window via callback
                if self.log_callback:
                    self.log_callback("sync", message)
            else:
                message = "No successful sensor synchronizations in last period"
                self.sync_logger.warning(message)
                
                # Forward to log window via callback
                if self.log_callback:
                    self.log_callback("sync", message)
    
    def _log_system_stats(self):
        """Log system resource statistics"""
        messages = []
        
        # CPU usage
        if self.cpu_usage:
            cpu_message = f"CPU Usage: Current={self.cpu_usage[-1]:.1f}%, Avg={np.mean(self.cpu_usage):.1f}%, Max={np.max(self.cpu_usage):.1f}%"
            self.logger.info(cpu_message)
            messages.append(cpu_message)
        
        # Memory usage
        if self.memory_usage:
            mem_message = f"Memory Usage: Current={self.memory_usage[-1]:.1f}%, Avg={np.mean(self.memory_usage):.1f}%"
            self.logger.info(mem_message)
            messages.append(mem_message)
        
        # Disk usage
        for path, usage in self.disk_usage.items():
            if usage:
                disk_message = f"Disk Usage ({path}): Current={usage[-1]:.1f}%"
                self.logger.info(disk_message)
                messages.append(disk_message)
        
        # Thread information
        thread_count = threading.active_count()
        thread_message = f"Active threads: {thread_count}"
        self.logger.info(thread_message)
        messages.append(thread_message)
        
        # Forward to log window via callback
        if self.log_callback:
            for message in messages:
                self.log_callback("system", message)


    def _log_performance_stats(self):
        """Log performance statistics - placeholder for module-specific metrics"""
        message = "Performance metrics available in performance.log"
        self.logger.info(message)
        
        # Forward to log window via callback
        if self.log_callback:
            self.log_callback("performance", message)

    def log_performance_metrics(self, module: str, execution_time: float, additional_info: Optional[Dict] = None):
        """Log performance metrics for a specific module"""
        if self.enabled_categories["performance"]:
            info_str = ""
            if additional_info:
                info_str = ", ".join(f"{k}={v}" for k, v in additional_info.items())
            
            message = f"Performance - {module}: {execution_time*1000:.2f}ms {info_str}"
            self.logger.info(message)
            
            # Forward to log window via callback
            if self.log_callback:
                self.log_callback("performance", message)