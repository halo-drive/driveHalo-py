import logging
import os
import time
import sys
from typing import Dict, Optional, List
import threading

class LoggingManager:
    """
    Centralized logging management system for the autonomous driving platform.
    
    Features:
    - Hierarchical logger organization
    - Dynamic log level adjustment at runtime
    - Custom log handlers (file, console)
    - Specialized logging categories (diagnostic, sensor metrics, sync)
    - Memory-efficient rotating log files
    """
    
    # Log level mapping for easy reference
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,       # 10
        'INFO': logging.INFO,         # 20
        'WARNING': logging.WARNING,   # 30
        'ERROR': logging.ERROR,       # 40
        'CRITICAL': logging.CRITICAL  # 50
    }
    
    def __init__(self, log_dir: str = 'logs', 
                 default_level: int = logging.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5):
        """
        Initialize the logging manager with configurable parameters.
        
        Args:
            log_dir: Directory to store log files
            default_level: Default logging level
            enable_console: Whether to log to console
            enable_file: Whether to log to files
            max_file_size_mb: Maximum size of each log file in MB
            backup_count: Number of backup log files to keep
        """
        # Create log directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Store configuration
        self.default_level = default_level
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        
        # Keep track of all loggers
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Special logger categories
        self.special_loggers = {
            'diagnostic': None,
            'sensor': None,
            'sync': None,
            'control': None,
            'performance': None
        }
        
        # Standard log format
        self.standard_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        
        # Detailed format for debugging
        self.detailed_format = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s [%(threadName)s] - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
        
        # State lock for thread safety
        self.lock = threading.RLock()
        
        # Global logging rate limiter (messages per module per level)
        self.rate_limiters = {}
        
        # Initialize root logger
        self._configure_root_logger()
        
        # Initialize special loggers
        self._initialize_special_loggers()
        
    def _configure_root_logger(self):
        """Configure the root logger with console and file handlers."""
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Remove any existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Set level
        root_logger.setLevel(self.default_level)
        
        # Add console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.standard_format)
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if self.enable_file:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                os.path.join(self.log_dir, 'main.log'),
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(self.detailed_format)
            root_logger.addHandler(file_handler)
        
        # Store the root logger
        self.loggers['root'] = root_logger
    
    def _initialize_special_loggers(self):
        """Initialize special purpose loggers."""
        for category in self.special_loggers:
            logger = self.get_logger(category)
            
            # Create a separate log file for each special logger
            if self.enable_file:
                from logging.handlers import RotatingFileHandler
                
                file_handler = RotatingFileHandler(
                    os.path.join(self.log_dir, f'{category}.log'),
                    maxBytes=self.max_file_size_mb * 1024 * 1024,
                    backupCount=self.backup_count
                )
                file_handler.setFormatter(self.detailed_format)
                logger.addHandler(file_handler)
            
            # Store the special logger
            self.special_loggers[category] = logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name, typically the module name
            
        Returns:
            Configured logger instance
        """
        with self.lock:
            if name in self.loggers:
                return self.loggers[name]
            
            # Create new logger
            logger = logging.getLogger(name)
            
            # Don't propagate to root logger if it's a special logger
            if name in self.special_loggers:
                logger.propagate = False
            
            # Store and return
            self.loggers[name] = logger
            return logger
    
    def set_level(self, level: int, loggers: Optional[List[str]] = None):
        """
        Set the logging level for specified loggers or all loggers.
        
        Args:
            level: Logging level (e.g., logging.INFO)
            loggers: List of logger names to update, or None for all loggers
        """
        with self.lock:
            if loggers is None:
                # Update all loggers including root
                for logger in self.loggers.values():
                    logger.setLevel(level)
            else:
                # Update only specified loggers
                for name in loggers:
                    if name in self.loggers:
                        self.loggers[name].setLevel(level)
    
    def get_diagnostic_logger(self):
        """Get the diagnostic logger for system health monitoring."""
        return self.special_loggers['diagnostic']
    
    def get_sensor_logger(self):
        """Get the sensor metrics logger."""
        return self.special_loggers['sensor']
    
    def get_sync_logger(self):
        """Get the synchronization logger."""
        return self.special_loggers['sync']
    
    def get_control_logger(self):
        """Get the control system logger."""
        return self.special_loggers['control']
    
    def get_performance_logger(self):
        """Get the performance monitoring logger."""
        return self.special_loggers['performance']
    
    def toggle_special_logger(self, category: str, enable: bool):
        """
        Enable or disable a special logger category.
        
        Args:
            category: Logger category name
            enable: Whether to enable or disable
        """
        if category in self.special_loggers and self.special_loggers[category]:
            level = logging.DEBUG if enable else logging.CRITICAL
            self.special_loggers[category].setLevel(level)
            
            # Log the state change to the root logger
            self.loggers['root'].info(f"{'Enabled' if enable else 'Disabled'} {category} logging")
    
    def rate_limited_log(self, logger: logging.Logger, level: int, message: str, 
                          limit_per_second: float = 1.0):
        """
        Rate-limited logging to prevent log flooding.
        
        Args:
            logger: Logger to use
            level: Log level
            message: Log message
            limit_per_second: Maximum messages per second
        """
        with self.lock:
            # Create unique key for this logger+level combination
            key = f"{logger.name}:{level}"
            
            current_time = time.time()
            
            # Initialize or check rate limiter for this key
            if key not in self.rate_limiters:
                self.rate_limiters[key] = current_time
                logger.log(level, message)
            else:
                last_log_time = self.rate_limiters[key]
                time_diff = current_time - last_log_time
                
                if time_diff >= (1.0 / limit_per_second):
                    logger.log(level, message)
                    self.rate_limiters[key] = current_time
    
    def log_system_info(self):
        """Log system information for diagnostics."""
        import platform
        import psutil
        
        logger = self.get_logger('system')
        
        # System info
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"Processor: {platform.processor()}")
        
        # Resource usage
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.percent}% used ({memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB)")
        
        for disk in psutil.disk_partitions():
            usage = psutil.disk_usage(disk.mountpoint)
            logger.info(f"Disk {disk.mountpoint}: {usage.percent}% used ({usage.used / (1024**3):.1f} GB / {usage.total / (1024**3):.1f} GB)")
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        logger.info(f"CPU: {cpu_percent}% used")
        
    def flush_all(self):
        """Flush all loggers to ensure logs are written."""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.flush()


# Global logging manager instance
_logging_manager = None

def initialize_logging(log_dir: str = 'logs', 
                      default_level: int = logging.INFO,
                      enable_console: bool = True,
                      enable_file: bool = True):
    """
    Initialize the global logging manager.
    
    Args:
        log_dir: Directory to store log files
        default_level: Default logging level
        enable_console: Whether to log to console
        enable_file: Whether to log to files
    """
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager(
            log_dir=log_dir,
            default_level=default_level,
            enable_console=enable_console,
            enable_file=enable_file
        )
    return _logging_manager

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = initialize_logging()
    return _logging_manager.get_logger(name)

def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = initialize_logging()
    return _logging_manager

# Convenience functions for special loggers
def get_diagnostic_logger():
    """Get the diagnostic logger."""
    return get_logging_manager().get_diagnostic_logger()

def get_sensor_logger():
    """Get the sensor metrics logger."""
    return get_logging_manager().get_sensor_logger()

def get_sync_logger():
    """Get the synchronization logger."""
    return get_logging_manager().get_sync_logger()

def get_control_logger():
    """Get the control system logger."""
    return get_logging_manager().get_control_logger()

def get_performance_logger():
    """Get the performance monitoring logger."""
    return get_logging_manager().get_performance_logger()