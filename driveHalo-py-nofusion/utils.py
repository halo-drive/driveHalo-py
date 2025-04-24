import logging
import time
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def measure_execution_time(func):
    """Decorator to measure function execution time"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # ms
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Function {func.__name__} executed in {execution_time:.2f} ms")
        return result

    return wrapper


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (416, 416)) -> np.ndarray:
    """Preprocess frame for TensorRT inference"""
    resized = cv2.resize(frame, target_size)
    normalized = resized.astype(np.float32) / 255.0
    chw = normalized.transpose((2, 0, 1))  # HWC to CHW
    batched = np.expand_dims(chw, axis=0)  # Add batch dimension
    return batched


def draw_overlay(frame: np.ndarray,
                 info_dict: Dict[str, Any],
                 position: str = 'bottom',
                 font_scale: float = 0.8,
                 thickness: int = 2,
                 color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw information overlay on frame"""
    height, width = frame.shape[:2]
    result = frame.copy()

    # Convert info_dict to list of strings
    lines = [f"{key}: {value}" for key, value in info_dict.items()]

    if position.lower() == 'bottom':
        y_start = height - 30 * len(lines)
    else:  # top
        y_start = 30

    for i, line in enumerate(lines):
        y = y_start + i * 30
        cv2.putText(result, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)

    return result