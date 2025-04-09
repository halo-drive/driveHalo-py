# lane_detection.py

import cv2
import torch
from ultralytics import YOLO
import logging
import time
from typing import Tuple, Optional
import numpy as np

class OptimizedCamera:
    """Optimized camera capture implementation using GStreamer pipeline"""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None

    def build_gstreamer_pipeline(self) -> str:
        """Build optimized GStreamer pipeline for video capture"""
        return (
            f"v4l2src device=/dev/video{self.device_id} ! "
            f"video/x-raw, width={self.width}, height={self.height}, "
            "format=YUY2, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink max-buffers=1 drop=true"
        )

    def initialize(self) -> bool:
        """Initialize camera with optimized settings"""
        try:
            # Try GStreamer pipeline first
            pipeline = self.build_gstreamer_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                self.logger.warning("GStreamer pipeline failed, falling back to default capture")
                self.cap = cv2.VideoCapture(self.device_id)

                if not self.cap.isOpened():
                    self.logger.error("Failed to open camera")
                    return False

                # Configure optimal camera settings
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify configuration
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(
                f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps"
            )
            return True

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with error handling"""
        if not self.cap or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                return False, None
            return True, frame

        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return False, None

    def release(self):
        """Safely release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """Context manager entry"""
        if self.initialize():
            return self
        raise RuntimeError("Failed to initialize camera")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

class LaneDetector:
    def __init__(self, model_path):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)

        # Initialize FPS counter
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0

    def run(self, source=0):
        """Run lane detection with optimized camera pipeline"""
        self.logger.info(f"Starting lane detection on source: {source}")

        try:
            with OptimizedCamera(device_id=source) as camera:
                while True:
                    success, frame = camera.read_frame()
                    if not success:
                        self.logger.error("Failed to read frame")
                        break

                    # Run inference
                    results = self.model(frame, verbose=False)

                    # Process results
                    annotated_frame = results[0].plot()

                    # Update FPS
                    self.frame_count += 1
                    if time.time() - self.fps_start_time >= 1:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.fps_start_time = time.time()

                    # Add FPS to frame
                    cv2.putText(annotated_frame, f'FPS: {self.fps}', (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Display the frame
                    cv2.imshow('Lane Detection', annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        finally:
            cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam)')
    args = parser.parse_args()

    detector = LaneDetector(args.model)
    detector.run(0 if args.source == '2' else args.source)


if __name__ == '__main__':
    main()