import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque
import argparse
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EnhancedLaneTracker:
    """
    Enhanced lane tracking with state persistence, RANSAC fitting,
    and gap handling for the Drive AGX Orin platform.
    """

    def __init__(self, engine_path, camera_id=0, debug_visualization=True):
        """Initialize the lane tracker with TensorRT engine"""
        logger.info(f"Initializing Enhanced Lane Tracker with engine: {engine_path}")
        
        # Lane tracking state
        self.LANE_STATE = {"UNKNOWN": 0, "LEFT": 1, "RIGHT": 2, "TRANSITIONING": 3}
        self.current_lane_state = self.LANE_STATE["UNKNOWN"]
        self.lane_state_counter = 0
        self.transition_progress = 0.0
        self.transition_target = self.LANE_STATE["UNKNOWN"]
        
        # Lane history for tracking and prediction
        self.lane_history = {
            'left': deque(maxlen=30),
            'right': deque(maxlen=30)
        }
        self.lane_timestamps = deque(maxlen=30)
        self.lane_confidences = {
            'left': deque(maxlen=30),
            'right': deque(maxlen=30)
        }
        
        # Current polynomial fits
        self.left_fit = None
        self.right_fit = None
        self.left_confidence = 0.0
        self.right_confidence = 0.0
        
        # Bird's-eye view transformation setup
        self.setup_perspective_transform()
        
        # Stanley controller parameters
        self.k_crosstrack = 0.8
        self.k_heading = 1.2
        self.k_lookahead = 0.5
        
        # Real-world scale parameters
        self.xm_per_pix = 0.01395  # meters per pixel in x dimension
        self.ym_per_pix = 0.0224   # meters per pixel in y dimension
        
        # Inference parameters - from ONNX model specs
        self.input_h = 416
        self.input_w = 416
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Visualization flag
        self.show_debug_visualizations = debug_visualization
        
        # Initialize TensorRT
        self.init_tensorrt(engine_path)
        
        # Camera initialization
        self.camera_id = camera_id
        self.init_camera()
        
        logger.info("Lane tracker initialized successfully")

    def init_tensorrt(self, engine_path):
        """Initialize TensorRT with dynamic tensor identification"""
        logger.info("Initializing TensorRT engine")

        # Create TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load engine from file
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Input shape
        self.input_shape = (1, 3, self.input_h, self.input_w)

        # Identify tensor bindings
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_tensor_shape(binding_name)
            logger.info(f"Binding {i}: {binding_name} with shape {binding_shape}")

        # Based on the logged binding information:
        self.input_idx = 0  # Input tensor at index 0
        self.prototype_idx = 1  # Prototype tensor at index 1
        self.detection_idx = 2  # Detection tensor at index 2

        # Define output shapes directly from the logs
        self.output0_shape = (1, 37, 3549)  # Detection output
        self.output1_shape = (1, 32, 104, 104)  # Prototype masks

        logger.info(
            f"Identified tensors: input={self.input_idx}, detection={self.detection_idx}, prototype={self.prototype_idx}")

        # Calculate buffer sizes
        input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        output0_size = trt.volume(self.output0_shape) * np.dtype(np.float32).itemsize
        output1_size = trt.volume(self.output1_shape) * np.dtype(np.float32).itemsize

        # Allocate device memory
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output0 = cuda.mem_alloc(output0_size)
        self.d_output1 = cuda.mem_alloc(output1_size)

        # Allocate host memory (page-locked)
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output0 = cuda.pagelocked_empty(self.output0_shape, dtype=np.float32)
        self.h_output1 = cuda.pagelocked_empty(self.output1_shape, dtype=np.float32)

        # Create CUDA stream
        self.stream = cuda.Stream()

        logger.info("TensorRT engine initialized successfully")

    def init_camera(self):
        """Initialize camera with optimized GStreamer pipeline for Jetson"""
        logger.info(f"Initializing camera #{self.camera_id}")

        # Simple v4l2src pipeline that works reliably on Jetson platforms
        gst_pipeline = (
            f"v4l2src device=/dev/video{self.camera_id} ! "
            "video/x-raw, width=640, height=480, format=YUY2, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink max-buffers=1 drop=true"
        )

        logger.info(f"Using pipeline: {gst_pipeline}")
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            logger.warning("Failed to open camera with GStreamer, falling back to default")
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Set optimal buffer size
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get actual camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height}")
    
    def setup_perspective_transform(self):
        """Setup bird's-eye view transformation matrices"""
        # These points should be calibrated for your specific camera
        self.warped_width = 300
        self.warped_height = 400
        
        # Source points in camera view (these need calibration)
        self.birdeye_src = np.float32([
            [177.0, 170.0], [410.0, 164.0],
            [58.0, 336.0],  [544.0, 327.0]
        ])
        
        # Destination points in bird's eye view
        self.birdeye_dst = np.float32([
            [0, 0],
            [self.warped_width, 0],
            [0, self.warped_height],
            [self.warped_width, self.warped_height]
        ])
        
        # Compute transformation matrices
        self.birdeye_matrix = cv2.getPerspectiveTransform(self.birdeye_src, self.birdeye_dst)
        self.birdeye_inv_matrix = cv2.getPerspectiveTransform(self.birdeye_dst, self.birdeye_src)
    
    def preprocess_image(self, frame):
        """Preprocess image for TensorRT inference with letterboxing"""
        # Calculate scaling ratio to maintain aspect ratio
        input_height, input_width = self.input_h, self.input_w
        image_height, image_width = frame.shape[:2]
        
        r = min(input_width / image_width, input_height / image_height)
        new_width, new_height = int(image_width * r), int(image_height * r)
        
        # Resize
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Create letterboxed image
        letterboxed = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        offset_x, offset_y = (input_width - new_width) // 2, (input_height - new_height) // 2
        letterboxed[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
        
        # Normalize and transpose
        preprocessed = letterboxed.astype(np.float32) / 255.0
        preprocessed = np.transpose(preprocessed, (2, 0, 1))  # HWC to CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        
        return np.ascontiguousarray(preprocessed), (offset_x, offset_y, new_width, new_height)

    def infer(self, input_data):
        """Execute TensorRT inference with proper memory management"""
        # Copy input data to GPU
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)

        # Execute inference with the correct binding order
        bindings = [0] * self.engine.num_bindings
        bindings[self.input_idx] = int(self.d_input)
        bindings[self.detection_idx] = int(self.d_output0)
        bindings[self.prototype_idx] = int(self.d_output1)

        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.stream.handle
        )

        # Copy outputs back to host
        cuda.memcpy_dtoh_async(self.h_output0, self.d_output0, self.stream)
        cuda.memcpy_dtoh_async(self.h_output1, self.d_output1, self.stream)

        # Synchronize stream to ensure all operations complete
        self.stream.synchronize()

        return [self.h_output0.copy(), self.h_output1.copy()]
    def process_yolo_outputs(self, outputs):
        """
        Process YOLOv8-seg model outputs to generate lane mask with shape handling.
        
        Parameters:
            outputs: List of tensors from TensorRT engine inference
                [0]: Detection output [1, 37, 3549]
                [1]: Prototype masks [1, 32, 104, 104]
        
        Returns:
            Processed binary mask (uint8)
        """
        # Log tensor shapes for diagnostics
        for i, output in enumerate(outputs):
            logger.debug(f"Output tensor[{i}] shape: {output.shape}, dtype: {output.dtype}")
        
        # Validate output count
        if len(outputs) < 2:
            logger.error(f"Insufficient outputs: expected 2, got {len(outputs)}")
            return np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        
        # Process detection tensor - first output
        detection_tensor = outputs[0]
        
        # Process prototype masks tensor - second output
        proto_tensor = outputs[1]
        
        # Ensure detection tensor has correct format [1, 37, 3549]
        if detection_tensor.shape[1] == 37:
            # Extract confidence scores (index 4) and mask coefficients (indices 5-36)
            conf = detection_tensor[0, 4, :]  # [3549]
            mask_coefs = detection_tensor[0, 5:, :]  # [32, 3549]
        else:
            logger.error(f"Unexpected detection tensor shape: {detection_tensor.shape}")
            return np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        
        # Ensure prototype masks tensor has correct format [1, 32, 104, 104]
        if len(proto_tensor.shape) == 4:
            proto_masks = proto_tensor[0]  # [32, 104, 104]
            mask_height, mask_width = proto_masks.shape[1], proto_masks.shape[2]
        else:
            logger.error(f"Unexpected prototype masks shape: {proto_tensor.shape}")
            return np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        
        # Create final mask
        final_mask = np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        
        # Apply confidence threshold
        valid_idx = np.where(conf > self.conf_threshold)[0]
        
        if len(valid_idx) > 0:
            # Process each valid detection
            for idx in valid_idx:
                try:
                    # Create empty base mask
                    mask_base = np.zeros((mask_height, mask_width), dtype=np.float32)
                    
                    # Combine prototype masks with coefficients
                    for c_idx in range(proto_masks.shape[0]):
                        mask_base += mask_coefs[c_idx, idx] * proto_masks[c_idx]
                    
                    # Apply sigmoid activation
                    mask_sigmoid = 1.0 / (1.0 + np.exp(-mask_base))
                    
                    # Resize to input dimensions and threshold
                    mask_resized = cv2.resize(mask_sigmoid, (self.input_w, self.input_h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Add to final mask
                    final_mask = cv2.bitwise_or(final_mask, mask_binary)
                except Exception as e:
                    logger.error(f"Error processing mask for detection {idx}: {str(e)}")
                    continue
        
        # Apply morphological operations for cleaning
        if cv2.countNonZero(final_mask) > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask_closed = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
            return mask_clean
        
        return final_mask
    
    def detect_lanes(self, frame):
        """Execute inference and generate lane mask"""
        # Preprocess image
        input_data, preprocess_info = self.preprocess_image(frame)
        
        # Convert to contiguous array for CUDA
        input_buffer = np.ascontiguousarray(input_data)
        
        # Execute inference
        outputs = self.infer(input_buffer)
        
        # Process outputs to generate mask
        lane_mask = self.process_yolo_outputs(outputs)
        
        # Transform to bird's-eye view
        warped_mask = cv2.warpPerspective(
            lane_mask, self.birdeye_matrix,
            (self.warped_width, self.warped_height),
            flags=cv2.INTER_LINEAR
        )
        
        return lane_mask, warped_mask
    
    def find_lane_contours(self, warped_mask):
        """Find and validate lane contours in bird's-eye view"""
        contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        left_contour, right_contour = None, None
        left_confidence, right_confidence = 0.0, 0.0
        center_x = self.warped_width // 2
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 100 or len(cnt) < 5:
                continue
            mean_x = np.mean(cnt[:, 0, 0])
            if mean_x < center_x:
                valid, conf = self.validate_lane_contour(cnt, [p for p in self.lane_history['left'] if p is not None])
                if valid and conf > left_confidence:
                    left_contour, left_confidence = cnt, conf
            else:
                valid, conf = self.validate_lane_contour(cnt, [p for p in self.lane_history['right'] if p is not None])
                if valid and conf > right_confidence:
                    right_contour, right_confidence = cnt, conf
        
        return left_contour, right_contour, left_confidence, right_confidence
    
    def validate_lane_contour(self, contour, prev_polys):
        """Validate contour as a lane boundary based on shape and temporal consistency"""
        if contour is None or len(contour) < 10:
            return False, 0.0
        
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, False)
        if area < 100 or length < 50:
            return False, 0.0
        
        rect = cv2.minAreaRect(contour)
        if rect[1][0] > 0 and rect[1][1] > 0:
            ratio = max(rect[1]) / min(rect[1])
            if ratio < 2.0 or ratio > 20.0:
                return False, 0.0
        
        # Temporal consistency check
        temporal_score = 1.0
        if prev_polys and len(prev_polys) >= 3:
            # Simple temporal consistency score
            temporal_score = 0.8
        
        confidence = min(1.0, (area / 1000.0) * 0.7 + temporal_score * 0.3)
        return True, confidence
    
    def expand_shifted_model(self, model, shift, scale):
        """
        Expand a polynomial x = a*(y')^2 + b*y' + c fit in shifted/scaled space
        back into x = A*y^2 + B*y + C in original y space.
        """
        a, b, c_ = model[0], model[1], model[2]
        
        # For y' = (y - shift)/scale,
        # x = a*((y - shift)/scale)^2 + b*((y - shift)/scale) + c_
        # Expand to get A, B, C in x = A*y^2 + B*y + C:
        A = a / (scale * scale)
        B = -2.0 * a * shift / (scale * scale) + b / scale
        C = (a * (shift ** 2) / (scale ** 2)) - (b * shift / scale) + c_
        
        return np.array([A, B, C], dtype=np.float32)

    def ransac_poly_fit(self, points, degree=2, max_trials=100, residual_threshold=2.0):
        """
        RANSAC polynomial fit with improved numerical stability
        """
        if points is None or len(points) < 5:
            return None, 0.0

        x_coords = points[:, 0].astype(np.float64)  # Use double precision
        y_coords = points[:, 1].astype(np.float64)

        # Check for sufficient variation in the data
        if np.std(y_coords) < 1e-6:
            logger.warning("Insufficient variation in y coordinates for reliable fitting")
            return None, 0.0

        # Calculate global shift and scale for enhanced numerical stability
        y_min = y_coords.min()
        y_max = y_coords.max()
        y_range = max(y_max - y_min, 1.0)

        SHIFT = y_min
        SCALE = y_range / 100.0  # Normalize to reasonable range

        # Apply y' = (y - SHIFT)/SCALE for all points
        y_shifted_full = (y_coords - SHIFT) / SCALE

        # Set rcond parameter for np.polyfit to control conditioning
        rcond = 1e-10  # Smaller values keep more information but may be less stable

        best_model = None
        best_score = 0.0
        best_inliers = None

        for _ in range(max_trials):
            # Sample minimal set with sufficient spacing
            if len(points) <= (degree + 1):
                sample_idx = np.arange(len(points))
            else:
                # Try to get well-distributed sample points
                sample_idx = np.random.choice(len(points), degree + 1, replace=False)

                # Check if selected points have good spread
                if np.std(y_coords[sample_idx]) < y_range * 0.1:
                    continue  # Skip this sample if points are too clustered

            try:
                # Fit in shifted domain with explicit rcond parameter
                sample_y_shifted = y_shifted_full[sample_idx]
                sample_x = x_coords[sample_idx]

                # Suppress warnings during trial fitting
                with np.errstate(all='ignore'):
                    model_shifted = np.polyfit(sample_y_shifted, sample_x, degree, rcond=rcond)
            except:
                continue

            # Evaluate residuals
            x_pred = np.polyval(model_shifted, y_shifted_full)
            residuals = np.abs(x_coords - x_pred)
            inliers = residuals < residual_threshold
            inlier_count = np.sum(inliers)

            # Score: more inliers + lower sum of residual
            if inlier_count > degree + 1:  # Ensure we have sufficient inliers
                score = inlier_count / (1.0 + np.sum(residuals[inliers]))
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_model = model_shifted
                best_inliers = inliers

        # Refit using all inliers with more robust parameters
        if best_model is not None and best_inliers is not None and np.sum(best_inliers) >= 5:
            y_in = y_shifted_full[best_inliers]
            x_in = x_coords[best_inliers]

            try:
                # Use vandermode matrix with better conditioning for final fit
                V = np.vander(y_in, degree + 1)
                final_shifted = np.linalg.lstsq(V, x_in, rcond=1e-13)[0]

                # Expand shifted poly back into original domain
                unshifted_poly = self.expand_shifted_model(final_shifted, SHIFT, SCALE)
                return unshifted_poly, best_score
            except np.linalg.LinAlgError:
                # Fall back to polyfit if vandermode approach fails
                try:
                    final_shifted = np.polyfit(y_in, x_in, degree, rcond=1e-10)
                    unshifted_poly = self.expand_shifted_model(final_shifted, SHIFT, SCALE)
                    return unshifted_poly, best_score * 0.9  # Reduce confidence slightly
                except:
                    pass

        return None, 0.0
    def predict_lane_polynomial(self, lane_side, timestamp):
        """Predict lane polynomial during detection gaps using weighted history"""
        history = self.lane_history[lane_side]
        confidences = self.lane_confidences[lane_side]
        
        if not history or all(p is None for p in history):
            return None, 0.0
        
        valid_polys = []
        valid_confidences = []
        valid_timestamps = []
        
        for i, poly in enumerate(history):
            if poly is not None and i < len(self.lane_timestamps):
                dt = timestamp - self.lane_timestamps[i]
                if dt < 1.0:  # Use last 1 second of data
                    valid_polys.append(poly)
                    conf_i = confidences[i] if i < len(confidences) else 0.5
                    valid_confidences.append(conf_i)
                    valid_timestamps.append(self.lane_timestamps[i])
        
        if not valid_polys:
            return None, 0.0
        
        # Calculate time and confidence weighted average
        weights = []
        for i, ts in enumerate(valid_timestamps):
            time_weight = np.exp(-3.0 * (timestamp - ts))  # Exponential time decay
            confidence_weight = valid_confidences[i]
            weights.append(time_weight * confidence_weight)
        
        weights = np.array(weights)
        sum_w = weights.sum()
        if sum_w == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights /= sum_w
        
        # Weighted average polynomial coefficients
        predicted_poly = np.zeros_like(valid_polys[0])
        for i, poly in enumerate(valid_polys):
            predicted_poly += weights[i] * poly
        
        # Calculate prediction confidence based on best weight
        pred_conf = 0.8 * np.max(weights)
        return predicted_poly, pred_conf
    
    def update_lane_selection(self):
        """State machine for lane tracking with transition handling"""
        left_conf = self.lane_confidences['left'][-1] if self.lane_confidences['left'] else 0.0
        right_conf = self.lane_confidences['right'][-1] if self.lane_confidences['right'] else 0.0
        
        if self.current_lane_state == self.LANE_STATE["UNKNOWN"]:
            # Initial lane selection based on confidence
            if left_conf > 0.6 and left_conf > right_conf:
                self.current_lane_state = self.LANE_STATE["LEFT"]
                self.lane_state_counter = 0
            elif right_conf > 0.6 and right_conf > left_conf:
                self.current_lane_state = self.LANE_STATE["RIGHT"]
                self.lane_state_counter = 0
        
        elif self.current_lane_state == self.LANE_STATE["LEFT"]:
            # Potential transition to RIGHT lane
            if right_conf > 0.7 and (left_conf < 0.3 or right_conf > left_conf + 0.4):
                self.lane_state_counter += 1
                if self.lane_state_counter > 15:  # Require consistency for transition
                    self.current_lane_state = self.LANE_STATE["TRANSITIONING"]
                    self.transition_target = self.LANE_STATE["RIGHT"]
                    self.transition_progress = 0.0
            else:
                self.lane_state_counter = max(0, self.lane_state_counter - 1)
        
        elif self.current_lane_state == self.LANE_STATE["RIGHT"]:
            # Potential transition to LEFT lane
            if left_conf > 0.7 and (right_conf < 0.3 or left_conf > right_conf + 0.4):
                self.lane_state_counter += 1
                if self.lane_state_counter > 15:  # Require consistency for transition
                    self.current_lane_state = self.LANE_STATE["TRANSITIONING"]
                    self.transition_target = self.LANE_STATE["LEFT"]
                    self.transition_progress = 0.0
            else:
                self.lane_state_counter = max(0, self.lane_state_counter - 1)
        
        elif self.current_lane_state == self.LANE_STATE["TRANSITIONING"]:
            # Gradual transition between lanes
            self.transition_progress += 0.05  # 5% progress per frame
            if self.transition_progress >= 1.0:
                self.current_lane_state = self.transition_target
                self.lane_state_counter = 0
        
        return self.current_lane_state
    
    def calculate_target_polynomial(self):
        """
        Determine target path polynomial based on lane state.
        Handles lane changes and offset from boundary.
        """
        timestamp = time.time()
        
        # Use predicted lane if current fit is unavailable
        if self.left_fit is None:
            left_poly, left_conf = self.predict_lane_polynomial('left', timestamp)
        else:
            left_poly = self.left_fit
            left_conf = self.lane_confidences['left'][-1] if self.lane_confidences['left'] else 0.5
        
        if self.right_fit is None:
            right_poly, right_conf = self.predict_lane_polynomial('right', timestamp)
        else:
            right_poly = self.right_fit
            right_conf = self.lane_confidences['right'][-1] if self.lane_confidences['right'] else 0.5
        
        state = self.current_lane_state
        if state == self.LANE_STATE["LEFT"] and left_poly is not None:
            # 1.5m offset from left boundary
            offset_px = 1.5 / self.xm_per_pix
            p = left_poly.copy()
            p[2] += offset_px  # Adjust constant term
            return p, state
        
        elif state == self.LANE_STATE["RIGHT"] and right_poly is not None:
            # -1.5m offset from right boundary
            offset_px = -1.5 / self.xm_per_pix
            p = right_poly.copy()
            p[2] += offset_px  # Adjust constant term
            return p, state
        
        elif state == self.LANE_STATE["TRANSITIONING"]:
            # Smooth lane transition
            if (left_poly is not None) and (right_poly is not None):
                # Weighted blend based on transition progress
                if self.transition_target == self.LANE_STATE["LEFT"]:
                    blend = right_poly * (1.0 - self.transition_progress) + left_poly * self.transition_progress
                else:
                    blend = left_poly * (1.0 - self.transition_progress) + right_poly * self.transition_progress
                return blend, state
            elif left_poly is not None:
                return left_poly, self.LANE_STATE["LEFT"]
            elif right_poly is not None:
                return right_poly, self.LANE_STATE["RIGHT"]
        
        # Default: pick whichever lane is available
        if left_poly is not None and right_poly is not None:
            center_poly = (left_poly + right_poly) / 2.0  # Lane center
            return center_poly, self.LANE_STATE["UNKNOWN"]
        elif left_poly is not None:
            return left_poly, self.LANE_STATE["LEFT"]
        elif right_poly is not None:
            return right_poly, self.LANE_STATE["RIGHT"]
        
        return None, self.LANE_STATE["UNKNOWN"]
    
    def calculate_stanley_control(self, lateral_error, heading_error, velocity=5.0):
        """Stanley controller for lateral control"""
        velocity_term = max(0.1, velocity)  # Avoid division by zero
        crosstrack_term = np.arctan2(self.k_crosstrack * lateral_error, velocity_term)
        heading_term = self.k_heading * heading_error
        steering = (crosstrack_term + heading_term) / np.pi  # Normalize to [-1, 1]
        return np.clip(steering, -1.0, 1.0)
    
    def calculate_lookahead_curvature(self, poly_coeffs, velocity=5.0):
        """Calculate curvature at lookahead point for predictive control"""
        if poly_coeffs is None:
            return 0.0
        
        # Dynamic lookahead distance based on velocity
        lookahead_distance = 5.0 + 0.5 * velocity
        
        # For polynomial x = A*y^2 + B*y + C:
        # First derivative: dx/dy = 2A*y + B
        # Second derivative: d²x/dy² = 2A
        first_deriv = 2.0 * poly_coeffs[0] * lookahead_distance + poly_coeffs[1]
        second_deriv = 2.0 * poly_coeffs[0]
        
        # Curvature formula: ? = |f''(x)| / (1 + (f'(x))²)^(3/2)
        denom = (1 + first_deriv**2) ** 1.5
        if denom < 1e-6:
            return 0.0
            
        curvature = second_deriv / denom
        return curvature
    
    def estimate_heading_error(self, poly_coeffs):
        """Estimate heading error from polynomial"""
        if poly_coeffs is None:
            return 0.0
        
        # Tangent angle at y=0 (current position)
        # dx/dy at y=0 is the coefficient of y¹, which is B
        lane_heading = np.arctan(poly_coeffs[1])
        return lane_heading
    
    def calculate_lateral_offset(self, poly):
        """Calculate lateral offset from lane center in meters"""
        if poly is None:
            return 0.0
            
        # Bottom point of the image (vehicle position)
        bottom_y = self.warped_height - 1
        image_center_x = self.warped_width / 2.0
        
        # Lane x-position at bottom
        lane_x = poly[0]*(bottom_y**2) + poly[1]*bottom_y + poly[2]
        
        # Offset in pixels
        offset_px = lane_x - image_center_x
        
        # Convert to meters
        offset_m = offset_px * self.xm_per_pix
        return offset_m
    
    def draw_poly_on_original(self, poly, color, frame):
        """Draw polynomial on original camera image"""
        if poly is None:
            return frame
        
        # Sample points along the polynomial
        sample_ys = np.linspace(0, self.warped_height, 20)
        sample_xs = poly[0]*(sample_ys**2) + poly[1]*sample_ys + poly[2]
        
        # Filter out points outside the image
        valid_indices = (sample_xs >= 0) & (sample_xs < self.warped_width)
        sample_xs = sample_xs[valid_indices]
        sample_ys = sample_ys[valid_indices]
        
        if len(sample_xs) < 2:
            return frame
        
        # Transform points from warped to original image space
        warped_pts = np.column_stack((sample_xs, sample_ys)).astype(np.float32).reshape(-1, 1, 2)
        unwarped_pts = cv2.perspectiveTransform(warped_pts, self.birdeye_inv_matrix).astype(np.int32)
        
        # Draw on frame
        result = frame.copy()
        cv2.polylines(result, [unwarped_pts], False, color, 2)
        return result

    def create_visualization(self, frame, warped_mask, target_poly, state):
        """Create visualization with separate windows for main view and BEV"""
        if not self.show_debug_visualizations:
            return frame

        # Create bird's-eye view visualization
        bev_with_mask = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)

        # Draw polynomials on bird's-eye view
        if self.left_fit is not None:
            y_vals = np.linspace(0, self.warped_height - 1, 25)
            x_vals = self.left_fit[0] * (y_vals ** 2) + self.left_fit[1] * y_vals + self.left_fit[2]
            pts = np.column_stack((x_vals, y_vals)).astype(np.int32)
            cv2.polylines(bev_with_mask, [pts], False, (0, 255, 0), 2)

        if self.right_fit is not None:
            y_vals = np.linspace(0, self.warped_height - 1, 25)
            x_vals = self.right_fit[0] * (y_vals ** 2) + self.right_fit[1] * y_vals + self.right_fit[2]
            pts = np.column_stack((x_vals, y_vals)).astype(np.int32)
            cv2.polylines(bev_with_mask, [pts], False, (0, 255, 255), 2)

        if target_poly is not None:
            y_vals = np.linspace(0, self.warped_height - 1, 25)
            x_vals = target_poly[0] * (y_vals ** 2) + target_poly[1] * y_vals + target_poly[2]
            pts = np.column_stack((x_vals, y_vals)).astype(np.int32)
            cv2.polylines(bev_with_mask, [pts], False, (255, 0, 255), 2)

        # Draw vehicle position
        cv2.circle(bev_with_mask, (self.warped_width // 2, self.warped_height - 1), 5, (255, 255, 255), -1)

        # Get lane state text
        state_names = {0: "UNKNOWN", 1: "LEFT", 2: "RIGHT", 3: "TRANSITIONING"}
        state_name = state_names.get(state, "UNKNOWN")

        # Draw polynomials on original frame
        result = self.draw_poly_on_original(target_poly, (255, 0, 255), frame)
        if self.left_fit is not None:
            result = self.draw_poly_on_original(self.left_fit, (0, 255, 0), result)
        if self.right_fit is not None:
            result = self.draw_poly_on_original(self.right_fit, (0, 255, 255), result)

        # Add text overlays to main view
        cv2.putText(result, f"Lane State: {state_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if state == self.LANE_STATE["TRANSITIONING"]:
            progress = int(self.transition_progress * 100)
            target = state_names.get(self.transition_target, "UNKNOWN")
            cv2.putText(result, f"Transition: {progress}% to {target}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add lateral error
        if target_poly is not None:
            lateral_error = self.calculate_lateral_offset(target_poly)
            cv2.putText(result, f"Lateral Error: {lateral_error:.2f}m",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Return both visualizations separately - do NOT embed BEV in main frame
        return result, bev_with_mask

    def process_frame(self, frame):
        """Process a single frame with enhanced lane tracking"""
        t_start = time.time()
        
        # Lane detection -> warp
        lane_mask, warped_mask = self.detect_lanes(frame)
        
        # Contour analysis
        l_cnt, r_cnt, l_conf, r_conf = self.find_lane_contours(warped_mask)
        
        # RANSAC polynomial fitting
        if l_cnt is not None:
            points_l = l_cnt.reshape(-1, 2)
            self.left_fit, ls = self.ransac_poly_fit(points_l)
            self.left_confidence = min(l_conf * ls, 1.0)
        else:
            self.left_fit = None
            self.left_confidence = 0.0
        
        if r_cnt is not None:
            points_r = r_cnt.reshape(-1, 2)
            self.right_fit, rs = self.ransac_poly_fit(points_r)
            self.right_confidence = min(r_conf * rs, 1.0)
        else:
            self.right_fit = None
            self.right_confidence = 0.0
        
        # Update history
        timestamp = time.time()
        self.lane_history['left'].append(self.left_fit)
        self.lane_history['right'].append(self.right_fit)
        self.lane_timestamps.append(timestamp)
        self.lane_confidences['left'].append(self.left_confidence)
        self.lane_confidences['right'].append(self.right_confidence)
        
        # Update lane selection state
        state = self.update_lane_selection()
        
        # Calculate target polynomial
        target_poly, active_state = self.calculate_target_polynomial()
        
        # Calculate control parameters
        lateral_error = self.calculate_lateral_offset(target_poly)
        heading_error = self.estimate_heading_error(target_poly)
        curvature = self.calculate_lookahead_curvature(target_poly)
        
        # Stanley control calculation for lateral control
        steering = self.calculate_stanley_control(lateral_error, heading_error)
        
        # Simplified longitudinal control based on curvature
        throttle = 0.5  # Base throttle
        brake = 0.0
        
        # Adjust for curves
        if abs(curvature) > 0.01:
            throttle *= max(0.5, 1.0 - abs(curvature) * 30.0)
            if abs(curvature) > 0.03:
                brake = min(0.3, abs(curvature) * 10.0)
        
        # Create visualization
        main_view, bev_view = self.create_visualization(frame, warped_mask, target_poly, active_state)
        
        # Add control information to visualization
        cv2.putText(main_view, f"Steering: {steering:.3f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(main_view, f"Throttle: {throttle:.2f}, Brake: {brake:.2f}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Processing time
        proc_time = (time.time() - t_start) * 1000
        cv2.putText(main_view, f"Processing: {proc_time:.1f}ms", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return main_view, bev_view, (steering, throttle, brake)
    
    def run(self):
        """Main processing loop"""
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Performance metrics
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    time.sleep(0.01)
                    continue
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - fps_start_time)
                    logger.info(f"FPS: {fps:.2f}")
                    frame_count = 0
                    fps_start_time = time.time()
                
                # Process frame
                main_view, bev_view, controls = self.process_frame(frame)
                
                # Display
                cv2.imshow("Enhanced Lane Tracking", main_view)
                cv2.imshow("Bird's eye view", bev_view)
                
                # Exit on 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup resources"""
        try:
            # Free GPU memory
            if hasattr(self, 'd_input'):
                self.d_input.free()
            if hasattr(self, 'd_output0'):
                self.d_output0.free()
            if hasattr(self, 'd_output1'):
                self.d_output1.free()
            
            # Release other resources
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            # Release TensorRT resources
            if hasattr(self, 'context'):
                del self.context
            if hasattr(self, 'engine'):
                del self.engine
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced Lane Tracking System")
    parser.add_argument("--engine", type=str, required=True,
                        help="Path to TensorRT engine file (.engine)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug visualizations")
    args = parser.parse_args()
    
    # Print system information
    logger.info(f"System Platform: {os.uname().sysname} {os.uname().release}")
    logger.info(f"TensorRT Version: {trt.__version__}")
    
    try:
        # Initialize and run lane tracker
        tracker = EnhancedLaneTracker(
            engine_path=args.engine,
            camera_id=args.camera,
            debug_visualization=args.debug
        )
        tracker.run()
    except Exception as e:
        logger.error(f"Error in lane tracker: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
