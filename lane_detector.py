import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
import time
import warnings
from utils import measure_execution_time
from trt_inference import TRTInference
from cuda_kernels import CUDAProcessor


class LaneDetector:
    """
    Lane detection and analysis using TensorRT and CUDA acceleration,
    separating left and right lane boundaries, applying a bird?s-eye transform,
    fitting polynomials with coordinate shifting, and optionally computing a centerline curvature.
    """

    def __init__(self, engine_path: str):
        self.logger = logging.getLogger("LaneDetector")
        self.logger.info(f"Initializing lane detector with engine: {engine_path}")

        # 1. TensorRT Inference
        self.trt_model = TRTInference(engine_path)
        self.cuda_processor = CUDAProcessor()

        # 2. Basic Lane Parameters
        self.ym_per_pix = 0.0224
        self.xm_per_pix = 0.01395
        self.lane_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.conf_threshold = 0.05

        # 3. Polynomials for left/right lane boundaries (in bird?s-eye space)
        self.poly_fit_left = None   # [a, b, c] for left boundary
        self.poly_fit_right = None  # [a, b, c] for right boundary
        self.smoothing_alpha = 0.9  # Exponential smoothing factor

        # 4. Bird?s-Eye (Perspective) Transform Setup
        #    NOTE: These points must be calibrated for your camera and scenario.
        # Correct format
        self.warped_width = 300
        self.warped_height = 400
        self.birdeye_src = np.float32([
            [177.0, 170.0], [410.0, 164.0], [58.0, 336.0], [544.0, 327.0]
        ])
        self.birdeye_dst = np.float32([
            [0, 0],
            [self.warped_width, 0],
            [0, self.warped_height],
            [self.warped_width, self.warped_height]
        ])
        self.birdeye_matrix = cv2.getPerspectiveTransform(self.birdeye_src, self.birdeye_dst)
        self.birdeye_inv_matrix = cv2.getPerspectiveTransform(self.birdeye_dst, self.birdeye_src)


        #5. Vehicle lateral offset parameters
        self.vehicle_center_x = None  # Will be set during detection
        self.logger.info("Lane detector initialized successfully")


    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts BGR->RGB, resizes to 416x416, normalizes to [0,1],
        and rearranges to NCHW format for TensorRT.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (416, 416))
        preprocessed = resized.astype(np.float32) / 255.0
        preprocessed = preprocessed.transpose((2, 0, 1))  # HWC->CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)
        return preprocessed


    def _fit_and_smooth(
        self,
        contour: np.ndarray,
        old_poly: Optional[np.ndarray],
        alpha: float = 0.8
    ) -> Optional[np.ndarray]:
        """
        Given a contour in bird?s-eye space, fit a 2nd-degree polynomial x = a*(y_shifted^2) + b*y_shifted + c,
        then exponentially smooth with the old polynomial if available.

        Lines ~65?90 in your code.
        """
        if contour is None or len(contour) < 3:
            return old_poly

        # Extract (x, y) from the contour (bird?s-eye coords)
        points = contour.reshape(-1, 2)
        x_coords = points[:, 0].astype(np.float32)
        y_coords = points[:, 1].astype(np.float32)

        if len(points) < 3:
            return old_poly

        # 1) SHIFT Y to reduce numeric range => helps avoid ill-conditioned fits
        y_min = y_coords.min()
        y_shifted = y_coords - y_min

        # You could also scale if needed: y_shifted /= 10.0

        # 2) Fit polynomial
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.RankWarning)  # treat RankWarning as error
            try:
                fit_new = np.polyfit(y_shifted, x_coords, 2)  # => [a, b, c]
            except (np.RankWarning, ValueError):
                # fallback to old polynomial if fit fails
                return old_poly

        # 3) Exponential smoothing
        if old_poly is None:
            return fit_new
        else:
            return alpha * old_poly + (1.0 - alpha) * fit_new

    def calculate_lane_center_offset(self, center_line_pts):
        """
        Calculate vehicle's lateral offset from lane center in meters.
        Positive: vehicle is right of center. Negative: vehicle is left of center.
        """
        if center_line_pts is None or len(center_line_pts) < 2:
            return 0.0

        # Vehicle center is at the bottom-middle of warped space
        image_center_x = self.warped_width / 2

        # Get lane center x-position at the bottom of warped image (closest to vehicle)
        bottom_y = self.warped_height - 1

        # For the center line polynomial, calculate x at bottom_y
        # Recall polynomial form: x = a*y^2 + b*y + c
        if self.poly_fit_left is not None and self.poly_fit_right is not None:
            # Average the left and right polynomials
            left_x_at_bottom = (
                    self.poly_fit_left[0] * (bottom_y ** 2) +
                    self.poly_fit_left[1] * bottom_y +
                    self.poly_fit_left[2]
            )

            right_x_at_bottom = (
                    self.poly_fit_right[0] * (bottom_y ** 2) +
                    self.poly_fit_right[1] * bottom_y +
                    self.poly_fit_right[2]
            )

            lane_center_x = (left_x_at_bottom + right_x_at_bottom) / 2
        elif len(center_line_pts) > 0:
            # Fall back to the provided center line points if available
            # Find the point with highest y value (closest to bottom)
            closest_pt_idx = np.argmax(center_line_pts[:, 1])
            lane_center_x = center_line_pts[closest_pt_idx, 0]
        else:
            # No valid data, assume centered
            return 0.0

        # Calculate offset in pixels and convert to meters
        offset_pixels = lane_center_x - image_center_x
        offset_meters = offset_pixels * self.xm_per_pix

        return offset_meters

    @measure_execution_time
    def detect_lane(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        1. Run YOLOv8-seg inference => final_mask
        2. Morphological ops => mask_clean
        3. Bird?s-eye warp => warped_mask
        4. Find all contours in warped_mask => separate left vs. right
        5. Fit & smooth polynomials in bird?s-eye space
        6. Sample polynomials => unwarp for drawing on original frame
        7. If both sides exist, compute centerline + curvature
        8. Return annotated frame + curvature
        """
        try:
            t_start = time.time()

            # (A) Preprocess & Infer
            preprocessed = self.preprocess_image(frame)
            outputs = self.trt_model.infer(preprocessed)

            import pycuda.driver as cuda
            cuda.Context.synchronize()

            # Typical YOLOv8-seg engine:
            # outputs[0] => proto_masks [1, 32, 104, 104]
            # outputs[1] => detection_output [1, 37, 3549]
            proto_masks = outputs[0]
            detection_output = outputs[1]
            dets = detection_output[0].transpose()  # shape => (3549, 37)

            height, width = frame.shape[:2]
            final_mask = np.zeros((height, width), dtype=np.uint8)
            proto = proto_masks[0]  # shape => (32, 104, 104)

            # (B) Build final_mask from YOLO detections
            obj_conf = dets[:, 4]
            coefs = dets[:, 5:37]  # 32 mask coefficients

            valid_idx = np.where(obj_conf > self.conf_threshold)[0]
            for idx in valid_idx:
                mask_104 = np.zeros((proto.shape[1], proto.shape[2]), dtype=np.float32)
                for c_idx in range(32):
                    mask_104 += coefs[idx, c_idx] * proto[c_idx]
                # Sigmoid
                mask_104 = 1.0 / (1.0 + np.exp(-mask_104))
                # Resize to original frame dims
                mask_resized = cv2.resize(mask_104, (width, height))
                # Threshold
                mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
                final_mask = cv2.bitwise_or(final_mask, mask_bin)

            # (C) Morphological ops
            kernel_vertical = np.ones((7, 3), np.uint8)
            mask_closed = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_vertical)
            kernel_open = np.ones((3, 3), np.uint8)
            mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

            # (D) Warp mask_clean to bird?s-eye space
            #     Use self.birdeye_matrix to get a top-down view
            warped_mask = cv2.warpPerspective(
                mask_clean,
                self.birdeye_matrix,
                (self.warped_width, self.warped_height),
                flags=cv2.INTER_LINEAR
            )

            # (E) Find contours in the warped (bird?s-eye) mask
            contours, _ = cv2.findContours(
                warped_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_TC89_KCOS
            )

            # Separate left vs. right in bird?s-eye => use warped_width // 2 as center
            left_contour = None
            right_contour = None
            beye_center_x = self.warped_width // 2

            for cnt in contours:
                x_vals = cnt[:, 0, 0]
                mean_x = np.mean(x_vals)
                if mean_x < beye_center_x:
                    # Potential left lane
                    if (left_contour is None) or (cv2.contourArea(cnt) > cv2.contourArea(left_contour)):
                        left_contour = cnt
                else:
                    # Potential right lane
                    if (right_contour is None) or (cv2.contourArea(cnt) > cv2.contourArea(right_contour)):
                        right_contour = cnt

            # (F) Fit & smooth polynomials for each side in bird?s-eye space
            self.poly_fit_left = self._fit_and_smooth(
                left_contour, self.poly_fit_left, self.smoothing_alpha
            )
            self.poly_fit_right = self._fit_and_smooth(
                right_contour, self.poly_fit_right, self.smoothing_alpha
            )

            annotated_frame = frame.copy()
            radius = float('inf')

            # (G) Draw each boundary => sample polynomial in warped coords, then unwarp
            def draw_poly_on_original(poly, color):
                """
                Sample polynomial in bird?s-eye coords, unwarp, then draw on annotated_frame.
                """
                if poly is None:
                    return

                # We'll sample y from 0..warped_height
                sample_ys = np.linspace(0, self.warped_height, 20)
                # SHIFT must match what we do in _fit_and_smooth => we used (y_min) but each frame can differ
                # For simplicity, we assume y_min=0 in bird?s-eye. So no extra shift here.
                # If we had used scaling, we'd apply it inversely here as well.

                # x = a*y^2 + b*y + c
                sample_xs = (
                    poly[0] * (sample_ys**2) +
                    poly[1] * sample_ys +
                    poly[2]
                )

                # Create Nx1x2 array for perspectiveTransform
                warped_pts = np.column_stack((sample_xs, sample_ys)).astype(np.float32)
                warped_pts = warped_pts.reshape(-1, 1, 2)

                # Unwarp
                unwarped_pts = cv2.perspectiveTransform(warped_pts, self.birdeye_inv_matrix)
                unwarped_pts_int = unwarped_pts.astype(np.int32)

                # Draw
                cv2.polylines(annotated_frame, [unwarped_pts_int], False, color, 2)

            # Draw left boundary in green
            draw_poly_on_original(self.poly_fit_left, (0, 255, 0))
            # Draw right boundary in yellow
            draw_poly_on_original(self.poly_fit_right, (0, 255, 255))

            lateral_offset = 0.0

            # (H) If both sides exist, compute centerline in bird?s-eye space
            if (self.poly_fit_left is not None) and (self.poly_fit_right is not None):
                sample_ys = np.linspace(0, self.warped_height, 20)

                left_xs = (
                    self.poly_fit_left[0]*(sample_ys**2) +
                    self.poly_fit_left[1]*sample_ys +
                    self.poly_fit_left[2]
                )
                right_xs = (
                    self.poly_fit_right[0]*(sample_ys**2) +
                    self.poly_fit_right[1]*sample_ys +
                    self.poly_fit_right[2]
                )
                center_xs = (left_xs + right_xs) / 2.0

                # Combine into Nx2
                center_pts = np.column_stack((center_xs, sample_ys)).astype(np.float32)

                # 1) Compute curvature in top-down coords => more physically accurate
                radius = self.cuda_processor.calculate_curvature(center_pts)

                # 2) Unwarp the centerline for display
                center_pts_reshaped = center_pts.reshape(-1,1,2)
                unwarped_center = cv2.perspectiveTransform(center_pts_reshaped, self.birdeye_inv_matrix)
                unwarped_center_int = unwarped_center.astype(np.int32)
                cv2.polylines(annotated_frame, [unwarped_center_int], False, (0, 0, 255), 2)

                lateral_offset = self.calculate_lane_center_offset(center_pts)
                cv2.putText(
                    annotated_frame,
                    f'Offset: {lateral_offset:.2f}m',
                    (20, 130),  # Position below curvature text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                # 3) Display curvature
                cv2.putText(
                    annotated_frame,
                    f'Curvature: {radius:.1f}m',
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # (I) Timing overlay
            t_elapsed = time.time() - t_start
            cv2.putText(
                annotated_frame,
                f'Processing: {t_elapsed*1000:.1f}ms',
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            return annotated_frame, radius, lateral_offset

        except Exception as e:
            self.logger.error(f"Lane detection failed: {str(e)}")
            return frame.copy(), float('inf'), 0.0
