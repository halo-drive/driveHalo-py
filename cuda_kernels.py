import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import logging
from typing import Optional, Tuple

# CUDA kernel for curvature calculation
CURVATURE_KERNEL_CODE = """
__global__ void calculate_curvature(
    float *x_points, float *y_points, int point_count,
    float ym_per_pix, float xm_per_pix, float *result) {

    // Skip calculation if insufficient points
    if (point_count < 3) {
        result[0] = INFINITY;
        return;
    }

    // Simplified polynomial fitting for curvature calculation
    float sum_x = 0.0f, sum_y = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;
    float sum_x_y2 = 0.0f, sum_y3 = 0.0f;
    float max_y = -INFINITY;

    // Process all points
    for (int i = 0; i < point_count; i++) {
        // Scale to real-world coordinates
        float x = x_points[i] * xm_per_pix;
        float y = y_points[i] * ym_per_pix;

        // Track maximum y for evaluation point
        if (y > max_y) {
            max_y = y;
        }

        // Accumulate sums for quadratic fit
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        sum_x_y2 += x * y * y;
        sum_y3 += y * y * y;
    }

    // Calculate polynomial coefficients (for y = ax² + bx + c)
    float n = (float)point_count;
    float det = n * sum_xx - sum_x * sum_x;

    if (fabs(det) < 1e-6f) {
        // Near-singular matrix, straight line
        result[0] = INFINITY;
        return;
    }

    // Solve for coefficients
    float a = (n * sum_x_y2 - sum_x * sum_y3) / det;
    float b = (sum_xx * sum_y3 - sum_x * sum_x_y2) / det;

    // Calculate curvature at maximum y value
    if (fabs(a) < 1e-4f) {
        // Nearly linear, large radius
        result[0] = INFINITY;
    } else {
        // Compute radius of curvature
        float deriv_first = 2.0f * a * max_y + b;
        float deriv_second = 2.0f * a;

        float curvature = powf(1.0f + deriv_first * deriv_first, 1.5f) / fabs(deriv_second);

        // Determine direction from coefficient sign
        float direction = (a > 0.0f) ? 1.0f : -1.0f;

        // Limit maximum value and apply direction
        result[0] = fminf(curvature, 100.0f) * direction;
    }
}
"""


class CUDAProcessor:
    """CUDA-accelerated processing for computationally intensive operations"""

    def __init__(self, ym_per_pix: float = 17.0 / 720.0, xm_per_pix: float = 17.0 / 1280.0):
        self.logger = logging.getLogger("CUDAProcessor")
        self.logger.info("Initializing CUDA processing kernels")

        try:
            # Compile CUDA kernels
            self.module = SourceModule(CURVATURE_KERNEL_CODE)
            self.curvature_kernel = self.module.get_function("calculate_curvature")

            # Store parameters
            self.ym_per_pix = np.float32(ym_per_pix)
            self.xm_per_pix = np.float32(xm_per_pix)

            self.logger.info("CUDA kernels compiled successfully")

        except Exception as e:
            self.logger.error(f"CUDA kernel compilation failed: {str(e)}")
            raise

    def calculate_curvature(self, lane_points: np.ndarray) -> float:
        """Calculate lane curvature using CUDA acceleration"""
        if len(lane_points) < 3:
            return float('inf')

        try:
            # Extract x and y coordinates
            x_points = lane_points[:, 0].astype(np.float32)
            y_points = lane_points[:, 1].astype(np.float32)

            # Allocate device memory
            x_gpu = cuda.mem_alloc(x_points.nbytes)
            y_gpu = cuda.mem_alloc(y_points.nbytes)
            result_gpu = cuda.mem_alloc(np.float32(0).nbytes)

            # Copy data to device
            cuda.memcpy_htod(x_gpu, x_points)
            cuda.memcpy_htod(y_gpu, y_points)

            # Execute kernel
            self.curvature_kernel(
                x_gpu, y_gpu, np.int32(len(lane_points)),
                self.ym_per_pix, self.xm_per_pix, result_gpu,
                block=(1, 1, 1), grid=(1, 1)
            )

            # Retrieve result
            result = np.empty(1, dtype=np.float32)
            cuda.memcpy_dtoh(result, result_gpu)

            # Free GPU memory
            x_gpu.free()
            y_gpu.free()
            result_gpu.free()

            return float(result[0])

        except Exception as e:
            self.logger.error(f"CUDA curvature calculation failed: {str(e)}")
            return float('inf')