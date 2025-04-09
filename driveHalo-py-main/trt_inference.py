import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import logging
from typing import List, Dict, Any, Tuple


class TRTInference:
    """TensorRT inference engine wrapper for optimized model execution"""

    def __init__(self, engine_path: str):
        self.logger = logging.getLogger("TRTInference")
        self.logger.info(f"Initializing TensorRT engine from: {engine_path}")

        # Load TensorRT engine
        self._load_engine(engine_path)

        # Allocate buffers
        self._allocate_buffers()

        # Create execution stream
        self.stream = cuda.Stream()

    def _load_engine(self, engine_path: str) -> None:
        """Load TensorRT engine from file"""
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if not self.engine:
                raise ValueError(f"Failed to load engine from {engine_path}")

            self.context = self.engine.create_execution_context()
            self.logger.info(f"Engine loaded successfully")

        except Exception as e:
            self.logger.error(f"Engine loading failed: {str(e)}")
            raise

    def _allocate_buffers(self) -> None:
        """Allocate device memory for inputs and outputs"""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            binding_shape = self.engine.get_binding_shape(binding_idx)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))

            # Calculate buffer size
            size = trt.volume(binding_shape) * np.dtype(binding_dtype).itemsize

            # Allocate device memory
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))

            # Record binding details
            binding_dict = {
                "name": binding_name,
                "shape": binding_shape,
                "dtype": binding_dtype,
                "size": size,
                "device_mem": device_mem
            }

            if self.engine.binding_is_input(binding_idx):
                self.inputs.append(binding_dict)
            else:
                self.outputs.append(binding_dict)

        self.logger.info(f"Allocated buffers for {len(self.inputs)} inputs and {len(self.outputs)} outputs")

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Execute inference on preprocessed input data"""
        try:
            # Ensure input data has correct shape and type
            if not isinstance(input_data, np.ndarray):
                raise TypeError("Input must be a numpy array")

            # Convert to contiguous array with correct data type
            input_buffer = np.ascontiguousarray(input_data.astype(self.inputs[0]["dtype"]))

            # Copy input to device
            cuda.memcpy_htod_async(self.inputs[0]["device_mem"], input_buffer, self.stream)

            # Execute inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # Retrieve outputs
            outputs = []
            for output in self.outputs:
                # Allocate host memory for output
                output_buffer = np.empty(tuple(output["shape"]), dtype=output["dtype"])

                # Copy output from device to host
                cuda.memcpy_dtoh_async(output_buffer, output["device_mem"], self.stream)
                outputs.append(output_buffer)

            # Synchronize to ensure all operations complete
            self.stream.synchronize()

            return outputs

        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            raise

    def __del__(self):
        """Clean up resources"""
        try:
            # Free GPU memory
            for binding in self.inputs + self.outputs:
                if "device_mem" in binding:
                    binding["device_mem"].free()

            # Release CUDA resources explicitly
            if hasattr(self, 'context') and self.context:
                del self.context

            if hasattr(self, 'engine') and self.engine:
                del self.engine

        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def synchronize_stream(self) -> None:
        """Explicitly synchronize the CUDA stream"""
        self.stream.synchronize()