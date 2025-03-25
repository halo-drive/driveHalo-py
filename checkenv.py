# test_env.py
import torch
import tensorrt as trt
import cv2
import os

# Set CUDA environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "all"
os.environ["CUDA_CACHE_DISABLE"] = "0"

# Optional: Set TensorRT optimization flags
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT"] = "1"
os.environ["TF_TRT_ABORT_CUDA_ENGINE_ON_ERROR"] = "1"

def print_versions():
    print(f"PyTorch version: {torch.__version__}")
    print(f"TensorRT version: {trt.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    print_versions()