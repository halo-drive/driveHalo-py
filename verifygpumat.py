import cv2

# Check if CUDA is available
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"CUDA Enabled: {cuda_available}")

if cuda_available:
    try:
        # Try allocating a GpuMat
        gpu_mat = cv2.cuda_GpuMat()
        print("GpuMat is supported.")
    except Exception as e:
        print(f"Error creating GpuMat: {e}")
