import cv2
print(cv2.getBuildInformation())
print("CUDA Devices:", cv2.cuda.getCudaEnabledDeviceCount())
try:
    gpu_mat = cv2.cuda_GpuMat()
    print("CUDA GpuMat works!")
except AttributeError:
    print("CUDA module missing.")
