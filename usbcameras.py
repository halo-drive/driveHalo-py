# camera_detect.py
import cv2
import subprocess
import os


def list_v4l2_devices():
    """List available V4L2 devices"""
    print("Available V4L2 devices:")
    if os.path.exists('/dev/video0'):
        result = subprocess.run(['v4l2-ctl', '--list-devices'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        print(result.stdout)
    else:
        print("No V4L2 devices found")


def test_camera_pipeline(device_id=0, width=640, height=480, fps=30):
    """Test camera with GStreamer pipeline"""
    pipeline = (
        f"v4l2src device=/dev/video{device_id} ! "
        f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1"
    )

    print(f"Testing pipeline: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open camera with pipeline")
        return

    print(f"Camera opened, reading frame...")
    ret, frame = cap.read()

    if ret:
        print(f"Successfully read frame: {frame.shape}")
        cv2.imwrite(f"camera_test_{device_id}.jpg", frame)
        print(f"Saved test image to camera_test_{device_id}.jpg")
    else:
        print("Failed to read frame")

    cap.release()


if __name__ == "__main__":
    list_v4l2_devices()

    for device_id in range(4):  # Check first 4 devices
        if os.path.exists(f'/dev/video{device_id}'):
            print(f"\nTesting device: /dev/video{device_id}")
            test_camera_pipeline(device_id)