#!/usr/bin/env python3
"""
Bird's-Eye Transform Calibration Tool

This script assists in calibrating the perspective transform for lane detection by:
1. Capturing frames from a camera or loading from file
2. Allowing interactive selection of source points
3. Computing transformation parameters based on physical measurements
4. Generating calibration parameters for the lane detection pipeline
"""

import cv2
import numpy as np
import argparse
import json
import os
from datetime import datetime


class CameraCalibrator:
    def __init__(self, source=0, resolution=(640, 480)):
        """Initialize the calibrator with a camera source or image file"""
        self.source = source
        self.resolution = resolution
        self.markers = []
        self.img = None
        self.warped_img = None
        self.calibration_points = []

        # Default warped rectangle dimensions (typically fixed)
        self.warped_width = 300
        self.warped_height = 400

        # Physical measurements (to be entered by user)
        self.width_near = 0.0
        self.width_far = 0.0
        self.length = 0.0

        # Calibration results
        self.birdeye_src = None
        self.birdeye_dst = None
        self.xm_per_pix = 0.0
        self.ym_per_pix = 0.0
        self.transform_matrix = None
        self.inverse_transform_matrix = None

    def capture_frame(self):
        """Capture a frame from the camera or load from file"""
        if isinstance(self.source, str) and os.path.isfile(self.source):
            self.img = cv2.imread(self.source)
            if self.img is None:
                print(f"Error: Could not read image file {self.source}")
                return False
            print(f"Loaded image from {self.source}")
        else:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"Error: Could not open camera source {self.source}")
                return False

            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            print("Press SPACE to capture frame or ESC to cancel...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    return False

                cv2.imshow("Camera Preview - SPACE to capture, ESC to exit", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                elif key == 32:  # SPACE key
                    self.img = frame.copy()
                    cv2.destroyAllWindows()
                    cap.release()
                    print("Frame captured")
                    break

        return True

    def click_event(self, event, x, y, flags, param):
        """Handle mouse clicks to select calibration points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Limit to 4 points for calibration
            if len(self.calibration_points) < 4:
                self.calibration_points.append((x, y))
                # Draw circle at the clicked position
                cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)
                # Display coordinates
                cv2.putText(self.img, f"({x},{y})", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Calibration Frame", self.img)

                # Print point being added with point type
                point_types = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
                print(
                    f"Point {len(self.calibration_points)}: ({x}, {y}) - {point_types[len(self.calibration_points) - 1]}")

    def select_points(self):
        """Interactive UI for selecting calibration points"""
        if self.img is None:
            print("No image available. Capture a frame first.")
            return False

        # Clear any previous points
        self.calibration_points = []
        img_copy = self.img.copy()
        self.img = img_copy

        cv2.namedWindow("Calibration Frame")
        cv2.setMouseCallback("Calibration Frame", self.click_event)

        print("\n=== CALIBRATION INSTRUCTIONS ===")
        print("Click to select 4 points in this specific order:")
        print("1. Top-Left (furthest visible left point)")
        print("2. Top-Right (furthest visible right point)")
        print("3. Bottom-Left (closest visible left point)")
        print("4. Bottom-Right (closest visible right point)")
        print("Press 'r' to reset points, 'c' to continue when done, ESC to cancel")

        while True:
            cv2.imshow("Calibration Frame", self.img)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return False
            elif key == ord('r'):  # Reset
                print("Resetting points")
                self.img = img_copy.copy()
                self.calibration_points = []
                cv2.imshow("Calibration Frame", self.img)
            elif key == ord('c') and len(self.calibration_points) == 4:  # Continue
                cv2.destroyAllWindows()
                return True

        return False

    def enter_measurements(self):
        """Prompt user to enter physical measurements"""
        print("\n=== PHYSICAL MEASUREMENTS ===")
        print("Enter the real-world measurements for your calibration points:")

        # Keep asking until valid inputs are provided
        while True:
            try:
                self.width_near = float(input("Width at bottom/near edge (meters): "))
                self.width_far = float(input("Width at top/far edge (meters): "))
                self.length = float(input("Length from near to far edge (meters): "))

                if self.width_near <= 0 or self.width_far <= 0 or self.length <= 0:
                    print("Error: Measurements must be positive values")
                    continue

                return True
            except ValueError:
                print("Error: Please enter valid numbers")

    def compute_calibration(self):
        """Compute calibration parameters based on selected points and measurements"""
        if len(self.calibration_points) != 4:
            print("Error: Need exactly 4 calibration points")
            return False

        # Arrange points in the order: [top-left, top-right, bottom-left, bottom-right]
        # Points should already be in this order from the UI
        self.birdeye_src = np.float32(self.calibration_points)

        # Fixed destination rectangle
        self.birdeye_dst = np.float32([
            [0, 0],
            [self.warped_width, 0],
            [0, self.warped_height],
            [self.warped_width, self.warped_height]
        ])

        # Compute transformation matrices
        self.transform_matrix = cv2.getPerspectiveTransform(self.birdeye_src, self.birdeye_dst)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(self.birdeye_dst, self.birdeye_src)

        # Compute meter-per-pixel ratios
        self.xm_per_pix = self.width_near / self.warped_width
        self.ym_per_pix = self.length / self.warped_height

        return True

    def visualize_transform(self):
        """Visualize the perspective transform to verify calibration"""
        if self.img is None or self.transform_matrix is None:
            print("Calibration not complete")
            return

        # Apply perspective transform
        self.warped_img = cv2.warpPerspective(
            self.img,
            self.transform_matrix,
            (self.warped_width, self.warped_height),
            flags=cv2.INTER_LINEAR
        )

        # Draw a grid on the warped image to help verify
        grid_img = self.warped_img.copy()
        # Draw horizontal lines every 50 pixels
        for y in range(0, self.warped_height, 50):
            cv2.line(grid_img, (0, y), (self.warped_width, y), (0, 255, 0), 1)
        # Draw vertical lines every 50 pixels
        for x in range(0, self.warped_width, 50):
            cv2.line(grid_img, (x, 0), (x, self.warped_height), (0, 255, 0), 1)

        # Display both original and warped images
        cv2.imshow("Original with Selected Points", self.img)
        cv2.imshow("Warped Bird's-Eye View", grid_img)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_calibration(self, filename=None):
        """Save calibration parameters to a JSON file"""
        if self.birdeye_src is None:
            print("Calibration not complete")
            return False

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_{timestamp}.json"

        # Create calibration data dictionary
        calibration_data = {
            "birdeye_src": self.birdeye_src.tolist(),
            "birdeye_dst": self.birdeye_dst.tolist(),
            "xm_per_pix": self.xm_per_pix,
            "ym_per_pix": self.ym_per_pix,
            "warped_width": self.warped_width,
            "warped_height": self.warped_height,
            "physical_measurements": {
                "width_near": self.width_near,
                "width_far": self.width_far,
                "length": self.length
            },
            "timestamp": datetime.now().isoformat()
        }

        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            print(f"Calibration saved to {filename}")

            # Also display the Python code to use these values
            self.generate_code_snippet()

            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def generate_code_snippet(self):
        """Generate code snippet with calibration parameters"""
        if self.birdeye_src is None:
            return

        code = f"""
# Bird's-Eye Transform Calibration Parameters
# Generated on {datetime.now().isoformat()}

# Source points in original image (trapezoid corners)
birdeye_src = np.float32([
    {self.birdeye_src.tolist()}
])

# Destination points in warped image (rectangle corners)
birdeye_dst = np.float32([
    {self.birdeye_dst.tolist()}
])

# Meter per pixel ratios
xm_per_pix = {self.xm_per_pix}  # meters per pixel in x dimension
ym_per_pix = {self.ym_per_pix}  # meters per pixel in y dimension

# Create transformation matrices
birdeye_matrix = cv2.getPerspectiveTransform(birdeye_src, birdeye_dst)
birdeye_inv_matrix = cv2.getPerspectiveTransform(birdeye_dst, birdeye_src)
"""

        print("\n=== CODE SNIPPET FOR YOUR LANE DETECTOR ===")
        print(code)


def main():
    """Main function to run the calibration tool"""
    parser = argparse.ArgumentParser(description="Bird's-Eye Transform Calibration Tool")
    parser.add_argument('--source', type=str, default='0',
                        help='Camera index (default: 0) or path to image file')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera capture width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera capture height (default: 480)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output calibration file (default: calibration_TIMESTAMP.json)')

    args = parser.parse_args()

    # Parse the source argument
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    # Create calibrator instance
    calibrator = CameraCalibrator(source=source, resolution=(args.width, args.height))

    # Run the calibration process
    print("=== Bird's-Eye Transform Calibration Tool ===")

    # Step 1: Capture or load frame
    if not calibrator.capture_frame():
        print("Calibration aborted: failed to acquire image")
        return

    # Step 2: Select calibration points
    if not calibrator.select_points():
        print("Calibration aborted: point selection cancelled")
        return

    # Step 3: Enter physical measurements
    if not calibrator.enter_measurements():
        print("Calibration aborted: measurements not provided")
        return

    # Step 4: Compute calibration parameters
    if not calibrator.compute_calibration():
        print("Calibration failed: could not compute parameters")
        return

    # Step 5: Visualize the transform
    calibrator.visualize_transform()

    # Step 6: Save calibration
    calibrator.save_calibration(args.output)

    print("Calibration completed successfully!")


if __name__ == "__main__":
    main()