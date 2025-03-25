import cv2
import torch
import numpy as np
from ultralytics import YOLO
import logging
import time

class LaneDetector:
    def __init__(self, model_path):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model = YOLO(model_path)
        
        # Initialize FPS counter
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Visualization parameters
        self.lane_color = (0, 255, 255)  # Yellow
        self.overlay_color = (0, 255, 0, 0.3)  # Semi-transparent green
        self.text_color = (255, 255, 255)  # White
        
        # Lane tracking parameters
        self.previous_lanes = []
        self.smoothing_factor = 0.8
        
        
        self.prev_lanes = None
        self.smoothing_factor = 0.8  # Adjust this for more/less smoothing
        self.buffer_size = 5
        self.lane_buffer = []
        self.min_confidence = 0.5
        
        # Lane persistence
        self.lanes_lost_frames = 0
        self.max_lost_frames = 10  # Number of frames to keep showing lanes when detection is lost

    def draw_lane_overlay(self, frame, results):
        """Draw a more aesthetically pleasing lane visualization"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.data
            
            # Sort boxes by x-position to separate left and right lanes
            boxes_sorted = sorted(boxes, key=lambda x: x[0])
            
            lane_points = []
            for box in boxes_sorted:
                x1, y1, x2, y2, conf, cls = map(float, box[:6])
                
                # Get bottom center point of each lane marking
                bottom_center = (int((x1 + x2) / 2), int(y2))
                # Get top center point
                top_center = (int((x1 + x2) / 2), int(y1))
                
                lane_points.append([bottom_center, top_center])
            
            # Draw lane area if we have at least two lanes
            if len(lane_points) >= 2:
                # Create arrays for fillPoly
                left_points = []
                right_points = []
                
                for i, points in enumerate(lane_points):
                    if i == 0:  # Leftmost lane
                        left_points = np.array([
                            points[0],  # Bottom point
                            points[1],  # Top point
                        ])
                    if i == len(lane_points) - 1:  # Rightmost lane
                        right_points = np.array([
                            points[0],  # Bottom point
                            points[1],  # Top point
                        ])
                
                if len(left_points) > 0 and len(right_points) > 0:
                    # Create drivable area polygon
                    lane_area = np.array([
                        left_points[0],   # Bottom left
                        left_points[1],   # Top left
                        right_points[1],  # Top right
                        right_points[0]   # Bottom right
                    ], np.int32)
                    
                    # Draw filled polygon for lane area
                    overlay_area = np.zeros_like(frame)
                    cv2.fillPoly(overlay_area, [lane_area], (0, 255, 0))
                    cv2.addWeighted(overlay_area, 0.3, frame, 1 - 0.3, 0, frame)
                    
                    # Draw lane lines
                    for points in lane_points:
                        # Draw smooth curve for lane line
                        pts = np.array([points[0], points[1]], np.int32)
                        cv2.polylines(frame, [pts], False, self.lane_color, 3, cv2.LINE_AA)
                    
                    # Add distance markers
                    self.draw_distance_markers(frame, lane_area)
                    
                    # Add driving guidance
                    self.draw_driving_guidance(frame, lane_area)
        
        return frame

    def draw_distance_markers(self, frame, lane_area):
        """Draw distance markers on the lane"""
        height, width = frame.shape[:2]
        
        # Draw horizontal markers at different distances
        distances = [10, 20, 30]  # meters
        for dist in distances:
            # Convert distance to y-coordinate (simplified)
            y = int(height - (dist * height / 40))  # Assuming 40m is max visible distance
            
            # Draw marker line
            cv2.line(frame, (0, y), (width, y), (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{dist}m", (10, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    def draw_driving_guidance(self, frame, lane_area):
        """Draw driving guidance arrows and info"""
        height, width = frame.shape[:2]
        
        # Calculate center line
        center_x = width // 2
        
        # Draw center guidance line
        cv2.line(frame, (center_x, height), (center_x, height-100),
                (0, 255, 255), 2, cv2.LINE_AA)
        
        # Add guidance arrow
        cv2.arrowedLine(frame, (center_x, height-50),
                       (center_x, height-100),
                       (0, 255, 255), 2, cv2.LINE_AA)

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run inference
            results = self.model(frame, verbose=False)
            
            # Draw enhanced visualization
            annotated_frame = self.draw_lane_overlay(frame, results)

            # Update FPS
            self.frame_count += 1
            if time.time() - self.fps_start_time >= 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.fps_start_time = time.time()

            # Add FPS counter
            cv2.putText(annotated_frame, f'FPS: {self.fps}', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Lane Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam)')
    args = parser.parse_args()

    detector = LaneDetector(args.model)
    detector.run(0 if args.source == '0' else args.source)

if __name__ == '__main__':
    main()