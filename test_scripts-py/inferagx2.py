import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
ENGINE_PATH = "/home/pomo/DriveGXO/Models/yolov8n.trt"
INPUT_SHAPE = (640, 640)
CONF_THRESHOLD = 0.4
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class YOLOv8nDetector:
    def __init__(self, engine_path):
        logger.info("Initializing YOLOv8n TensorRT detector...")
        self.logger = trt.Logger(trt.Logger.INFO)

        # Load the TensorRT engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers with correct shapes from ONNX/TRT specs
        self.input_shape = (1, 3, *INPUT_SHAPE)  # [1, 3, 640, 640]
        self.output_shape = (1, 84, 8400)  # Corrected to match model specs

        # Calculate buffer sizes
        input_size = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        output_size = int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize)

        # Allocate device memory
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)

        # Allocate host memory
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

        # Verify bindings
        assert self.engine.get_binding_shape(0) == self.input_shape, \
            f"Input binding shape mismatch: {self.engine.get_binding_shape(0)} vs {self.input_shape}"
        assert self.engine.get_binding_shape(1) == self.output_shape, \
            f"Output binding shape mismatch: {self.engine.get_binding_shape(1)} vs {self.output_shape}"

        logger.info("YOLOv8n TensorRT detector initialized successfully")

    def preprocess(self, image):
        """Preprocess image to match model input requirements"""
        # Resize maintaining aspect ratio
        input_height, input_width = INPUT_SHAPE
        image_height, image_width = image.shape[:2]

        # Calculate scaling ratio
        r = min(input_width / image_width, input_height / image_height)
        new_width, new_height = int(image_width * r), int(image_height * r)

        # Resize
        resized = cv2.resize(image, (new_width, new_height))

        # Create letterboxed image
        letterboxed = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        offset_x, offset_y = (input_width - new_width) // 2, (input_height - new_height) // 2
        letterboxed[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized

        # Normalize and transpose
        preprocessed = letterboxed.astype(np.float32) / 255.0
        preprocessed = np.transpose(preprocessed, (2, 0, 1))  # HWC to CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension

        return preprocessed, (offset_x, offset_y, new_width, new_height, image.shape[0], image.shape[1])

    def postprocess(self, raw_output, preprocess_info, conf_threshold=0.25, iou_threshold=0.45):
        """
        Postprocess raw model output
        raw_output shape: [1, 84, 8400] where:
        - 84 channels: [x, y, w, h, 80 class scores]
        - 8400: number of predictions
        """
        # Reshape output to [8400, 84] for easier processing
        predictions = raw_output[0].transpose(1, 0)  # [8400, 84]

        # Extract boxes and scores
        boxes = predictions[:, :4]  # [8400, 4]
        scores = predictions[:, 4:]  # [8400, 80]

        # Get max class scores and corresponding class IDs
        max_scores = np.max(scores, axis=1)  # [8400]
        class_ids = np.argmax(scores, axis=1)  # [8400]

        # Filter by confidence threshold
        mask = max_scores > conf_threshold
        if not np.any(mask):
            return np.array([]), np.array([]), np.array([])

        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            max_scores.tolist(),
            conf_threshold,
            iou_threshold
        )

        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        indices = indices.flatten()

        # Get final detections
        final_boxes = boxes_xyxy[indices]
        final_scores = max_scores[indices]
        final_class_ids = class_ids[indices]

        # Unpack preprocessing info
        offset_x, offset_y, new_width, new_height, orig_h, orig_w = preprocess_info

        # Calculate scale and padding compensation
        gain = min(INPUT_SHAPE[0] / orig_w, INPUT_SHAPE[1] / orig_h)
        pad_x = (INPUT_SHAPE[0] - orig_w * gain) / 2
        pad_y = (INPUT_SHAPE[1] - orig_h * gain) / 2

        # Denormalize coordinates to original image space
        final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - pad_x) / gain
        final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - pad_y) / gain

        # Clip to image boundaries
        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, orig_w)
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, orig_h)

        return final_boxes, final_scores, final_class_ids

    def infer(self, image):
        """Run inference with proper shape handling"""
        # Preprocess image
        input_data, preprocess_info = self.preprocess(image)

        # Copy input data to GPU
        cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)

        # Run inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )

        # Copy output back to CPU
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        # Post-process detections
        return self.postprocess(output, preprocess_info)

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'd_input'):
                self.d_input.free()
            if hasattr(self, 'd_output'):
                self.d_output.free()
            if hasattr(self, 'context'):
                del self.context
            if hasattr(self, 'engine'):
                del self.engine
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def draw_detections(image, boxes, scores, class_ids):
    """Draw bounding boxes and labels on the image"""
    h, w = image.shape[:2]

    # Use a fixed color palette for consistency
    color_palette = np.array([
        [56, 56, 255], [151, 157, 255], [31, 112, 255], [29, 178, 255],
        [49, 210, 207], [10, 249, 72], [23, 204, 146], [134, 219, 61],
        [52, 147, 26], [187, 212, 0], [168, 153, 44], [255, 194, 0],
        [255, 148, 0], [255, 103, 0], [255, 59, 0]
    ])

    for box, score, class_id in zip(boxes, scores, class_ids):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box.astype(np.int32)

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Get color from palette
        color = color_palette[int(class_id) % len(color_palette)].tolist()

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Create label
        label = f'{CLASS_NAMES[int(class_id)]} {score:.2f}'

        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline - 10),
            (x1 + label_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return image


def init_camera():
    """Initialize USB camera with optimized GStreamer pipeline"""
    logger.info("Initializing USB camera with GStreamer pipeline...")
    gst_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=640, height=480, format=YUY2, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=true"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to initialize camera with GStreamer pipeline")

    # Set optimal buffer size
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    detector = YOLOv8nDetector(ENGINE_PATH)
    cap = init_camera()

    frame_count = 0
    fps_start_time = time.time()

    try:
        logger.info("Starting inference loop...")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start_time)
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                fps_start_time = time.time()

            # Run detection
            boxes, scores, class_ids = detector.infer(frame)

            # Log detections if any
            if len(boxes) > 0:
                detected_classes = [CLASS_NAMES[int(i)] for i in class_ids]
                logger.info(f"Detections: {list(zip(detected_classes, scores))}")

            # Draw results
            frame = draw_detections(frame, boxes, scores, class_ids)

            # Display
            cv2.namedWindow("v8-nano Inference", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("v8-nano Inference", 1280, 720)
            cv2.imshow("v8-nano Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        del detector


if __name__ == "__main__":
    main()
