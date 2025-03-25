import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging
import time
import can
import cantools
import crc8
import asyncio
import gi
import signal
import sys
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
ENGINE_PATH = "./yolov8n.trt"
INPUT_SHAPE = (640, 640)
CONF_THRESHOLD = 0.8
STOP_SIGN_CLASS_ID = 11  # COCO dataset index for 'stop sign'
BRAKE_FORCE = 0.4  # 40% brake force
BRAKE_DURATION = 10.0  # seconds
DETECTION_TO_BRAKE_DELAY = 1.0  # seconds

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

class MCM:
    """
    Motor Control Module for brake control via CAN bus communication.
    """
    def __init__(self, channel: str):
        self.db = cantools.database.Database()
        self.db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        self.bus = can.Bus(bustype='socketcan', channel=channel, bitrate=500000)
        self.control_count = 0
        self.bus_address = 1

    def calc_crc8(self, data: bytearray) -> int:
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]

    async def enable_control(self, module: str):
        control_enable_msg = self.db.get_message_by_name('ControlEnable')
        interface = {'brake': 0, 'accel': 1, 'steer': 2}.get(module)
        data = bytearray(control_enable_msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': interface,
            'Enable': 1,
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_enable_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        await asyncio.sleep(0.1)

    async def update_brake_setpoint(self, value: float):
        control_cmd_msg = self.db.get_message_by_name('ControlCommand')
        data = bytearray(control_cmd_msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 0,
            'Count8': self.control_count,
            'Value': value,
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_cmd_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        logging.info(f"Sent brake command with ID: {msg.arbitration_id}, data: {msg.data.hex()}")
        self.control_count = (self.control_count + 1) % 256

    async def apply_brakes(self, percentage: float, duration: float):
        await self.enable_control('brake')
        logging.info(f"Applying {percentage * 100}% brake force for {duration} seconds")
        await self.update_brake_setpoint(percentage)
        await asyncio.sleep(duration)
        await self.update_brake_setpoint(0)
        logging.info("Brake force released")


class GstreamerPipelineManager:
    def __init__(self):
        Gst.init(None)
        self.pipeline = None
        self.bus = None

    def create_pipeline(self, pipeline_str):
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.bus = self.pipeline.get_bus()
            return True
        except Exception as e:
            logging.error(f"Pipeline creation failed: {e}")
            return False

    def set_state_null(self):
        """Explicitly transition pipeline to NULL state"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            self.pipeline.set_state(Gst.State.READY)
            self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.get_state(Gst.CLOCK_TIME_NONE)

    def cleanup(self):
        """Release pipeline resources"""
        if self.pipeline:
            self.set_state_null()
            if self.bus:
                self.bus.remove_signal_watch()
            self.pipeline = None
            self.bus = None

class CameraManager:
    def __init__(self):
        self.cap = None
        self.gst_manager = GstreamerPipelineManager()
        self.device_path = "/dev/video0"
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def build_pipeline(self):
        return (
            f"v4l2src device={self.device_path} ! "
            "video/x-raw, width=640, height=480, format=YUY2, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink name=appsink0"
        )

    def init_camera(self):
        logging.info("Initializing USB camera with GStreamer pipeline...")
        try:
            pipeline_str = self.build_pipeline()
            self.gst_manager.create_pipeline(pipeline_str)

            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to initialize camera")

            return self.cap

        except Exception as e:
            logging.error(f"Camera initialization failed: {str(e)}")
            self.cleanup()
            raise

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.gst_manager:
            self.gst_manager.set_state_null()

        cv2.destroyAllWindows()

    def _signal_handler(self, signum, frame):
        logging.info(f"Received signal {signum}")
        self.cleanup()
        sys.exit(0)

class YOLOv8nDetector:
    def __init__(self, engine_path):
        logger.info("Initializing YOLOv8n TensorRT detector...")
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_shape = (1, 3, *INPUT_SHAPE)
        self.output_shape = (1, 84, 8400)
        input_size = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        output_size = int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize)
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)
        assert self.engine.get_binding_shape(0) == self.input_shape, \
            f"Input binding shape mismatch: {self.engine.get_binding_shape(0)} vs {self.input_shape}"
        assert self.engine.get_binding_shape(1) == self.output_shape, \
            f"Output binding shape mismatch: {self.engine.get_binding_shape(1)} vs {self.output_shape}"
        logger.info("YOLOv8n TensorRT detector initialized successfully")

    def preprocess(self, image):
        input_height, input_width = INPUT_SHAPE
        image_height, image_width = image.shape[:2]
        r = min(input_width / image_width, input_height / image_height)
        new_width, new_height = int(image_width * r), int(image_height * r)
        resized = cv2.resize(image, (new_width, new_height))
        letterboxed = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        offset_x, offset_y = (input_width - new_width) // 2, (input_height - new_height) // 2
        letterboxed[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
        preprocessed = letterboxed.astype(np.float32) / 255.0
        preprocessed = np.transpose(preprocessed, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)
        return preprocessed, (offset_x, offset_y, new_width, new_height, image.shape[0], image.shape[1])

    def postprocess(self, raw_output, preprocess_info, conf_threshold=0.25, iou_threshold=0.45):
        predictions = raw_output[0].transpose(1, 0)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        mask = max_scores > conf_threshold
        if not np.any(mask):
            return np.array([]), np.array([]), np.array([])
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), max_scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        indices = indices.flatten()
        final_boxes = boxes_xyxy[indices]
        final_scores = max_scores[indices]
        final_class_ids = class_ids[indices]
        offset_x, offset_y, new_width, new_height, orig_h, orig_w = preprocess_info
        gain = min(INPUT_SHAPE[0] / orig_w, INPUT_SHAPE[1] / orig_h)
        pad_x = (INPUT_SHAPE[0] - orig_w * gain) / 2
        pad_y = (INPUT_SHAPE[1] - orig_h * gain) / 2
        final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - pad_x) / gain
        final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - pad_y) / gain
        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, orig_w)
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, orig_h)
        return final_boxes, final_scores, final_class_ids

    def infer(self, image):
        input_data, preprocess_info = self.preprocess(image)
        cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.postprocess(output, preprocess_info)

    def __del__(self):
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
    h, w = image.shape[:2]
    color_palette = np.array([
        [56, 56, 255], [151, 157, 255], [31, 112, 255], [29, 178, 255],
        [49, 210, 207], [10, 249, 72], [23, 204, 146], [134, 219, 61],
        [52, 147, 26], [187, 212, 0], [168, 153, 44], [255, 194, 0],
        [255, 148, 0], [255, 103, 0], [255, 59, 0]
    ])
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(np.int32)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = color_palette[int(class_id) % len(color_palette)].tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f'{CLASS_NAMES[int(class_id)]} {score:.2f}'
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_height - baseline - 10), (x1 + label_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

async def main():
    detector = YOLOv8nDetector(ENGINE_PATH)
    mcm = MCM(channel='can0')
    camera = CameraManager()
    cap = None

    frame_count = 0
    fps_start_time = time.time()
    detection_state = {
        'stop_sign_detected': False,
        'braking_in_progress': False,
        'brake_start_time': None,
        'critical_frame_captured': False,
        'last_detection_time': None,
        'consecutive_detections': 0
    }

    try:
        cap = camera.init_camera()
        logger.info("Starting inference loop...")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start_time)
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                fps_start_time = time.time()

            boxes, scores, class_ids = detector.infer(frame)

            if len(boxes) > 0:
                detected_classes = [CLASS_NAMES[int(i)] for i in class_ids]
                logger.info(f"Detections: {list(zip(detected_classes, scores))}")

            frame = draw_detections(frame, boxes, scores, class_ids)

            current_time = time.time()

            for box, score, class_id in zip(boxes, scores, class_ids):
                if int(class_id) == STOP_SIGN_CLASS_ID and score > CONF_THRESHOLD:
                    if not detection_state['stop_sign_detected'] and not detection_state['braking_in_progress']:
                        detection_state['stop_sign_detected'] = True
                        detection_state['braking_in_progress'] = True
                        detection_state['brake_start_time'] = current_time
                        detection_state['last_detection_time'] = current_time

                        logger.info(f"Stop sign detected with confidence {score:.2f}")
                        logger.info(f"Initiating braking sequence with {BRAKE_FORCE * 100}% force")
                        await mcm.apply_brakes(BRAKE_FORCE, BRAKE_DURATION)
                    break

            if detection_state['braking_in_progress'] and detection_state['brake_start_time']:
                if current_time - detection_state['brake_start_time'] >= BRAKE_DURATION:
                    await handle_brake_release(detection_state, mcm)

            cv2.imshow("YOLOv8n Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    finally:
        camera.cleanup()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        del detector

async def handle_brake_release(state: dict, mcm: MCM) -> None:
    logging.info("Brake duration completed, releasing brakes")
    await mcm.update_brake_setpoint(0)
    state['braking_in_progress'] = False
    state['stop_sign_detected'] = False
    state['brake_start_time'] = None
    state['critical_frame_captured'] = False
    state['last_detection_time'] = None
    state['consecutive_detections'] = 0

if __name__ == "__main__":
    asyncio.run(main())
