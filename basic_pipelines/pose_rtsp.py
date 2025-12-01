from pathlib import Path
import argparse
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import json
import socket
import subprocess
import signal
import threading
import time
import cv2
import hailo
import copy
from collections import deque

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    QUEUE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
    get_default_parser,
)

# RTSP configuration
RTSP_URI = "rtsp://192.168.1.41:554/stream/main"
RTSP_TRANSPORT = "tcp"  # Passed to ffmpeg's -rtsp_transport
FFMPEG_BIN = "ffmpeg"

# Telemetry publishing
UDP_TARGET = ("192.168.1.30", 6666)
UDP_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Video configuration
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
FRAME_RATE = 30

# Filtering
MIN_CONFIDENCE = 0.6
KEYPOINTS_OF_INTEREST = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def rtsp_source_pipeline(
    video_width: int = 640,
    video_height: int = 360,
    frame_rate: int = 30,
    sync: str = "false",
    name: str = "source",
    video_format: str = "RGB",
):
    """Source pipeline that ingests raw RGB frames pushed into an appsrc."""
    fps_caps = f"video/x-raw, framerate={frame_rate}/1" if sync == "true" else "video/x-raw"

    pipeline = (
        # Keep the appsrc queue tiny and drop old frames to avoid latency buildup.
        f"appsrc name={name}_appsrc is-live=true do-timestamp=true format=time "
        f"block=true max-buffers=2 "
        f"caps=video/x-raw,format={video_format},width={video_width},height={video_height},"
        f"framerate={frame_rate}/1,pixel-aspect-ratio=1/1 ! "
        f'{QUEUE(name=f"{name}_src_q", max_size_buffers=1, leaky="downstream")} ! '
        f"videoconvert n-threads=3 name={name}_convert qos=false ! "
        f"videoscale name={name}_videoscale n-threads=2 ! "
        f"video/x-raw, pixel-aspect-ratio=1/1, format={video_format}, width={video_width}, height={video_height} ! "
        f"videorate name={name}_videorate ! capsfilter name={name}_fps_caps caps=\"{fps_caps}\" "
    )

    return pipeline


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self._payload_lock = threading.Lock()
        self._latest_payload = None
        self._latest_payload_bytes = None
        self._payload_event = threading.Event()
        self._rate_lock = threading.Lock()
        self._last_rate_time = time.time()
        self._last_count = 0
        self._inference_rate = 0.0
        self._timing_lock = threading.Lock()
        self._push_times = deque(maxlen=400)  # FIFO arrival times
        self._last_push_ts = None
        self._last_inference_latency = None

    def set_payload(self, payload: dict):
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with self._payload_lock:
            self._latest_payload = payload
            self._latest_payload_bytes = payload_bytes
            self._payload_event.set()

    def get_payload(self):
        with self._payload_lock:
            return copy.deepcopy(self._latest_payload)

    def get_payload_bytes(self):
        with self._payload_lock:
            return self._latest_payload_bytes

    def wait_for_payload(self, timeout=None):
        if not self._payload_event.wait(timeout):
            return None
        with self._payload_lock:
            payload_bytes = self._latest_payload_bytes
        self._payload_event.clear()
        return payload_bytes

    def notify_publishers(self):
        # Wake any waiting publisher threads (e.g., during shutdown).
        self._payload_event.set()

    def update_inference_rate(self) -> float:
        # Compute a smoothed inference rate over ~1 second windows.
        with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_rate_time
            if elapsed >= 1.0:
                current_count = self.get_count()
                self._inference_rate = (current_count - self._last_count) / elapsed
                self._last_count = current_count
                self._last_rate_time = now
            return self._inference_rate

    def record_push_time(self, ts: float):
        # Track when a buffer entered the pipeline so we can compute latency later.
        with self._timing_lock:
            self._push_times.append(ts)

    def pop_push_time(self):
        with self._timing_lock:
            if self._push_times:
                return self._push_times.popleft()
            return None

    def note_inference(self, push_ts: float, inference_latency: float):
        with self._timing_lock:
            self._last_push_ts = push_ts
            self._last_inference_latency = inference_latency

    def get_last_timing(self):
        with self._timing_lock:
            return self._last_push_ts, self._last_inference_latency


# -----------------------------------------------------------------------------------------------
# Helper for COCO keypoints ordering
# -----------------------------------------------------------------------------------------------
def get_keypoints():
    coco_keypoints = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
    }
    return {name: coco_keypoints[name] for name in KEYPOINTS_OF_INTEREST}


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    inference_rate = user_data.update_inference_rate()
    push_time = user_data.pop_push_time()
    inference_latency_ms = (time.time() - push_time) * 1000.0 if push_time is not None else None
    if inference_latency_ms is not None:
        user_data.note_inference(push_time, inference_latency_ms / 1000.0)
    string_to_print = (
        f"Frame count: {user_data.get_count()}\n"
        f"Inference rate: {inference_rate:.2f} FPS\n"
    )
    if inference_latency_ms is not None:
        string_to_print += f"Latency (push->callback): {inference_latency_ms:.1f} ms\n"

    format, width, height = get_caps_from_pad(pad)
    image_width = int(width) if width is not None else VIDEO_WIDTH
    image_height = int(height) if height is not None else VIDEO_HEIGHT
    center_x = image_width / 2.0
    center_y = image_height / 2.0

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    keypoints = get_keypoints()

    payload = {
        "stream_path": RTSP_URI,
        "objects": [],
    }

    people_count = 0
    for detection in list(detections):
        label = detection.get_label()
        confidence = detection.get_confidence()
        if label != "person" or confidence < MIN_CONFIDENCE:
            continue

        bbox = detection.get_bbox()
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        string_to_print += f"Person ID: {track_id} Confidence: {confidence:.2f}\n"
        people_count += 1

        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) == 0:
            continue

        points = landmarks[0].get_points()
        error_x = None
        error_y = None
        keypoints_payload = {}
        head_points = []
        for name, idx in keypoints.items():
            point = points[idx]
            x = (point.x() * bbox.width() + bbox.xmin()) * image_width
            y = (point.y() * bbox.height() + bbox.ymin()) * image_height
            string_to_print += f"{name}: x={x:.1f} y={y:.1f}\n"
            keypoints_payload[name] = [float(x), float(y)]
            head_points.append((x, y))
            if user_data.use_frame and frame is not None:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        if head_points:
            head_x = sum(p[0] for p in head_points) / len(head_points)
            head_y = sum(p[1] for p in head_points) / len(head_points)
            string_to_print += f"head: x={head_x:.1f} y={head_y:.1f}\n"
            keypoints_payload["head"] = [float(head_x), float(head_y)]
            error_x = float(head_x - center_x)
            error_y = float(head_y - center_y)
            string_to_print += f"error_x={error_x:.1f} error_y={error_y:.1f}\n"
            if user_data.use_frame and frame is not None:
                cv2.circle(frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)
        payload["objects"].append(
            {
                "id": int(track_id),
                "class": label,
                "confidence": float(confidence),
                "keypoints": keypoints_payload,
                "error_x": error_x if head_points else None,
                "error_y": error_y if head_points else None,
            }
        )

    if user_data.use_frame and frame is not None:
        cv2.putText(frame, f"People: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame_bgr)

    # Let the publisher thread send at a fixed cadence.
    user_data.set_payload(payload)

    print(string_to_print)
    return Gst.PadProbeReturn.OK


class RTSPPoseEstimationApp(GStreamerPoseEstimationApp):
    """Pose estimation app wired to an RTSP source."""

    def __init__(self, app_callback, user_data, parser=None, headless: bool = False):
        self.ffmpeg_proc = None
        self.ffmpeg_thread = None
        self.appsrc = None
        self.running = True
        self.publish_thread = None
        self.publish_interval = 0.01  # used as timeout fallback
        self.user_data_ref = user_data
        self.headless = headless
        super().__init__(app_callback, user_data, parser)

    def start_ffmpeg(self):
        """Launch ffmpeg to pull RTSP and emit raw RGB frames to stdout."""
        cmd = [
            FFMPEG_BIN,
            "-loglevel",
            "warning",
            "-rtsp_transport",
            RTSP_TRANSPORT,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            RTSP_URI,
            "-an",
            "-vf",
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT},fps={FRAME_RATE}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
        self.ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def pump_ffmpeg_to_appsrc(self):
        """Read raw frames from ffmpeg stdout and push them into appsrc."""
        if self.ffmpeg_proc is None or self.ffmpeg_proc.stdout is None:
            print("ffmpeg process not started or stdout unavailable.")
            return

        frame_size = VIDEO_WIDTH * VIDEO_HEIGHT * 3  # rgb24
        pts = 0
        frame_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FRAME_RATE)
        while self.running:
            data = self.ffmpeg_proc.stdout.read(frame_size)
            if data is None or len(data) < frame_size:
                break
            buf = Gst.Buffer.new_allocate(None, frame_size, None)
            buf.fill(0, data)
            buf.pts = pts
            buf.duration = frame_duration
            self.user_data_ref.record_push_time(time.time())
            pts += frame_duration
            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print(f"appsrc push-buffer returned {ret}")
                break
        if self.appsrc:
            self.appsrc.emit("end-of-stream")

    def create_pipeline(self):
        self.video_source = RTSP_URI
        self.source_type = "rtsp"
        self.sync = "false"
        self.frame_rate = FRAME_RATE
        self.video_width = VIDEO_WIDTH
        self.video_height = VIDEO_HEIGHT
        if self.headless:
            # Drop rendering pipeline and frame conversions for lower latency.
            self.video_sink = "fakesink"
            self.show_fps = False
            self.user_data_ref.use_frame = False
        self.start_ffmpeg()
        super().create_pipeline()
        self.appsrc = self.pipeline.get_by_name("source_appsrc")
        if self.appsrc is None:
            raise RuntimeError("source_appsrc not found in pipeline")
        self.ffmpeg_thread = threading.Thread(target=self.pump_ffmpeg_to_appsrc, daemon=True)
        self.ffmpeg_thread.start()
        self.publish_thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.publish_thread.start()

    def publish_loop(self):
        """Send latest payload as soon as it is available, with a short timeout fallback."""
        while self.running:
            payload_bytes = self.user_data_ref.wait_for_payload(timeout=self.publish_interval)
            if not self.running:
                break
            if payload_bytes:
                try:
                    push_ts, _ = self.user_data_ref.get_last_timing()
                    send_ts = time.time()
                    publish_latency_ms = (send_ts - push_ts) * 1000.0 if push_ts is not None else None
                    if publish_latency_ms is not None:
                        print(f"Latency (push->UDP send): {publish_latency_ms:.1f} ms")
                    UDP_SOCKET.sendto(payload_bytes, UDP_TARGET)
                except OSError as exc:
                    print(f"Failed to publish keypoints: {exc}")

    def get_pipeline_string(self):
        source_pipeline = rtsp_source_pipeline(
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        infer_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size,
        )
        infer_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(infer_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=0)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline_str = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{infer_pipeline_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline_str}"
        )
        print(pipeline_string)
        return pipeline_string

    def shutdown(self, signum=None, frame=None):
        self.running = False
        if hasattr(self.user_data_ref, "notify_publishers"):
            self.user_data_ref.notify_publishers()
        if getattr(self, "ffmpeg_proc", None):
            self.ffmpeg_proc.send_signal(signal.SIGTERM)
            try:
                self.ffmpeg_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
        if self.ffmpeg_thread:
            self.ffmpeg_thread.join(timeout=2)
        if self.publish_thread:
            self.publish_thread.join(timeout=2)
        super().shutdown(signum, frame)


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--no-display", action="store_true", help="Disable display/rendering to reduce latency.")
    # Peek at args to set headless before handing parser to the app (parser will be parsed again inside).
    parsed_args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    if parsed_args.no_display:
        user_data.use_frame = False
    app = RTSPPoseEstimationApp(app_callback, user_data, parser=parser, headless=parsed_args.no_display)
    app.run()
