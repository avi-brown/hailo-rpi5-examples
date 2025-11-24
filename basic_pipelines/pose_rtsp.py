from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import subprocess
import signal
import threading
import cv2
import hailo

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
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

# RTSP configuration
RTSP_URI = "rtsp://192.168.1.41:554/stream/main"
RTSP_TRANSPORT = "tcp"  # Passed to ffmpeg's -rtsp_transport
FFMPEG_BIN = "ffmpeg"

# Video configuration
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FRAME_RATE = 30

# Filtering
MIN_CONFIDENCE = 0.7


def rtsp_source_pipeline(
    video_width: int = 1280,
    video_height: int = 720,
    frame_rate: int = 30,
    sync: str = "false",
    name: str = "source",
    video_format: str = "RGB",
):
    """Source pipeline that ingests raw RGB frames pushed into an appsrc."""
    fps_caps = f"video/x-raw, framerate={frame_rate}/1" if sync == "true" else "video/x-raw"

    pipeline = (
        f"appsrc name={name}_appsrc is-live=true do-timestamp=true format=time "
        f"block=true max-bytes=0 "
        f"caps=video/x-raw,format={video_format},width={video_width},height={video_height},"
        f"framerate={frame_rate}/1,pixel-aspect-ratio=1/1 ! "
        f'{QUEUE(name=f"{name}_src_q")} ! '
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


# -----------------------------------------------------------------------------------------------
# Helper for COCO keypoints ordering
# -----------------------------------------------------------------------------------------------
def get_keypoints():
    return {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    format, width, height = get_caps_from_pad(pad)
    image_width = int(width) if width is not None else VIDEO_WIDTH
    image_height = int(height) if height is not None else VIDEO_HEIGHT

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    keypoints = get_keypoints()

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
        for name, idx in keypoints.items():
            point = points[idx]
            x = (point.x() * bbox.width() + bbox.xmin()) * image_width
            y = (point.y() * bbox.height() + bbox.ymin()) * image_height
            string_to_print += f"{name}: x={x:.1f} y={y:.1f}\n"
            if user_data.use_frame and frame is not None:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    if user_data.use_frame and frame is not None:
        cv2.putText(frame, f"People: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame_bgr)

    print(string_to_print)
    return Gst.PadProbeReturn.OK


class RTSPPoseEstimationApp(GStreamerPoseEstimationApp):
    """Pose estimation app wired to an RTSP source."""

    def __init__(self, app_callback, user_data, parser=None):
        self.ffmpeg_proc = None
        self.ffmpeg_thread = None
        self.appsrc = None
        self.running = True
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
        self.start_ffmpeg()
        super().create_pipeline()
        self.appsrc = self.pipeline.get_by_name("source_appsrc")
        if self.appsrc is None:
            raise RuntimeError("source_appsrc not found in pipeline")
        self.ffmpeg_thread = threading.Thread(target=self.pump_ffmpeg_to_appsrc, daemon=True)
        self.ffmpeg_thread.start()

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
        if getattr(self, "ffmpeg_proc", None):
            self.ffmpeg_proc.send_signal(signal.SIGTERM)
            try:
                self.ffmpeg_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
        if self.ffmpeg_thread:
            self.ffmpeg_thread.join(timeout=2)
        super().shutdown(signum, frame)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    app = RTSPPoseEstimationApp(app_callback, user_data)
    app.run()
