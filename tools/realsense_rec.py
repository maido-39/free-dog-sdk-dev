#!/usr/bin/env python3
"""
Record Realsense RGB stream at 50 FPS to MP4 and display with OpenCV.
Start recording: press 'n'
Stop recording: press 'm'
Quit: press 'q' or close window

The script prefers pyrealsense2; if not available it will use the default camera via OpenCV.
Output files are saved under `videos/` as YYYY-MM-DD_HH-MM-SS-realsense.mp4
"""
import os
import time
from datetime import datetime

import cv2
import numpy as np

# try to import pyrealsense2; if not available, fallback to cv2.VideoCapture
try:
    import pyrealsense2 as rs
    HAS_RS = True
except Exception:
    HAS_RS = False

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
os.makedirs(OUT_DIR, exist_ok=True)

DESIRED_FPS = 50
DESIRED_W = 1280
DESIRED_H = 720
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_realsense():
    # return (capture_source, getter_function, width, height, fps)
    if not HAS_RS:
        return None
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        # try to enable color at requested resolution and fps
        cfg.enable_stream(rs.stream.color, DESIRED_W, DESIRED_H, rs.format.bgr8, DESIRED_FPS)
        pipeline.start(cfg)
        # give it a moment
        time.sleep(0.1)
        return (pipeline, None, DESIRED_W, DESIRED_H, DESIRED_FPS)
    except Exception:
        try:
            # fallback to default color with any available fps
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, DESIRED_FPS)
            pipeline.start(cfg)
            time.sleep(0.1)
            return (pipeline, None, 640, 480, DESIRED_FPS)
        except Exception:
            return None


def read_realsense_frame(pipeline):
    # returns BGR frame or None
    try:
        frames = pipeline.wait_for_frames(timeout_ms=500)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        img = np.asanyarray(color_frame.get_data())
        return img
    except Exception:
        return None


def open_cv_cam():
    cap = cv2.VideoCapture(0)
    # try to set resolution and fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)
    # read properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or DESIRED_W)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or DESIRED_H)
    fps = cap.get(cv2.CAP_PROP_FPS) or DESIRED_FPS
    return (cap, None, w, h, fps)


def main():
    print("Starting realsense recorder (n=start, m=stop, q=quit)")

    # choose source
    source = None
    if HAS_RS:
        source = open_realsense()
        if source:
            print("Using pyrealsense2 pipeline for color stream.")
    if not source:
        print("pyrealsense2 not available or failed -> falling back to OpenCV VideoCapture(0)")
        source = open_cv_cam()

    if source is None:
        print("No camera available. Exiting.")
        return

    cap_obj, _, width, height, fps = source

    writer = None
    recording = False
    last_fname = None

    try:
        # if using realsense pipeline, cap_obj is pipeline and reading is different
        while True:
            frame = None
            if HAS_RS and isinstance(cap_obj, rs.pipeline):
                # lazy import numpy only when needed
                import numpy as np
                frames = cap_obj.poll_for_frames()
                if frames:
                    color = frames.get_color_frame()
                    if color:
                        frame = np.asanyarray(color.get_data())
            else:
                ret, frame = cap_obj.read()
                if not ret:
                    frame = None

            if frame is None:
                # show a black frame while waiting
                frame = 255 * np.zeros((int(height), int(width), 3), dtype='uint8') if 'np' in globals() else None

            # overlay recording indicator
            if recording:
                cv2.putText(frame, 'REC', (10, 30), FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Realsense RGB', frame)

            # write if recording
            if recording and writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                if not recording:
                    # start new file
                    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    fname = f"{ts}-realsense.mp4"
                    fpath = os.path.join(OUT_DIR, fname)
                    writer = cv2.VideoWriter(fpath, FOURCC, DESIRED_FPS, (int(width), int(height)))
                    if not writer.isOpened():
                        print("Failed to open VideoWriter, aborting start")
                        writer = None
                    else:
                        recording = True
                        last_fname = fpath
                        print(f"Recording started -> {fpath}")
            elif key == ord('m'):
                if recording:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print(f"Recording stopped -> {last_fname}")

    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        if HAS_RS and isinstance(cap_obj, rs.pipeline):
            try:
                cap_obj.stop()
            except Exception:
                pass
        else:
            try:
                cap_obj.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Exited cleanly")


if __name__ == '__main__':
    main()
