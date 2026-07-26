"""Run-directory, frame-overlay, and video artifact helpers."""

import os
import shutil
import subprocess
from datetime import datetime

import cv2
import numpy as np


def prepare_run_dir(output_dir, output_group=None):
    """Create a timestamped run directory, optionally inside one output group."""
    if output_group:
        if (
            not isinstance(output_group, str)
            or output_group.strip() != output_group
            or output_group in {".", ".."}
            or "/" in output_group
            or "\\" in output_group
        ):
            raise ValueError(
                "output_group must be one simple directory name inside "
                "--output-dir"
            )
        output_dir = os.path.join(output_dir, output_group)
    os.makedirs(output_dir, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(output_dir, run_name)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(output_dir, f"{run_name}_{suffix:02d}")
        suffix += 1
    frame_dir = os.path.join(run_dir, "frames")
    os.makedirs(frame_dir)
    return run_dir, frame_dir


def rgb_to_bgr(rgb):
    """Convert Habitat RGB channel order for OpenCV output."""
    return rgb[:, :, [2, 1, 0]].copy()


def draw_status(bgr, step, action, valid_action, collision):
    """Draw compact execution status on one video frame."""
    color = (255, 255, 255) if valid_action else (0, 165, 255)
    lines = [
        f"step={step:03d}",
        f"action={action}",
        f"valid={valid_action} collision={collision}",
    ]
    height, width = bgr.shape[:2]
    font_scale = min(0.65, max(0.38, width / 900.0))
    thickness = 2 if width >= 480 else 1
    padding = max(6, width // 100)
    text_height = cv2.getTextSize(
        "Ag",
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )[0][1]
    line_height = text_height + max(7, text_height // 2)
    overlay_height = min(height, padding * 2 + line_height * len(lines))
    status_area = bgr[:overlay_height, :].copy()
    black = np.zeros_like(status_area)
    bgr[:overlay_height, :] = cv2.addWeighted(
        status_area,
        0.35,
        black,
        0.65,
        0.0,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            bgr,
            line,
            (padding, padding + text_height + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def transcode_video_to_h264(source_path, output_path):
    """Convert one silent video to browser-compatible H.264 with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, "ffmpeg is not installed"

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffmpeg exited with an error"
    return os.path.isfile(output_path), ""


def write_video(frame_paths, video_path, fps):
    """Encode frames into H.264 MP4, with an mp4v fallback if needed."""
    if not frame_paths:
        return False

    first = cv2.imread(frame_paths[0])
    if first is None:
        return False

    root, extension = os.path.splitext(video_path)
    temporary_path = f"{root}.mp4v{extension or '.mp4'}"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        temporary_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        return False
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()

    converted, error = transcode_video_to_h264(temporary_path, video_path)
    if converted:
        os.remove(temporary_path)
        return True

    if os.path.exists(video_path):
        os.remove(video_path)
    os.replace(temporary_path, video_path)
    print(f"warning: H.264 conversion failed; kept mp4v video: {error}")
    return os.path.exists(video_path)
