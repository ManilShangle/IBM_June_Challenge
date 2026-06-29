"""Extract frames from uploaded video footage so they can be passed to
Granite Vision. Supports both a single middle frame (legacy) and a set of
evenly-spaced frames for richer multi-frame analysis.
"""
import os
import tempfile

import cv2

from src.config import VIDEO_FRAMES


class VideoExtractionError(RuntimeError):
    pass


def extract_middle_frame_jpeg(video_bytes: bytes) -> bytes:
    """Single-frame extraction: returns the frame at the video midpoint."""
    frames = extract_frames_jpeg(video_bytes, num_frames=1)
    return frames[0]


def extract_frames_jpeg(video_bytes: bytes, num_frames: int = VIDEO_FRAMES) -> list[bytes]:
    """Extract ``num_frames`` evenly-spaced frames from ``video_bytes`` and
    return each as JPEG bytes. At least one frame is always returned.

    Writes the uploaded bytes to a temp file because cv2.VideoCapture requires
    a real file path for most container formats. The temp file is deleted before
    returning regardless of success or failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            raise VideoExtractionError("Could not open the uploaded video file.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise VideoExtractionError("Could not determine frame count for the uploaded video.")

        # Clamp num_frames to what the video actually contains
        num_frames = max(1, min(num_frames, frame_count))

        # Pick evenly-spaced positions: for num_frames=1 this gives [midpoint]
        if num_frames == 1:
            positions = [frame_count // 2]
        else:
            step = frame_count / num_frames
            positions = [int(step * i + step / 2) for i in range(num_frames)]

        results: list[bytes] = []
        for pos in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(pos, frame_count - 1))
            success, frame = capture.read()
            if not success or frame is None:
                continue
            encoded, buffer = cv2.imencode(".jpg", frame)
            if encoded:
                results.append(buffer.tobytes())

        capture.release()

        if not results:
            raise VideoExtractionError("Could not read any frames from the uploaded video.")
        return results
    finally:
        os.unlink(tmp_path)
