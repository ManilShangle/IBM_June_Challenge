"""Extract a single representative frame from uploaded video footage so it
can be passed to Granite Vision the same way a still image would be.
"""
import os
import tempfile

import cv2


class VideoExtractionError(RuntimeError):
    pass


def extract_middle_frame_jpeg(video_bytes: bytes) -> bytes:
    """Writes the uploaded video to a temp file (cv2 needs a real file path
    for most container formats), grabs the frame at the midpoint of the
    clip, and returns it re-encoded as JPEG bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            raise VideoExtractionError("Could not open the uploaded video file.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(frame_count // 2, 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            raise VideoExtractionError("Could not read a frame from the uploaded video.")

        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            raise VideoExtractionError("Could not encode the extracted frame as JPEG.")
        return buffer.tobytes()
    finally:
        os.unlink(tmp_path)
