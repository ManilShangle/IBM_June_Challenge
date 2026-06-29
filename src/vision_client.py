"""IBM Granite Vision client: turns an uploaded image (or video frames) into a
text description of what is visible in the incident, which then feeds into the
same text-based prediction pipeline as the situation description.

For video clips, ``describe_video_frames`` accepts multiple evenly-spaced frames
and sends them together in a single Granite Vision call, giving the model
temporal context across the clip rather than a single frozen moment.
"""
import io

from src.config import REPLICATE_API_TOKEN, REPLICATE_VISION_MODEL

VISION_PROMPT = (
    "You are analyzing footage from a football (soccer) match for a VAR review. "
    "If multiple frames are provided they are evenly spaced through the same clip "
    "in chronological order. Describe only what is visible and relevant to a "
    "refereeing decision: player positions relative to each other and the goal "
    "line, ball position, body contact between players, arm/hand position "
    "relative to the body, player movement across frames if multiple are shown, "
    "and where on the pitch this is happening. Be factual and concise, 3-5 "
    "sentences. Do not guess at a ruling, only describe what is visible."
)


class VisionClientError(RuntimeError):
    pass


def _replicate_client():
    if not REPLICATE_API_TOKEN:
        raise VisionClientError(
            "REPLICATE_API_TOKEN missing. Footage analysis requires the "
            "Replicate backend regardless of GRANITE_BACKEND for text "
            "prediction, since IBM Granite Vision is only wired up there."
        )
    import replicate
    return replicate.Client(api_token=REPLICATE_API_TOKEN)


def describe_image(image_bytes: bytes) -> str:
    """Describe a single still image frame."""
    return describe_video_frames([image_bytes])


def describe_video_frames(frames: list[bytes]) -> str:
    """Send one or more frames to Granite Vision in a single call.

    Passing multiple frames lets the model reason about motion, trajectory,
    and player positions across time rather than from a single frozen moment.
    Frames should be ordered chronologically (earliest first).
    """
    client = _replicate_client()

    image_files = []
    for i, frame_bytes in enumerate(frames):
        buf = io.BytesIO(frame_bytes)
        buf.name = f"frame_{i:02d}.jpg"
        image_files.append(buf)

    output = client.run(
        REPLICATE_VISION_MODEL,
        input={
            "images": image_files,
            "prompt": VISION_PROMPT,
            "max_tokens": 400,
            "temperature": 0.2,
        },
    )
    if isinstance(output, list):
        return "".join(str(chunk) for chunk in output)
    return str(output)
