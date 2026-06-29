"""IBM Granite Vision client: turns an uploaded image (or a video frame) into
a text description of what's visible in the incident, which then feeds into
the same text-based prediction pipeline as the situation description.
"""
import io

from src.config import REPLICATE_API_TOKEN, REPLICATE_VISION_MODEL

VISION_PROMPT = (
    "You are analyzing a single frame from football (soccer) match footage "
    "for a VAR review. Describe only what is visible and relevant to a "
    "refereeing decision: player positions relative to each other and the "
    "goal line, ball position, body contact between players, arm/hand "
    "position relative to the body, and where on the pitch this is "
    "happening. Be factual and concise, 2-4 sentences. Do not guess at a "
    "ruling, only describe what is visible."
)


class VisionClientError(RuntimeError):
    pass


def describe_image(image_bytes: bytes) -> str:
    if not REPLICATE_API_TOKEN:
        raise VisionClientError(
            "REPLICATE_API_TOKEN missing. Footage analysis requires the "
            "Replicate backend regardless of GRANITE_BACKEND for text "
            "prediction, since IBM Granite Vision is only wired up there."
        )
    import replicate

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    image_file = io.BytesIO(image_bytes)
    image_file.name = "frame.jpg"
    output = client.run(
        REPLICATE_VISION_MODEL,
        input={
            "images": [image_file],
            "prompt": VISION_PROMPT,
            "max_tokens": 300,
            "temperature": 0.2,
        },
    )
    if isinstance(output, list):
        return "".join(str(chunk) for chunk in output)
    return str(output)
