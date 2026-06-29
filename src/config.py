import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PDF_PATH = DATA_DIR / "raw" / "ifab_laws_2025_26.pdf"
LAW_SECTIONS_MD_PATH = DATA_DIR / "processed" / "law_sections.md"
LAW_SECTIONS_JSON_PATH = DATA_DIR / "processed" / "law_sections.json"
LAW_EMBEDDINGS_PATH = DATA_DIR / "processed" / "law_embeddings.npy"
SCENARIOS_PATH = DATA_DIR / "scenarios" / "preset_scenarios.json"

GRANITE_BACKEND = os.getenv("GRANITE_BACKEND", "replicate")

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "ibm-granite/granite-4.0-h-small")
REPLICATE_VISION_MODEL = os.getenv("REPLICATE_VISION_MODEL", "ibm-granite/granite-vision-3.3-2b")

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "ibm-granite/granite-3.1-8b-instruct")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite3.1-dense:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

LANGFLOW_API_URL = os.getenv("LANGFLOW_API_URL", "http://localhost:7860")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID", "1faee69d-d060-4a1d-b573-001ca4180798")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 4
VIDEO_FRAMES = int(os.getenv("VIDEO_FRAMES", "5"))
