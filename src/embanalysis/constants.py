from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "output" / "embeddings.duckdb"

HF_MODEL_ALIASES = {
    "olmo": "allenai/OLMo-2-1124-7B",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "phi": "microsoft/Phi-4-mini-instruct",
}

DEFAULT_SEED = 1234
