"""
Download the default Style-Bert-VITS2 Japanese voice (jvnv-F1-jp) into
models/sbv2/model_assets/.

    python scripts/download_sbv2_model.py

The per-language BERT models (DeBERTa) are downloaded automatically the first
time the engine runs, so you do not need to fetch them here.
"""

import os
import sys

# make the project root importable so we can read config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SBV2_ASSETS_DIR, SBV2_MODEL_NAME, SBV2_MODEL_FILE, SBV2_CONFIG_FILE, SBV2_STYLE_FILE

REPO = "litagin/style_bert_vits2_jvnv"


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. Install deps first: pip install -r requirements.txt")
        sys.exit(1)

    os.makedirs(SBV2_ASSETS_DIR, exist_ok=True)
    files = [SBV2_MODEL_FILE, SBV2_CONFIG_FILE, SBV2_STYLE_FILE]
    print(f"Downloading '{SBV2_MODEL_NAME}' from {REPO} → {SBV2_ASSETS_DIR}")
    for fname in files:
        rel = f"{SBV2_MODEL_NAME}/{fname}"
        print(f"  ↓ {rel}")
        path = hf_hub_download(REPO, rel, local_dir=SBV2_ASSETS_DIR)
        print(f"    → {path}")
    print("✅ Style-Bert-VITS2 voice ready.")


if __name__ == "__main__":
    main()
