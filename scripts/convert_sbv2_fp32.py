"""
Convert the Style-Bert-VITS2 voice model to float32 (in place).

The default jvnv-F1-jp weights are stored in float16.  On CPU that causes
mixed-precision errors during synthesis:
    "Input type (struct c10::Half) and bias type (float) should be the same"
because some layers load as fp16 while others stay fp32.

This script rewrites the .safetensors with every tensor cast to float32, so the
whole generator is fp32 and CPU inference works.  The original fp16 file is kept
as a .fp16.bak backup.  Run once, then restart main.py.

    python scripts/convert_sbv2_fp32.py
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SBV2_ASSETS_DIR, SBV2_MODEL_NAME, SBV2_MODEL_FILE, SBV2_STYLE_FILE


def _convert_weights(path: str) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors, meta, n_fp16 = {}, {}, 0
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            t = f.get_tensor(key)
            if t.dtype in (torch.float16, torch.bfloat16):
                t = t.float()
                n_fp16 += 1
            tensors[key] = t

    if n_fp16 == 0:
        print(f"  weights: already float32 ({os.path.basename(path)}).")
        return
    backup = path + ".fp16.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    save_file(tensors, path, metadata=meta)
    print(f"  weights: converted {n_fp16} tensors to float32 → {os.path.basename(path)}")


def _convert_style(path: str) -> None:
    import numpy as np
    arr = np.load(path)
    if arr.dtype == np.float32:
        print(f"  style:   already float32 ({os.path.basename(path)}).")
        return
    backup = path + ".orig.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    np.save(path, arr.astype(np.float32))
    print(f"  style:   converted {arr.dtype} → float32 ({os.path.basename(path)})")


def main() -> None:
    try:
        import torch  # noqa: F401
        import numpy  # noqa: F401
        from safetensors import safe_open  # noqa: F401
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Activate the venv and install requirements first.")
        sys.exit(1)

    base = os.path.join(SBV2_ASSETS_DIR, SBV2_MODEL_NAME)
    weights = os.path.join(base, SBV2_MODEL_FILE)
    style = os.path.join(base, SBV2_STYLE_FILE)
    if not os.path.exists(weights):
        print(f"Model not found: {weights}\nRun: python scripts/download_sbv2_model.py")
        sys.exit(1)

    print(f"Forcing float32 for '{SBV2_MODEL_NAME}':")
    _convert_weights(weights)
    if os.path.exists(style):
        _convert_style(style)
    print("Done. Now restart:  python main.py")


if __name__ == "__main__":
    main()
