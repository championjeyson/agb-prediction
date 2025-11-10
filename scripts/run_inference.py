"""
Run AGB Inference
=================

Entry-point script to execute the full Above-Ground Biomass (AGB)
prediction pipeline.

Usage
-----
From the repository root:

    python scripts/run_inference.py

This will:
1. Load configuration files from `configs/`.
2. Process imagery inputs and normalize them.
3. Run inference using the pretrained model.
4. Save the predicted AGB raster to the configured output directory.
"""

import sys
from pathlib import Path

# Ensure repository root is in the Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.inference import run_inference

if __name__ == "__main__":
    try:
        output_path = run_inference()
        print(f"✅ Inference completed successfully.\nOutput saved to: {output_path}")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise

