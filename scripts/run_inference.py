import sys
from pathlib import Path

# Add repo root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import run_inference

if __name__ == "__main__":
    run_inference()

