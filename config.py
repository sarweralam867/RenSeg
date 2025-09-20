import os
from pathlib import Path

# Root directory for dynamic path handling
BASE_DIR = Path(__file__).resolve().parent

# Example subdirectories (update if needed)
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure folders exist
for folder in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)
