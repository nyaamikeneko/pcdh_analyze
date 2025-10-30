# notebooks/_init_notebook.py
"""
Helper to initialize notebook environment for this project.
Usage: import notebooks._init_notebook  # will ensure project root is on sys.path
"""
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Optionally, expose common environment variables here or perform other setup.
# Importing this module ensures that `src` is importable from notebooks.

# Example convenience import (not required):
# from src.config import PROCESSED_DIR, DATA_DIR

print(f"Notebook init: PROJECT_ROOT={PROJECT_ROOT}")
