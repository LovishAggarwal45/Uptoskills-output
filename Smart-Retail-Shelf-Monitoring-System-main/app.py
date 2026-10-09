"""
Shelvex Enterprise Retail Vision - Root Entrypoint
Delegates execution to the main application module while ensuring correct paths.
"""
from pathlib import Path
import sys
import runpy

ROOT_DIR = Path(__file__).resolve().parent
INNER_DIR = ROOT_DIR / "Smart-Retail-Shelf-Monitoring-System-main"
TARGET_APP = INNER_DIR / "app.py"

if not TARGET_APP.exists():
    raise FileNotFoundError(f"Cannot find main application at {TARGET_APP}")

# Execute the application script
runpy.run_path(str(TARGET_APP), run_name="__main__")
