"""Convenience launcher for the FastAPI service."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "api.py")] + sys.argv[1:]
    print(f"Launching API: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nAPI stopped by user.")
