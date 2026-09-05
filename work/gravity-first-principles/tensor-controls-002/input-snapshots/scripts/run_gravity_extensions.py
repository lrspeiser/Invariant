"""Run the opt-in synthetic extension workflow from a checkout, on Windows or Linux."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invariant_gravity_extensions.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
