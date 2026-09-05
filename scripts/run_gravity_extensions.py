"""Run the opt-in synthetic extension workflow from a checkout, on Windows or Linux."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invariant_gravity_extensions.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
