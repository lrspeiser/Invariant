"""Repository wrapper for the standalone source-release builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from sigma_theory_compiler.standalone_release import main

    return main(("build", "--project-root", str(ROOT), *sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(_main())
