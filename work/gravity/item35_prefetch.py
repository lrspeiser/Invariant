from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CACHE = (ROOT / "work/gravity/item-35-manga-maps-cache").resolve()
MANIFEST = ROOT / "runs/gravity/roadmap/item-35-modified-inertia-v1-source/sample-manifest.json"
BASE = (
    "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/"
    "HYB10-MILESHC-MASTARSSP"
)


def fetch(identity: str) -> str:
    plate, ifu = identity.split("-")
    if not (plate.isdigit() and ifu.isdigit()):
        raise ValueError(f"invalid plateifu: {identity}")
    filename = f"manga-{identity}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
    destination = (CACHE / filename).resolve()
    if destination.parent != CACHE:
        raise ValueError("cache target escaped exact directory")
    if destination.is_file() and destination.stat().st_size > 0:
        return identity
    url = f"{BASE}/{plate}/{ifu}/{filename}"
    temporary = destination.with_suffix(destination.suffix + ".prefetch-part")
    last_error: Exception | None = None
    for _ in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item35/1.0"})
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = response.read()
            if not payload:
                raise OSError("empty MAPS payload")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            return identity
        except (OSError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError(f"failed to prefetch {identity}") from last_error


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    identities = sorted(
        str(row["plateifu"]) for row in manifest["objects"] if row["role"] == "exploration"
    )
    if len(identities) != 160 or len(set(identities)) != 160:
        raise ValueError("unexpected frozen exploration identities")
    with ThreadPoolExecutor(max_workers=12) as executor:
        completed = list(executor.map(fetch, identities))
    if len(completed) != 160:
        raise RuntimeError("incomplete Item 35 prefetch")
    print(f"cached={len(list(CACHE.glob('*.fits.gz')))}")


if __name__ == "__main__":
    main()
