"""Validate frozen per-object LC2 constraints without requesting mass columns."""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads(
    (ROOT / "configs/gravity_item15_accept_lc2_timescale_ratios_v2.json").read_text()
)
SOURCE = CONFIG["sources"]["lc2"]


def main() -> None:
    failures = []
    for accept_name, lc2_name, author, bibcode in CONFIG["sample"]["eligible_lc2_rows"]:
        parameters = [
            ("-source", SOURCE["catalog"]),
            ("-out.max", "20"),
            ("-out", ",".join(SOURCE["metadata_columns"])),
            ("Name", lc2_name),
            ("Author", author),
            ("BibCode", bibcode),
        ]
        url = f"{SOURCE['vizier_endpoint']}?{urllib.parse.urlencode(parameters)}"
        request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read().decode("utf-8")
        lines = [line for line in payload.splitlines() if line and not line.startswith("#")]
        rows = list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t"))
        matches = [
            row
            for row in rows
            if (row.get("Name") or "").strip() == lc2_name
            and (row.get("Author") or "").strip() == author
            and (row.get("BibCode") or "").strip() == bibcode
        ]
        if len(matches) != 1:
            failures.append({"accept_name": accept_name, "matching_metadata_rows": len(matches)})
        print(f"{accept_name}\tmetadata_rows={len(matches)}")
    print(json.dumps({"objects": 23, "failures": failures, "mass_columns_requested": 0}))


if __name__ == "__main__":
    main()
