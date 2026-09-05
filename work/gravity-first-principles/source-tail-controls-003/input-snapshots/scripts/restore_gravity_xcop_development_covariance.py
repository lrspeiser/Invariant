"""Restore only eight already-exposed pressure covariance files from a fixed archive."""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from hashlib import file_digest, sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invariant_gravity_extensions.cluster_pressure import DEVELOPMENT_CLUSTERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT/"work/gravity-first-principles/xcop-covariance-acquisition-001.json").read_bytes())
    with args.archive.open("rb") as handle:
        if file_digest(handle, "sha256").hexdigest() != manifest["archive_sha256"]:
            raise ValueError("archive SHA-256 mismatch")
    records = manifest["records"]
    if len(records) != len(DEVELOPMENT_CLUSTERS) or {r["cluster"] for r in records} != DEVELOPMENT_CLUSTERS:
        raise ValueError("manifest population must be exactly the eight development clusters")
    targets = []
    for record in records:
        cluster = record["cluster"]
        stem = "ZwCl1215.1+040" if cluster == "ZW1215" else cluster
        member = f"{cluster}/{stem}_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits"
        relative = f"work/private/xcop-pressure-covariance/{cluster}.fits"
        if record["member"] != member or record["path"] != relative:
            raise ValueError("unapproved member or destination in manifest")
        target = (ROOT/relative).resolve()
        if not target.is_relative_to((ROOT/"work/private/xcop-pressure-covariance").resolve()):
            raise ValueError("destination escaped approved cache")
        targets.append((record, target))
    restored, verified = [], []
    with tarfile.open(args.archive, "r:gz") as bundle:
        for record, target in targets:
            if target.exists():
                if sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"existing covariance hash mismatch: {record['cluster']}")
                verified.append(record["cluster"])
                continue
            member = bundle.getmember(record["member"])
            if not member.isfile() or member.size != record["bytes"]:
                raise ValueError("unexpected archive entry type or size")
            with bundle.extractfile(member) as handle:
                payload = handle.read()
            if sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("covariance payload SHA-256 mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
            restored.append(record["cluster"])
    print(json.dumps({"restored": restored, "already_present_hash_verified": verified,
                      "reserved_payloads_extracted": 0, "numeric_tables_parsed": False}))


if __name__ == "__main__":
    main()
