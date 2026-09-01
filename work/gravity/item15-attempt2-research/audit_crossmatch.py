"""Response-blind ACCEPT x LC2 metadata overlap audit for gravity Item 15.

This script intentionally requests no LC2 mass or mass-error column.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERBONNET_METADATA_SOURCE = Path(__file__).with_name("herbonnet-tex") / "masses.tex"
ACCEPT_METADATA_URL = (
    "https://web.archive.org/web/20141022053600id_/"
    "http://www.pa.msu.edu/astro/MC2/accept/accept_main.tab"
)
LC2_METADATA_URL = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?" + urllib.parse.urlencode(
    {
        "-source": "J/MNRAS/450/3665/single",
        "-out.max": "unlimited",
        "-out": "Name,_RAJ2000,_DEJ2000,z,NameNED,Author,BibCode",
    }
)


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def normalize(value: str) -> str:
    text = value.upper().replace("ABELL", "A").replace("ZWICKY", "ZW")
    text = text.replace("ZWCL", "ZW").replace("RX J", "RXJ").replace("MACS J", "MACS")
    text = text.replace("SPT-CL J", "SPT").replace("ACT-CL J", "ACT")
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return re.sub(r"^A0+(\d+)$", r"A\1", text)


def hms_to_degrees(value: str) -> float:
    hours, minutes, seconds = (float(part) for part in value.split(":"))
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def dms_to_degrees(value: str) -> float:
    sign = -1.0 if value.startswith("-") else 1.0
    degrees, minutes, seconds = (float(part) for part in value.lstrip("+-").split(":"))
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def separation_arcmin(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 60.0


def parse_accept(payload: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in payload.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 13:
            raise RuntimeError(f"unexpected ACCEPT row: {line}")
        rows.append(
            {
                "name": fields[0],
                "normalized": normalize(fields[0]),
                "ra": hms_to_degrees(fields[1]),
                "dec": dms_to_degrees(fields[2]),
                "z": float(fields[3]),
                "k0": float(fields[4]),
                "k100": float(fields[5]),
                "alpha": float(fields[6]),
                "temperature": float(fields[7]),
                "lbol": float(fields[8]),
            }
        )
    return rows


def parse_lc2(payload: str) -> list[dict[str, object]]:
    data_lines = [line for line in payload.splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(data_lines)), delimiter="\t")
    rows: list[dict[str, object]] = []
    for raw in reader:
        name = (raw.get("Name") or "").strip()
        if not name or name.startswith("-") or (raw.get("_RAJ2000") or "").strip() == "deg":
            continue
        rows.append(
            {
                "name": name,
                "name_ned": (raw.get("NameNED") or "").strip(),
                "normalized": normalize(name),
                "normalized_ned": normalize(raw.get("NameNED") or ""),
                "ra": float(raw["_RAJ2000"]),
                "dec": float(raw["_DEJ2000"]),
                "z": float(raw["z"]),
                "author": (raw.get("Author") or "").strip(),
                "bibcode": (raw.get("BibCode") or "").strip(),
            }
        )
    return rows


def prior_normalized_names() -> set[str]:
    names: set[str] = set()
    g4 = json.loads((ROOT / "runs/gravity/g4/cluster-lensing-exploration-v7.json").read_text())
    names.update(g4["mechanism_transfer"]["fold_assignment"])

    xcop = json.loads((ROOT / "configs/gravity_item3_smooth_density_profiles_v2.json").read_text())
    lane = xcop["cluster_lane"]
    names.update(lane["exploration_objects"])
    names.update(lane["reserved_confirmation_objects"])

    item6 = json.loads((ROOT / "configs/gravity_item6_thermodynamic_accept_hecs_v1.json").read_text())
    names.update(item6["sample"]["exploration"])
    names.update(item6["sample"]["reserved_confirmation"])
    names.update(item6["sample"]["accept_name_map"].values())

    item5 = json.loads((ROOT / "configs/gravity_item5_pressure_cross_support_v2.json").read_text())
    names.update(item5["sample"]["exploration"])
    names.update(item5["sample"]["reserved_confirmation"])
    names.update(row["name"] for row in herbonnet_prior_metadata())
    return {normalize(name) for name in names}


def herbonnet_prior_metadata() -> list[dict[str, object]]:
    """Read only the public sample identities/coordinates, never its lensing responses.

    The paper source was downloaded during source research before the attempt-2 sample
    freeze.  Conservatively excluding all 100 physical systems prevents even an
    accidentally viewed response row from entering this experiment.
    """

    text = HERBONNET_METADATA_SOURCE.read_text(encoding="utf-8")
    anchor = text.index("Here we document all the cluster properties")
    end = text.index(r"\caption{Basic information", anchor)
    rows: dict[int, dict[str, object]] = {}
    for line in text[anchor:end].splitlines():
        fields = [field.strip() for field in line.split("&")]
        if len(fields) < 5 or not re.fullmatch(r"\d{1,3}", fields[0]):
            continue
        ordinal = int(fields[0])
        if not 1 <= ordinal <= 100:
            continue
        ra_text = fields[3]
        dec_text = re.sub(r"\\hspace\{[^}]+\}", "", fields[4])
        if dec_text.startswith("--"):
            dec_text = "-" + dec_text[2:]
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", ra_text):
            continue
        if not re.fullmatch(r"[+-]\d{2}:\d{2}:\d{2}(?:\.\d+)?", dec_text):
            continue
        rows[ordinal] = {
            "name": fields[1].replace("~", "").replace("{", "").replace("}", ""),
            "ra": hms_to_degrees(ra_text),
            "dec": dms_to_degrees(dec_text),
        }
    if sorted(rows) != list(range(1, 101)):
        missing = sorted(set(range(1, 101)) - set(rows))
        raise RuntimeError(
            "Herbonnet metadata-only exclusion must contain exactly ordinals 1..100; "
            f"missing={missing}"
        )
    return [rows[index] for index in range(1, 101)]


def overlaps_prior_alias(aliases: set[str], prior_names: set[str]) -> bool:
    for alias in aliases:
        if not alias:
            continue
        for prior in prior_names:
            if alias == prior:
                return True
            shorter, longer = sorted((alias, prior), key=len)
            if len(shorter) >= 5 and longer.startswith(shorter):
                return True
    return False


def spt_prior_coordinates() -> list[tuple[float, float]]:
    config = json.loads((ROOT / "configs/gravity_item5_pressure_cross_support_v2.json").read_text())
    coordinates: list[tuple[float, float]] = []
    for token in config["sample"]["exploration"] + config["sample"]["reserved_confirmation"]:
        match = re.fullmatch(r"(\d{2})(\d{2})([+-])(\d{2})(\d{2})", token)
        if not match:
            continue
        hh, mm, sign, dd, dm = match.groups()
        ra = 15.0 * (int(hh) + int(mm) / 60.0)
        dec = (int(dd) + int(dm) / 60.0) * (-1.0 if sign == "-" else 1.0)
        coordinates.append((ra, dec))
    return coordinates


def main() -> None:
    accept = parse_accept(fetch(ACCEPT_METADATA_URL))
    lc2 = parse_lc2(fetch(LC2_METADATA_URL))
    prior_names = prior_normalized_names()
    spt_coordinates = spt_prior_coordinates()
    herbonnet_coordinates = [
        (float(row["ra"]), float(row["dec"])) for row in herbonnet_prior_metadata()
    ]
    cdx_path = Path(__file__).with_name("accept-profile-cdx.json")
    profile_names: set[str] = set()
    if cdx_path.exists():
        cdx_rows = json.loads(cdx_path.read_text(encoding="utf-8"))
        for row in cdx_rows[1:]:
            filename = urllib.parse.urlparse(row[1]).path.rsplit("/", 1)[-1]
            if filename.lower().endswith("_profiles.dat"):
                profile_names.add(normalize(filename[: -len("_profiles.dat")]))
    matches: list[dict[str, object]] = []
    ambiguous: list[tuple[str, int]] = []
    for source in accept:
        candidates: list[tuple[float, dict[str, object]]] = []
        for target in lc2:
            separation = separation_arcmin(
                float(source["ra"]), float(source["dec"]), float(target["ra"]), float(target["dec"])
            )
            name_match = source["normalized"] in {target["normalized"], target["normalized_ned"]}
            if (name_match or separation <= 5.0) and abs(float(source["z"]) - float(target["z"])) <= 0.02:
                candidates.append((separation, target))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], str(item[1]["bibcode"]), str(item[1]["name"])))
        if len(candidates) > 1:
            ambiguous.append((str(source["name"]), len(candidates)))
        for separation, target in candidates[:1]:
            excluded: list[str] = []
            aliases = {
                str(source["normalized"]),
                str(target["normalized"]),
                str(target["normalized_ned"]),
            }
            if overlaps_prior_alias(aliases, prior_names):
                excluded.append("prior_name")
            nearest_herbonnet = min(
                separation_arcmin(
                    float(source["ra"]), float(source["dec"]), prior_ra, prior_dec
                )
                for prior_ra, prior_dec in herbonnet_coordinates
            )
            if nearest_herbonnet <= 20.0:
                excluded.append("prior_herbonnet_coordinate")
            if spt_coordinates:
                nearest_spt = min(
                    separation_arcmin(
                        float(source["ra"]), float(source["dec"]), prior_ra, prior_dec
                    )
                    for prior_ra, prior_dec in spt_coordinates
                )
                if nearest_spt <= 20.0:
                    excluded.append("prior_spt_coordinate")
            else:
                nearest_spt = math.inf
            if profile_names and str(source["normalized"]) not in profile_names:
                excluded.append("profile_not_in_wayback_cdx")
            matches.append(
                {
                    **source,
                    "lc2_name": target["name"],
                    "lc2_ned": target["name_ned"],
                    "author": target["author"],
                    "bibcode": target["bibcode"],
                    "separation_arcmin": separation,
                    "nearest_spt_arcmin": nearest_spt,
                    "nearest_herbonnet_arcmin": nearest_herbonnet,
                    "excluded": excluded,
                }
            )

    fresh = [row for row in matches if not row["excluded"]]
    fresh_names = {str(row["normalized"]) for row in fresh}
    print(f"ACCEPT rows={len(accept)} LC2 metadata rows={len(lc2)}")
    print(f"matched rows={len(matches)} matched ACCEPT identities={len({r['normalized'] for r in matches})}")
    print(f"fresh rows={len(fresh)} fresh ACCEPT identities={len(fresh_names)}")
    print(f"ambiguous ACCEPT identities={len(ambiguous)}")
    print(
        "HERBONNET_COORDINATE_EXCLUSIONS="
        + json.dumps(
            [
                {
                    "accept_name": row["name"],
                    "separation_arcmin": round(float(row["nearest_herbonnet_arcmin"]), 6),
                }
                for row in matches
                if float(row["nearest_herbonnet_arcmin"]) <= 20.0
            ],
            sort_keys=True,
        )
    )
    by_author: dict[str, int] = {}
    for row in fresh:
        by_author[str(row["author"])] = by_author.get(str(row["author"]), 0) + 1
    print("fresh source rows=" + json.dumps(dict(sorted(by_author.items(), key=lambda item: (-item[1], item[0])))))
    print("fresh candidates:")
    for row in fresh:
        print(
            "\t".join(
                [
                    str(row["name"]),
                    str(row["lc2_name"]),
                    str(row["author"]),
                    str(row["bibcode"]),
                    f"sep={float(row['separation_arcmin']):.3f}",
                    f"z={float(row['z']):.4f}",
                    f"T={float(row['temperature']):.2f}",
                    f"K0={float(row['k0']):.2f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
