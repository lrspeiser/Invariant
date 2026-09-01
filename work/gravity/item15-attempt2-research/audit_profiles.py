"""Fetch and audit only ACCEPT predictor profiles selected by metadata overlap."""

from __future__ import annotations

import contextlib
import io
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import audit_crossmatch


HERE = Path(__file__).resolve().parent
RADII_MPC = (0.02, 0.05, 0.1)
MU_E = 1.17
PROTON_MASS_KG = 1.67262192369e-27
MPC_M = 3.085677581491367e22
SOLAR_MASS_KG = 1.98847e30


def selected_names() -> list[str]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit_crossmatch.main()
    lines = stream.getvalue().splitlines()
    start = lines.index("fresh candidates:") + 1
    return [line.split("\t", 1)[0] for line in lines[start:] if line.strip()]


def registry() -> dict[str, list[str]]:
    rows = json.loads((HERE / "accept-profile-cdx.json").read_text(encoding="utf-8"))
    output: dict[str, list[str]] = {}
    for row in rows[1:]:
        timestamp, original = row[0], row[1]
        filename = urllib.parse.urlparse(original).path.rsplit("/", 1)[-1]
        if filename.lower().endswith("_profiles.dat"):
            output[audit_crossmatch.normalize(filename[: -len("_profiles.dat")])] = [
                timestamp,
                original,
            ]
    return output


def fetch(url: str, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retrying archival transport
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"archival retrieval failed after {attempts} attempts: {error}")


def parse(payload: bytes) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 20:
            raise RuntimeError(f"unexpected profile row width {len(values)}")
        fields = [
            "rin",
            "rout",
            "ne",
            "ne_err",
            "k_itpl",
            "k_flat",
            "k_err",
            "p_itpl",
            "p_flat",
            "p_err",
            "m_grav",
            "m_grav_err",
            "temperature",
            "temperature_err",
            "cooling_function",
            "tcool_5_2",
            "tcool_5_2_err",
            "tcool_3_2",
            "tcool_3_2_err",
        ]
        rows.append(dict(zip(fields, map(float, values[1:]), strict=True)))
    return sorted(rows, key=lambda row: row["rin"])


def shell_value(rows: list[dict[str, float]], radius: float, field: str) -> float:
    enclosing = [row for row in rows if row["rin"] <= radius <= row["rout"]]
    if len(enclosing) != 1:
        raise RuntimeError(f"radius {radius} has {len(enclosing)} enclosing shells")
    value = enclosing[0][field]
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid {field} at {radius}")
    return value


def gas_mass(rows: list[dict[str, float]], radius: float) -> float:
    mass_kg = 0.0
    for row in rows:
        lo = max(0.0, row["rin"])
        hi = min(radius, row["rout"])
        if hi <= lo or row["ne"] <= 0:
            continue
        volume_m3 = 4.0 * math.pi / 3.0 * ((hi * MPC_M) ** 3 - (lo * MPC_M) ** 3)
        electron_density_m3 = row["ne"] * 1.0e6
        mass_kg += MU_E * PROTON_MASS_KG * electron_density_m3 * volume_m3
    return mass_kg / SOLAR_MASS_KG


def main() -> None:
    names = selected_names()
    archived = registry()
    passing: list[str] = []
    feature_rows: list[dict[str, float | str]] = []
    failures: dict[str, str] = {}
    for index, name in enumerate(names, start=1):
        key = audit_crossmatch.normalize(name)
        if key not in archived:
            failures[name] = "not archived"
            continue
        timestamp, original = archived[key]
        url = f"https://web.archive.org/web/{timestamp}id_/{original}"
        try:
            rows = parse(fetch(url))
            for radius in RADII_MPC:
                shell_value(rows, radius, "tcool_3_2")
                shell_value(rows, radius, "temperature")
                shell_value(rows, radius, "ne")
                if gas_mass(rows, radius) <= 0:
                    raise RuntimeError(f"nonpositive gas mass at {radius}")
            passing.append(name)
            feature_rows.append(
                {
                    "name": name,
                    "temperature": shell_value(rows, 0.1, "temperature"),
                    "tcool20": shell_value(rows, 0.02, "tcool_3_2"),
                    "tcool100": shell_value(rows, 0.1, "tcool_3_2"),
                    "mgas100": gas_mass(rows, 0.1),
                }
            )
            print(
                f"{index:02d}/{len(names)} PASS {name} bins={len(rows)} "
                f"range={rows[0]['rin']:.4f}-{rows[-1]['rout']:.4f} "
                f"tc20={shell_value(rows, 0.02, 'tcool_3_2'):.4g} "
                f"tc100={shell_value(rows, 0.1, 'tcool_3_2'):.4g} "
                f"Mgas100={gas_mass(rows, 0.1):.4g}"
            )
        except Exception as exc:  # noqa: BLE001 - audit retains all failures
            failures[name] = str(exc)
            print(f"{index:02d}/{len(names)} FAIL {name}: {exc}")
    print(f"SUMMARY selected={len(names)} passing={len(passing)} failing={len(failures)}")
    print("PASSING=" + json.dumps(passing))
    print("FAILURES=" + json.dumps(failures, sort_keys=True))
    if feature_rows:
        temperatures = sorted(float(row["temperature"]) for row in feature_rows)
        cooling = sorted(math.log10(float(row["tcool20"])) for row in feature_rows)
        middle = len(feature_rows) // 2
        temperature_median = temperatures[middle]
        cooling_median = cooling[middle]
        cells: dict[str, int] = {}
        for row in feature_rows:
            cell = (
                ("short_cooling" if math.log10(float(row["tcool20"])) <= cooling_median else "long_cooling")
                + "|"
                + ("low_temperature" if float(row["temperature"]) <= temperature_median else "high_temperature")
            )
            cells[cell] = cells.get(cell, 0) + 1
        print(f"TEMPERATURE_MEDIAN={temperature_median:.12g}")
        print(f"LOG_TCOOL20_MEDIAN={cooling_median:.12g}")
        print("CELL_COUNTS=" + json.dumps(dict(sorted(cells.items()))))
        print("FEATURES=" + json.dumps(feature_rows, sort_keys=True))


if __name__ == "__main__":
    main()
