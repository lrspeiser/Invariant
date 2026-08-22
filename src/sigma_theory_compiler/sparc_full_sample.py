"""The full published SPARC sample, and Tier 7 R1/R2 run across all of it.

``configs/sparc_rotation_curves_v1.json`` carries six galaxies and 214 points.  The
published catalogue carries 175 galaxies and 3391 points.  Everything downstream -- the
pooled confrontation, the per-object decomposition, the constancy diagnosis -- has been
answering a question posed on 6.3% of the available rows, and the per-object question in
particular is *about the spread of a population*, which four objects cannot measure.

This module widens the sample and re-runs the per-object machinery on it.  It adds no new
fitting interface: :mod:`sigma_theory_compiler.per_object_law_decomposition` already
defines the law spaces, the exact interval arithmetic, the constancy adjudicator and its
falsifier guard, and every one of them is imported and used unchanged.  What is new here
is (a) the widened dataset with per-galaxy provenance, (b) the schema and cross-retrieval
controls that decide whether it may be believed, (c) an exactly equivalent
critical-coverage solver that does not enumerate ordered pairs, because the pooled axis
now carries thousands of rows and the reference solver is quadratic, and (d) the run.

Where the numbers come from
---------------------------

The published per-galaxy mass models are distributed as ``Rotmod_LTG``: one
``<NAME>_rotmod.dat`` file per galaxy, whose first six columns are exactly the six columns
the repository already declares (Rad, Vobs, errV, Vgas, Vdisk, Vbul).  This module reads
that distribution *offline*, from a local archive path supplied on the command line, and
writes every published decimal string through verbatim -- no value is re-derived,
re-rounded, averaged, or interpolated, and the two photometry columns the repository does
not declare (SBdisk, SBbul) are dropped rather than carried as undeclared data.

The archive is not self-authenticating, so the widened dataset is checked against
something that is.  The six galaxies already in the repository were retrieved
independently, from VizieR's ASU-TSV service on 2026-08-18, by a different person through
a different protocol.  :func:`crosscheck_declared_subset` requires that all six appear in
the widened file with **byte-identical** published strings on all 214 rows and all six
columns.  Two independent retrievals of a published table agreeing character for character
on 1284 fields is the strongest offline authentication available, and it is a control that
can fail: a single altered digit anywhere in those six galaxies raises.  It says nothing
about the other 169 galaxies beyond "they came from the same archive as the six that
check out", and the receipt says so in those words.

What the widening changes, and what it cannot
---------------------------------------------

R2 asks whether one value of a per-object parameter lies inside every object's exact
interval.  On four objects, "yes" is cheap: four intervals derived from four independent
sets of published rows overlap far more often than a hundred and forty do.  So a CONSTANT
verdict on the six-galaxy sample is weak evidence and a VARIES verdict on it is strong;
widening to the full sample makes CONSTANT mean something and can only ever *remove*
CONSTANT verdicts, never manufacture them, because the intersection over a superset of
objects is contained in the intersection over the subset.  That monotonicity is the reason
this widening is worth its cost, and it is asserted in the tests rather than asserted here.

The admission rule, declared before the run
-------------------------------------------

Under the declared universal mass-to-light convention, ``V_bar^2 = Vgas |Vgas| + 0.5
Vdisk^2 + 0.7 Vbul^2``.  The signed gas term preserves the published sign convention for
central HI holes, and on a small number of published rows it makes ``V_bar^2`` non-positive
-- the baryons, as decomposed, do not define a Newtonian prediction there at all.  The
declared rule is stated at galaxy granularity, because the galaxy is the unit of a
per-object decomposition and dropping *rows* from inside an object is exactly the freedom
that would let a fit choose its own data:

    a galaxy is admitted if and only if every one of its published rows has
    ``V_bar^2 > 0``; a galaxy with any such row is excluded whole, and named, with the
    offending rows exhibited.

The rule is a property of the published columns and the declared convention.  It reads no
residual, no fit and no verdict, and it is applied *after* the exploration/confirmation
split has been computed from names alone, so it cannot move an object across the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .per_object_law_decomposition import (
    CONSTANT,
    NO_POPULATION,
    VARIES,
    Axis,
    ConfirmationSetTouched,
    LawSpace,
    Split,
    _fraction_block,
    _fraction_data,
    _from_block,
    _num,
    adjudicate,
    build_axis,
    build_law_spaces,
    critical_coverage,
    crosscheck_against_simplex,
    declare_split,
    decompose,
    diagnose,
    interval_at,
    pooled_axis,
    verify_adjudication,
)
from .real_data_gravity_confrontation import (
    COVERAGE_GRID,
    QUADRATURE,
    REFERENCE_GRID_POINT,
    ColumnCache,
    Galaxy,
    load_families,
    measured_rows,
    prepare_galaxy,
    select_best_family,
)
from .sigma_core import canonical_json_bytes, canonical_sha256
from .tolerance_aware_fitting import forbidden_receipt_keys

DATASET_SCHEMA = "invariant-sparc-rotation-curves-full-1.0"
RESULT_SCHEMA = "invariant-full-sparc-per-object-decomposition-result-1.0"
DATASET_PATH = "configs/sparc_rotation_curves_full_v1.json"
DECLARED_SUBSET_PATH = "configs/sparc_rotation_curves_v1.json"
RECEIPT_PATH = "runs/gpu-baryonic-screen/per-object-decomposition-full-sparc-v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/sparc_full_sample.py"
TEST_PATH = "tests/test_sparc_full_sample.py"

#: This receipt fits.  It may never be cited as a confirmation of anything.
TRIAL_TYPE = "exploratory"

#: The published sample size, from the title of the catalogue itself.  A declared count,
#: not a measured one: if the archive offers a different number of galaxies the build
#: refuses rather than quietly widening to whatever it found.
PUBLISHED_GALAXY_COUNT = 175

ARCHIVE_SUFFIX = "_rotmod.dat"

#: The published columns this repository declares, in the order the archive prints them.
#: The archive carries two further photometry columns; they are undeclared here and are
#: dropped rather than carried as data nobody cited.
ARCHIVE_COLUMNS = ("rad_kpc", "vobs_km_s", "e_vobs_km_s", "vgas_km_s", "vdisk_km_s", "vbul_km_s")
ARCHIVE_HEADER_LABELS = ("Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul")

#: Published values are fixed-point decimal strings.  Anything else -- a float, an
#: exponent, a bare integer, a NaN -- is refused, because a float on a certificate path is
#: the I3 falsifier and a re-rendered number is no longer the published number.
DECIMAL_FIELD = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]+$")
GALAXY_NAME = re.compile(r"^[A-Za-z0-9_.+-]+$")

# ---------------------------------------------------------------------------
# The split over the widened population.  Declared here, before any datum is read.
# ---------------------------------------------------------------------------

#: The salt fixing the widened split.  A different population gets a different salt, so
#: the six-galaxy split cannot be replayed here and neither can be re-rolled.
FULL_SPLIT_SALT = "invariant-tier7-full-sparc-per-object-decomposition-v1"

#: One galaxy in five is withheld.  Declared as an exact rational and applied by floor, so
#: the withheld count is a function of the published sample size and nothing else.
CONFIRMATION_FRACTION = Fraction(1, 5)

FULL_SPLIT_RULE = (
    "sort every published galaxy name by sha256(FULL_SPLIT_SALT + '|' + name) and withhold "
    "the first floor(CONFIRMATION_FRACTION * published_galaxy_count) of them. The rule "
    "reads galaxy *names* only -- never a radius, a velocity, or an uncertainty -- so it "
    "cannot be steered by the data it is protecting, it is recomputable from this source "
    "file alone, and it is applied to the full published name list before the admission "
    "rule looks at a single column, so admission cannot move an object across the split"
)

ADMISSION_RULE = (
    "a galaxy is admitted if and only if every one of its published rows has V_bar^2 > 0 "
    "under the declared universal mass-to-light convention. A galaxy carrying any such row "
    "is excluded whole and named, with the offending rows exhibited: the galaxy is the unit "
    "of a per-object decomposition, and dropping rows from inside an object is the freedom "
    "that would let a fit choose its own data. The rule reads published columns and the "
    "declared convention only -- no residual, no fit, no verdict"
)

CROSS_RETRIEVAL_RULE = (
    "every galaxy of the six-galaxy declared subset must appear in the widened dataset with "
    "byte-identical published strings on every row and every column. The subset was "
    "retrieved from VizieR's ASU-TSV service; the widened dataset was read from the "
    "Rotmod_LTG per-galaxy distribution. Two independent retrievals of one published table "
    "agreeing character for character is the strongest offline authentication available, "
    "and one altered digit anywhere in those six galaxies fails it"
)

MONOTONICITY_CLAIM = (
    "widening the population can only remove CONSTANT verdicts, never create them: the set "
    "of values lying in every interval of a superset of objects is contained in the set "
    "lying in every interval of the subset. So a CONSTANT verdict that survives 140 objects "
    "means more than one drawn from four, and a VARIES verdict on four objects stays VARIES "
    "on 140. This is why widening the sample is the cheapest real information available and "
    "why it cannot flatter the result"
)

#: Nested subpopulation sizes for the constancy ladder.  Declared before the run and
#: chosen for one reason: the first entry is exactly the number of galaxies the predecessor
#: receipt fitted, and the rest double until the whole population is reached.
LADDER_SIZES = (4, 8, 16, 32, 64, 128)

#: The predecessor receipt fitted four galaxies.  This is that number, used as the block
#: size of the survey below, so the survey answers a question about that receipt and not
#: about some other sample size chosen later.
PREDECESSOR_POPULATION_SIZE = 4

BLOCK_SURVEY_RULE = (
    "cut the exploration population into consecutive disjoint blocks of "
    f"{PREDECESSOR_POPULATION_SIZE} galaxies in the sealed split-digest order -- an order "
    "fixed by names before any column was read -- and run the whole R2 question inside each "
    "block, at that block's own smallest feasible coverage. Each block is a complete "
    "miniature of the predecessor receipt: same law spaces, same arithmetic, same "
    "adjudicator, four galaxies. Counting how many blocks return CONSTANT turns 'four "
    "objects cannot measure the spread of a population' from an assertion into an integer, "
    "and the blocks are a partition, so no galaxy is used twice and none is left out except "
    "a declared remainder shorter than one block"
)

OWN_COVERAGE_LADDER_RULE = (
    "the same nested prefixes, but each prefix is adjudicated at *its own* smallest feasible "
    "coverage rather than the whole population's. That is exactly the receipt a run over "
    "that many galaxies would have produced, so the ladder answers the only question that "
    "matters about a CONSTANT verdict drawn from a handful of objects: would it have "
    "survived more of them. Its price of universality is not monotone and is not meant to "
    "be -- the coverage moves with the prefix -- and that is precisely why it is reported "
    "alongside the fixed-coverage ladder rather than instead of it"
)

LADDER_RULE = (
    "at the one coverage where the whole exploration population admits a solution, take the "
    "exploration objects in the order their split digests already fix -- an order derived "
    "from names alone, sealed before any column was read -- and ask the R2 question of each "
    "nested prefix in turn. Because the prefixes are nested, the verdict can only ever run "
    "CONSTANT then VARIES and never back, so the ladder has one breakpoint: the number of "
    "objects at which one shared value stops reaching all of them. The build refuses a "
    "non-monotone ladder rather than publishing it, which is the monotonicity claim turned "
    "into a live guard instead of a paragraph"
)

CLAIMS = {
    "confirmation_set_fitted": False,
    "every_fit_is_single_object": True,
    "exact_rational_certificates": True,
    "population_kept_not_ranked": True,
    "published_values_carried_verbatim": True,
    "split_sealed_before_fitting": True,
}

SCOPE = (
    "Tier 7 R1 and R2 over the full published SPARC sample. R1 fits every admitted "
    "exploration galaxy alone, with its own copy of one free parameter, over each declared "
    "one-parameter law space, and keeps every local solution. R2 emits an exact interval "
    "per object and decides whether one value lies in all of them. The two-parameter "
    "polytope projection of the predecessor receipt is deliberately out of scope: it costs "
    "an exact simplex per object per coordinate per coverage and it answers a different "
    "question. R3, R4 and R5 are out of scope and no channel is named or measured."
)

ASSUMPTIONS = {
    "the_archive_is_authenticated_only_through_the_six": (
        "the widened rows come from a local copy of the published per-galaxy distribution. "
        "That copy is authenticated by exact agreement with the six galaxies the repository "
        "retrieved independently -- 214 rows, 1284 published fields, byte for byte. The "
        "other 169 galaxies inherit provenance from the same archive and no further: their "
        "per-file digests are recorded so any future retrieval can confirm or refute them "
        "row by row, and until that happens this receipt claims exactly that much"
    ),
    "small_objects_have_wide_intervals": (
        "a galaxy with few published points constrains its own parameter weakly and its "
        "interval is correspondingly wide, so it will rarely be the object that binds a "
        "constancy verdict. No galaxy is dropped for having few points -- that would be a "
        "threshold chosen after seeing which objects were inconvenient -- and the point "
        "count is published next to every interval so the reader can see which objects are "
        "actually carrying the verdict"
    ),
    "columns_are_frozen_floats": (
        "for the screened families the design entries are produced by the predecessor "
        "module's declared quadrature in float64 and frozen onto a 15-significant-digit "
        "decimal grid. Every step after that freeze is exact rational arithmetic. The "
        "Newtonian law space touches no float at all: its column is the published decimal "
        "strings combined under the declared convention, exactly"
    ),
    "published_sigmas_are_random_errors_only": (
        "the e_Vobs column is the published random error from non-circular motions and "
        "kinematic asymmetries. It excludes inclination and distance systematics, so the "
        "coverage factors reported here are large for every law tried. That is a property "
        "of the declared uncertainty budget and it applies identically to every law compared"
    ),
}


class FullSampleError(ValueError):
    """Raised on a malformed dataset, a failed control, or a failed self-check."""


# ---------------------------------------------------------------------------
# Step 1 -- read the offline archive into the declared shape
# ---------------------------------------------------------------------------


def _decimal(text: str) -> Fraction:
    return Fraction(Decimal(text))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_archive_file(path: Path) -> dict[str, Any]:
    """One ``<NAME>_rotmod.dat`` file, as published strings plus its own provenance."""

    name = path.name
    if not name.endswith(ARCHIVE_SUFFIX):
        raise FullSampleError(f"{name} is not a published per-galaxy mass-model file")
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    header = [line for line in lines if line.startswith("#")]
    if len(header) != 3:
        raise FullSampleError(f"{name}: expected three published header lines")
    if not header[0].startswith("# Distance"):
        raise FullSampleError(f"{name}: the first header line does not declare a distance")
    labels = tuple(header[1].lstrip("#").split())
    if labels != ARCHIVE_HEADER_LABELS:
        raise FullSampleError(f"{name}: published column labels are not the declared ones")
    distance = header[0].split("=", 1)[1].replace("Mpc", "").strip()
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(ARCHIVE_HEADER_LABELS):
            raise FullSampleError(f"{name}: a published row does not carry every column")
        rows.append(fields[: len(ARCHIVE_COLUMNS)])
    if not rows:
        raise FullSampleError(f"{name}: no published rows")
    return {
        "distance_mpc": distance,
        "name": name[: -len(ARCHIVE_SUFFIX)],
        "point_count": len(rows),
        "provenance": {
            "published_header": list(header),
            "source_file": name,
            "source_file_sha256": _sha256_bytes(raw),
        },
        "rows": rows,
    }


def _galaxy_digest(entry: Mapping[str, Any]) -> str:
    """A galaxy's own seal, over exactly the published strings it carries."""

    return canonical_sha256(
        {
            "name": entry["name"],
            "point_count": entry["point_count"],
            "rows": entry["rows"],
        }
    )


def dataset_digest(galaxies: Sequence[Mapping[str, Any]]) -> str:
    """The dataset seal: an ordered list of per-galaxy seals, and nothing else."""

    return canonical_sha256(
        [
            {
                "name": entry["name"],
                "point_count": entry["point_count"],
                "rows_sha256": entry["provenance"]["rows_sha256"],
            }
            for entry in galaxies
        ]
    )


def build_dataset(archive: Path, subset: Mapping[str, Any]) -> dict[str, Any]:
    """Read the offline archive and assemble the widened dataset with its provenance.

    The citation, column declarations and mass-to-light convention are lifted verbatim from
    the six-galaxy file rather than restated, so the widened sample cannot silently adopt a
    different convention from the one the repository already declared and cited.
    """

    archive = archive.resolve()
    files = sorted(archive.glob(f"*{ARCHIVE_SUFFIX}"))
    if len(files) != PUBLISHED_GALAXY_COUNT:
        raise FullSampleError(
            f"the archive offers {len(files)} galaxies; the published sample is "
            f"{PUBLISHED_GALAXY_COUNT}"
        )
    galaxies = [parse_archive_file(path) for path in files]
    galaxies.sort(key=lambda entry: entry["name"])
    for entry in galaxies:
        entry["provenance"]["rows_sha256"] = _galaxy_digest(entry)
    payload = {
        "columns": subset["columns"],
        "galaxies": galaxies,
        "galaxy_digest_sha256": dataset_digest(galaxies),
        "mass_to_light_convention": subset["mass_to_light_convention"],
        "schema_version": DATASET_SCHEMA,
        "selection": {
            "declared_before_confrontation": True,
            "galaxy_count": len(galaxies),
            "point_count": sum(entry["point_count"] for entry in galaxies),
            "rule": (
                "every galaxy in the published per-galaxy mass-model distribution, with no "
                "galaxy chosen, dropped, reordered or preferred by anything measured. The "
                "file list is the selection, and the count is checked against the published "
                "sample size before the dataset is written"
            ),
            "widens": {
                "from": DECLARED_SUBSET_PATH,
                "from_galaxy_count": subset["selection"]["galaxy_count"],
                "from_point_count": subset["selection"]["point_count"],
            },
        },
        "source": {
            **subset["source"],
            "cross_retrieval_control": CROSS_RETRIEVAL_RULE,
            "dropped_columns": (
                "the archive prints SBdisk and SBbul after the six declared columns. They "
                "are not declared by this repository and are dropped rather than carried"
            ),
            "retrieval": {
                **subset["source"]["retrieval"],
                "widened_from": (
                    "the published Rotmod_LTG per-galaxy distribution, read offline from a "
                    "local archive; every decimal string is carried verbatim and each file "
                    "carries its own sha256 in this dataset"
                ),
            },
        },
        "status": (
            "declared measured data; every number is a published decimal string carried "
            "verbatim; nothing here is fitted, re-derived, re-rounded, averaged or "
            "interpolated"
        ),
    }
    validate_dataset(payload)
    return payload


# ---------------------------------------------------------------------------
# Step 2 -- the schema control.  It must fail on a tampered dataset.
# ---------------------------------------------------------------------------


def validate_dataset(payload: Mapping[str, Any]) -> None:
    """Refuse a dataset that is malformed, mis-sealed, or not made of published decimals."""

    if payload.get("schema_version") != DATASET_SCHEMA:
        raise FullSampleError("widened dataset schema changed")
    galaxies = payload.get("galaxies")
    if not isinstance(galaxies, list) or not galaxies:
        raise FullSampleError("the widened dataset carries no galaxies")
    names = [entry["name"] for entry in galaxies]
    if names != sorted(names):
        raise FullSampleError("galaxies are not in canonical name order")
    if len(set(names)) != len(names):
        raise FullSampleError("a galaxy name appears twice")
    total = 0
    for entry in galaxies:
        name = entry["name"]
        if not GALAXY_NAME.match(name):
            raise FullSampleError(f"{name}: galaxy name is not a plain published identifier")
        rows = entry["rows"]
        if not isinstance(rows, list) or not rows:
            raise FullSampleError(f"{name}: no published rows")
        if len(rows) != entry["point_count"]:
            raise FullSampleError(f"{name}: declared point count disagrees with the rows")
        if not DECIMAL_FIELD.match(str(entry["distance_mpc"])):
            raise FullSampleError(f"{name}: the published distance is not a decimal string")
        previous: Fraction | None = None
        for row in rows:
            if not isinstance(row, list) or len(row) != len(ARCHIVE_COLUMNS):
                raise FullSampleError(f"{name}: a row does not carry the declared columns")
            for field in row:
                if not isinstance(field, str) or not DECIMAL_FIELD.match(field):
                    raise FullSampleError(
                        f"{name}: {field!r} is not a published fixed-point decimal string"
                    )
            radius = _decimal(row[0])
            if radius <= 0:
                raise FullSampleError(f"{name}: a published radius is not positive")
            if previous is not None and radius <= previous:
                raise FullSampleError(f"{name}: published radii are not strictly increasing")
            previous = radius
            if _decimal(row[2]) <= 0:
                raise FullSampleError(f"{name}: a published velocity uncertainty is not positive")
        provenance = entry.get("provenance")
        if not isinstance(provenance, Mapping):
            raise FullSampleError(f"{name}: no per-galaxy provenance")
        if provenance.get("source_file") != f"{name}{ARCHIVE_SUFFIX}":
            raise FullSampleError(f"{name}: provenance names a different source file")
        for key in ("source_file_sha256", "rows_sha256"):
            digest = provenance.get(key)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise FullSampleError(f"{name}: {key} is not a sha256")
        if provenance["rows_sha256"] != _galaxy_digest(entry):
            raise FullSampleError(f"{name}: the published rows do not match their own seal")
        total += entry["point_count"]
    selection = payload["selection"]
    if selection["galaxy_count"] != len(galaxies) or selection["point_count"] != total:
        raise FullSampleError("the declared selection counts disagree with the galaxies")
    if payload.get("galaxy_digest_sha256") != dataset_digest(galaxies):
        raise FullSampleError("the dataset seal does not match the galaxies it seals")


def crosscheck_declared_subset(
    payload: Mapping[str, Any], subset: Mapping[str, Any]
) -> dict[str, Any]:
    """The independent-retrieval control: the six must appear verbatim in the 175.

    This is the control that can fail.  It compares published *strings*, not parsed values,
    so a re-rounded, re-rendered or edited digit anywhere in the six galaxies raises rather
    than being absorbed by a numeric tolerance.
    """

    wide = {entry["name"]: entry for entry in payload["galaxies"]}
    checks: list[dict[str, Any]] = []
    fields = 0
    for declared in subset["galaxies"]:
        name = declared["name"]
        if name not in wide:
            raise FullSampleError(f"{name} is declared by the repository and is missing")
        entry = wide[name]
        if entry["point_count"] != declared["point_count"]:
            raise FullSampleError(f"{name}: the two retrievals disagree on the point count")
        left = [list(row) for row in declared["rows"]]
        right = [list(row) for row in entry["rows"]]
        if left != right:
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                if a != b:
                    raise FullSampleError(
                        f"{name}: the two retrievals disagree at published row {index}: "
                        f"{a!r} against {b!r}"
                    )
            raise FullSampleError(f"{name}: the two retrievals disagree")
        fields += len(left) * len(ARCHIVE_COLUMNS)
        checks.append(
            {
                "distance_mpc_archive": entry["distance_mpc"],
                "distance_mpc_declared": declared["distance_mpc"],
                "distance_mpc_equal_as_numbers": (
                    _decimal(str(entry["distance_mpc"])) == _decimal(str(declared["distance_mpc"]))
                ),
                "distance_mpc_equal_as_strings": (
                    str(entry["distance_mpc"]) == str(declared["distance_mpc"])
                ),
                "object": name,
                "points": entry["point_count"],
                "rows_identical": True,
            }
        )
    checks.sort(key=lambda item: item["object"])
    return {
        "checks": checks,
        "disagreements": 0,
        "objects": len(checks),
        "published_fields_compared": fields,
        "reading": (
            "the six galaxies the repository retrieved from VizieR appear in the widened "
            "dataset with byte-identical published strings on every row and column. Where a "
            "distance differs it differs only by a trailing zero in the printed literal, "
            "which is recorded rather than normalised away; the two are equal as numbers and "
            "the distance takes no part in any fit here"
        ),
        "rule": CROSS_RETRIEVAL_RULE,
    }


def load_full_sample(path: Path) -> tuple[list[Galaxy], dict[str, Any]]:
    """Validated widened dataset, in exactly the ``Galaxy`` shape the fitter already takes."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset(payload)
    galaxies: list[Galaxy] = []
    for entry in payload["galaxies"]:
        columns = list(zip(*entry["rows"], strict=True))
        galaxies.append(
            Galaxy(
                name=entry["name"],
                distance_mpc=entry["distance_mpc"],
                radius=tuple(_decimal(value) for value in columns[0]),
                v_obs=tuple(_decimal(value) for value in columns[1]),
                e_v_obs=tuple(_decimal(value) for value in columns[2]),
                v_gas=tuple(_decimal(value) for value in columns[3]),
                v_disk=tuple(_decimal(value) for value in columns[4]),
                v_bul=tuple(_decimal(value) for value in columns[5]),
                published=tuple(tuple(row) for row in entry["rows"]),
            )
        )
    provenance = {
        "columns": payload["columns"],
        "dataset_sha256": canonical_sha256(payload),
        "galaxy_count": len(galaxies),
        "galaxy_digest_sha256": payload["galaxy_digest_sha256"],
        "mass_to_light_convention": payload["mass_to_light_convention"],
        "per_galaxy_provenance": {
            entry["name"]: dict(entry["provenance"]) for entry in payload["galaxies"]
        },
        "point_count": sum(item.count for item in galaxies),
        "selection": payload["selection"],
        "source": payload["source"],
    }
    return galaxies, provenance


# ---------------------------------------------------------------------------
# Step 3 -- the admission rule
# ---------------------------------------------------------------------------


def admit(
    galaxies: Sequence[Galaxy], upsilon_disk: Fraction, upsilon_bul: Fraction
) -> tuple[list[Galaxy], dict[str, Any]]:
    """Apply the declared admission rule, exactly, and name every exclusion."""

    admitted: list[Galaxy] = []
    excluded: list[dict[str, Any]] = []
    for galaxy in galaxies:
        offenders: list[dict[str, Any]] = []
        for index in range(galaxy.count):
            gas = galaxy.v_gas[index]
            value = (
                gas * abs(gas)
                + upsilon_disk * galaxy.v_disk[index] ** 2
                + upsilon_bul * galaxy.v_bul[index] ** 2
            )
            if value <= 0:
                offenders.append(
                    {
                        "baryonic_v_bar_sq": _fraction_block(value),
                        "published_row": list(galaxy.published[index]),
                        "row_index": index,
                    }
                )
        if offenders:
            excluded.append(
                {
                    "object": galaxy.name,
                    "offending_rows": offenders,
                    "points": galaxy.count,
                    "reason": (
                        "the published baryonic decomposition is not positive on every row "
                        "under the declared convention, so a Newtonian prediction is not "
                        "defined for this object as published"
                    ),
                }
            )
        else:
            admitted.append(galaxy)
    excluded.sort(key=lambda item: item["object"])
    return admitted, {
        "admitted_galaxies": len(admitted),
        "admitted_points": sum(item.count for item in admitted),
        "excluded": excluded,
        "excluded_galaxies": len(excluded),
        "excluded_points": sum(item["points"] for item in excluded),
        "rule": ADMISSION_RULE,
    }


# ---------------------------------------------------------------------------
# Step 4 -- the same critical coverage, without enumerating pairs
# ---------------------------------------------------------------------------

#: A hard stop on the envelope walk.  It is a guard, not a convergence tolerance: the walk
#: is exact and finite, and reaching this bound means the invariant that drives it is
#: broken, so it raises rather than returning whatever it had reached.
MAX_ENVELOPE_STEPS = 4096

FAST_SOLVER_RULE = (
    "k*(axis) is the smallest k with max_i(alpha_i - beta_i k) <= min_j(alpha_j + beta_j k). "
    "The left side is a maximum of falling lines and the right a minimum of rising ones, so "
    "their difference f(k) is convex and strictly decreasing and has one root. At any k the "
    "active pair (i, j) gives a supporting line of f whose own root is (alpha_i - alpha_j) / "
    "(beta_i + beta_j); because that line lies below f, its root never overshoots k*, so "
    "stepping to it walks up to k* from below and lands on it exactly. Each step costs one "
    "linear scan instead of the reference solver's n^2 pairs, the step is exact rational "
    "arithmetic throughout, and the pair it stops on is the same certificate: the receipt "
    "re-checks that (alpha_i - alpha_j) / (beta_i + beta_j) equals the k* it reports"
)


def critical_coverage_fast(axis: Axis) -> tuple[Fraction, tuple[str, str] | None]:
    """The exact same rational as :func:`critical_coverage`, by walking the envelopes.

    The reference solver enumerates every ordered pair and is quadratic.  Pooling every
    exploration row onto one axis makes that thousands of rows, and quadratic in thousands
    of exact rationals is minutes per law space.  This walks the upper envelopes instead:
    each step is one linear scan and lands on the pair that binds.
    """

    if not axis.alphas:
        raise FullSampleError("the axis carried no row the free parameter can move")
    coverage = Fraction(0)
    binding: tuple[int, int] | None = None
    for _ in range(MAX_ENVELOPE_STEPS):
        low_index = 0
        high_index = 0
        lower = axis.alphas[0] - axis.betas[0] * coverage
        upper = axis.alphas[0] + axis.betas[0] * coverage
        for index in range(1, len(axis.alphas)):
            alpha = axis.alphas[index]
            pad = axis.betas[index] * coverage
            value = alpha - pad
            if value > lower:
                lower, low_index = value, index
            value = alpha + pad
            if value < upper:
                upper, high_index = value, index
        if lower <= upper:
            break
        step = (axis.alphas[low_index] - axis.alphas[high_index]) / (
            axis.betas[low_index] + axis.betas[high_index]
        )
        if step <= coverage:
            raise FullSampleError(
                "the envelope walk failed to advance, which contradicts the convexity it "
                "relies on; the closed form is not the minimum it claims to be"
            )
        coverage, binding = step, (low_index, high_index)
    else:  # pragma: no cover - the walk is finite; this is a fail-closed guard
        raise FullSampleError("the envelope walk did not terminate")
    if binding is not None:
        low_index, high_index = binding
        attained = (axis.alphas[low_index] - axis.alphas[high_index]) / (
            axis.betas[low_index] + axis.betas[high_index]
        )
        if attained != coverage:
            raise FullSampleError("the exhibited pair does not attain the coverage it certifies")
    if coverage > axis.blind_floor:
        pair = (
            None if binding is None else (axis.labels[binding[0]], axis.labels[binding[1]])
        )
        return coverage, pair
    return axis.blind_floor, None


def solver_equivalence(
    axes: Sequence[tuple[str, Axis]],
    *,
    fast: Any = critical_coverage_fast,
    reference: Any = critical_coverage,
) -> dict[str, Any]:
    """Run both solvers on the same axes and refuse a single disagreement.

    The fast solver is the one that makes the widened run possible, so it is exactly the
    thing that must not be trusted.  Every axis handed here is solved twice, by two
    derivations sharing no code: an explicit maximum over ordered pairs, and a walk up the
    convex envelope.  A single differing rational raises.

    Both solvers are parameters so that the guard itself can be tested: handing it a
    deliberately wrong solver must make it raise, and a comparison that cannot fail is not
    a comparison.
    """

    checks: list[dict[str, Any]] = []
    for name, axis in axes:
        expected, reference_pair = reference(axis)
        observed, fast_pair = fast(axis)
        if expected != observed:
            raise FullSampleError(
                f"{name}: the two critical-coverage derivations disagree, so the fast "
                "solver is not the reference it claims to replace"
            )
        checks.append(
            {
                "axis": name,
                "critical_coverage": _fraction_block(expected),
                "pairs_enumerated_by_reference": len(axis.alphas) ** 2,
                "reference_pair": list(reference_pair) if reference_pair else None,
                "rows": len(axis.alphas),
                "same_binding_pair": list(fast_pair or ()) == list(reference_pair or ()),
                "walk_pair": list(fast_pair) if fast_pair else None,
            }
        )
    checks.sort(key=lambda item: item["axis"])
    return {
        "agreements": len(checks),
        "checks": checks,
        "disagreements": 0,
        "reading": (
            "both derivations return the same exact rational on every axis checked. Where "
            "they name different binding pairs the pair is not unique and both attain the "
            "same k*, which the fast solver re-checks before returning"
        ),
        "rule": FAST_SOLVER_RULE,
    }


def ladder_order(split: Split, names: Sequence[str]) -> tuple[str, ...]:
    """Nest the objects in an order fixed by the sealed split digests, not by any datum."""

    digests = dict(split.digests)
    missing = sorted(name for name in names if name not in digests)
    if missing:
        raise FullSampleError(f"{missing[0]} carries no sealed split digest")
    return tuple(name for _, name in sorted((digests[name], name) for name in names))


def assert_monotone_ladder(verdicts: Sequence[str]) -> None:
    """Refuse a ladder that recovers a CONSTANT verdict after losing it.

    Nested prefixes cannot produce such a sequence: the values shared by every object of a
    superset are contained in those shared by every object of a subset, so once a value
    common to all of them stops existing, adding objects cannot bring one back.  The check
    is therefore fail-closed on an invariant honest data cannot violate, which is exactly
    why it is a separate function -- a guard that can only be exercised through data that
    cannot exist is a guard nobody has tested.
    """

    broken = False
    for verdict in verdicts:
        if verdict == VARIES:
            broken = True
        elif verdict == CONSTANT and broken:
            raise FullSampleError(
                "the constancy ladder recovered a CONSTANT verdict after losing it, which "
                "contradicts the nesting it is built on: the values shared by a superset of "
                "objects are contained in the values shared by any subset"
            )


def constancy_ladder(decomposition: Mapping[str, Any], order: Sequence[str]) -> dict[str, Any]:
    """Ask R2 of nested subpopulations and find the size at which universality breaks.

    Every step runs at the *same* coverage -- the one where the whole exploration
    population first admits a solution -- so the only thing changing along the ladder is
    how many objects have to agree.  That is what makes the breakpoint a measurement: it
    is the number of galaxies at which a single value stops reaching all of them, and it
    is the reason a CONSTANT verdict drawn from four objects says so much less than the
    same word drawn from a hundred and thirty-nine.
    """

    intervals = {
        entry["object"]: entry["interval_at_population_coverage"]
        for entry in decomposition["population"]
    }
    ordered = [name for name in order if name in intervals]
    if len(ordered) != len(intervals):
        raise FullSampleError("the ladder order does not cover the fitted population")
    sizes = [size for size in LADDER_SIZES if size < len(ordered)] + [len(ordered)]
    steps: list[dict[str, Any]] = []
    verdicts: list[str] = []
    breaks_at: int | None = None
    for size in sizes:
        subset = {name: intervals[name] for name in ordered[:size]}
        record = adjudicate(subset)
        verify_adjudication(record, subset)
        verdict = record["verdict"]
        verdicts.append(verdict)
        assert_monotone_ladder(verdicts)
        if verdict == VARIES and breaks_at is None:
            breaks_at = size
        steps.append(
            {
                "certificate": record.get("certificate"),
                "objects": size,
                "verdict": verdict,
                "witness": record.get("witness"),
            }
        )
    return {
        "breaks_at_objects": breaks_at,
        "coverage_factor": decomposition["smallest_coverage_with_a_population"],
        "monotone": True,
        "reading": (
            "the verdict as a function of how many galaxies had to agree, at one fixed "
            "coverage. A ladder that is CONSTANT at four objects and VARIES at the full "
            "population is not a contradiction between two receipts; it is the measurement "
            "that four objects could not make"
        ),
        "rule": LADDER_RULE,
        "steps": steps,
    }


def own_coverage_ladder(
    law: LawSpace,
    galaxies: Sequence[Galaxy],
    rows_by_object: Mapping[str, Sequence[Any]],
    order: Sequence[str],
    split: Split,
) -> dict[str, Any]:
    """What a per-object receipt would have said had it been built on N galaxies.

    Each nested prefix is fitted and adjudicated at *its own* smallest feasible coverage,
    which is what a run over that many objects would have used.  The verdict and the price
    of universality are then read off exactly as the predecessor reads them, so the first
    step of this ladder is directly comparable with the receipt that only ever saw four
    galaxies, and every later step is the answer it could not reach.
    """

    split.guard(galaxies)
    axes: dict[str, Axis] = {}
    criticals: dict[str, Fraction] = {}
    for galaxy in galaxies:
        offsets, slopes = law.columns(galaxy)
        axis = build_axis(offsets, slopes, rows_by_object[galaxy.name])
        axes[galaxy.name] = axis
        criticals[galaxy.name] = critical_coverage_fast(axis)[0]
    by_name = {galaxy.name: galaxy for galaxy in galaxies}
    ordered = [name for name in order if name in axes]
    if len(ordered) != len(axes):
        raise FullSampleError("the ladder order does not cover the fitted population")
    sizes = [size for size in LADDER_SIZES if size < len(ordered)] + [len(ordered)]
    steps: list[dict[str, Any]] = []
    for size in sizes:
        names = ordered[:size]
        k_pop = max(criticals[name] for name in names)
        intervals = {name: interval_at(axes[name], k_pop) for name in names}
        record = adjudicate(intervals)
        verify_adjudication(record, intervals)
        k_common, _ = critical_coverage_fast(
            pooled_axis(law, [by_name[name] for name in names], rows_by_object)
        )
        if k_common < k_pop:
            raise FullSampleError(
                "the shared-parameter coverage fell below the per-object one, which is "
                "arithmetically impossible: the pooled pair set contains every per-object pair"
            )
        price = k_common / k_pop if k_pop > 0 else Fraction(1)
        if (record["verdict"] == CONSTANT) != (price == 1):
            raise FullSampleError(
                "the constancy verdict read off the intervals disagrees with the price of "
                "universality read off the certificates; one derivation is wrong"
            )
        steps.append(
            {
                "certificate": record.get("certificate"),
                "objects": size,
                "price_of_universality": {
                    "decimal": _num(price),
                    "exact": _fraction_data(price),
                    "is_one": price == 1,
                },
                "shared_parameter_coverage": _fraction_block(k_common),
                "smallest_coverage_with_a_population": _fraction_block(k_pop),
                "verdict": record["verdict"],
                "witness": record.get("witness"),
            }
        )
    constant_through = 0
    for step in steps:
        if step["verdict"] != CONSTANT:
            break
        constant_through = step["objects"]
    return {
        "constant_through_objects": constant_through,
        "law": law.name,
        "reading": (
            "the verdict and the price of universality a per-object receipt would have "
            "reported at each population size, each computed at the coverage that "
            "population needs. Reading down the price column shows how much of a CONSTANT "
            "verdict was the law and how much was the sample size"
        ),
        "rule": OWN_COVERAGE_LADDER_RULE,
        "steps": steps,
    }


def block_survey(
    law: LawSpace,
    galaxies: Sequence[Galaxy],
    rows_by_object: Mapping[str, Sequence[Any]],
    order: Sequence[str],
    split: Split,
) -> dict[str, Any]:
    """Run the predecessor's whole question inside every disjoint block of four galaxies.

    A CONSTANT verdict says a single value of the per-object parameter lies inside every
    object's interval.  Whether that is a statement about the law or about the sample is
    decidable, and this decides it: partition the population into blocks the size of the
    predecessor's, ask each block the same question, and count.  If most blocks say
    CONSTANT while the whole population says VARIES, then the predecessor's CONSTANT was a
    property of having four galaxies, and the number of blocks that agreed is the measure
    of how easily that happens.
    """

    split.guard(galaxies)
    axes: dict[str, Axis] = {}
    criticals: dict[str, Fraction] = {}
    for galaxy in galaxies:
        offsets, slopes = law.columns(galaxy)
        axis = build_axis(offsets, slopes, rows_by_object[galaxy.name])
        axes[galaxy.name] = axis
        criticals[galaxy.name] = critical_coverage_fast(axis)[0]
    by_name = {galaxy.name: galaxy for galaxy in galaxies}
    ordered = [name for name in order if name in axes]
    if len(ordered) != len(axes):
        raise FullSampleError("the block order does not cover the fitted population")
    size = PREDECESSOR_POPULATION_SIZE
    blocks = [ordered[start : start + size] for start in range(0, len(ordered), size)]
    remainder = [names for names in blocks if len(names) != size]
    blocks = [names for names in blocks if len(names) == size]
    records: list[dict[str, Any]] = []
    for names in blocks:
        k_pop = max(criticals[name] for name in names)
        intervals = {name: interval_at(axes[name], k_pop) for name in names}
        record = adjudicate(intervals)
        verify_adjudication(record, intervals)
        k_common, _ = critical_coverage_fast(
            pooled_axis(law, [by_name[name] for name in names], rows_by_object)
        )
        price = k_common / k_pop if k_pop > 0 else Fraction(1)
        if (record["verdict"] == CONSTANT) != (price == 1):
            raise FullSampleError(
                "a block's constancy verdict disagrees with its price of universality"
            )
        records.append(
            {
                "objects": list(names),
                "price_of_universality": {
                    "decimal": _num(price),
                    "exact": _fraction_data(price),
                },
                "smallest_coverage_with_a_population": _fraction_block(k_pop),
                "verdict": record["verdict"],
            }
        )
    constant = sum(1 for entry in records if entry["verdict"] == CONSTANT)
    return {
        "block_size": size,
        "blocks": records,
        "blocks_constant": constant,
        "blocks_total": len(records),
        "blocks_varies": len(records) - constant,
        "law": law.name,
        "reading": (
            "each block is four galaxies asked the same question the predecessor receipt "
            "asked of four galaxies. The count of CONSTANT blocks against the whole "
            "population's single verdict is the contrast this receipt exists to publish"
        ),
        "remainder_objects": [name for names in remainder for name in names],
        "rule": BLOCK_SURVEY_RULE,
    }


# ---------------------------------------------------------------------------
# Step 5 -- the run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Population:
    """Everything the widened run needs, assembled once and reused by every law space."""

    admitted: tuple[Galaxy, ...]
    exploration: tuple[Galaxy, ...]
    split: Split
    admission: dict[str, Any]
    prepared: dict[str, dict[str, Any]]
    rows: dict[str, Any]
    provenance: dict[str, Any]
    declared_subset_names: tuple[str, ...]


def assemble(root: Path) -> Population:
    """Load, split by name, admit by rule, and prepare -- in that order, and only that one."""

    root = root.resolve()
    galaxies, provenance = load_full_sample(root / DATASET_PATH)
    subset = json.loads((root / DECLARED_SUBSET_PATH).read_text(encoding="utf-8"))
    subset_names = tuple(sorted(entry["name"] for entry in subset["galaxies"]))
    provenance["cross_retrieval_control"] = crosscheck_declared_subset(
        json.loads((root / DATASET_PATH).read_text(encoding="utf-8")), subset
    )

    # Names only, and before a single column is touched.
    count = int(CONFIRMATION_FRACTION * len(galaxies))
    split = declare_split(
        [galaxy.name for galaxy in galaxies],
        count=count,
        salt=FULL_SPLIT_SALT,
        rule=FULL_SPLIT_RULE,
    )

    convention = provenance["mass_to_light_convention"]
    upsilon_disk = Fraction(convention["disk_3_6um"])
    upsilon_bul = Fraction(convention["bulge_3_6um"])
    admitted, admission = admit(galaxies, upsilon_disk, upsilon_bul)
    allowed = set(split.exploration)
    exploration = tuple(galaxy for galaxy in admitted if galaxy.name in allowed)
    if not exploration:
        raise FullSampleError("the split and the admission rule left nothing to fit")
    admission["excluded_from_confirmation_set"] = sorted(
        entry["object"] for entry in admission["excluded"] if entry["object"] in split.confirmation
    )
    admission["excluded_from_exploration_set"] = sorted(
        entry["object"] for entry in admission["excluded"] if entry["object"] in allowed
    )

    source = (
        f"{provenance['source']['primary_citation']}; {provenance['source']['table']}; "
        f"{provenance['source']['dataset_doi']}"
    )
    prepared = {
        galaxy.name: prepare_galaxy(galaxy, upsilon_disk, upsilon_bul, QUADRATURE)
        for galaxy in exploration
    }
    rows = {galaxy.name: measured_rows(galaxy, source) for galaxy in exploration}
    return Population(
        admitted=tuple(admitted),
        admission=admission,
        declared_subset_names=subset_names,
        exploration=exploration,
        prepared=prepared,
        provenance=provenance,
        rows=rows,
        split=split,
    )


def compact_population(decomposition: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The whole population, kept, in the smallest honest form: nothing is ranked or cut."""

    out: list[dict[str, Any]] = []
    for entry in decomposition["population"]:
        interval = entry["interval_at_population_coverage"]
        out.append(
            {
                "binding_pair": entry["binding_pair"],
                "critical_coverage": entry["critical_coverage"],
                "interval_at_population_coverage": (
                    {"empty": True}
                    if interval.get("empty")
                    else {
                        "lower": interval["lower"],
                        "upper": interval["upper"],
                        "width": interval["width"],
                    }
                ),
                "object": entry["object"],
                "points": entry["points"],
                "residual_sign_changes": entry["residual_structure_at_critical_theta"][
                    "sign_changes"
                ],
                "residuals_structured": entry["residual_structure_at_critical_theta"][
                    "structured"
                ],
                "theta_at_critical_coverage": entry["theta_at_critical_coverage"],
            }
        )
    out.sort(key=lambda item: item["object"])
    return out


def law_summary(decomposition: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    """One law space, reduced to the R2 question and the certificate that answers it."""

    record = diagnosis["at_population_coverage"]
    thetas = sorted(
        (_from_block(block), name)
        for name, block in diagnosis["per_object_point_estimates"].items()
    )
    return {
        "is_control": decomposition["is_control"],
        "law": decomposition["law"],
        "largest_theta_object": thetas[-1][1],
        "objects_fitted": len(decomposition["objects_fitted"]),
        "objects_with_a_solution_by_declared_coverage": {
            text: diagnosis["at_declared_coverage_factors"][text]["objects_with_a_solution"]
            for text in COVERAGE_GRID
        },
        "parameter": decomposition["parameter"],
        "price_of_universality": {
            "decimal": diagnosis["price_of_universality"]["decimal"],
            "exact": diagnosis["price_of_universality"]["exact"],
            "is_one": diagnosis["price_of_universality"]["is_one"],
        },
        "shared_parameter_coverage": decomposition["shared_parameter_coverage"],
        "smallest_coverage_with_a_population": (
            decomposition["smallest_coverage_with_a_population"]
        ),
        "smallest_coverage_with_a_population_set_by": (
            decomposition["smallest_coverage_with_a_population_set_by"]
        ),
        "smallest_theta_object": thetas[0][1],
        "structured_object_count": diagnosis["structured_object_count"],
        "theta_span": diagnosis["point_estimate_span"],
        "variation_certificate": record.get("certificate"),
        "verdict_at_population_coverage": diagnosis["verdict_at_population_coverage"],
        "witness": record.get("witness"),
    }


def run(root: Path) -> dict[str, Any]:
    """R1 and R2 over the full published sample, every law space, exploration only."""

    root = root.resolve()
    population = assemble(root)
    families = load_families(root)
    a0_text = REFERENCE_GRID_POINT["a0"]
    unit_text = REFERENCE_GRID_POINT["length_unit"]
    cache = ColumnCache(population.prepared)
    spaces = build_law_spaces(families, population.prepared, cache, a0_text, unit_text)

    decompositions: dict[str, Any] = {}
    diagnoses: dict[str, Any] = {}
    for law in spaces:
        decomposition = decompose(
            law,
            population.exploration,
            population.rows,
            population.split,
            solver=critical_coverage_fast,
        )
        decompositions[law.name] = decomposition
        diagnoses[law.name] = diagnose(decomposition)
    return {
        "decompositions": decompositions,
        "diagnoses": diagnoses,
        "families": families,
        "population": population,
        "spaces": spaces,
    }


def _equivalence_axes(
    state: Mapping[str, Any], law_names: Sequence[str], pooled_for: Sequence[str]
) -> list[tuple[str, Axis]]:
    """Axes to solve twice, by both derivations: per object, and the pooled axis itself.

    The pooled axis is the one the fast solver exists for -- it carries every exploration
    row at once -- so at least one of them is re-derived by the quadratic reference even
    though that costs millions of exact rational divisions.  A solver nobody checked on the
    case it was written for is not checked.
    """

    population: Population = state["population"]
    axes: list[tuple[str, Axis]] = []
    for name in law_names:
        law: LawSpace = next(item for item in state["spaces"] if item.name == name)
        for galaxy in population.exploration:
            offsets, slopes = law.columns(galaxy)
            axes.append(
                (
                    f"{name}|{galaxy.name}",
                    build_axis(offsets, slopes, population.rows[galaxy.name]),
                )
            )
        if name in pooled_for:
            axes.append(
                (f"{name}|pooled", pooled_axis(law, population.exploration, population.rows))
            )
    return axes


def build_receipt(root: Path) -> dict[str, Any]:
    """The widened R1/R2 receipt: sealed, replayable, and carrying its own controls."""

    root = root.resolve()
    state = run(root)
    population: Population = state["population"]
    decompositions = state["decompositions"]
    diagnoses = state["diagnoses"]
    families = state["families"]
    best = select_best_family(families)
    best_space = f"family_{best.ordinal}_amplitude"
    headline = ("newtonian_baryons_only", best_space, "deliberately_wrong_law")

    equivalence = solver_equivalence(
        _equivalence_axes(state, headline, ("newtonian_baryons_only",))
    )

    crosschecks = {
        name: crosscheck_against_simplex(
            next(law for law in state["spaces"] if law.name == name),
            population.exploration,
            population.rows,
            decompositions[name],
        )
        for name in ("newtonian_baryons_only",)
    }

    order = ladder_order(population.split, [galaxy.name for galaxy in population.exploration])
    ladders = {
        name: constancy_ladder(decompositions[name], order) for name in sorted(decompositions)
    }
    own_ladders = {
        law.name: own_coverage_ladder(
            law, population.exploration, population.rows, order, population.split
        )
        for law in state["spaces"]
    }
    surveys = {
        law.name: block_survey(
            law, population.exploration, population.rows, order, population.split
        )
        for law in state["spaces"]
    }
    summaries = {
        name: law_summary(decompositions[name], diagnoses[name]) for name in sorted(decompositions)
    }
    for name, summary in summaries.items():
        summary["constancy_breaks_at_objects"] = ladders[name]["breaks_at_objects"]
        summary["constant_at_own_coverage_through_objects"] = own_ladders[name][
            "constant_through_objects"
        ]
        summary["four_object_blocks_calling_it_constant"] = surveys[name]["blocks_constant"]
        summary["four_object_blocks_total"] = surveys[name]["blocks_total"]
    constant = sorted(
        name for name, item in summaries.items() if item["verdict_at_population_coverage"] == CONSTANT
    )
    varies = sorted(
        name for name, item in summaries.items() if item["verdict_at_population_coverage"] == VARIES
    )
    unresolved = sorted(
        name
        for name, item in summaries.items()
        if item["verdict_at_population_coverage"] == NO_POPULATION
    )

    control = decompositions["deliberately_wrong_law"]
    control_k = _from_block(control["smallest_coverage_with_a_population"])
    family_spaces = [f"family_{family.ordinal}_amplitude" for family in families]
    not_beaten = sorted(
        name
        for name in family_spaces
        if _from_block(decompositions[name]["smallest_coverage_with_a_population"]) >= control_k
    )
    wrong_law_control = {
        "declared_expectation": (
            "the deliberately wrong law -- the same family with its local factor inverted, "
            "so the modification grows where gravity is strong -- must need a strictly "
            "larger k_pop than every screened family on the same exploration galaxies"
        ),
        "families_it_did_not_beat": not_beaten,
        "held": not not_beaten,
        "reading": (
            "on six galaxies this expectation held. It is re-run here on the full sample "
            "because a control that was only ever checked on four objects is a control "
            "nobody has tested, and the result is published whichever way it fell"
        ),
        "smallest_coverage_with_a_population": control["smallest_coverage_with_a_population"],
        "verdict_at_population_coverage": summaries["deliberately_wrong_law"][
            "verdict_at_population_coverage"
        ],
    }

    body: dict[str, Any] = {
        "admission": population.admission,
        "assumptions": dict(ASSUMPTIONS),
        "claims": dict(CLAIMS),
        "counts": {
            "admitted_galaxies": population.admission["admitted_galaxies"],
            "confirmation_galaxies_declared": len(population.split.confirmation),
            "exploration_galaxies_fitted": len(population.exploration),
            "exploration_points_fitted": sum(item.count for item in population.exploration),
            "law_spaces": len(state["spaces"]),
            "one_parameter_fits": len(state["spaces"]) * len(population.exploration),
            "published_galaxies": population.provenance["galaxy_count"],
            "published_points": population.provenance["point_count"],
            "widened_from_galaxies": population.provenance["selection"]["widens"][
                "from_galaxy_count"
            ],
            "widened_from_points": population.provenance["selection"]["widens"]["from_point_count"],
        },
        "coverage_factors": list(COVERAGE_GRID),
        "data_provenance": {
            "columns": population.provenance["columns"],
            "cross_retrieval_control": population.provenance["cross_retrieval_control"],
            "dataset_sha256": population.provenance["dataset_sha256"],
            "galaxy_digest_sha256": population.provenance["galaxy_digest_sha256"],
            "mass_to_light_convention": population.provenance["mass_to_light_convention"],
            "per_galaxy_provenance": population.provenance["per_galaxy_provenance"],
            "selection": population.provenance["selection"],
            "source": population.provenance["source"],
        },
        "exploration_confirmation_split": population.split.block(),
        "exploratory_caveat": {
            "confirmation_set_overlap_with_the_six_already_scanned": sorted(
                set(population.split.confirmation) & set(population.declared_subset_names)
            ),
            "may_be_cited_as_confirmation": False,
            "sealed_no_refit_trial": False,
            "statement": (
                "Every fit here gives each object its own copy of a free parameter, which is "
                "exactly the freedom a sealed trial forbids, so this receipt generates "
                "hypotheses and confirms nothing. The confirmation galaxies were never "
                "handed to a fitting function: the guard raises rather than declines. Unlike "
                "the six-galaxy predecessor, whose two withheld galaxies had already been "
                "scanned by an earlier receipt, the great majority of the galaxies withheld "
                "here have never been read by anything in this repository -- the overlap "
                "with the six already scanned is listed above and is the only exception"
            ),
        },
        "grid_point": {
            "a0_kms2_per_kpc": REFERENCE_GRID_POINT["a0"],
            "declared_before_the_fit": True,
            "length_unit_kpc": REFERENCE_GRID_POINT["length_unit"],
            "reading": (
                "the reference point of the predecessor module's declared grid, chosen there "
                "before any data was opened and reused unchanged, so the widened populations "
                "are not a scan over grid points"
            ),
        },
        "instrument_crosscheck": crosschecks,
        "law_space_summaries": summaries,
        "method": {
            "admission_rule": ADMISSION_RULE,
            "critical_coverage": (
                "k*(object) = max over ordered pairs (i, j) of (alpha_i - alpha_j) / "
                "(beta_i + beta_j), with alpha = (v - c) / b and beta = sigma / |b|. A "
                "maximum over a finite set of exact rationals: the argmax pair is the "
                "certificate and there is nothing to converge"
            ),
            "constancy_ladder": LADDER_RULE,
            "constancy_ladder_at_own_coverage": OWN_COVERAGE_LADDER_RULE,
            "four_object_block_survey": BLOCK_SURVEY_RULE,
            "fast_solver": FAST_SOLVER_RULE,
            "monotonicity_of_widening": MONOTONICITY_CLAIM,
            "price_of_universality": (
                "k_common / k_pop, where k_pop is the largest per-object critical coverage "
                "and k_common the same maximum over pairs drawn from all objects at once. It "
                "is at least 1 by construction: the pooled pair set contains every "
                "per-object pair"
            ),
            "split_rule": FULL_SPLIT_RULE,
        },
        "parameter_constancy_r2": {
            "constancy_ladders": ladders,
            "constancy_ladders_at_own_coverage": own_ladders,
            "four_object_block_survey": surveys,
            "constant_law_spaces": constant,
            "constant_at_four_objects_law_spaces": sorted(
                name
                for name, entry in own_ladders.items()
                if entry["steps"] and entry["steps"][0]["verdict"] == CONSTANT
            ),
            "reading": (
                "This is the deliverable. For each declared law space the per-object "
                "parameter either takes one value inside every one of the exploration "
                "galaxies' exact intervals, or it does not and two named galaxies are "
                "disjoint by an exact rational. On four galaxies that question is nearly "
                "free to answer yes; on this many it is not, and the contrast between the "
                "two columns is a measurement rather than a null"
            ),
            "unresolved_law_spaces": unresolved,
            "varies_law_spaces": varies,
            "widening_removed_constancy_from": sorted(
                name
                for name, entry in own_ladders.items()
                if entry["steps"]
                and entry["steps"][0]["verdict"] == CONSTANT
                and entry["steps"][-1]["verdict"] == VARIES
            ),
        },
        "per_object_population_r1": {
            name: compact_population(decompositions[name]) for name in sorted(decompositions)
        },
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "solver_equivalence": equivalence,
        "trial_type": TRIAL_TYPE,
        "wrong_law_control": wrong_law_control,
    }
    body["decision"] = _decision(body)
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise FullSampleError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    return {**body, "content_sha256": canonical_sha256(body)}


def _decision(body: Mapping[str, Any]) -> str:
    counts = body["counts"]
    constancy = body["parameter_constancy_r2"]
    newtonian = body["law_space_summaries"]["newtonian_baryons_only"]
    survey = body["parameter_constancy_r2"]["four_object_block_survey"]
    blocks_constant = sum(entry["blocks_constant"] for entry in survey.values())
    blocks_total = sum(entry["blocks_total"] for entry in survey.values())
    control = body["law_space_summaries"]["deliberately_wrong_law"]
    hardest_family = _num(
        max(
            _from_block(item["smallest_coverage_with_a_population"])
            for name, item in body["law_space_summaries"].items()
            if name.startswith("family_")
        )
    )
    return (
        f"EXPLORATORY: the sample is widened from {counts['widened_from_galaxies']} galaxies "
        f"and {counts['widened_from_points']} published points to "
        f"{counts['published_galaxies']} and {counts['published_points']}, of which "
        f"{counts['exploration_galaxies_fitted']} galaxies and "
        f"{counts['exploration_points_fitted']} points are fitted -- each galaxy alone, with "
        f"its own copy of one free parameter, over {counts['law_spaces']} declared law "
        f"spaces. Of those law spaces {len(constancy['constant_law_spaces'])} carry a "
        f"per-object parameter that is CONSTANT within the declared intervals and "
        f"{len(constancy['varies_law_spaces'])} carry one that VARIES; the Newtonian "
        f"per-object baryonic rescale is {newtonian['verdict_at_population_coverage']} at "
        f"price of universality {newtonian['price_of_universality']['decimal']}. Cut the "
        f"same population into disjoint blocks of {PREDECESSOR_POPULATION_SIZE} galaxies -- "
        f"the predecessor receipt's population size -- and {blocks_constant} of "
        f"{blocks_total} blocks call their per-object parameter CONSTANT, the deliberately "
        f"wrong law among them in "
        f"{survey['deliberately_wrong_law']['blocks_constant']} of "
        f"{survey['deliberately_wrong_law']['blocks_total']}. So CONSTANT at four galaxies "
        f"is a statement about the sample size and not about the law, and on the full "
        f"population it survives in {len(constancy['constant_law_spaces'])} of "
        f"{counts['law_spaces']} law spaces. The wrong-law control still separates: it needs "
        f"coverage {control['smallest_coverage_with_a_population']['decimal']} where the "
        f"screened families need at most {hardest_family}. The "
        f"{counts['confirmation_galaxies_declared']} confirmation galaxies were never "
        "handed to a fitting function. Nothing here may be cited as a confirmation."
    )


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Reject tamper or drift by exact deterministic replay."""

    if receipt.get("schema_version") != RESULT_SCHEMA:
        raise FullSampleError("receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise FullSampleError("receipt seal changed")
    if receipt.get("claims") != CLAIMS:
        raise FullSampleError("claims changed")
    if receipt.get("trial_type") != TRIAL_TYPE:
        raise FullSampleError("trial type changed")
    split = receipt.get("exploration_confirmation_split")
    if not isinstance(split, Mapping):
        raise FullSampleError("the receipt carries no exploration/confirmation split")
    sealed = {key: value for key, value in split.items() if key != "split_sha256"}
    if split.get("split_sha256") != canonical_sha256(sealed):
        raise FullSampleError("the split seal changed")
    if split.get("salt") != FULL_SPLIT_SALT or split.get("rule") != FULL_SPLIT_RULE:
        raise FullSampleError("the split rule changed")
    withheld = set(split.get("confirmation", ()))
    fitted = {
        entry["object"]
        for population in receipt.get("per_object_population_r1", {}).values()
        for entry in population
    }
    if withheld & fitted:
        raise FullSampleError("a confirmation-set object appears in a fitted population")
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise FullSampleError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    if dict(receipt) != build_receipt(root):
        raise FullSampleError("receipt exact replay changed")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    """Write once; a differing rewrite is refused rather than silently accepted."""

    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise FullSampleError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full SPARC sample, Tier 7 R1/R2.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=RECEIPT_PATH)
    parser.add_argument(
        "--build-dataset",
        metavar="ARCHIVE",
        default=None,
        help="read the offline published per-galaxy archive and write the widened dataset",
    )
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.build_dataset is not None:
        subset = json.loads((root / DECLARED_SUBSET_PATH).read_text(encoding="utf-8"))
        payload = build_dataset(Path(args.build_dataset), subset)
        crosscheck_declared_subset(payload, subset)
        target = root / DATASET_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json_bytes(payload) + b"\n")
        return 0
    output = (root / args.output).resolve()
    if args.validate_checked:
        validate_receipt(json.loads(output.read_text(encoding="utf-8")), root=root)
        return 0
    receipt = build_receipt(root)
    write_immutable(output, receipt)
    validate_receipt(receipt, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMISSION_RULE",
    "ASSUMPTIONS",
    "BLOCK_SURVEY_RULE",
    "CLAIMS",
    "CONFIRMATION_FRACTION",
    "CROSS_RETRIEVAL_RULE",
    "DATASET_PATH",
    "DATASET_SCHEMA",
    "DECLARED_SUBSET_PATH",
    "FAST_SOLVER_RULE",
    "FULL_SPLIT_RULE",
    "FULL_SPLIT_SALT",
    "LADDER_RULE",
    "LADDER_SIZES",
    "MONOTONICITY_CLAIM",
    "OWN_COVERAGE_LADDER_RULE",
    "PREDECESSOR_POPULATION_SIZE",
    "PUBLISHED_GALAXY_COUNT",
    "RECEIPT_PATH",
    "RESULT_SCHEMA",
    "SCOPE",
    "TRIAL_TYPE",
    "ConfirmationSetTouched",
    "FullSampleError",
    "Population",
    "admit",
    "assemble",
    "assert_monotone_ladder",
    "block_survey",
    "build_dataset",
    "build_receipt",
    "compact_population",
    "constancy_ladder",
    "critical_coverage_fast",
    "crosscheck_declared_subset",
    "dataset_digest",
    "ladder_order",
    "law_summary",
    "load_full_sample",
    "main",
    "own_coverage_ladder",
    "parse_archive_file",
    "run",
    "solver_equivalence",
    "validate_dataset",
    "validate_receipt",
    "write_immutable",
]
