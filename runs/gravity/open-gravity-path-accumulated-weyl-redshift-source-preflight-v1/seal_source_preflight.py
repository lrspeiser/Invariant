"""Seal the zero-response Lane-1 public-source preflight.

Only archive metadata, the opaque-download manifest, and the already-public
eight-row exploration prediction file are parsed. FITS and spectral table payloads
are hash-checked as bytes and are never decompressed or interpreted.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DOWNLOAD_MANIFEST = ROOT / "download-manifest.json"
BOUNDARY = ROOT / "authorization-and-access-boundary.json"
PREDICTIONS = (
    REPO
    / "runs/gravity/open-gravity-path-accumulated-weyl-redshift-v1/artifacts"
    / "exploration-lens-predictions.csv"
)

EXPECTED_BINDINGS = {
    "sealed_config": (
        REPO / "configs/open_gravity_path_accumulated_weyl_redshift_v1.json",
        "2d1414fae7bb4c626e0c3ea45acd0f1957f01e7abc37a27682ebca8909e4fbce",
    ),
    "sealed_module": (
        REPO / "src/sigma_theory_compiler/open_gravity_path_accumulated_weyl_redshift_v1.py",
        "fb0ce705135dd64e55f77ab9fcdd6413a50d08caadb549164dced018fdf75313",
    ),
    "sealed_test": (
        REPO / "tests/test_open_gravity_path_accumulated_weyl_redshift_v1.py",
        "5baa3104aca0b1ed0858a530022ce029d6fca49a06ac8d17a2e8f3e20ac69b3a",
    ),
    "sealed_receipt": (
        REPO / "runs/gravity/open-gravity-path-accumulated-weyl-redshift-v1/receipt.json",
        "e11048029f44eaca87f25cbcc3172694c7f9b4c83ec8b04edeb2b37e4e94b50c",
    ),
    "exploration_predictions": (
        PREDICTIONS,
        "6a0c462d11702445b0fc8cf4f08702e54076e1e3ab901a087d6c7ff479435bdf",
    ),
}

EXPECTED_NAMES = [
    "SDSS J0832+0404",
    "SDSS J1226-0006",
    "SDSS J1320+1644",
    "SDSS J1335+0118",
    "SDSS J1349+1227",
    "SDSS J1455+1447",
    "SDSS J1515+1511",
    "SDSS J1620+1203",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_url(product_id: str) -> str:
    return f"https://dataportal.eso.org/dataPortal/file/{product_id}"


def eso_products(*ids: str) -> list[dict[str, Any]]:
    return [{"product_id": value, "url": source_url(value)} for value in ids]


def smoka_products(*ids: str) -> list[dict[str, Any]]:
    return [
        {
            "frame_id": value,
            "metadata_url": (
                "https://smoka.nao.ac.jp/info.jsp?"
                f"frameid={value}&date_obs=2009-02-01&i=0"
            ),
            "raw_retrieval": "BLOCKED_FREE_SMOKA_ACCOUNT_REQUIRED",
        }
        for value in ids
    ]


def load_predictions() -> dict[str, dict[str, Any]]:
    with PREDICTIONS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["name"] for row in rows] != EXPECTED_NAMES:
        raise RuntimeError("exploration identity/order changed")
    if len(rows) != 8 or any(row["response_opened"] != "False" for row in rows):
        raise RuntimeError("sealed exploration response boundary changed")
    return {
        row["name"]: {
            "fold": int(row["fold"]),
            "z_lens": float(row["z_lens"]),
            "z_source": float(row["z_source"]),
            "image_separation_arcsec": float(row["image_separation_arcsec"]),
            "image_flux_ratio": float(row["image_flux_ratio"]),
            "delta_velocity_km_s_per_alpha": float(row["delta_velocity_km_s_per_alpha"]),
            "source_model": row["source_model"],
            "response_opened": False,
        }
        for row in rows
    }


def source_ledger(predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    paper = {
        "J0832": "https://arxiv.org/abs/0708.0871",
        "J1226_J1335": "https://arxiv.org/abs/astro-ph/0511026",
        "J1320": "https://arxiv.org/abs/1206.2011",
        "J1349_J1455_J1620": "https://arxiv.org/abs/0912.1462",
        "J1515": "https://arxiv.org/abs/1701.04272",
        "MILLON": "https://arxiv.org/abs/2002.05736",
        "TDCARMA": "https://arxiv.org/abs/2207.09327",
    }
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "name": "SDSS J0832+0404",
            "predictor": predictions["SDSS J0832+0404"],
            "source_status": "SOURCE_BLOCKED",
            "official_products": eso_products(
                "EFOSC.2005-12-31T06:26:10.175",
                "EFOSC.2005-12-31T06:36:44.141",
                "EFOSC.2005-12-31T06:47:18.079",
            ),
            "archive": {
                "name": "ESO Science Archive",
                "program_id": "076.A-0519(A)",
                "ob_id": "100187792",
                "all_science_bytes_downloaded_opaque": True,
                "calselector_raw2raw_url": (
                    "https://archive.eso.org/calselector/v1/associations?"
                    "dp_id=EFOSC.2005-12-31T06:26:10.175&mode=raw2raw"
                ),
                "calselector_result": "UNKNOWN_CATEGORY_NO_ASSOCIATION_ESTABLISHED",
            },
            "spectroscopy": {
                "instrument_mode": "ESO 3.6m EFOSC2 long slit, grism #13, 1.0 arcsec slit",
                "epochs_utc": ["2005-12-31T06:26:10.175", "2005-12-31T06:36:44.141", "2005-12-31T06:47:18.079"],
                "total_exposure_s": 1800,
                "wavelength_range_angstrom": [3700, 9300],
                "dispersion_angstrom_per_pixel_approx": 5,
                "spatial_scale_arcsec_per_pixel": 0.31,
                "resolving_power_approx": 400,
                "image_resolution": "A extracted; B not separated from lens galaxy",
            },
            "time_delay": {
                "published_delta_t_ab_days": -125.3,
                "uncertainty_days": {"minus": 23.4, "plus": 12.8},
                "source": paper["MILLON"],
                "spectral_epochs_can_be_phase_aligned": False,
            },
            "primary_sources": [paper["J0832"], paper["MILLON"]],
            "missing": [
                "image-separated B spectrum",
                "archive-associated bias/flat/arc/standard products",
                "wavelength-solution residuals and covariance",
                "second spectral epoch separated by the measured delay",
                "precision lens-model/environment covariance",
            ],
            "decisive_block": "B_IS_LENS_BLENDED_AND_NO_CALIBRATION_OR_DELAY_ALIGNED_PAIR",
        }
    )

    rows.append(
        {
            "name": "SDSS J1226-0006",
            "predictor": predictions["SDSS J1226-0006"],
            "source_status": "SOURCE_BLOCKED",
            "official_products": eso_products(
                "FORS1.2005-05-16T01:01:39.042",
                "FORS1.2005-05-16T01:27:07.535",
                "FORS1.2005-05-16T01:56:46.427",
                "FORS1.2005-05-16T02:22:14.919",
                "FORS1.2005-05-16T03:16:03.097",
                "FORS1.2005-05-16T03:41:31.599",
                "FORS1.2005-05-16T04:10:13.347",
                "FORS1.2005-05-16T04:35:41.758",
            ),
            "archive": {
                "name": "ESO Science Archive",
                "program_id": "075.A-0377(B)",
                "ob_ids": ["197376", "197375", "197372"],
                "science_and_complete_raw2raw_calibration_union_downloaded_opaque": True,
            },
            "spectroscopy": {
                "instrument_mode": "VLT/FORS1 MOS, G300V+GG435, SR collimator, 1.0 arcsec slit",
                "epoch_utc": "2005-05-16",
                "exposures": 8,
                "exposure_each_s": 1400,
                "total_exposure_s": 11200,
                "wavelength_range_angstrom": [4450, 8650],
                "dispersion_angstrom_per_pixel": 2.69,
                "spatial_scale_arcsec_per_pixel": 0.2,
                "resolving_power_at_5900_angstrom": 400,
                "extraction": "MCS point-source and extended-channel method described; no exact public A/B extracted table found",
            },
            "time_delay": {
                "published_delta_t_ab_days": 33.7,
                "uncertainty_days": {"minus": 2.7, "plus": 2.7},
                "source": paper["MILLON"],
                "spectral_epochs_can_be_phase_aligned": False,
            },
            "primary_sources": [paper["J1226_J1335"], paper["MILLON"]],
            "missing": [
                "public extracted image-A/image-B wavelength-flux products",
                "per-pixel uncertainty/covariance and wavelength-fit residuals",
                "second spectral epoch separated by 33.7 days",
                "precision lens-model/environment covariance",
            ],
            "decisive_block": "RAW_DATA_EXIST_BUT_NO_PUBLIC_IMAGE_PAIR_RESPONSE_OR_DELAY_ALIGNED_EPOCH",
        }
    )

    rows.append(
        {
            "name": "SDSS J1320+1644",
            "predictor": predictions["SDSS J1320+1644"],
            "source_status": "SOURCE_BLOCKED",
            "official_products": [],
            "archive": {
                "name": "ARC 3.5m/APO DIS",
                "public_exact_product_identified": False,
                "published_spectral_table_identified": False,
            },
            "spectroscopy": {
                "instrument_mode": "ARC 3.5m DIS, B400/R300, 1.5 arcsec slit",
                "epoch_utc": "2009-02-19",
                "exposures": 1,
                "exposure_each_s": 1500,
                "wavelength_range_angstrom": [3700, 10000],
                "resolving_power_approx": 500,
                "image_resolution": "both quasar components on one slit",
            },
            "time_delay": {
                "accepted_measured_delay_identified": False,
                "spectral_epochs_can_be_phase_aligned": False,
            },
            "primary_sources": [paper["J1320"], paper["MILLON"]],
            "missing": [
                "public raw or extracted exact spectrum product",
                "calibration products and covariance",
                "accepted measured delay",
                "second spectral epoch",
                "secure lens-vs-binary classification and precision group/cluster lens model",
            ],
            "decisive_block": "NO_EXACT_PUBLIC_SPECTRAL_PRODUCT_AND_NO_MEASURED_DELAY",
        }
    )

    rows.append(
        {
            "name": "SDSS J1335+0118",
            "predictor": predictions["SDSS J1335+0118"],
            "source_status": "SOURCE_BLOCKED",
            "official_products": eso_products(
                "FORS1.2005-02-03T08:12:22.553",
                "FORS1.2005-02-03T08:37:50.882",
                "FORS1.2005-03-03T07:39:57.422",
                "FORS1.2005-03-03T08:05:25.946",
                "FORS1.2005-03-03T08:34:10.974",
                "FORS1.2005-03-03T08:59:40.116",
            ),
            "archive": {
                "name": "ESO Science Archive",
                "program_id": "074.A-0563(B)",
                "ob_ids": ["182508", "182507", "182504"],
                "science_and_complete_raw2raw_calibration_union_downloaded_opaque": True,
            },
            "spectroscopy": {
                "instrument_mode": "VLT/FORS1 MOS, G300V+GG435, HR collimator, 1.0 arcsec slit",
                "epochs_utc": ["2005-02-03", "2005-03-03"],
                "exposures": 6,
                "exposure_each_s": 1400,
                "total_exposure_s": 8400,
                "wavelength_range_angstrom": [4450, 8650],
                "dispersion_angstrom_per_pixel": 2.69,
                "spatial_scale_arcsec_per_pixel": 0.1,
                "resolving_power_at_5900_angstrom": 210,
                "extraction": "MCS method described; published figure is lens-galaxy spectrum, not an exact A/B response table",
            },
            "time_delay": {
                "published_delta_t_ab_days": -56.0,
                "uncertainty_days": {"minus": 6.1, "plus": 5.7},
                "source": paper["MILLON"],
                "spectral_epoch_separation_days": 28,
                "spectral_epochs_can_be_phase_aligned": False,
            },
            "primary_sources": [paper["J1226_J1335"], paper["MILLON"]],
            "missing": [
                "public extracted image-A/image-B wavelength-flux products",
                "per-pixel uncertainty/covariance and wavelength-fit residuals",
                "spectrum pair separated by the 56-day delay",
                "precision lens-model/environment covariance",
            ],
            "decisive_block": "TWO_EPOCHS_EXIST_BUT_NOT_DELAY_ALIGNED_AND_NO_PUBLIC_IMAGE_PAIR_RESPONSE",
        }
    )

    rows.append(
        {
            "name": "SDSS J1349+1227",
            "predictor": predictions["SDSS J1349+1227"],
            "source_status": "SOURCE_BLOCKED_ARCHIVE_AUTH",
            "official_products": smoka_products("FCSA00104201", "FCSA00104202"),
            "archive": {
                "name": "SMOKA/NAOJ",
                "exposure_id": "FCSE00104201",
                "metadata_downloaded_and_hashed": True,
                "raw_fits_downloaded": False,
                "raw_retrieval_rule": "FREE_USER_REGISTRATION_REQUIRED_BY_SMOKA",
            },
            "spectroscopy": {
                "instrument_mode": "Subaru/FOCAS, 300B+L600, 1.0 arcsec slit, two detector-chip frames",
                "epoch_utc": "2009-02-01T11:44:13.271",
                "exposure_s": 480,
                "wavelength_range_angstrom": [3700, 6000],
                "resolving_power": 400,
                "image_resolution": "A/B one-dimensional spectra described and plotted; no machine-readable spectra found",
            },
            "time_delay": {
                "published_delta_t_days": 432.05,
                "uncertainty_days": 1.95,
                "source": paper["TDCARMA"],
                "spectral_epochs_can_be_phase_aligned": False,
            },
            "primary_sources": [paper["J1349_J1455_J1620"], paper["TDCARMA"]],
            "missing": [
                "SMOKA account authorization for raw FITS retrieval",
                "machine-readable image-separated spectra",
                "calibration products/residuals and covariance",
                "second spectral epoch separated by 432.05 days",
                "precision lens-model/environment covariance",
            ],
            "decisive_block": "SMOKA_ACCOUNT_REQUIRED_AND_ONLY_ONE_UNTABLED_SPECTRAL_EPOCH",
        }
    )

    rows.append(
        {
            "name": "SDSS J1455+1447",
            "predictor": predictions["SDSS J1455+1447"],
            "source_status": "SOURCE_BLOCKED_ARCHIVE_AUTH",
            "official_products": smoka_products("FCSA00104251", "FCSA00104252"),
            "archive": {
                "name": "SMOKA/NAOJ",
                "exposure_id": "FCSE00104251",
                "metadata_downloaded_and_hashed": True,
                "raw_fits_downloaded": False,
                "raw_retrieval_rule": "FREE_USER_REGISTRATION_REQUIRED_BY_SMOKA",
            },
            "spectroscopy": {
                "instrument_mode": "Subaru/FOCAS, 300B+L600, 1.0 arcsec slit, two detector-chip frames",
                "epoch_utc": "2009-02-01T13:13:52.356",
                "exposure_s": 480,
                "wavelength_range_angstrom": [3700, 6000],
                "resolving_power": 400,
                "image_resolution": "A/B one-dimensional spectra described and plotted; no machine-readable spectra found",
            },
            "time_delay": {
                "published_delta_t_days": 45.36,
                "uncertainty_days": 1.93,
                "source": paper["TDCARMA"],
                "spectral_epochs_can_be_phase_aligned": False,
            },
            "primary_sources": [paper["J1349_J1455_J1620"], paper["TDCARMA"]],
            "missing": [
                "SMOKA account authorization for raw FITS retrieval",
                "machine-readable image-separated spectra",
                "calibration products/residuals and covariance",
                "second spectral epoch separated by 45.36 days",
                "precision lens-model/environment covariance",
            ],
            "decisive_block": "SMOKA_ACCOUNT_REQUIRED_AND_ONLY_ONE_UNTABLED_SPECTRAL_EPOCH",
        }
    )

    rows.append(
        {
            "name": "SDSS J1515+1511",
            "predictor": predictions["SDSS J1515+1511"],
            "source_status": "SOURCE_PARTIAL_BLOCKED_CALIBRATION_COVARIANCE",
            "official_products": [
                {
                    "catalog_id": "J/ApJ/836/14",
                    "file": "table2.dat",
                    "url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table2.dat",
                    "records": 1024,
                    "content": "GTC-OSIRIS-R500B wavelength plus A/B flux",
                },
                {
                    "catalog_id": "J/ApJ/836/14",
                    "file": "table3.dat",
                    "url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table3.dat",
                    "records": 916,
                    "content": "GTC-OSIRIS-R500R wavelength plus A/B/G1 flux",
                },
                {
                    "catalog_id": "J/ApJ/836/14",
                    "file": "table4.dat",
                    "url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table4.dat",
                    "records": 880,
                    "content": "LT-SPRAT 2015 August wavelength plus A/B flux",
                },
                {
                    "catalog_id": "J/ApJ/836/14",
                    "file": "table5.dat",
                    "url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table5.dat",
                    "records": 880,
                    "content": "LT-SPRAT 2016 March wavelength plus A/B flux",
                },
            ],
            "archive": {
                "name": "CDS/VizieR",
                "catalog_id": "J/ApJ/836/14",
                "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/ReadMe",
                "all_four_spectral_tables_downloaded_opaque": True,
                "gtc_raw_archive_search_observation": "PUBLIC_ARCHIVE_ASSERTED_BY_PAPER_BUT_CURRENT_SEARCH_FORM_RETURNED_HTTP_500",
            },
            "spectroscopy": {
                "GTC_OSIRIS": {
                    "modes": ["R500B", "R500R"],
                    "epochs_utc": ["2015-04-15", "2015-04-16"],
                    "slit_arcsec": 1.23,
                    "exposures": ["1x1800 s R500B", "3x1800 s R500R"],
                    "wavelength_range_angstrom": [3570, 9250],
                    "resolving_power": [300, 400],
                },
                "LT_SPRAT": {
                    "modes": ["red", "blue"],
                    "epochs_utc": ["2015-08-16", "2015-08-18", "2016-03-17"],
                    "slit_arcsec": 1.8,
                    "exposures": ["5x600 s red", "5x600 s blue", "5x600 s blue"],
                    "wavelength_range_angstrom": [4000, 8000],
                    "resolving_power_at_6000_angstrom": 350,
                },
            },
            "time_delay": {
                "published_delta_t_ab_days": 211,
                "uncertainty_days": 5,
                "leading_image": "A",
                "source": paper["J1515"],
                "spectral_epochs_can_be_phase_aligned": True,
                "phase_pair": "A on 2015-08-16/18 versus B on 2016-03-17",
            },
            "primary_sources": [paper["J1515"], paper["TDCARMA"]],
            "missing": [
                "per-bin spectral uncertainties",
                "inter-bin and common-mode wavelength covariance",
                "arc-line wavelength-solution products and residuals",
                "slit-centering/flexure covariance needed for a differential centroid",
                "published differential A/B centroid observable with uncertainty",
                "precision lens-model/environment covariance tied to the sealed path predictor",
            ],
            "decisive_block": "PHASE_ALIGNED_SPECTRA_EXIST_BUT_NO_CENTROID_UNCERTAINTY_COVARIANCE_OR_CALIBRATION_RESIDUALS",
        }
    )

    rows.append(
        {
            "name": "SDSS J1620+1203",
            "predictor": predictions["SDSS J1620+1203"],
            "source_status": "SOURCE_BLOCKED_ARCHIVE_AUTH",
            "official_products": smoka_products("FCSA00104303", "FCSA00104304"),
            "archive": {
                "name": "SMOKA/NAOJ",
                "exposure_id": "FCSE00104303",
                "metadata_downloaded_and_hashed": True,
                "raw_fits_downloaded": False,
                "raw_retrieval_rule": "FREE_USER_REGISTRATION_REQUIRED_BY_SMOKA",
            },
            "spectroscopy": {
                "instrument_mode": "Subaru/FOCAS, 300B+SY47, 1.0 arcsec slit, two detector-chip frames",
                "epoch_utc": "2009-02-01T14:50:49.933",
                "exposure_s": 720,
                "wavelength_range_angstrom": [4700, 9100],
                "resolving_power": 400,
                "image_resolution": "faint B spectrum is explicitly contaminated by bright lens galaxy; no machine-readable spectra found",
            },
            "time_delay": {
                "published_delta_t_ab_days": -171.5,
                "uncertainty_days": {"minus": 8.7, "plus": 8.7},
                "source": paper["MILLON"],
                "spectral_epochs_can_be_phase_aligned": False,
            },
            "primary_sources": [paper["J1349_J1455_J1620"], paper["MILLON"]],
            "missing": [
                "SMOKA account authorization for raw FITS retrieval",
                "unblended image-B spectrum",
                "machine-readable image-separated spectra",
                "calibration products/residuals and covariance",
                "second spectral epoch separated by 171.5 days",
                "precision lens-model/environment covariance",
            ],
            "decisive_block": "SMOKA_ACCOUNT_REQUIRED_B_IS_LENS_BLENDED_AND_ONLY_ONE_UNTABLED_EPOCH",
        }
    )
    if [row["name"] for row in rows] != EXPECTED_NAMES:
        raise RuntimeError("source ledger identity/order changed")
    return rows


def verify_downloads(manifest: dict[str, Any]) -> None:
    if manifest["accounting"] != {
        "association_xml_files": 14,
        "confirmation_products_requested": 0,
        "downloaded_bytes": 420288947,
        "downloaded_files": 157,
        "failed_downloads": 0,
        "fits_files_decompressed": 0,
        "requested_download_files": 157,
        "response_scores_computed": 0,
        "spectral_rows_parsed": 0,
        "spectral_values_read": 0,
    }:
        raise RuntimeError("download accounting changed")
    for row in manifest["association_files"] + manifest["downloaded_files"]:
        path = ROOT / row["local_path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"source binding changed: {row['local_path']}")
    if manifest["failures"]:
        raise RuntimeError("download failures retained")


def package_bindings() -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for role, (path, expected) in EXPECTED_BINDINGS.items():
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f"sealed binding changed: {role}")
        observed[role] = {
            "path": path.relative_to(REPO).as_posix(),
            "sha256": digest,
        }
    observed["download_manifest"] = {
        "path": DOWNLOAD_MANIFEST.relative_to(REPO).as_posix(),
        "sha256": sha256(DOWNLOAD_MANIFEST),
    }
    observed["authorization_boundary"] = {
        "path": BOUNDARY.relative_to(REPO).as_posix(),
        "sha256": sha256(BOUNDARY),
    }
    observed["acquisition_script"] = {
        "path": (ROOT / "acquire_public_sources.py").relative_to(REPO).as_posix(),
        "sha256": sha256(ROOT / "acquire_public_sources.py"),
    }
    observed["seal_script"] = {
        "path": Path(__file__).resolve().relative_to(REPO).as_posix(),
        "sha256": sha256(Path(__file__).resolve()),
    }
    return observed


def write_csv(manifest: dict[str, Any]) -> None:
    columns = ["source_class", "product_id", "local_path", "bytes", "sha256", "url"]
    with (ROOT / "downloaded-file-sha256.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in manifest["downloaded_files"]:
            writer.writerow({key: row.get(key, "") for key in columns})


def main() -> None:
    manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    verify_downloads(manifest)
    predictions = load_predictions()
    lenses = source_ledger(predictions)
    ledger = {
        "schema": "invariant-gravity-path-accumulated-weyl-redshift-source-ledger-1.0",
        "package_id": "open-gravity-path-accumulated-weyl-redshift-source-preflight-v1",
        "status": "SEALED_SOURCE_PREFLIGHT_ALL_EIGHT_RESPONSE_BLOCKED_ONE_PHASE_ALIGNED_PARTIAL",
        "lens_count": 8,
        "lenses": lenses,
        "required_scored_observable": {
            "definition": "image-separated differential narrow-line centroid Δv=c Δln(1+z)",
            "required_fields": [
                "image and line identity",
                "wavelength or velocity centroid",
                "uncertainty and covariance",
                "wavelength dependence",
                "observation epoch",
                "time-delay-aligned source phase or a justified static absorber",
                "precision lens/path predictor and covariance",
            ],
            "catalog_redshift_substitution_allowed": False,
        },
        "ordinary_controls_required": [
            "differential extinction and chromatic throughput",
            "microlensing and differential magnification of velocity-resolved source structure",
            "intrinsic source variability convolved with image delay",
            "moving-lens frequency shift",
            "slit centering, IFU/long-slit wavelength calibration, flexure, and atmospheric dispersion",
            "different absorption systems or transverse absorber velocity offsets",
            "lens-galaxy blending and extraction leakage",
            "mass sheet, environment, and lens-model covariance",
        ],
        "claim_boundary": (
            "Source availability only. No spectral row or FITS pixel was decoded; no response "
            "observable was computed; no lens was scored; no theory parameter or source choice was tuned."
        ),
        "conclusion": {
            "ready_to_score_lenses": [],
            "phase_alignable_lenses": ["SDSS J1515+1511"],
            "phase_alignable_but_not_scoreable": ["SDSS J1515+1511"],
            "hard_source_blocked_lenses": [
                "SDSS J0832+0404",
                "SDSS J1226-0006",
                "SDSS J1320+1644",
                "SDSS J1335+0118",
                "SDSS J1349+1227",
                "SDSS J1455+1447",
                "SDSS J1620+1203",
            ],
            "minimum_next_empirical_action": (
                "Obtain at least one wavelength-stable, image-separated, uncertainty-bearing spectrum pair "
                "at delay-aligned source phase, with calibration residuals/covariance and a precision lens/path model."
            ),
        },
        "access_accounting": {
            **manifest["accounting"],
            "paper_text_used_for_source_metadata": True,
            "archive_metadata_rows_read": True,
            "spectral_response_rows_decoded": 0,
            "spectral_response_values_read": 0,
            "confirmation_identities_opened": 0,
            "confirmation_response_rows_opened": 0,
            "formula_or_parameter_tuning_events": 0,
            "paid_calls": 0,
            "model_calls": 0,
        },
        "bindings": package_bindings(),
    }
    ledger["content_sha256"] = content_sha256(ledger)
    ledger_path = ROOT / "per-lens-source-ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(manifest)

    receipt = {
        "schema": "invariant-gravity-path-accumulated-weyl-redshift-source-preflight-receipt-1.0",
        "package_id": ledger["package_id"],
        "status": ledger["status"],
        "decision": "BLOCK_RESPONSE_SCORING_NO_LENS_MEETS_FULL_SOURCE_CONTRACT",
        "ledger": {
            "path": ledger_path.relative_to(REPO).as_posix(),
            "sha256": sha256(ledger_path),
            "content_sha256": ledger["content_sha256"],
        },
        "downloaded_file_ledger": {
            "path": (ROOT / "downloaded-file-sha256.csv").relative_to(REPO).as_posix(),
            "sha256": sha256(ROOT / "downloaded-file-sha256.csv"),
            "files": manifest["accounting"]["downloaded_files"],
            "bytes": manifest["accounting"]["downloaded_bytes"],
        },
        "mechanical_checks": {
            "eight_exploration_lenses_exact": len(lenses) == 8,
            "sealed_v1_bindings_match": True,
            "all_downloaded_bytes_hash_verified": True,
            "all_acquisition_failures_zero": not manifest["failures"],
            "confirmation_products_requested_zero": manifest["accounting"]["confirmation_products_requested"] == 0,
            "spectral_rows_parsed_zero": manifest["accounting"]["spectral_rows_parsed"] == 0,
            "spectral_values_read_zero": manifest["accounting"]["spectral_values_read"] == 0,
            "response_scores_zero": manifest["accounting"]["response_scores_computed"] == 0,
            "only_j1515_phase_alignable": [r["name"] for r in lenses if r["time_delay"].get("spectral_epochs_can_be_phase_aligned")]
            == ["SDSS J1515+1511"],
            "ready_to_score_count_zero": True,
        },
        "claim_boundary": ledger["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    receipt_path = ROOT / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = """# Lane 1 source-only preflight

Status: `SEALED_SOURCE_PREFLIGHT_ALL_EIGHT_RESPONSE_BLOCKED_ONE_PHASE_ALIGNED_PARTIAL`

The public-source search did not produce a scoreable response dataset for any of the eight frozen exploration lenses. Seven are hard source-blocked. SDSS J1515+1511 has a genuinely time-delay-aligned A/B spectral pair in the CDS tables, but the tables contain wavelength and flux only: no per-bin uncertainties, wavelength covariance, calibration residuals, or published differential centroid with uncertainty. It therefore remains source-partial, not scoreable.

Opaque acquisition copied and SHA-256 verified 157 public files (420,288,947 bytes), including 17 ESO science exposures, the complete 135-product FORS1 Raw2Raw union for J1226/J1335, four J1515 spectral tables plus ReadMe, seven primary papers, and seven SMOKA metadata pages. No FITS was decompressed and no spectral row/value was parsed. Subaru raw products are account-gated by SMOKA; J1320 has no exact public spectral product identified. Confirmation systems stayed sealed.

The next honest empirical step is new or recovered image-separated spectroscopy at source phases separated by the measured delay, with wavelength-solution diagnostics, centroid uncertainty/covariance, and a precision lens/path model. Catalog redshifts cannot substitute for that response.
"""
    (ROOT / "SOURCE_PREFLIGHT.md").write_text(summary, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": ledger["status"],
                "decision": receipt["decision"],
                "ledger_sha256": sha256(ledger_path),
                "receipt_sha256": sha256(receipt_path),
                "download_manifest_sha256": sha256(DOWNLOAD_MANIFEST),
                "downloaded_file_ledger_sha256": sha256(ROOT / "downloaded-file-sha256.csv"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
