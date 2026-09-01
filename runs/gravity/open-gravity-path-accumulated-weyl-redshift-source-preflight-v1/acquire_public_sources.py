"""Acquire and hash only frozen Lane-1 public source products.

The downloader never decompresses FITS, parses FITS, or reads tabular spectral rows.
It streams exact bytes to disk and records transport metadata and SHA-256 digests.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sources"
MANIFEST_PATH = ROOT / "download-manifest.json"

J0832_SCIENCE = [
    "EFOSC.2005-12-31T06:26:10.175",
    "EFOSC.2005-12-31T06:36:44.141",
    "EFOSC.2005-12-31T06:47:18.079",
]

J1226_SCIENCE = [
    "FORS1.2005-05-16T01:01:39.042",
    "FORS1.2005-05-16T01:27:07.535",
    "FORS1.2005-05-16T01:56:46.427",
    "FORS1.2005-05-16T02:22:14.919",
    "FORS1.2005-05-16T03:16:03.097",
    "FORS1.2005-05-16T03:41:31.599",
    "FORS1.2005-05-16T04:10:13.347",
    "FORS1.2005-05-16T04:35:41.758",
]

J1335_SCIENCE = [
    "FORS1.2005-02-03T08:12:22.553",
    "FORS1.2005-02-03T08:37:50.882",
    "FORS1.2005-03-03T07:39:57.422",
    "FORS1.2005-03-03T08:05:25.946",
    "FORS1.2005-03-03T08:34:10.974",
    "FORS1.2005-03-03T08:59:40.116",
]

PAPERS = {
    "inada2008_j0832.pdf": "https://arxiv.org/pdf/0708.0871",
    "eigenbrod2006_j1226_j1335.pdf": "https://arxiv.org/pdf/astro-ph/0511026",
    "rusu2013_j1320.pdf": "https://arxiv.org/pdf/1206.2011",
    "kayo2010_j1349_j1455_j1620.pdf": "https://arxiv.org/pdf/0912.1462",
    "shalyapin2017_j1515.pdf": "https://arxiv.org/pdf/1701.04272",
    "millon2020_cosmograil_delays.pdf": "https://arxiv.org/pdf/2002.05736",
    "meyer2022_tdcarma_delays.pdf": "https://arxiv.org/pdf/2207.09327",
}

CDS = {
    "ReadMe": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/ReadMe",
    "table2.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table2.dat",
    "table3.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table3.dat",
    "table4.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table4.dat",
    "table5.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/14/table5.dat",
}

SMOKA_METADATA = {
    "howto_search.html": "https://smoka.nao.ac.jp/help/howto_search.jsp",
    "FCSA00104201.html": "https://smoka.nao.ac.jp/info.jsp?frameid=FCSA00104201&date_obs=2009-02-01&i=0",
    "FCSA00104202.html": "https://smoka.nao.ac.jp/info.jsp?frameid=FCSA00104202&date_obs=2009-02-01&i=0",
    "FCSA00104251.html": "https://smoka.nao.ac.jp/info.jsp?frameid=FCSA00104251&date_obs=2009-02-01&i=0",
    "FCSA00104252.html": "https://smoka.nao.ac.jp/info.jsp?frameid=FCSA00104252&date_obs=2009-02-01&i=0",
    "FCSA00104303.html": "https://smoka.nao.ac.jp/info.jsp?frameid=FCSA00104303&date_obs=2009-02-01&i=0",
    "FCSA00104304.html": "https://smoka.nao.ac.jp/info.jsp?frameid=FCSA00104304&date_obs=2009-02-01&i=0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_with_retry(url: str, *, stream: bool = False) -> requests.Response:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=120, stream=stream)
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as error:
            last = error
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed after retries: {url}") from last


def download(url: str, directory: Path, fallback_name: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    safe_fallback = re.sub(r'[<>:"/\\|?*]', "-", fallback_name)
    existing = sorted(directory.glob(f"{safe_fallback}*"))
    if len(existing) == 1 and existing[0].is_file():
        path = existing[0]
        return {
            "url": url,
            "local_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "content_type": None,
            "content_disposition": None,
            "payload_decoded": False,
        }
    response = request_with_retry(url, stream=True)
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^";]+)', disposition)
    filename = match.group(1) if match else fallback_name
    filename = re.sub(r'[<>:"/\\|?*]', "-", filename)
    path = directory / filename
    if path.is_file():
        response.close()
        return {
            "url": url,
            "local_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "content_type": response.headers.get("Content-Type"),
            "content_disposition": disposition or None,
            "payload_decoded": False,
        }
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    return {
        "url": url,
        "local_path": path.relative_to(ROOT).as_posix(),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "content_type": response.headers.get("Content-Type"),
        "content_disposition": disposition or None,
        "payload_decoded": False,
    }


def acquire_associations() -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    products: dict[str, str] = {}
    association_dir = SOURCE_DIR / "eso" / "associations"
    association_dir.mkdir(parents=True, exist_ok=True)
    for science_id in J1226_SCIENCE + J1335_SCIENCE:
        url = (
            "https://archive.eso.org/calselector/v1/associations"
            f"?dp_id={science_id}&mode=raw2raw"
        )
        response = request_with_retry(url)
        root = ET.fromstring(response.content)
        if root.attrib.get("mode") != "Raw2Raw" or root.attrib.get("complete") != "true":
            raise RuntimeError(f"incomplete Raw2Raw association: {science_id}")
        safe_science_id = science_id.replace(":", "-")
        path = association_dir / f"{safe_science_id}.raw2raw.xml"
        path.write_bytes(response.content)
        rows.append(
            {
                "science_product_id": science_id,
                "url": url,
                "local_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "association_category": root.attrib.get("category"),
                "complete": True,
                "payload_decoded": "XML_METADATA_ONLY",
            }
        )
        for element in root.findall(".//mainFiles/file"):
            products[element.attrib["name"]] = element.attrib["category"]
    return rows, products


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    association_rows, associated_products = acquire_associations()

    tasks: list[tuple[str, str, Path, str, dict[str, Any]]] = []
    for product_id in J0832_SCIENCE:
        tasks.append(
            (
                "ESO_J0832_SCIENCE",
                f"https://dataportal.eso.org/dataPortal/file/{product_id}",
                SOURCE_DIR / "eso" / "J0832",
                product_id,
                {"product_id": product_id, "category": "SCIENCE"},
            )
        )
    for product_id, category in sorted(associated_products.items()):
        target = "J1226_J1335_SHARED" if category != "SCIENCE_MOS" else (
            "J1226" if product_id in J1226_SCIENCE else "J1335"
        )
        tasks.append(
            (
                "ESO_FORS1_RAW2RAW",
                f"https://dataportal.eso.org/dataPortal/file/{product_id}",
                SOURCE_DIR / "eso" / target,
                product_id,
                {"product_id": product_id, "category": category},
            )
        )
    for name, url in PAPERS.items():
        tasks.append(("PRIMARY_PAPER", url, SOURCE_DIR / "papers", name, {}))
    for name, url in CDS.items():
        tasks.append(("CDS_J1515", url, SOURCE_DIR / "cds" / "J1515", name, {}))
    for name, url in SMOKA_METADATA.items():
        tasks.append(("SMOKA_METADATA", url, SOURCE_DIR / "smoka", name, {}))

    downloaded: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def worker(task: tuple[str, str, Path, str, dict[str, Any]]) -> dict[str, Any]:
        source_class, url, directory, fallback_name, extra = task
        row = download(url, directory, fallback_name)
        return {"source_class": source_class, **extra, **row}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(worker, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            try:
                downloaded.append(future.result())
            except Exception as error:  # retain every acquisition failure
                errors.append(
                    {
                        "source_class": task[0],
                        "url": task[1],
                        "fallback_name": task[3],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    manifest = {
        "schema": "invariant-gravity-path-accumulated-weyl-redshift-download-manifest-1.0",
        "package_id": "open-gravity-path-accumulated-weyl-redshift-source-preflight-v1",
        "transport_rule": "BYTE_COPY_AND_SHA256_ONLY_NO_FITS_DECOMPRESSION_NO_SPECTRAL_ROW_PARSE",
        "association_files": sorted(
            association_rows, key=lambda row: row["science_product_id"]
        ),
        "downloaded_files": sorted(downloaded, key=lambda row: row["local_path"]),
        "failures": sorted(errors, key=lambda row: row["url"]),
        "accounting": {
            "association_xml_files": len(association_rows),
            "requested_download_files": len(tasks),
            "downloaded_files": len(downloaded),
            "failed_downloads": len(errors),
            "downloaded_bytes": sum(row["bytes"] for row in downloaded),
            "spectral_rows_parsed": 0,
            "spectral_values_read": 0,
            "fits_files_decompressed": 0,
            "confirmation_products_requested": 0,
            "response_scores_computed": 0,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["accounting"], indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
