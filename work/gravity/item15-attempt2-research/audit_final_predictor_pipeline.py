"""Exercise the exact final predictor pipeline without writing frozen artifacts."""

from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler import gravity_item15_accept_lc2_timescale_ratios as pipeline


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    config = pipeline.load_config(ROOT)
    payload = pipeline._fetch(config["sources"]["accept"]["metadata_url"], attempts=4)
    metadata = {
        row["accept_name"]: row for row in pipeline.parse_accept_metadata(payload)
    }
    passing = 0
    for accept_name, lc2_name, author, bibcode in config["sample"]["eligible_lc2_rows"]:
        profile_payload, _ = pipeline._profile_payload(ROOT, accept_name, config)
        values = pipeline.derive_cluster_features(
            pipeline.parse_accept_profile(profile_payload),
            metadata[accept_name],
            {"lc2_name": lc2_name, "lc2_author": author, "lc2_bibcode": bibcode},
            config,
        )
        passing += 1
        print(
            f"{accept_name}\ttcool20={values['tcool20_gyr']:.6g}"
            f"\ttff20={values['tff_baryon20_gyr']:.6g}"
            f"\ttsound20={values['tsound20_gyr']:.6g}"
        )
    print(f"PASS predictor_objects={passing} lc2_mass_response_rows=0")


if __name__ == "__main__":
    main()
