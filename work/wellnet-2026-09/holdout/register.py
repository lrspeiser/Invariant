"""register.py -- register the seal in the wellnet run registry."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402

RUN_ID = "BP-holdout-seal"
NOTE = ("Gold-cluster holdout seal. Splits Run BO's 605 clusters with a public "
        "raw weak-lensing channel into 304 SEALED and 301 OPEN, stratified by "
        "terciles of z x CTS500 x shear depth, assigned by keyed sha256 with no "
        "random state, and committed by digest. Made BEFORE any shape, response "
        "or per-source redshift was read from any cluster in the pool, which is "
        "what makes the sealed half a CONFIRMATION set rather than validation. "
        "Enforcement is by IDENTITY at loader.py, not by path token: the sealed "
        "half is rows in a shared public archive, so provenance.py's filename "
        "matching cannot see it. Public catalogue metadata stays readable for "
        "sealed clusters; the OUTCOME needs a one-shot token whose reason is "
        "recorded before use. CI gate: .github/workflows/holdout-seal.yml runs "
        "test_seal.py on every PR and fails if a token has been spent. "
        "Opens no data and computes no statistic.")

if __name__ == "__main__":
    rec = registry.register(
        RUN_ID, "holdout",
        depends_on=("holdout_seal", "confirmation_status",
                    "catalogue_validation"),
        outputs=("work/wellnet-2026-09/holdout/holdout_split.json",
                 "work/wellnet-2026-09/holdout/SEAL.md",
                 "work/wellnet-2026-09/holdout/loader.py",
                 "work/wellnet-2026-09/holdout/tests.json"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"], rec["depends_on"])
