"""register.py -- register this run in the wellnet run registry."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402
RUN_ID = "BS-clusterjoint"
NOTE = ("Baryons and gravity on the SAME radial x azimuthal cells for 52 clusters "
        "of the OPEN half: DECADE per-source shear one side, Chandra ACIS photons "
        "the other, nothing azimuthally averaged. 2080 rows, 1201 usable. Asks "
        "whether baryon ASYMMETRY predicts lensing beyond radius. Answer at this "
        "depth: NO -- gain +0.0146 against an azimuthal-permutation null of "
        "-0.0091 +- 0.0144, z=+1.64, p=0.055. NINTH ARTEFACT CAUGHT AND IT WAS "
        "OURS: at 40 permutations the same data gave z=+2.67, p=0.000 and a "
        "'detection'; at 200 the null widened from sd 0.0095 to 0.0144 and the "
        "signal vanished. Undersized null, nothing else. Two controls passed: "
        "n_phot alone carries nothing (z=+0.63) so the gain is not a brightness "
        "proxy, and the g_x parity-odd target is null (z=-0.28) so it is not cell "
        "geometry or a shear systematic. Limiting factor measured: median S/N per "
        "cell 0.78. Sealed half never queried.")
if __name__ == "__main__":
    rec = registry.register(RUN_ID, "clusterjoint",
        depends_on=("holdout_seal", "confirmation_status", "catalogue_validation"),
        outputs=("work/wellnet-2026-09/clusterjoint/table.csv",
                 "work/wellnet-2026-09/clusterjoint/learn.json",
                 "work/wellnet-2026-09/clusterjoint/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"])
