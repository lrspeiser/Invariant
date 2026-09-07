"""register_bu.py -- register the modification search in the wellnet registry."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402

RUN_ID = "BU-modifications"
NOTE = (
    "Extended the gas density from the Chandra events already on disk out to "
    "0.94-4.6 Mpc: beta-model fits to the 0.5-2 keV surface brightness, anchored "
    "to ACCEPT where the two overlap (0.04-0.19 dex, 21-59 anchor bins each). One "
    "cluster REJECTED for a degenerate fit at Rc=0 kpc, n0=1.4e5 cm^-3 -- five "
    "orders of magnitude above any cluster core, reproducing the surface "
    "brightness while being physically impossible. Ten survive, taking Run BT's 15 "
    "usable points to 65 rows and sharpening the residual from 0.456 at 0.65 dex "
    "to 0.620 at 0.52 dex. "
    "Then seven one-parameter modification families multiplying the RAR: constant "
    "slip, a0 shift, density^p, temperature^p, radius^p, (1+z)^p, gas mass^p -- the "
    "axes this programme has spent eighteen months on. EVERY ONE drives the median "
    "to within a few percent of unity; NOT ONE reduces the scatter below 0.52 dex "
    "and four make it worse. Each refitted 300x with its variable permuted across "
    "clusters; none beats its own null. The discriminator is the design: anything "
    "with a free amplitude fixes a median, so only a scatter reduction is evidence "
    "of a real dependence. "
    "CONCLUSION: the cluster residual is an AMPLITUDE, not a dependence. It tracks "
    "neither local gas density, temperature, position, redshift nor gas mass. "
    "Consistent with a roughly universal missing-mass fraction; inconsistent with "
    "the environmental modifications tested. Binding limits, both stated: the 0.52 "
    "dex scatter (shape noise over 6,974-25,199 sources per cluster split ten "
    "ways), and the analytic ACIS vignetting model standing in for CIAO exposure "
    "maps, which biases the outer surface brightness, hence beta, hence the outer "
    "density. Sealed half never queried.")

if __name__ == "__main__":
    rec = registry.register(
        RUN_ID, "clusterfirst",
        depends_on=("holdout_seal", "confirmation_status", "temperature_support"),
        outputs=("work/wellnet-2026-09/clusterfirst/modifications.json",
                 "work/wellnet-2026-09/clusterfirst/gas_extended.json",
                 "work/wellnet-2026-09/clusterfirst/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"])
