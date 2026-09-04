#!/usr/bin/env bash
# Run BH -- cdm-separation.  Order matters: size first, then power, then verdicts.
set -e
cd "$(dirname "$0")"
python ../registry/registry.py                       # the run must be registered
python test_lane.py                                  # tests before results
N_HALF=${N_HALF:-1000} python run_power.py           # sizing, power, rates on CDM
N_G45=${N_G45:-300}   python run_g45.py              # galaxy misspecified-axis control
N_DIAG=${N_DIAG:-400} python run_mechanism.py        # Job 1: the mechanism
python run_library_axis.py                           # the shared library's own axis alignment
N_F=${N_F:-500}       python run_forward.py          # inverse-crime control + alignment scan
N_C6=${N_C6:-400}     python run_c6.py               # out-of-grammar injection (C6)
N_J=${N_J:-500}       python run_joint_scan.py       # the joint procedure across the scan
N_NG=${N_NG:-200}     python run_ngal.py             # sqrt(N_gal) law for the galaxy channel
python certify.py                                    # Stage 4 certificates
python write_report.py                               # REPORT.md, rendered from the JSONs
