#!/usr/bin/env bash
# Run BL -- extraction.  Order matters: register, test, generate, size-then-locate,
# counterfactuals, closure, distil, certify, render.
set -e
cd "$(dirname "$0")"
python ../registry/registry.py                              # the run must be registered
python test_lane.py                                         # tests before results
N_MAIN=${N_MAIN:-1000} N_HELD=${N_HELD:-400} N_SCAN=${N_SCAN:-300} N_CF=${N_CF:-300} N_G2=${N_G2:-600} \
    python run_generate.py                                  # every paired pool
python run_discriminate.py                                  # sizing, discriminator, ablations, scans, transfer
python run_counterfactual.py                                # dO/dB
python run_closure.py                                       # P(G|B)
python distil.py                                            # 1-3 sparse invariants
python certify.py                                           # Stage 4 certificates
python write_report.py                                      # REPORT.md, rendered from the JSONs
