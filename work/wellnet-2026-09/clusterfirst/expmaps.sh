#!/usr/bin/env bash
# expmaps.sh -- real Chandra exposure maps, replacing the analytic vignetting guess.
#
# WHY. Every result in this lane is limited by one approximation: with no CIAO
# available, `extend_gas.py` corrected ACIS vignetting with
#     V(theta) = (1 + (theta/11')^2)^-0.8
# which reproduces the 1 keV falloff to about 10% and nothing else. That
# approximation sets the outer surface brightness, which sets the beta-model
# slope, which sets the enclosed gas mass, which IS the prediction. Run BW's
# error budget put a conservative 15% on it and that alone moved the
# universality test from p = 0.009 to p = 0.086.
#
# A real exposure map is not an approximation of the same kind. It is computed
# per observation from the aspect solution, the instrument map and the CALDB
# effective area, so it carries the actual dither, chip gaps, bad pixels, the
# time-dependent contamination layer and the energy-weighted vignetting for the
# band being used.
#
# WHAT THIS PRODUCES, per observation:
#     <root>_0.5-2.0_thresh.img     counts, 0.5-2 keV, binned
#     <root>_0.5-2.0_thresh.expmap  exposure map, cm^2 s, same grid
# and the surface brightness is then sum(counts)/sum(expmap) per annulus, which
# is exposure-weighted correctly rather than divided by a guessed curve.
#
# Runs inside WSL. CIAO does not run natively on Windows.
#
#   wsl.exe -e bash <this script>

# NOT `set -u`: CIAO's own activation hook references IPYTHONDIR without a
# default and aborts under it. Declaring the variable first is not enough
# because the hook is sourced by micromamba, so the safe order is to define it
# and leave unset-checking off for the activation.
export IPYTHONDIR="${IPYTHONDIR:-$HOME/.ipython}"
MAMBA_ROOT_PREFIX="$HOME/micromamba"
export MAMBA_ROOT_PREFIX
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate ciao

WIN_LANE="/mnt/c/Users/henry/Documents/Codex/2026-09-04/pu-2/work/Invariant/work/wellnet-2026-09"
RAW="$WIN_LANE/clusterxray/raw"
WORK="$HOME/ciao-work"
OUTDIR="$WIN_LANE/clusterfirst/expmaps"
mkdir -p "$WORK" "$OUTDIR"

echo "CIAO: $(punlearn dmlist 2>/dev/null; ciaover 2>/dev/null | head -2)"
echo "CALDB: ${CALDB:-unset}"
echo

# The eleven clusters that carry gas + shear. Only their observations are
# processed; the other 41 overlap clusters are not needed for this test.
CLUSTERS=$(python3 - <<'PY'
import json, io, os
p = "/mnt/c/Users/henry/Documents/Codex/2026-09-04/pu-2/work/Invariant/work/wellnet-2026-09/clusterfirst/accept_overlap.json"
c = "/mnt/c/Users/henry/Documents/Codex/2026-09-04/pu-2/work/Invariant/work/wellnet-2026-09/clusterxray/overlap_centres.json"
match = json.load(io.open(p))
centres = json.load(io.open(c))
key = {v[6]: k for k, v in centres.items()}
print(" ".join(key[m["erass"]] for m in match if m["erass"] in key))
PY
)

n_ok=0; n_fail=0
for CL in $CLUSTERS; do
  for EVT in "$RAW"/${CL}_*evt2*; do
    [ -e "$EVT" ] || continue
    BASE=$(basename "$EVT" | sed 's/\.fits\.gz$//;s/\.fits$//')
    if [ -e "$OUTDIR/${BASE}_0.5-2.0_thresh.expmap" ]; then
      echo "  $BASE  cached"; n_ok=$((n_ok+1)); continue
    fi
    # work on WSL's own filesystem: /mnt/c is slow and CIAO writes a lot
    cp "$EVT" "$WORK/" 2>/dev/null
    LOCAL="$WORK/$(basename "$EVT")"

    # Stage the aspect solution under its ORIGINAL name. fluximage reads the
    # ASOLFILE keyword out of the event header and looks for exactly that
    # filename, so the cluster prefix this lane adds for tidiness has to come
    # back off. Without it fluximage stops with
    #   ERROR ASOLFILE=pcadf....asol1.fits not found
    # which is the whole reason the first batch produced nothing.
    OBSID=$(echo "$BASE" | sed 's/.*acisf0*\([0-9]\+\)N.*/\1/')
    for A in "$RAW"/${CL}_pcadf*asol1.fits*; do
      [ -e "$A" ] || continue
      ORIG=$(basename "$A" | sed "s/^${CL}_//")
      case "$ORIG" in
        *"$(printf '%05d' "$OBSID")"*|*"$OBSID"*) cp -f "$A" "$WORK/$ORIG" 2>/dev/null ;;
      esac
    done
    # CIAO reads gzipped FITS transparently, but the header names the file
    # without .gz; provide both spellings.
    ( cd "$WORK" && for G in pcadf*asol1.fits.gz; do
        [ -e "$G" ] && [ ! -e "${G%.gz}" ] && gunzip -kf "$G" 2>/dev/null
      done; true )

    ( cd "$WORK" && \
      punlearn fluximage && \
      fluximage infile="$LOCAL" outroot="$WORK/$BASE" \
        bands=0.5:2.0:1.0 binsize=4 psfecf=0.9 clobber=yes mode=h > "$WORK/$BASE.log" 2>&1 )
    if [ -e "$WORK/${BASE}_0.5-2.0_thresh.expmap" ]; then
      cp "$WORK/${BASE}_0.5-2.0_thresh.img" "$WORK/${BASE}_0.5-2.0_thresh.expmap" "$OUTDIR/" 2>/dev/null
      echo "  $BASE  OK"; n_ok=$((n_ok+1))
    else
      echo "  $BASE  FAILED -- $(tail -2 "$WORK/$BASE.log" 2>/dev/null | tr '\n' ' ' | cut -c1-140)"
      n_fail=$((n_fail+1))
    fi
    rm -f "$LOCAL"
  done
done

echo
echo "exposure maps: $n_ok made or cached, $n_fail failed"
echo "written to $OUTDIR"
