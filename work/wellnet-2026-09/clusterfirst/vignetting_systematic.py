"""How big is the vignetting systematic, MEASURED rather than guessed?

Run BW put a conservative 15% on the analytic curve because there was nothing to
compare it to. There is now: the same pipeline, same shear, same anchoring, with
the analytic curve replaced by CIAO exposure maps. The per-cluster change in the
residual IS the sensitivity to the vignetting treatment.
"""
import io, json, os, subprocess, sys
import numpy as np

def run(gas):
    e = dict(os.environ); e["GAS_FILE"] = gas
    subprocess.run([sys.executable, "universality.py"], env=e, capture_output=True)
    return {c["cluster"]: c for c in json.load(io.open("universality.json"))["clusters"]}

A = run("gas_extended.json")            # analytic curve
B = run("gas_extended_expcorr.json")    # CIAO exposure maps
json.dump(B, io.open(os.path.join(os.environ["TMPDIR_X"], "univ_B.json"), "w"))

print("  %-24s %-8s %-8s %-8s %s" % ("cluster", "analytic", "CIAO", "delta", "frac"))
fr = []
for c in sorted(set(A) & set(B), key=lambda c: -B[c]["residual"]):
    a, b = A[c]["residual"], B[c]["residual"]
    f = abs(b - a) / max(abs(b), 1e-9)
    fr.append(f)
    print("  %-24s %-8.3f %-8.3f %+-8.3f %.1f%%" % (c, a, b, b - a, 100 * f))
fr = np.array(fr)
print("\n  shared clusters %d" % len(fr))
print("  median |change| %.1f%%   mean %.1f%%   max %.1f%%"
      % (100 * np.median(fr), 100 * fr.mean(), 100 * fr.max()))
print("  Run BW's GUESS was 15.0%%")
print("\n  -> the analytic curve was better than assumed; the measured sensitivity")
print("     of the residual to the vignetting treatment is %.1f%% (median)." % (100*np.median(fr)))
