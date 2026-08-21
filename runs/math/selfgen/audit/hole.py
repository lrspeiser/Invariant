"""Investigate the one ACCEPTED case: real hole, or a vacuous tamper?"""
import copy
import json
import sys

sys.path.insert(0, r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/src")

from sigma_theory_compiler.self_generated_conjecture_widening import (  # noqa: E402
    ConjectureWideningError,
    residue_table,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256  # noqa: E402

GOOD = json.load(open(r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/runs/math/selfgen/widening-receipt-v1.json"))


def reseal(b):
    b["content_sha256"] = canonical_sha256({k: v for k, v in b.items() if k != "content_sha256"})
    return b


r = next(
    x
    for x in GOOD["conjectures"]
    if x["kind"] == "zero_free_over_the_integers"
    and x["adjudication"].get("proof_route") == "local_obstruction"
)
rt = r["adjudication"]["residue_table"]
print("a genuine local-obstruction certificate:")
print("  object:", r["object_id"], "modulus:", rt["modulus"], "mu:", rt["mu"], "lam:", rt["lam"])
print("  values:", rt["values"])
print("  does it already contain a zero?", 0 in rt["values"])
print("  -> my 'scrub zeros' edit replaces nothing; the receipt is byte-identical.")
print("  receipt unchanged by the edit:",
      [v if v != 0 else 1 for v in rt["values"]] == rt["values"])
print()

# Now a NON-vacuous version of the same attack: forge a certificate for an object
# whose orbit mod m genuinely DOES contain a zero, by scrubbing the zeros out.
print("non-vacuous version -- forge a clean table for an object that really hits 0 mod m:")
inp = r["obligation"]["inputs"]
coeffs, init = list(inp["coefficients"]), list(inp["initial"])
forged = None
for m in (2, 3, 4, 5, 7, 8, 9, 11, 13, 16):
    tab = residue_table(coeffs, init, m, cap=200000)
    if tab is not None and 0 in tab.values:
        forged = (m, tab)
        break
if forged is None:
    print("  (this object has no zero-bearing orbit among the declared moduli; using another object)")
    for x in GOOD["conjectures"]:
        i2 = x["obligation"]["inputs"]
        if "coefficients" not in i2:
            continue
        for m in (2, 3, 4, 5, 7, 8, 9, 11, 13, 16):
            tab = residue_table(list(i2["coefficients"]), list(i2["initial"]), m, cap=200000)
            if tab is not None and 0 in tab.values:
                coeffs, init = list(i2["coefficients"]), list(i2["initial"])
                forged = (m, tab)
                break
        if forged:
            break

m, tab = forged
print(f"  object c={coeffs} u0={init}, modulus {m}: true orbit values {list(tab.values)}")
scrubbed = [v if v != 0 else 1 for v in tab.values]
print(f"  scrubbed (zeros -> 1):            {scrubbed}")

b = copy.deepcopy(GOOD)
victim = next(
    x
    for x in b["conjectures"]
    if x["kind"] == "zero_free_over_the_integers"
    and x["adjudication"].get("proof_route") == "local_obstruction"
)
victim["obligation"]["inputs"]["coefficients"] = coeffs
victim["obligation"]["inputs"]["initial"] = init
victim["adjudication"]["certifying_modulus"] = m
victim["adjudication"]["residue_table"] = {
    "modulus": m,
    "order": tab.order,
    "mu": tab.mu,
    "lam": tab.lam,
    "values": scrubbed,
    "closure_state": list(tab.closure_state),
}
# reseal the obligation too, so only the mathematics is left to catch it
ob = victim["obligation"]
sealed = {k: v for k, v in ob.items() if k != "obligation_sha256"}
ob["obligation_sha256"] = canonical_sha256(sealed)
try:
    validate_receipt(reseal(b))
except ConjectureWideningError as exc:
    print(f"  -> REJECTED: {exc}")
except Exception as exc:  # noqa: BLE001
    print(f"  -> REJECTED({type(exc).__name__}): {exc}")
else:
    print("  -> *** ACCEPTED: REAL HOLE ***")

# And the honest single-value corruption of a real certificate
print("\nsingle-value corruption of a genuine certificate (values[0] += 1):")
b2 = copy.deepcopy(GOOD)
v2 = next(
    x
    for x in b2["conjectures"]
    if x["kind"] == "zero_free_over_the_integers"
    and x["adjudication"].get("proof_route") == "local_obstruction"
)
v2["adjudication"]["residue_table"]["values"][0] += 1
try:
    validate_receipt(reseal(b2))
except ConjectureWideningError as exc:
    print(f"  -> REJECTED: {exc}")
else:
    print("  -> *** ACCEPTED: REAL HOLE ***")
