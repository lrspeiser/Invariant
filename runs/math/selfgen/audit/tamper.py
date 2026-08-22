"""Adversarial battery: does the verifier actually REJECT invalid objects?
Every case below SHOULD raise. A case that passes silently is a hole."""
import copy
import json
import sys

sys.path.insert(0, r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/src")

from sigma_theory_compiler.self_generated_conjecture_widening import (  # noqa: E402
    ConjectureWideningError,
    WideningConfig,
    adjudicate_zero_free,
    positivity_certificate,
    recheck_zero_free_certificate,
    residue_table,
    validate_receipt,
    validate_residue_table,
)

RECEIPT = r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/runs/math/selfgen/widening-receipt-v1.json"
GOOD = json.load(open(RECEIPT))

results = []


def case(name, fn):
    try:
        fn()
    except ConjectureWideningError as exc:
        results.append((name, "REJECTED", str(exc)[:90]))
    except Exception as exc:  # noqa: BLE001
        results.append((name, f"REJECTED({type(exc).__name__})", str(exc)[:90]))
    else:
        results.append((name, "*** ACCEPTED -- HOLE ***", ""))


# 0. baseline: the real receipt must validate
try:
    validate_receipt(GOOD)
    print("baseline: pristine receipt VALIDATES\n")
except Exception as exc:  # noqa: BLE001
    print("baseline FAILED:", exc)
    raise SystemExit(1)

# --- receipt-level tampering ---
def t_hash():
    b = copy.deepcopy(GOOD)
    b["counts"]["adjudicated"] = 383
    validate_receipt(b)


def t_float():
    b = copy.deepcopy(GOOD)
    b["counts"]["triage"]["open"] = 0.0
    b["content_sha256"] = __import__(
        "sigma_theory_compiler.canonical_json", fromlist=["canonical_sha256"]
    ).canonical_sha256({k: v for k, v in b.items() if k != "content_sha256"})
    validate_receipt(b)


def t_claim_flip():
    b = copy.deepcopy(GOOD)
    b["claims"]["prior_art_absence_establishes_novelty"] = True
    from sigma_theory_compiler.canonical_json import canonical_sha256

    b["content_sha256"] = canonical_sha256({k: v for k, v in b.items() if k != "content_sha256"})
    validate_receipt(b)


def t_census_short():
    """Census that visited fewer than its declared box -- the 'skipped orbits' failure mode."""
    b = copy.deepcopy(GOOD)
    from sigma_theory_compiler.canonical_json import canonical_sha256

    sw = b["skolem_census_ladder"][2]
    sw["visited"] = 6000
    sw["counts"]["has_zero_with_explicit_witness"] = 5104
    sw["counts"]["total"] = 6000
    b["content_sha256"] = canonical_sha256({k: v for k, v in b.items() if k != "content_sha256"})
    validate_receipt(b)


def t_census_unsettled_hidden():
    """Move unsettled members into the zero-free bucket without a certificate."""
    b = copy.deepcopy(GOOD)
    from sigma_theory_compiler.canonical_json import canonical_sha256

    sw = b["skolem_census_ladder"][2]
    sw["counts"]["zero_free_with_modulus_certificate"] += 36
    sw["counts"]["unsettled"] = 0
    sw["triage"]["open"]["count"] = 0
    sw["triage"]["proved_and_prior_art_not_found"]["count"] += 36
    b["content_sha256"] = canonical_sha256({k: v for k, v in b.items() if k != "content_sha256"})
    validate_receipt(b)


case("receipt: adjudicated count bumped 382->383", t_hash)
case("receipt: float 0.0 injected, hash resealed", t_float)
case("receipt: novelty claim flipped to True, hash resealed", t_claim_flip)
case("census: visited < declared box (skipped members)", t_census_short)
case("census: 36 unsettled relabelled zero-free, hash resealed", t_census_unsettled_hidden)

# --- certificate-level tampering ---
ZF = [
    r
    for r in GOOD["conjectures"]
    if r["kind"] == "zero_free_over_the_integers"
    and r["adjudication"].get("proof_route") == "local_obstruction"
]
PZ = [
    r
    for r in GOOD["conjectures"]
    if r["kind"] == "zero_free_over_the_integers"
    and r["adjudication"].get("proof_route") == "positivity_induction"
]
print(f"zero-free certs available: local_obstruction={len(ZF)} positivity={len(PZ)}\n")


def t_lo_lam():
    r = copy.deepcopy(ZF[0])
    inp = r["obligation"]["inputs"]
    r["adjudication"]["residue_table"]["lam"] += 1
    recheck_zero_free_certificate(inp["coefficients"], inp["initial"], r["adjudication"])


def t_lo_values():
    """Delete the zero-bearing evidence: shorten the table so it looks zero-free."""
    r = copy.deepcopy(ZF[0])
    inp = r["obligation"]["inputs"]
    rt = r["adjudication"]["residue_table"]
    rt["values"] = rt["values"][:-1]
    recheck_zero_free_certificate(inp["coefficients"], inp["initial"], r["adjudication"])


def t_lo_wrong_object():
    """Attach a valid certificate to a DIFFERENT object that really does have a zero."""
    r = copy.deepcopy(ZF[0])
    recheck_zero_free_certificate([1, 1], [1, -1], r["adjudication"])  # Fibonacci-like, u(2)=0


def t_lo_swap_modulus():
    """Claim a modulus whose orbit genuinely contains a zero."""
    r = copy.deepcopy(ZF[0])
    inp = r["obligation"]["inputs"]
    tab = residue_table(inp["coefficients"], inp["initial"], 2, cap=200000)
    if tab is None or 0 not in tab.values:
        raise ConjectureWideningError("setup: need a modulus whose orbit has a zero")
    r["adjudication"]["certifying_modulus"] = 2
    r["adjudication"]["residue_table"] = {
        "modulus": 2,
        "order": tab.order,
        "mu": tab.mu,
        "lam": tab.lam,
        "values": list(tab.values),
        "closure_state": list(tab.closure_state),
    }
    recheck_zero_free_certificate(inp["coefficients"], inp["initial"], r["adjudication"])


def t_pos_coeffs():
    r = copy.deepcopy(PZ[0])
    inp = r["obligation"]["inputs"]
    r["adjudication"]["positivity_certificate"]["class_proofs"][0]["twisted_coefficients"] = [9, 9]
    recheck_zero_free_certificate(inp["coefficients"], inp["initial"], r["adjudication"])


def t_pos_drop_class():
    """Drop a residue class -- the 'silently skips orbits' failure mode, in miniature."""
    r = copy.deepcopy(PZ[0])
    inp = r["obligation"]["inputs"]
    cert = r["adjudication"]["positivity_certificate"]
    if cert["decimation"] < 2:
        raise ConjectureWideningError("setup: need decimation >= 2")
    cert["class_proofs"] = cert["class_proofs"][:-1]
    recheck_zero_free_certificate(inp["coefficients"], inp["initial"], r["adjudication"])


def t_pos_dup_class():
    """Cover class 0 twice and class k never, keeping the count right."""
    r = copy.deepcopy(PZ[0])
    inp = r["obligation"]["inputs"]
    cert = r["adjudication"]["positivity_certificate"]
    if cert["decimation"] < 2:
        raise ConjectureWideningError("setup: need decimation >= 2")
    cert["class_proofs"][-1] = copy.deepcopy(cert["class_proofs"][0])
    recheck_zero_free_certificate(inp["coefficients"], inp["initial"], r["adjudication"])


def t_pos_base_case():
    r = copy.deepcopy(PZ[0])
    inp = r["obligation"]["inputs"]
    cp = r["adjudication"]["positivity_certificate"]["class_proofs"][0]
    cp["induction_start"] = cp["induction_start"] + 1
    recheck_zero_free_certificate(inp["coefficients"], inp["initial"], r["adjudication"])


case("cert LO: lam incremented by 1", t_lo_lam)
case("cert LO: last table value deleted", t_lo_values)
case("cert LO: valid cert reattached to an object with a zero", t_lo_wrong_object)
case("cert LO: certifying modulus swapped for one whose orbit has a zero", t_lo_swap_modulus)
case("cert POS: twisted coefficients replaced by [9,9]", t_pos_coeffs)
case("cert POS: one residue class dropped", t_pos_drop_class)
case("cert POS: class 0 duplicated so a class is never covered", t_pos_dup_class)
case("cert POS: induction start shifted off its base case", t_pos_base_case)

print(f"{'case':<62} {'outcome':<26} detail")
print("-" * 120)
holes = 0
for n, o, d in results:
    if "HOLE" in o:
        holes += 1
    print(f"{n:<62} {o:<26} {d}")
print()
print("HOLES FOUND:", holes)

# --- soundness the other way: objects that genuinely HAVE a zero must be REFUTED, never PROVED ---
print("\n--- live adjudication of objects with a genuine zero ---")
cfg = WideningConfig()
planted = [
    ([1, 1], [1, -1]),      # u: 1,-1,0
    ([2, -1], [3, 2]),      # u: 3,2,1,0
    ([1, 0], [5, 0]),       # u(1)=0
    ([0, 1], [1, 0]),       # u: 1,0,1,0,...
    ([3, -2], [1, 1]),      # u: 1,1,1,... never 0 -> must NOT be refuted
    ([-1, -1], [-1, -1]),   # period 3, never 0 -> must NOT be refuted
]
for c, u0 in planted:
    vals = []
    d = len(c)
    v = list(u0)
    while len(v) < 60:
        v.append(sum(c[i] * v[-1 - i] for i in range(d)))
    truth_zero = next((n for n, x in enumerate(v[:60]) if x == 0), None)
    rec = {"coefficients": c, "initial": u0, "values": v[: cfg.window]}
    _, adj = adjudicate_zero_free(rec, cfg)
    verdict = adj["verdict"]
    expect = "REFUTED" if truth_zero is not None else "PROVED/OPEN"
    ok = (verdict == "REFUTED") == (truth_zero is not None)
    print(f"  c={c} u0={u0}  true zero at n={truth_zero}  verdict={verdict:<8} expect={expect:<12} {'OK' if ok else '*** WRONG ***'}")

# --- can positivity_certificate ever certify something with a zero? brute sweep ---
print("\n--- brute sweep: does positivity_certificate ever certify a sequence that has a zero? ---")
from itertools import product  # noqa: E402

bad = 0
checked = 0
for d in (2, 3):
    for c in product(range(-2, 3), repeat=d):
        for u0 in product(range(-2, 3), repeat=d):
            v = list(u0)
            while len(v) < 200:
                v.append(sum(c[i] * v[-1 - i] for i in range(d)))
            has_zero = any(x == 0 for x in v[:200])
            cert = positivity_certificate(c, u0, max_start=12, window=24)
            checked += 1
            if cert is not None and has_zero:
                bad += 1
                if bad <= 5:
                    print(f"  *** c={c} u0={u0} certified zero-free but u has a zero")
print(f"  swept {checked} objects (orders 2-3, coeffs and inits in [-2,2]); false certificates: {bad}")
