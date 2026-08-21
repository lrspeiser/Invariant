"""Receipt-level tamper cases, redone with the correct sealing function.
Each case reseals the hash so the tamper is NOT caught by the hash alone --
the structural checks must catch it. Every case SHOULD raise."""
import copy
import json
import sys

sys.path.insert(0, r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/src")

from sigma_theory_compiler.self_generated_conjecture_widening import (  # noqa: E402
    ConjectureWideningError,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256  # noqa: E402

GOOD = json.load(open(r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/runs/math/selfgen/widening-receipt-v1.json"))


def reseal(b):
    b["content_sha256"] = canonical_sha256({k: v for k, v in b.items() if k != "content_sha256"})
    return b


results = []


def case(name, fn):
    try:
        fn()
    except ConjectureWideningError as exc:
        results.append((name, "REJECTED", str(exc)[:95]))
    except Exception as exc:  # noqa: BLE001
        results.append((name, f"ERROR({type(exc).__name__})", str(exc)[:95]))
    else:
        results.append((name, "*** ACCEPTED -- HOLE ***", ""))


validate_receipt(GOOD)
print("baseline: pristine receipt VALIDATES")
print("sealing function reproduces stored hash:", canonical_sha256({k: v for k, v in GOOD.items() if k != "content_sha256"}) == GOOD["content_sha256"])
print()


def t_float():
    b = copy.deepcopy(GOOD)
    b["counts"]["triage"]["open"] = 0.0
    validate_receipt(reseal(b))


def t_claim_flip():
    b = copy.deepcopy(GOOD)
    b["claims"]["prior_art_absence_establishes_novelty"] = True
    validate_receipt(reseal(b))


def t_proved_means_novel():
    b = copy.deepcopy(GOOD)
    b["claims"]["proved_means_novel"] = True
    validate_receipt(reseal(b))


def t_census_short():
    """Census that visited fewer members than its declared box: skipped members."""
    b = copy.deepcopy(GOOD)
    sw = b["skolem_census_ladder"][2]
    sw["visited"] = 6000
    sw["counts"]["has_zero_with_explicit_witness"] = 5104
    sw["counts"]["total"] = 6000
    validate_receipt(reseal(b))


def t_census_unsettled_hidden():
    """Relabel the 36 unsettled order-4 members as proved zero-free."""
    b = copy.deepcopy(GOOD)
    sw = b["skolem_census_ladder"][2]
    sw["counts"]["zero_free_with_modulus_certificate"] += 36
    sw["counts"]["unsettled"] = 0
    sw["triage"]["open"]["count"] = 0
    sw["triage"]["proved_and_prior_art_not_found"]["count"] += 36
    validate_receipt(reseal(b))


def t_census_partition_flag():
    """Claim exactness while the counts do not reach the box."""
    b = copy.deepcopy(GOOD)
    sw = b["skolem_census_ladder"][3]
    sw["counts"]["unsettled"] = 0
    sw["counts"]["total"] = 59049 - 536
    validate_receipt(reseal(b))


def t_route_histogram():
    b = copy.deepcopy(GOOD)
    sw = b["skolem_census_ladder"][1]
    sw["zero_free_proof_route_histogram"]["local_obstruction"] += 10
    validate_receipt(reseal(b))


def t_open_verdict_smuggled():
    """Turn an OPEN into a PROVED with no certificate."""
    b = copy.deepcopy(GOOD)
    r = next(x for x in b["conjectures"] if x["adjudication"]["verdict"] == "REFUTED")
    r["adjudication"]["verdict"] = "PROVED"
    r["adjudication"].pop("witness", None)
    validate_receipt(reseal(b))


def t_zf_cert_stripped():
    """A zero-free PROVED whose residue table has been made to look clean."""
    b = copy.deepcopy(GOOD)
    r = next(
        x
        for x in b["conjectures"]
        if x["kind"] == "zero_free_over_the_integers"
        and x["adjudication"].get("proof_route") == "local_obstruction"
    )
    rt = r["adjudication"]["residue_table"]
    rt["values"] = [v if v != 0 else 1 for v in rt["values"]]
    validate_receipt(reseal(b))


def t_unadmitted():
    b = copy.deepcopy(GOOD)
    b["conjectures"][0]["admission"]["admitted"] = False
    validate_receipt(reseal(b))


def t_control_fail():
    b = copy.deepcopy(GOOD)
    b["controls"][0]["passed"] = False
    validate_receipt(reseal(b))


def t_obligation_hash():
    b = copy.deepcopy(GOOD)
    b["conjectures"][0]["obligation"]["inputs"]["m"] = 999
    validate_receipt(reseal(b))


case("float 0.0 injected, hash resealed", t_float)
case("claim prior_art_absence_establishes_novelty -> True", t_claim_flip)
case("claim proved_means_novel -> True", t_proved_means_novel)
case("census order4: visited 6000 < declared 6561 (skipped members)", t_census_short)
case("census order4: 36 unsettled relabelled proved zero-free", t_census_unsettled_hidden)
case("census order5: 536 unsettled dropped, total short of box", t_census_partition_flag)
case("census order3: route histogram inflated by 10", t_route_histogram)
case("conjecture: REFUTED promoted to PROVED, witness stripped", t_open_verdict_smuggled)
case("conjecture: zero residues scrubbed from a zero-free table", t_zf_cert_stripped)
case("conjecture: unadmitted candidate reached adjudication", t_unadmitted)
case("controls: one control marked failed", t_control_fail)
case("conjecture: sealed obligation input altered", t_obligation_hash)

print(f"{'case':<64} {'outcome':<26} detail")
print("-" * 125)
holes = 0
for n, o, d in results:
    if "HOLE" in o:
        holes += 1
    print(f"{n:<64} {o:<26} {d}")
print("\nHOLES FOUND:", holes)
