"""Independent re-verification. Shares no code with the module under audit.
Every routine below is written from scratch against the mathematical definition."""
import json
import re
from math import gcd

RECEIPT = r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/runs/math/selfgen/widening-receipt-v1.json"
REC = json.load(open(RECEIPT))


def parse_id(oid):
    m = re.fullmatch(r"L(\d+):c(.+):u(.+)", oid)
    d = int(m.group(1))
    c = [int(x) for x in m.group(2).split("_")]
    u = [int(x) for x in m.group(3).split("_")]
    assert len(c) == d and len(u) == d, oid
    return d, c, u


def seq(c, u0, N):
    d = len(c)
    v = list(u0)
    while len(v) < N:
        v.append(sum(c[i] * v[-1 - i] for i in range(d)))
    return v[:N]


def orbit(c, u0, m, cap=400000):
    d = len(c)
    st = tuple(x % m for x in u0)
    pos = {}
    vals = []
    i = 0
    while i < cap:
        if st in pos:
            mu = pos[st]
            return mu, i - mu, vals
        pos[st] = i
        vals.append(st[0])
        nxt = sum(c[k] * st[d - 1 - k] for k in range(d)) % m
        st = (*st[1:], nxt)
        i += 1
    return None


def res_at(mu, lam, vals, n):
    return vals[n] if n < mu + lam else vals[mu + (n - mu) % lam]


def lcm(a, b):
    return a * b // gcd(a, b)


BRUTE = 4000
fail = []
stats = {}

for rec in REC["conjectures"]:
    kind = rec["kind"]
    verdict = rec["adjudication"]["verdict"]
    stats[(kind, verdict)] = stats.get((kind, verdict), 0) + 1
    p = rec["parameters"]
    oid = rec["object_id"]
    d, c, u0 = parse_id(oid)
    inp = rec["obligation"]["inputs"]
    if "coefficients" in inp and (list(inp["coefficients"]) != c or list(inp["initial"]) != u0):
        fail.append((oid, kind, "obligation inputs disagree with object_id"))

    if kind == "divisibility_index_set":
        m, q, j = p["m"], p["q"], p["j"]
        mu, lam, vals = orbit(c, u0, m)
        H = mu + lcm(q, lam)
        agree = all((res_at(mu, lam, vals, n) == 0) == (n % q == j) for n in range(H))
        V = seq(c, u0, BRUTE)
        agree_brute = all((V[n] % m == 0) == (n % q == j) for n in range(BRUTE))
        want = verdict == "PROVED"
        if agree != want or agree_brute != want:
            fail.append((oid, kind, f"divisibility verdict={verdict} orbit={agree} brute={agree_brute}"))
        if verdict == "REFUTED":
            w = rec["adjudication"]["witness"]["n"]
            if (V[w] % m == 0) == (w % q == j):
                fail.append((oid, kind, f"refutation witness n={w} does not disagree"))

    elif kind == "modular_pure_period":
        m, P = p["m"], p["P"]
        mu, lam, vals = orbit(c, u0, m)
        V = seq(c, u0, BRUTE)
        R = [x % m for x in V]
        half = BRUTE // 2
        least = next((pp for pp in range(1, half) if all(R[n] == R[n + pp] for n in range(half))), None)
        want = verdict == "PROVED"
        got = mu == 0 and lam == P
        if got != want:
            fail.append((oid, kind, f"period verdict={verdict} mu={mu} lam={lam} P={P}"))
        if want and least != P:
            fail.append((oid, kind, f"PROVED least period {P} but brute force says {least}"))
        if not want and least == P and mu == 0:
            fail.append((oid, kind, f"REFUTED but brute force agrees period is {P}"))

    elif kind == "cross_object_congruence":
        m, a, b = p["m"], p["alpha"], p["beta"]
        _, cl, ul = parse_id(p["left_id"])
        _, cr, ur = parse_id(p["right_id"])
        T = seq(c, u0, BRUTE)
        L = seq(cl, ul, BRUTE)
        R = seq(cr, ur, BRUTE)
        ok = all((T[n] - a * L[n] - b * R[n]) % m == 0 for n in range(BRUTE))
        want = verdict == "PROVED"
        if ok != want:
            fail.append((oid, kind, f"cross verdict={verdict} brute={ok}"))
        if verdict == "REFUTED":
            w = rec["adjudication"]["witness"]["n"]
            if (T[w] - a * L[w] - b * R[w]) % m == 0:
                fail.append((oid, kind, f"cross refutation witness n={w} actually holds"))

    elif kind == "zero_free_over_the_integers":
        V = seq(c, u0, BRUTE)
        z = next((n for n, x in enumerate(V) if x == 0), None)
        if verdict == "PROVED":
            if z is not None:
                fail.append((oid, kind, f"PROVED zero-free but u({z})=0 !!!"))
            adj = rec["adjudication"]
            route = adj["proof_route"]
            if route == "local_obstruction":
                mo = adj["certifying_modulus"]
                mu, lam, vals = orbit(c, u0, mo)
                if 0 in vals:
                    fail.append((oid, kind, f"certifying modulus {mo} orbit DOES contain 0"))
                if (mu, lam) != (adj["residue_table"]["mu"], adj["residue_table"]["lam"]):
                    fail.append((oid, kind, "my mu/lam disagree with sealed table"))
            elif route == "positivity_induction":
                cert = adj["positivity_certificate"]
                per = cert["decimation"]
                cps = cert["class_proofs"]
                if sorted(cp["residue_class"] for cp in cps) != list(range(per)):
                    fail.append((oid, kind, "positivity residue classes are not a partition"))
                for cp in cps:
                    tw = cp["twisted_coefficients"]
                    r = cp["residue_class"]
                    sg = cp["sign"]
                    om = cp["omega"]
                    if per == 1:
                        w = [sg * om**n * V[n] for n in range(BRUTE)]
                    else:
                        w = [sg * V[n] for n in range(r, BRUTE, per)]
                    dd = len(tw)
                    holds = all(
                        w[t + dd] == sum(tw[i] * w[t + dd - 1 - i] for i in range(dd))
                        for t in range(len(w) - dd)
                    )
                    if not holds:
                        fail.append((oid, kind, f"twisted recurrence {tw} does NOT hold for class {r}"))
                    flat = all(x >= 0 for x in tw) and sum(tw) >= 1
                    tn = sum(x for x in tw[1:] if x < 0)
                    mono = tw[0] + tn >= 1
                    if cp["hypothesis"] == "flat" and not flat:
                        fail.append((oid, kind, f"flat hypothesis does not close for {tw}"))
                    if cp["hypothesis"] == "monotone" and not mono:
                        fail.append((oid, kind, f"monotone hypothesis does not close for {tw}"))
                    st = cp["induction_start"]
                    bc = w[st : st + dd]
                    if any(x < 1 for x in bc):
                        fail.append((oid, kind, f"base case {bc} not >= 1"))
                    if cp["hypothesis"] == "monotone" and any(bc[i] > bc[i + 1] for i in range(dd - 1)):
                        fail.append((oid, kind, f"base case {bc} not monotone"))
                cf = max(cp["covers_indices_from"] for cp in cps)
                if any(V[n] == 0 for n in range(cf)):
                    fail.append((oid, kind, "zero below the induction start"))
        else:
            w = rec["adjudication"]["witness"]["n"]
            if V[w] != 0:
                fail.append((oid, kind, f"REFUTED witness n={w} has u={V[w]} != 0"))

print("verdict census by kind:")
for k in sorted(stats, key=str):
    print("  ", k, stats[k])
print("total adjudicated:", len(REC["conjectures"]))
print("DISAGREEMENTS:", len(fail))
for f in fail[:40]:
    print("  !!", f)
