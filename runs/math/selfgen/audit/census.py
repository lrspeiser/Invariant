"""Independent census re-run. My own enumeration, my own orbit code, my own positivity route."""
import json
import sys
from itertools import product

RECEIPT = r"C:/Users/henry/Documents/Codex/2026-08-06/for/wt-self/runs/math/selfgen/widening-receipt-v1.json"
REC = json.load(open(RECEIPT))
MODULI = (2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 25, 27, 32)
SCAN = 100
DEEP = 300  # exact deep re-scan, well past their 100, to catch a false zero-free


def seq(c, u0, N):
    d = len(c)
    v = list(u0)
    while len(v) < N:
        v.append(sum(c[i] * v[-1 - i] for i in range(d)))
    return v[:N]


def orbit_has_zero(c, u0, m, cap=60000):
    """True/False if orbit closes; None if cap reached."""
    d = len(c)
    st = tuple(x % m for x in u0)
    pos = set()
    vals = []
    i = 0
    while i < cap:
        if st in pos:
            return 0 in vals
        pos.add(st)
        vals.append(st[0])
        nxt = sum(c[k] * st[d - 1 - k] for k in range(d)) % m
        st = (*st[1:], nxt)
        i += 1
    return None


def matmul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def companion(c):
    d = len(c)
    M = [[0] * d for _ in range(d)]
    for i in range(d - 1):
        M[i][i + 1] = 1
    for i in range(d):
        M[d - 1][i] = c[d - 1 - i]
    return M


def charrec(M):
    """char poly coefficients a with x(t+d)=a_1 x(t+d-1)+...+a_d x(t), via Leverrier over Q."""
    from fractions import Fraction

    n = len(M)
    cur = [row[:] for row in M]
    bs = []
    for k in range(1, n + 1):
        tr = sum(cur[i][i] for i in range(n))
        v = Fraction(-tr, k)
        assert v.denominator == 1
        b = int(v)
        bs.append(b)
        if k < n:
            sh = [[cur[i][j] + (b if i == j else 0) for j in range(n)] for i in range(n)]
            cur = matmul(M, sh)
    return [-b for b in bs]


def closes(tw):
    if all(x >= 0 for x in tw) and sum(tw) >= 1:
        return "flat"
    tn = sum(x for x in tw[1:] if x < 0)
    if tw[0] + tn >= 1:
        return "monotone"
    return None


def base_ok(win, hyp):
    if any(x < 1 for x in win):
        return False
    if hyp == "monotone":
        return all(win[i] <= win[i + 1] for i in range(len(win) - 1))
    return True


def positivity(c, u0, max_start, window, decs=(1, 2, 3, 4)):
    d = len(c)
    span = max(window, (max_start + d + 1) * max(decs) + max(decs))
    V = seq(c, u0, span)
    M = companion(c)
    for p in decs:
        classes = []
        if p == 1:
            for om in (1, -1):
                tw = [om ** (i + 1) * c[i] for i in range(d)]
                for sg in (1, -1):
                    classes.append((0, tw, [sg * om**i * V[i] for i in range(len(V))]))
        else:
            base = charrec(matmul_pow(M, p))
            for r in range(p):
                dec = [V[i] for i in range(r, len(V), p)]
                for sg in (1, -1):
                    classes.append((r, base, [sg * x for x in dec]))
        proofs = {}
        for r, tw, ser in classes:
            if r in proofs:
                continue
            hyp = closes(tw)
            if hyp is None:
                continue
            for st in range(max_start + 1):
                if st + d > len(ser):
                    break
                if not base_ok(ser[st : st + d], hyp):
                    continue
                cf = st * p + r
                if any(V[i] == 0 for i in range(min(cf, len(V)))):
                    break
                proofs[r] = cf
                break
        if len(proofs) == p:
            cf = max(proofs.values())
            if any(V[i] == 0 for i in range(min(cf, len(V)))):
                continue
            return True
    return False


def matmul_pow(M, e):
    n = len(M)
    R = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    B = [row[:] for row in M]
    while e:
        if e & 1:
            R = matmul(R, B)
        B = matmul(B, B)
        e >>= 1
    return R


orders = [int(x) for x in sys.argv[1:]] or [2, 3, 4, 5]
ladder = {int(s["config"]["order"]): s for s in REC["skolem_census_ladder"]}

for d in orders:
    has_zero = 0
    pos_ct = 0
    lo_ct = 0
    unsettled = 0
    visited = 0
    deep_violation = []
    seen = set()
    for c in product(range(-1, 2), repeat=d):
        for u0 in product(range(-1, 2), repeat=d):
            visited += 1
            seen.add((c, u0))
            V = seq(c, u0, SCAN)
            if any(x == 0 for x in V):
                has_zero += 1
                continue
            # deep exact rescan: anything called zero-free must survive far past the scan
            VD = seq(c, u0, DEEP)
            zd = next((n for n, x in enumerate(VD) if x == 0), None)
            if zd is not None:
                deep_violation.append((c, u0, zd))
            if positivity(c, u0, max_start=d + 4, window=SCAN):
                pos_ct += 1
                continue
            cert = None
            for m in MODULI:
                r = orbit_has_zero(c, u0, m)
                if r is False:
                    cert = m
                    break
            if cert is not None:
                lo_ct += 1
            else:
                unsettled += 1
    box = 9**d
    them = ladder[d]
    tc = them["counts"]
    tr = them["zero_free_proof_route_histogram"]
    print(f"--- order {d} ---")
    print(f"  box 9^{d} = {box} | my visited={visited} distinct={len(seen)} | their visited={them['visited']} their declared={them['config']['declared_box_size']}")
    print(f"  has_zero   mine={has_zero:6d} theirs={tc['has_zero_with_explicit_witness']:6d} {'OK' if has_zero==tc['has_zero_with_explicit_witness'] else 'MISMATCH'}")
    print(f"  zero_free  mine={pos_ct+lo_ct:6d} theirs={tc['zero_free_with_modulus_certificate']:6d} {'OK' if pos_ct+lo_ct==tc['zero_free_with_modulus_certificate'] else 'MISMATCH'}")
    print(f"    positivity mine={pos_ct:6d} theirs={tr.get('positivity_induction',0):6d}")
    print(f"    local_obst mine={lo_ct:6d} theirs={tr.get('local_obstruction',0):6d}")
    print(f"  unsettled  mine={unsettled:6d} theirs={tc['unsettled']:6d} {'OK' if unsettled==tc['unsettled'] else 'MISMATCH'}")
    print(f"  partition sums: {has_zero+pos_ct+lo_ct+unsettled} == {box} ? {has_zero+pos_ct+lo_ct+unsettled==box}")
    print(f"  deep exact rescan to n<{DEEP}: zero-free members that later hit 0: {len(deep_violation)}")
    for v in deep_violation[:5]:
        print("    !!", v)
