"""Patch job3_unbounded.py: add the ungated control (p = 0), a second q
definition, an overflow guard, and the diagnostic that names the mechanism
behind the R1-vs-R2 conflict."""
p = 'job3_unbounded.py'
s = open(p, encoding='utf-8').read()

s = s.replace(
    '    "exp_growth":    lambda qb, x, a, p, n: np.exp(a * qb ** p * x ** n),',
    '    "exp_growth":    lambda qb, x, a, p, n: np.exp(\n'
    '        np.minimum(a * qb ** p * x ** n, 500.0)),')

OLD = s[s.index("def screen_unbounded():"):s.index("# ==========================================================================\ndef main():")]

NEW = '''def gate_diagnostic():
    """Why R1 and R2 are mutually exclusive: the gate subtracts from D.

    For F = 1 + a G(qbar(r)) H(r),

        D = F - r F' = 1 + a G (H - r H') - a r G'(qbar) qbar'(r) H

    and the LAST term is strictly negative whenever G is increasing in qbar
    (the modification grows in voids) and qbar is increasing in r (paths get
    more void-like outwards).  So the very r-dependence that switches the
    modification on SUBTRACTS from the force, and it scales with the same
    amplitude a that the flat part needs.  That is the whole conflict.
    """
    head("Why R1 (D > 0) and R2 (D proportional to r) exclude each other")
    say("For F = 1 + a G(qbar(r)) H(r),")
    say("   D = 1 + a G (H - r H')  -  a r G'(qbar) qbar'(r) H")
    say("The last term is strictly NEGATIVE when the modification grows in "
        "voids")
    say("(G' > 0) and paths get more void-like outwards (qbar' > 0).  Raising "
        "a to")
    say("lengthen the flat part raises the negative term by the same factor.")
    out = {}
    r, q, qb = qbar_profile(6.0e10, 3.0, 1e6)
    qbf = lambda x: np.interp(np.log(np.maximum(x, 1e-30)), np.log(r),
                              np.log(np.maximum(qb, 1e-300)))
    ro = np.geomspace(1.0, 300.0, 60)
    L, n = 10.0, 1.0
    H = lambda x: (x / L) * np.log1p(n * L / np.maximum(x, 1e-30))
    for a in (0.3, 1.0, 3.0, 10.0):
        for p_ in (0.0, 1.0, 2.0):
            Ff = lambda x: 1 + a * np.exp(p_ * qbf(x)) * H(x)
            Fv, D = D_of(Ff, ro)
            sl = np.gradient(np.log(np.abs(D)), np.log(ro))
            sel = (ro >= 6.0) & (ro <= 60.0)
            out[f"a={a}|p={p_}"] = dict(
                min_D=float(np.min(D)),
                mean_slope=float(np.mean(sl[sel])),
                gated=bool(p_ > 0))
            say(f"   a={a:<5g} p={p_:<4g} ({'gated  ' if p_ else 'UNGATED'})"
                f"  min D over 1-300 kpc {np.min(D):+11.4g}   "
                f"mean dlnD/dlnr over 2-20 R_d {np.mean(sl[sel]):+.3f}")
    say("")
    say("   The p = 0 rows are the SAME H(r) with the gate removed: D stays")
    say("   positive and the slope still reaches 1.  Turning the gate on at "
        "the")
    say("   same a drives min D negative.  An ungated form is not a "
        "candidate,")
    say("   because without a gate the modification is present in the solar")
    say("   system too -- which is what R3 tests.")
    RES["gate_diagnostic"] = out


'''

NEW += OLD.replace(
    "        (0.3, 0.5, 1.0, 1.5), (1.0, 10.0, 100.0), (1e5, 1e6)))",
    "        (0.3, 0.5, 1.0, 1.5), (1.0, 10.0, 100.0), (1e5, 1e6),\n"
    "        (\"delta\", \"smooth\")))").replace(
    "    for atom, a, p, n, L, rho_ref in grid:",
    "    for atom, a, p, n, L, rho_ref, qkind in grid:").replace(
    "            r, q, qb, _, _ = prof[(name, rho_ref)]",
    "            r, q, qb, _, _ = prof[(name, rho_ref, qkind)]").replace(
    "        r, q, qb, M0, rd0 = prof[(\"MW_like\", rho_ref)]",
    "        r, q, qb, M0, rd0 = prof[(\"MW_like\", rho_ref, qkind)]").replace(
    "            r, q, qb = qbar_profile(M, rd, rho_ref)\n"
    "            prof[(name, rho_ref)] = (r, q, qb, M, rd)",
    "            for qkind in (\"delta\", \"smooth\"):\n"
    "                r, q, qb = qbar_profile(M, rd, rho_ref, qkind)\n"
    "                prof[(name, rho_ref, qkind)] = (r, q, qb, M, rd)").replace(
    "        (0.5, 1.0, 2.0),", "        (0.0, 0.5, 1.0, 2.0),").replace(
    "            atom=atom, unbounded=atom in UNBOUNDED, a=a, p=p, n=n, L=L,",
    "            atom=atom, unbounded=atom in UNBOUNDED, a=a, p=p, n=n, L=L,\n"
    "            qkind=qkind, gated=bool(p > 0),")

s = s[:s.index("def screen_unbounded():")] + NEW + s[s.index("# ==========================================================================\ndef main():"):]
s = s.replace("    t3_linearity_btfr()\n    screen_unbounded()",
              "    t3_linearity_btfr()\n    gate_diagnostic()\n"
              "    screen_unbounded()")

# report R1&R2 near-misses and split by gated/ungated
s = s.replace('''    surv = [r for r in rows if all(r[k] for k in keys)]''',
'''    for tag, sub in (("gated (p > 0)", [r for r in rows if r["gated"]]),
                     ("ungated (p = 0)",
                      [r for r in rows if not r["gated"]])):
        n12 = sum(r["R1_positive"] and r["R2_flat"] for r in sub)
        say(f"   R1 and R2 together, {tag:<16s}: {n12:5d} / {len(sub)}")
    surv = [r for r in rows if all(r[k] for k in keys)]''')

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("patched", len(s))
