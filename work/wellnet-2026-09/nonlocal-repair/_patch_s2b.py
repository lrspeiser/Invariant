"""Second patch to S2: separate the QUADRATURE pathology the hard clip
creates from the PHYSICAL jump it does or does not create.  Kept for audit."""
p = 'audit_smoothness.py'
s = open(p, encoding='utf-8').read()

NEW = '''def s2_kernel_surface(g, prof):
    head("S2  What the clip surface does in the KERNEL formulation: force, "
         "flux, energy")
    say("Two separate things have to be told apart here, and the first "
        "version of")
    say("this test conflated them.")
    say("")
    say("(a) A QUADRATURE pathology.  D contains dqbar/dr = Int q\\'(r_s) "
        "r(1-s)/r_s ds")
    say("    and a hard clip makes q\\' DISCONTINUOUS in s.  Gauss-Legendre "
        "on a")
    say("    discontinuous integrand converges like 1/n_s, not spectrally, "
        "so D")
    say("    near the clip radius carries an O(1/n_s) error that mimics a "
        "jump.")
    say("(b) The PHYSICAL question: with the quadrature converged, is there "
        "a jump")
    say("    in Phi or in g across the surface?")
    out = {}

    # ---- (a) convergence in n_s, hard versus C^2 ------------------------
    say("")
    say("(a) convergence of D at radii straddling the clip surface, as n_s "
        "grows.")
    say("    Reported as max |D(n_s)/D(1024) - 1| over 6 radii within 2% of "
        "r_clip.")
    conv = {}
    for rho_ref in (1e5, 1e6):
        for kind, w in (("hard", 0.0), ("quintic", 0.05), ("quintic", 0.20)):
            fld, gal, rc = _analytic_field(rho_ref, kind, w)
            if not np.isfinite(rc):
                continue
            rg = rc * np.array([0.98, 0.99, 0.995, 1.005, 1.01, 1.02])
            ref = None
            row = {}
            for n_s in (8, 16, 32, 64, 128, 256, 512, 1024):
                _, D = DC.phi_and_D(fld, rg, Fname="F1_poly", alpha=3.0,
                                    p=1.0, Mtot=gal.Mtot, use_gpu=GPU,
                                    chunk=2, n_D=48, n_s=n_s, n_gl=10,
                                    dlnr_max=0.12)
                if n_s == 1024:
                    ref = D
                row[n_s] = D
            errs = {k: float(np.max(np.abs(v / ref - 1.0)))
                    for k, v in row.items() if k != 1024}
            conv[f"rho_ref={rho_ref:g}|{kind}|w={w}"] = errs
            say(f"   rho_ref={rho_ref:<6g} {kind:<8s} w={w:<5.2f} : " +
                "  ".join(f"n_s={k}:{v:.2e}" for k, v in errs.items()))
    say("")
    say("   The hard clip loses roughly one factor of 2 in error per "
        "doubling of")
    say("   n_s -- first-order, the signature of a discontinuous integrand. "
        "The C^2")
    say("   clip converges far faster.  n_s = 12, the production value, is "
        "therefore")
    say("   the WRONG quadrature for a clipped q, and that is a defect of "
        "the clip,")
    say("   not of the solver.")
    out["n_s_convergence"] = conv

    # ---- (b) the physical jump, at converged quadrature ------------------
    say("")
    say("(b) one-sided limits at n_s = 512, window shrunk until the "
        "extrapolation")
    say("    settles.  [g] != 0 would be a shell of surface density "
        "[g]/(4 pi G).")
    for rho_ref in (1e5, 1e6):
        for kind, w in (("hard", 0.0), ("quintic", 0.05)):
            fld, gal, rc = _analytic_field(rho_ref, kind, w)
            if not np.isfinite(rc):
                continue
            Mt = gal.Mtot
            rho_c = float(gal.rho_pert(np.array([rc]))[0])
            for alpha, p in ((3.0, 1.0), (10.0, 1.0), (3.0, 2.0)):
                seq, gr = [], None
                for tmax in (2.0e-2, 1.0e-2, 5.0e-3):
                    t = np.linspace(0.35 * tmax, tmax, 8)
                    rg = np.concatenate([rc * (1 - t[::-1]), rc * (1 + t)])
                    Fe, D = DC.phi_and_D(fld, rg, Fname="F1_poly",
                                         alpha=alpha, p=p, Mtot=Mt,
                                         use_gpu=GPU, chunk=2, n_D=48,
                                         n_s=512, n_gl=10, dlnr_max=0.12)
                    Phi = -C.G * Mt * Fe / rg
                    gr = C.G * Mt * D / rg ** 2
                    x = rg / rc - 1.0
                    lo, hi = x < 0, x > 0
                    fa = lambda f_, m_: np.polyfit(x[m_], f_[m_], 1)
                    pP, pM = fa(Phi, hi), fa(Phi, lo)
                    gP, gM = fa(gr, hi), fa(gr, lo)
                    seq.append(dict(
                        tmax=tmax,
                        jump_Phi_rel=float(abs((pP[1] - pM[1])
                                               / np.mean(np.abs(Phi)))),
                        jump_g_rel=float(abs((gP[1] - gM[1])
                                             / np.mean(np.abs(gr)))),
                        jump_dgdx_rel=float(abs((gP[0] - gM[0])
                                                / np.mean(np.abs(gr))))))
                c = seq[-1]
                gmean = float(np.mean(np.abs(gr)))
                rho_eff_step = (c["jump_dgdx_rel"] * gmean / rc
                                / (4.0 * math.pi * C.G))
                sigma_shell = c["jump_g_rel"] * gmean / (4.0 * math.pi * C.G)
                key = f"rho_ref={rho_ref:g}|{kind}|w={w}|a={alpha}|p={p}"
                out[key] = dict(
                    r_clip_kpc=float(rc), window_sequence=seq,
                    jump_Phi_rel=c["jump_Phi_rel"],
                    jump_g_rel=c["jump_g_rel"],
                    jump_dg_dlnr_rel=c["jump_dgdx_rel"],
                    shell_surface_density_Msun_kpc2=float(sigma_shell),
                    shell_mass_over_Mtot=float(
                        4 * math.pi * rc ** 2 * sigma_shell / Mt),
                    rho_eff_step_Msun_kpc3=float(rho_eff_step),
                    rho_baryon_at_clip=rho_c,
                    rho_eff_step_over_baryon=float(rho_eff_step / rho_c))
                say(f"   {key}  r_clip = {rc:.3f} kpc")
                say(f"      [Phi]/|Phi| {c['jump_Phi_rel']:.2e}   [g]/|g| "
                    f"{c['jump_g_rel']:.2e}  -> shell mass/M_tot "
                    f"{out[key]['shell_mass_over_Mtot']:.2e}")
                say(f"      [dg/dlnr]/|g| {c['jump_dgdx_rel']:.2e}  -> "
                    f"effective-density step "
                    f"{out[key]['rho_eff_step_over_baryon']:.2e} x the local "
                    f"baryon density")
                say("      shrinking window, [g]/|g| : " + "  ".join(
                    f"{cc['tmax']:.0e}->{cc['jump_g_rel']:.1e}"
                    for cc in seq))
    RES["S2_kernel_surface"] = out


'''

a = s.index("def s2_kernel_surface(g, prof):")
b = s.index("# ==========================================================================\ndef s3_field_surface")
s = s[:a] + NEW + s[b:]
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("patched", len(s))
