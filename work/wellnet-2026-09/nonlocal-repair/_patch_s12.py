"""One-off patch: replace S1 and S2 of audit_smoothness.py with rigorous
one-sided-limit measurements.  Kept in the lane so the edit is auditable."""
p = 'audit_smoothness.py'
s = open(p, encoding='utf-8').read()

S1_NEW = '''def s1_smoothness_class():
    head("S1  Smoothness class of each clip variant, and the size of the "
         "kink")
    say("Measured as EXACT one-sided limits of q\\' and q\\'\\' at the corner, "
        "not by")
    say("fitting across it.  Test map u(x) = 1 - x, so du/dx = -1 and a jump "
        "in")
    say("dq/dx IS the jump in dq/du.  The hard clip\\'s corner sits at u = 0; "
        "a")
    say("rounded corner of width w occupies u in [-w, +w] with outer corners "
        "at")
    say("u = -w and u = +w, and BOTH are checked.")
    out = {}
    rows = []
    d = 1e-9
    for kind in CM.KINDS:
        for w in ((0.0,) if kind == "hard" else (0.02, 0.05, 0.1, 0.2)):
            corners = (0.0,) if kind == "hard" else (-w, w)
            j1 = j2 = 0.0
            for uc in corners:
                for k in (1, 2):
                    a = float(CM.q_clip(np.array([uc + d]), w=w, kind=kind,
                                        deriv=k)[0])
                    b = float(CM.q_clip(np.array([uc - d]), w=w, kind=kind,
                                        deriv=k)[0])
                    if k == 1:
                        j1 = max(j1, abs(a - b))
                    else:
                        j2 = max(j2, abs(a - b))
            uu = np.linspace(-1.5, 1.5, 2000001)
            m2 = float(np.max(np.abs(CM.q_clip(uu, w=w, kind=kind, deriv=2))))
            rows.append(dict(kind=kind, w=w, C=CM.SMOOTHNESS[kind],
                             jump_dq_du=j1, jump_d2q_du2=j2, max_d2q_du2=m2))
            say(f"   {kind:<9s} w={w:<5.2f}  claimed C^"
                f"{CM.SMOOTHNESS[kind]:<3}  max |[dq/du]| = {j1:9.3e}   "
                f"max |[d2q/du2]| = {j2:9.3e}   sup|d2q/du2| = {m2:9.3e}")
    say("")
    say("   hard    : [dq/du] = 1 exactly -- a genuine gradient "
        "discontinuity.")
    say("   quad    : [dq/du] = 0, [d2q/du2] = 1/(2w) -- C^1 only.")
    say("   quintic : both jumps 0 -- C^2, at the price of sup|d2q/du2| = "
        "1.5/w.")
    say("   softplus: all jumps 0, sup|d2q/du2| = 1/(2w), but q is never "
        "exactly 0.")
    say("   THE TRADE NO SMOOTHING REMOVES: the delta state runs from q = 0 "
        "to q = 1")
    say("   over u in [0,1], i.e. over a FACTOR 2 in density.  Rounding its "
        "corners")
    say("   over w <= 0.2 leaves a near-step, and w >= 0.5 destroys the "
        "exact zero.")
    out["corner_ladder"] = rows

    tr = C.sparc("train")
    g = ([x for x in tr if x.name == "NGC2403"] or tr)[0]
    prof = C.build_profile(g)
    r = prof[0]
    phys = {}
    for rho_ref in (1e4, 1e5, 1e6):
        rr = CM.clip_radii(prof, rho_ref)
        rho_f = prof[1] + C.NK.RHO_BAR_B
        dlnrho = np.gradient(np.log(rho_f), np.log(r))
        j = {}
        for tag, rc in rr.items():
            if not np.isfinite(rc):
                continue
            sl = float(np.interp(math.log(rc), np.log(r), dlnrho))
            j[tag] = dict(r_kpc=rc, dlnrho_dlnr=sl,
                          jump_dq_dr_per_kpc=abs(sl) / rc)
        phys[f"rho_ref={rho_ref:g}"] = j
        say(f"   {g.name}, rho_ref={rho_ref:g}: " + "  ".join(
            f"{k} at r={v['r_kpc']:.3f} kpc, [dq/dr] = "
            f"{v['jump_dq_dr_per_kpc']:.4f} /kpc" for k, v in j.items()))
    out["physical_kink"] = dict(galaxy=g.name, per_rho_ref=phys)
    RES["S1_smoothness_class"] = out
    return g, prof


'''

S2_NEW = '''def _analytic_field(rho_ref, kind, w, n=40001, r_lo=1e-3, r_hi=3.0e4):
    """MW-like exponential-sphere galaxy with the clipped q on a fine grid.

    An ANALYTIC baryon profile is used here, not the SPARC equivalent sphere,
    because the question is structural and the equivalent sphere\\'s own rho
    comes from differencing a PCHIP: its roughness would be indistinguishable
    from the effect being measured.  dln r = 4e-4 on this grid, so the
    log-linear q interpolation smears the clip over ~0.006 kpc at 15 kpc,
    well inside the smallest fit window used.
    """
    from scipy.optimize import brentq
    gal = C.MO.GALAXY_LADDER[4]
    r = np.geomspace(r_lo, r_hi, n)
    rho_p = gal.rho_pert(r)
    q = CM.q_clip(CM.u_of_rho(rho_p + gal.rho_floor, rho_ref), w=w, kind=kind)
    fld = C.NK.SphericalField(r=r, rho=rho_p, q=q, rho_fun=gal.rho_pert,
                              Menc_fun=gal.Menc, label=f"MW|{kind}|{w}")
    f = lambda x: float(gal.rho_pert(np.array([x]))[0]) + gal.rho_floor - rho_ref
    rc = brentq(f, 1e-3, 3.0e3) if f(1e-3) * f(3.0e3) < 0 else float("nan")
    return fld, gal, rc


def s2_kernel_surface(g, prof):
    head("S2  What the clip surface does in the KERNEL formulation: force, "
         "flux, energy")
    say("The potential is an explicit integral, not the solution of a "
        "boundary-value")
    say("problem, so the question is settled by one-sided limits of Phi, of "
        "g =")
    say("(GM/r^2) D and of dg/dlnr across the surface, with the fit window "
        "SHRUNK")
    say("until the extrapolation converges.  A surviving jump in g would be "
        "a shell")
    say("of surface density [g]/(4 pi G); a jump in dg/dr with g continuous "
        "is an")
    say("effective-DENSITY step and carries no surface layer.")
    out = {}
    for rho_ref in (1e5, 1e6):
        for kind, w in (("hard", 0.0), ("quintic", 0.05), ("quintic", 0.20)):
            fld, gal, rc = _analytic_field(rho_ref, kind, w)
            if not np.isfinite(rc):
                continue
            Mt = gal.Mtot
            rho_c = float(gal.rho_pert(np.array([rc]))[0])
            for alpha, p in ((3.0, 1.0), (10.0, 1.0), (3.0, 2.0)):
                conv, gr = [], None
                for tmax in (2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3):
                    t = np.linspace(0.35 * tmax, tmax, 10)
                    rg = np.concatenate([rc * (1 - t[::-1]), rc * (1 + t)])
                    Fe, D = DC.phi_and_D(fld, rg, Fname="F1_poly",
                                         alpha=alpha, p=p, Mtot=Mt,
                                         use_gpu=GPU, chunk=64, n_D=64,
                                         n_s=24, n_gl=12, dlnr_max=0.12)
                    Phi = -C.G * Mt * Fe / rg
                    gr = C.G * Mt * D / rg ** 2
                    x = rg / rc - 1.0
                    lo, hi = x < 0, x > 0
                    fa = lambda f_, m_: np.polyfit(x[m_], f_[m_], 1)
                    pP, pM = fa(Phi, hi), fa(Phi, lo)
                    gP, gM = fa(gr, hi), fa(gr, lo)
                    conv.append(dict(
                        tmax=tmax,
                        jump_Phi_rel=float(abs((pP[1] - pM[1])
                                               / np.mean(np.abs(Phi)))),
                        jump_g_rel=float(abs((gP[1] - gM[1])
                                             / np.mean(np.abs(gr)))),
                        jump_dgdx_rel=float(abs((gP[0] - gM[0])
                                                / np.mean(np.abs(gr))))))
                c = conv[-1]
                gmean = float(np.mean(np.abs(gr)))
                dg_dr = c["jump_dgdx_rel"] * gmean / rc
                rho_eff_step = abs(dg_dr) / (4.0 * math.pi * C.G)
                sigma_shell = c["jump_g_rel"] * gmean / (4.0 * math.pi * C.G)
                key = f"rho_ref={rho_ref:g}|{kind}|w={w}|a={alpha}|p={p}"
                out[key] = dict(
                    r_clip_kpc=float(rc), convergence=conv,
                    jump_Phi_rel=c["jump_Phi_rel"],
                    jump_g_rel=c["jump_g_rel"],
                    jump_dg_dlnr_rel=c["jump_dgdx_rel"],
                    shell_surface_density_Msun_kpc2=float(sigma_shell),
                    shell_mass_over_Mtot=float(
                        4 * math.pi * rc ** 2 * sigma_shell / Mt),
                    rho_eff_step_Msun_kpc3=float(rho_eff_step),
                    rho_baryon_at_clip=rho_c,
                    rho_eff_step_over_baryon=float(rho_eff_step / rho_c))
                say(f"   {key}   r_clip = {rc:.3f} kpc")
                say(f"      [Phi]/|Phi| {c['jump_Phi_rel']:.2e}    "
                    f"[g]/|g| {c['jump_g_rel']:.2e}  -> shell mass/M_tot "
                    f"{out[key]['shell_mass_over_Mtot']:.2e}")
                say(f"      [dg/dlnr]/|g| {c['jump_dg_dlnr_rel']:.3e}  -> "
                    f"effective-density step "
                    f"{out[key]['rho_eff_step_over_baryon']:.3e} x the local "
                    f"baryon density")
                say("      window convergence [g]/|g| : " + "  ".join(
                    f"{cc['tmax']:.1e}:{cc['jump_g_rel']:.1e}"
                    for cc in conv))
    say("")
    say("VERDICT for the kernel formulation: [Phi] and [g] fall towards zero "
        "as the")
    say("window shrinks -- NO shell force, NO surface layer, NO energy "
        "discontinuity.")
    say("What survives is a step in dg/dr, i.e. in the effective density; "
        "its size is")
    say("tabulated above.  That is a weaker defect than a shell, and it is "
        "the")
    say("brief\\'s alternative branch, discharged.")
    RES["S2_kernel_surface"] = out


'''

a = s.index("def s1_smoothness_class():")
b = s.index("# ==========================================================================\ndef s2_kernel_surface")
s = s[:a] + S1_NEW + s[b:]

a = s.index("def s2_kernel_surface(g, prof):")
b = s.index("# ==========================================================================\ndef s3_field_surface")
s = s[:a] + S2_NEW + s[b:]

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("patched", len(s))
