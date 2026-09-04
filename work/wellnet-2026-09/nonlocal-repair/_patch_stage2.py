"""Rewrite stage2_full3d: report BOOSTS (which isolate the K effect from the
disk-versus-sphere baryon geometry) and run an explicit f_T = 0 control beside
every directional candidate, which is the only way to tell whether the
anisotropy does independent work."""
p = 'job3_tensor.py'
s = open(p, encoding='utf-8').read()

NEW = '''def stage2_full3d(cands, n=64, L=160.0):
    head("STAGE 2  Full 3-D anisotropic solves on a DISK")
    say("Two things are being checked: whether the spherical proxy's BOOST "
        "survives")
    say("real disk geometry, and whether f_T decouples the radial from the "
        "vertical")
    say("response.  Everything is reported as a BOOST relative to the "
        "Newtonian")
    say("solve on the SAME density, because the raw speeds differ between a "
        "disk")
    say("and the exponential sphere of the Stage-1 proxy by 30-40% from "
        "geometry")
    say("alone and that difference has nothing to do with K.")
    import families as FA
    import fieldsolve as FS
    from scipy.ndimage import map_coordinates
    KPC, MSUN = FA.KPC, FA.MSUN
    box = FA.Box(n, L)
    Mg, Rd, hz = 6.0e10, 3.0, 1.0
    rho = FA.expdisk_rho(box.pts, Mg * MSUN, Rd * KPC,
                         hz * KPC).reshape(box.shape)
    rho = FA.normalise_mass(rho, box.vol, Mg * MSUN)
    say(f"   grid {n}^3, box {L} kpc, h = {box.h / KPC:.2f} kpc; "
        f"exponential disk M = {Mg:.2g} Msun, R_d = {Rd} kpc, h_z = {hz} kpc")
    say("   NOTE the resolution limit, stated rather than hidden: h = "
        f"{box.h / KPC:.2f} kpc does")
    say("   NOT resolve the |z| = 1.1 kpc Oort column, so the vertical force "
        "is")
    say("   reported at z = 3 and 6 kpc and the Oort number stays with the "
        "slab")
    say("   proxy of Stage 1.")
    rN = FS.solve_newton(rho, box, tol=1e-10, maxiter=4000)
    say(f"   Newtonian reference: {rN['iters']} iters, resid "
        f"{rN['resid']:.2e}")
    That = FA.tidal_hat(rN["Psi"], box.h, dict(eps_T=1e-30))
    #  is That really diag(-2,1,1)/sqrt6 in the outskirts?
    rr = box.r.ravel()
    sel = (rr > 40 * KPC) & (rr < 70 * KPC)
    nh = (np.stack([box.X.ravel(), box.Y.ravel(), box.Z.ravel()], 1)
          / np.maximum(rr, 1e-30)[:, None])
    trad = np.einsum("pi,pij,pj->p", nh[sel], That[sel], nh[sel])
    say(f"   orientation check: n.That.n over 40-70 kpc has mean "
        f"{trad.mean():+.4f}, sd {trad.std():.4f}")
    say(f"      (the exterior monopole value is -2/sqrt6 = "
        f"{-2 / S6:+.4f}; the Stage-1 proxy assumes it)")

    ns = 24
    sarr = (np.arange(ns) + 0.5) / ns
    P = np.stack([box.X.ravel(), box.Y.ravel(), box.Z.ravel()], 1)
    Rq = np.array([8.0, 16.0, 24.0, 32.0, 48.0]) * KPC
    vN = FS.vcirc_axis(rN["Psi"], box, Rq)
    gzN = [abs(FS.force_at(rN["Psi"], box, np.array(
        [R0_SUN * KPC, 0.0, z * KPC]))[2]) for z in (3.0, 6.0)]

    out = dict(orientation=dict(n_That_n_mean=float(trad.mean()),
                                n_That_n_sd=float(trad.std()),
                                monopole_value=float(-2 / S6)),
               grid=dict(n=n, L_kpc=L, h_kpc=float(box.h / KPC),
                         M=Mg, Rd=Rd, hz=hz),
               runs=[])
    seen = set()
    for cd in cands:
        key = (cd["f_nl"], cd["f_T"], cd["a"], cd["p"], cd["c"], cd["m"],
               cd["rho_ref"], cd["qdef"], cd["nonlocal_qbar"])
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 3:
            break
        rho_ref = cd["rho_ref"]
        rho_ms = rho.ravel() / MSUN * KPC ** 3 + C.NK.RHO_BAR_B
        if cd["qdef"] == "delta":
            q = np.clip(rho_ref / rho_ms - 1.0, 0.0, 1 - 1e-15)
        else:
            q = 1.0 / (1.0 + rho_ms / rho_ref)
        if cd["nonlocal_qbar"]:
            acc = np.zeros(P.shape[0])
            qg = q.reshape(box.shape)
            for sv in sarr:
                idx = ((P * (1.0 - sv) / box.h) + (n - 1) / 2.0).T
                acc += map_coordinates(qg, idx, order=1, mode="nearest")
            qb = acc / ns
            del acc, qg
        else:
            qb = q
        fn = f_nl(cd["f_nl"], qb, cd["a"], cd["p"])
        for ctag, cval in ((f"f_T={cd['f_T']} c={cd['c']:g}", cd["c"]),
                           ("f_T = 0 CONTROL", 0.0)):
            ft = (f_T(cd["f_T"], qb, cval, cd["m"]) if cval else
                  np.zeros_like(qb))
            eye = np.eye(3)[None]
            M = (np.clip(fn, -EXP_CLIP, EXP_CLIP)[:, None, None] * eye
                 + ft[:, None, None] * That)
            try:
                K = FA._sym_expm(M, "K = exp[f_nl I + f_T That]")
                rK = FS.solve_K(rho, K, box, tol=1e-10, maxiter=6000,
                                Mtot=float(rho.sum() * box.vol))
            except Exception as e:                       # noqa: BLE001
                say(f"   {ctag}: SOLVE FAILED {type(e).__name__}: {e}")
                continue
            vK = FS.vcirc_axis(rK["Psi"], box, Rq)
            gzK = [abs(FS.force_at(rK["Psi"], box, np.array(
                [R0_SUN * KPC, 0.0, z * KPC]))[2]) for z in (3.0, 6.0)]
            slK = float(np.mean(np.gradient(np.log(np.maximum(vK, 1e-30)),
                                            np.log(Rq))))
            slN = float(np.mean(np.gradient(np.log(np.maximum(vN, 1e-30)),
                                            np.log(Rq))))
            #  the Stage-1 spherical proxy's BOOST at the same radii
            rr_, Menc, qbs = galaxy_qbar(Mg, Rd, rho_ref, cd["qdef"],
                                         cd["nonlocal_qbar"])
            krr = kr_kt_kz(f_nl(cd["f_nl"], qbs, cd["a"], cd["p"]),
                           (f_T(cd["f_T"], qbs, cval, cd["m"]) if cval
                            else np.zeros_like(qbs)))[0]
            bproxy = np.interp(np.log(Rq / KPC), np.log(rr_), 1.0 / krr)
            b3d = (vK / vN) ** 2
            row = dict(setting={k: cd[k] for k in
                                ("f_nl", "f_T", "a", "p", "c", "m",
                                 "rho_ref", "qdef", "nonlocal_qbar")},
                       variant=ctag, c_used=float(cval),
                       R_kpc=(Rq / KPC).tolist(),
                       radial_boost_3d=b3d.tolist(),
                       radial_boost_proxy=bproxy.tolist(),
                       proxy_boost_rel_err=float(np.max(np.abs(
                           b3d / np.maximum(bproxy, 1e-30) - 1.0))),
                       vertical_boost_z3=float(gzK[0] / gzN[0]),
                       vertical_boost_z6=float(gzK[1] / gzN[1]),
                       outer_logslope_3d=slK,
                       outer_logslope_newton=slN,
                       shell_spread=float(rK["shell_spread"]),
                       iters=int(rK["iters"]), resid=float(rK["resid"]))
            out["runs"].append(row)
            s_ = row["setting"]
            say(f"   f_nl={s_['f_nl']} a={s_['a']:g} p={s_['p']:g} "
                f"q={s_['qdef']} nl={int(s_['nonlocal_qbar'])}  |  {ctag}")
            say("      radial boost 3-D   : " + " ".join(f"{x:6.3f}"
                                                         for x in b3d))
            say("      radial boost proxy : " + " ".join(f"{x:6.3f}"
                                                         for x in bproxy))
            say(f"      VERTICAL boost g_z/g_z,N: {row['vertical_boost_z3']:.3f}"
                f" at z=3 kpc, {row['vertical_boost_z6']:.3f} at z=6 kpc")
            say(f"      outer dlnv/dlnr {slK:+.3f} (Newton {slN:+.3f}); "
                f"proxy boost error {row['proxy_boost_rel_err'] * 100:.0f}%")
            del K, rK
        del qb, fn
    RES["stage2_full3d"] = out
    say("")
    say("   READ THE CONTROL ROWS.  If the directional run and its f_T = 0 "
        "control")
    say("   have the SAME ratio of vertical to radial boost, the anisotropy "
        "is only")
    say("   rescaling G and the Stage-1 no-go stands.  If they differ, f_T "
        "is doing")
    say("   independent work and the atom is a genuine new direction.")
    for r_ in out["runs"]:
        rb = float(np.mean(r_["radial_boost_3d"]))
        say(f"      {r_['variant']:<22s} radial {rb:6.3f}  vertical(z=3) "
            f"{r_['vertical_boost_z3']:6.3f}  ratio "
            f"{r_['vertical_boost_z3'] / rb:6.3f}")


'''

a = s.index("def stage2_full3d(cands, n=64, L=160.0):")
b = s.index("# ==========================================================================\ndef main():")
s = s[:a] + NEW + s[b:]
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("patched", len(s))
