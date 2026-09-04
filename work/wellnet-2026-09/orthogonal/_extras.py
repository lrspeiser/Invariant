"""Gates that do not need the long runs: frame validation, the algebraic
law's curl defect measured on the ANALYTIC field (so Newton returns the
machine-precision zero it must), and the integrator's energy drift."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbit_model as OM
import adyn_same_object as S


def curl_analytic(bar, law, h=0.05):
    """max |curl g| * (10 kpc) / |g| for the ALGEBRAIC law, on the analytic
    field over the region the streams actually occupy.  Newton must return
    ~1e-9; anything larger is the law's own inconsistency, not the grid's."""
    R = np.geomspace(6.0, 80.0, 40)
    z = np.geomspace(1.0, 60.0, 40)
    RR, ZZ = np.meshgrid(R, z, indexing="ij")
    def g(rr, zz):
        return law.g_algebraic(bar, rr, zz)
    gR_zp, _ = g(RR, ZZ + h)
    gR_zm, _ = g(RR, ZZ - h)
    _, gz_Rp = g(RR + h, ZZ)
    _, gz_Rm = g(RR - h, ZZ)
    dgR_dz = (gR_zp - gR_zm) / (2 * h * OM.KPC)
    dgz_dR = (gz_Rp - gz_Rm) / (2 * h * OM.KPC)
    gRv, gzv = g(RR, ZZ)
    mag = np.sqrt(gRv ** 2 + gzv ** 2) / (10.0 * OM.KPC)
    q = np.abs(dgR_dz - dgz_dR) / mag
    return float(np.max(q)), float(np.median(q))


if __name__ == "__main__":
    out = {"frame_validation": OM.validate_frame()}
    R, v, e, prov = S.load_eilers()
    cur = {}
    drift = {}
    for law in OM.frozen_laws():
        law.completion = S.COMPLETION[law.name]
        bar, sol, ref, info = S.calibrate_and_solve(law, R, v, e, verbose=False)
        mx, md = curl_analytic(bar, law)
        cur[law.name] = dict(max=mx, median=md)
        fl = OM.DeformedField(sol, 1.0, refine=2)
        w = np.zeros((6, 6))
        w[:, 0] = np.array([10., 20., 30., 50., 70., 90.]) * OM.KPC
        w[:, 2] = np.array([5., 10., 15., 25., 35., 45.]) * OM.KPC
        gR, _ = fl.force(np.array([10., 20., 30., 50., 70., 90.]),
                         np.array([5., 10., 15., 25., 35., 45.]))
        w[:, 4] = np.sqrt(gR * w[:, 0])
        drift[law.name] = dict(
            refine2=OM.energy_drift(fl, w, 1.5e6 * 3.1557e7, 2600),
            refine1=OM.energy_drift(OM.DeformedField(sol, 1.0, refine=1), w,
                                    1.5e6 * 3.1557e7, 2600))
        print(law.name, "curl max %.3e median %.3e" % (mx, md),
              "| E drift", drift[law.name], flush=True)
    out["curl_defect_analytic"] = cur
    out["energy_drift"] = drift
    json.dump(out, open("_extras.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
