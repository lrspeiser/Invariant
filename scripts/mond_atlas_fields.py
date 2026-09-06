"""NumPy-only finite-domain Newtonian/QUMOND field engine.

Solve the two Poisson equations, with face-centered nonlinear flux. Boundary
potentials are REQUIRED inputs: zero boundary is not silently equated to isolated
MOND. Physical units must be consistent across rho, G, a0, spacing and potentials.
This engine does not reconstruct mass, infer exterior fields or solve AQUAL.
"""
from __future__ import annotations

import numpy as np


def dst1(x, axis):
    """Orthonormal DST-I via an odd extension, self-inverse."""
    n = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = 2*(n+1)
    extended = np.zeros(shape, dtype=float)
    pos, neg = [slice(None)]*x.ndim, [slice(None)]*x.ndim
    pos[axis], neg[axis] = slice(1,n+1), slice(n+2,None)
    extended[tuple(pos)] = x
    extended[tuple(neg)] = -np.flip(x, axis=axis)
    return -np.fft.fft(extended, axis=axis).imag[tuple(pos)] / np.sqrt(2*(n+1))


def laplacian(phi, spacing):
    center = phi[1:-1,1:-1,1:-1]
    return (phi[2:,1:-1,1:-1]+phi[:-2,1:-1,1:-1]
            +phi[1:-1,2:,1:-1]+phi[1:-1,:-2,1:-1]
            +phi[1:-1,1:-1,2:]+phi[1:-1,1:-1,:-2]-6*center)/spacing**2


def poisson(rhs, boundary, spacing):
    rhs, boundary = np.asarray(rhs, float), np.asarray(boundary, float)
    if (rhs.ndim != 3 or min(rhs.shape) < 5 or rhs.shape != boundary.shape
            or spacing <= 0 or not np.isfinite(spacing)
            or not np.isfinite(rhs).all() or not np.isfinite(boundary).all()):
        raise ValueError("invalid finite-domain Poisson input")
    b = rhs[1:-1,1:-1,1:-1].copy()
    h2 = spacing**2
    b[0,:,:] -= boundary[0,1:-1,1:-1]/h2
    b[-1,:,:] -= boundary[-1,1:-1,1:-1]/h2
    b[:,0,:] -= boundary[1:-1,0,1:-1]/h2
    b[:,-1,:] -= boundary[1:-1,-1,1:-1]/h2
    b[:,:,0] -= boundary[1:-1,1:-1,0]/h2
    b[:,:,-1] -= boundary[1:-1,1:-1,-1]/h2
    eigenvalues = []
    for n in b.shape:
        eigenvalues.append(-4*np.sin(np.pi*np.arange(1,n+1)/(2*(n+1)))**2/h2)
    for axis in range(3):
        b = dst1(b, axis)
    b /= eigenvalues[0][:,None,None]+eigenvalues[1][None,:,None]+eigenvalues[2][None,None,:]
    for axis in range(3):
        b = dst1(b, axis)
    phi = boundary.copy()
    phi[1:-1,1:-1,1:-1] = b
    return phi


def qumond_source(phi_newton, spacing, a0):
    if a0 <= 0 or not np.isfinite(a0):
        raise ValueError("a0 must be positive and finite")
    gradients = np.gradient(phi_newton, spacing, edge_order=2)
    divergence = np.zeros_like(phi_newton)
    for axis in range(3):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(None,-1), slice(1,None)
        left, right = tuple(left), tuple(right)
        normal = (phi_newton[right]-phi_newton[left])/spacing
        norm2 = normal**2
        for transverse in range(3):
            if transverse != axis:
                norm2 += (.5*(gradients[transverse][left]+gradients[transverse][right]))**2
        magnitude = np.sqrt(norm2)
        # Evaluate nu*g stably at exactly zero field without a fictitious floor.
        factor = np.zeros_like(magnitude)
        positive = magnitude > 0
        factor[positive] = .5+np.sqrt(.25+a0/magnitude[positive])
        flux = factor*normal
        upper, lower = [slice(1,-1)]*3, [slice(1,-1)]*3
        upper[axis], lower[axis] = slice(1,None), slice(None,-1)
        divergence[1:-1,1:-1,1:-1] += (flux[tuple(upper)]-flux[tuple(lower)])/spacing
    return divergence


def solve_fields(density, spacing, newton_boundary, mond_boundary, *, gravity_constant, a0):
    density = np.asarray(density, float)
    if not np.isfinite(density).all() or (density < 0).any() or gravity_constant <= 0:
        raise ValueError("finite nonnegative baryon density and positive G required")
    rhs = 4*np.pi*gravity_constant*density
    newton = poisson(rhs, newton_boundary, spacing)
    source = qumond_source(newton, spacing, a0)
    mond = poisson(source, mond_boundary, spacing)
    residual = lambda phi, r: float(np.max(np.abs(laplacian(phi,spacing)-r[1:-1,1:-1,1:-1])) /
                                   max(np.max(np.abs(r[1:-1,1:-1,1:-1])), np.finfo(float).eps))
    return dict(newton_potential=newton, mond_potential=mond,
                newton_residual=residual(newton,rhs), mond_residual=residual(mond,source))


def acceleration(potential, spacing):
    return tuple(-g for g in np.gradient(potential,spacing,edge_order=2))


def plummer_fixture(nodes, half_width=8., mass=1., scale=1., a0=1.):
    axis = np.linspace(-half_width,half_width,nodes)
    h = axis[1]-axis[0]
    xyz = np.meshgrid(axis,axis,axis,indexing="ij")
    radius = np.sqrt(sum(a*a for a in xyz))
    rho = 3*mass/(4*np.pi*scale**3)*(1+(radius/scale)**2)**(-2.5)
    phi_n = -mass/np.sqrt(radius**2+scale**2)
    # Independent 1D integral of the exact spherical MOND force; not the discrete PDE.
    t = np.linspace(0,float(radius.max())*1.001,60001)
    gn = mass*t/(t*t+scale*scale)**1.5
    gm = .5*gn+np.sqrt(.25*gn*gn+a0*gn)
    phi = np.concatenate(([0.],np.cumsum(.5*(gm[1:]+gm[:-1])*np.diff(t))))
    phi_m = np.interp(radius,t,phi)
    return axis,h,xyz,radius,rho,phi_n,phi_m


def validate():
    """Independent analytic and metamorphic checks, with explicit scope."""
    resolution = []
    for nodes in (33,49,65):
        axis,h,xyz,radius,rho,pn,pm = plummer_fixture(nodes)
        result = solve_fields(rho,h,pn,pm,gravity_constant=1.,a0=1.)
        gn = acceleration(result["newton_potential"],h)
        gm = acceleration(result["mond_potential"],h)
        use = (radius >= 1.5) & (radius <= 4.)
        analytic_n = radius/(1+radius**2)**1.5
        analytic_m = .5*analytic_n+np.sqrt(.25*analytic_n**2+analytic_n)
        numerical_n = np.sqrt(sum(g*g for g in gn))
        numerical_m = np.sqrt(sum(g*g for g in gm))
        resolution.append(dict(nodes=nodes, spacing=h,
            newton_relative_force_rms=float(np.sqrt(np.mean((numerical_n[use]/analytic_n[use]-1)**2))),
            mond_relative_force_rms=float(np.sqrt(np.mean((numerical_m[use]/analytic_m[use]-1)**2))),
            newton_linear_residual=result["newton_residual"], mond_linear_residual=result["mond_residual"]))
    # A nonspherical density and fixed external boundary exercise the vector solve.
    axis,h,xyz,radius,_,_,_ = plummer_fixture(33,half_width=6.)
    x,y,z = xyz
    rho = np.exp(-.5*((x/1.3)**2+(y/.8)**2+(z/.5)**2))/(1.3*.8*.5*(2*np.pi)**1.5)
    safe = np.maximum(radius,h)
    pn,pm = -1/safe, np.log(safe)-.5/safe
    field = solve_fields(rho,h,pn,pm,gravity_constant=1.,a0=1.)
    swapped = solve_fields(rho.transpose(1,0,2),h,pn.transpose(1,0,2),pm.transpose(1,0,2),gravity_constant=1.,a0=1.)
    rotation_error = float(np.max(np.abs(field["mond_potential"]-swapped["mond_potential"].transpose(1,0,2))))
    center = tuple([len(axis)//2]*3)
    center_force = float(np.linalg.norm([a[center] for a in acceleration(field["mond_potential"],h)]))
    empty = np.zeros_like(rho)
    ext = 2*x+.3*y-.1*z
    norm = np.sqrt(4+.09+.01)
    ext_m = ext*(.5+np.sqrt(.25+1/norm))
    efield = solve_fields(empty,h,ext,ext_m,gravity_constant=1.,a0=1.)
    external_error = float(np.max(np.abs(efield["mond_potential"]-ext_m)))
    high = solve_fields(rho,h,pn,pn,gravity_constant=1.,a0=1e-8)
    use = (radius>1) & (radius<3)
    an = acceleration(high["newton_potential"],h)
    am = acceleration(high["mond_potential"],h)
    high_error = float(np.sqrt(np.mean(sum((a[use]-b[use])**2 for a,b in zip(an,am)))) /
                       np.sqrt(np.mean(sum(a[use]**2 for a in an))))
    gates = dict(poisson_residual=all(max(r["newton_linear_residual"],r["mond_linear_residual"])<1e-10 for r in resolution),
        spherical_newton=resolution[-1]["newton_relative_force_rms"]<.03,
        spherical_mond=resolution[-1]["mond_relative_force_rms"]<.03,
        resolution_improves=all(resolution[i+1]["mond_relative_force_rms"] < resolution[i]["mond_relative_force_rms"] for i in (0,1)),
        axis_rotation=rotation_error<1e-10, reflection_symmetry=center_force<1e-10,
        constant_external_field=external_error<1e-10, high_acceleration_limit=high_error<1e-5)
    return dict(status="NUMERICAL_FOUNDATION_ONLY", all_pass=all(gates.values()), gates=gates,
        spherical_resolution=resolution, rotation_error=rotation_error, center_force=center_force,
        constant_external_field_error=external_error, high_acceleration_relative_error=high_error,
        astronomical_galaxies_solved=0, aqual_validated=False,
        boundary_limitation="Boundary potentials supplied explicitly. Real isolated/external-field boundaries and box-size convergence remain to be validated for each observed source.",
        sources=["https://arxiv.org/abs/0911.5464"],
        nonclaims=["No observed 3D reconstruction", "No galaxy cube likelihood", "No AQUAL comparison", "No astrophysical external-field inference"])


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from mond_atlas_common import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["all_pass"] else 1)
