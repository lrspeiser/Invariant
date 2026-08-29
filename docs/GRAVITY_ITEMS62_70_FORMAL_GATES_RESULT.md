# Items 62–70 formal and external-domain gate results

## Overall result

The current survivor is a useful phenomenological cluster-acceleration formula, not a
complete alternative theory of gravity. None of Items 62–70 passed as a full physical
gate. The audit distinguishes three scientifically different outcomes:

- **not applicable to this candidate:** Items 62 and 63;
- **missing the equations required to test:** Items 65–68 and 70;
- **exact tested scaffold or extrapolation fails:** Items 64 and 69.

No formula family was pruned, and no empirical singleton was used as a veto.

| Item | Result | Meaning |
|---:|---|---|
| 62 time variation | Inconclusive / not applicable | The survivor is static; time-dependent families remain untested. |
| 63 massive mode | Inconclusive / not applicable | It has no carrier mass, range, dispersion, or polarization prediction. |
| 64 distance running | Exact scaffold not passed | The frozen curve lost to the galaxy RAR in X-COP development and used object-specific `R500`, not a universal physical scale. |
| 65 lensing slip | Blocked | No common dynamical and lensing potentials or photon coupling. |
| 66 conservation | Blocked | No action, stress-energy identity, matter coupling, constraints, or gauge identities. |
| 67 stability | Blocked | No degrees-of-freedom count, quadratic action, Hamiltonian, or perturbation spectrum. |
| 68 causality | Blocked | No evolution equations, principal symbol, characteristic cones, or initial-value problem. |
| 69 strong field | Exact universal representation vetoed | Without screening, its high-acceleration limit is about 2.5 times ordinary gravity throughout the Solar System. |
| 70 cosmology | Blocked | No background expansion or BBN/CMB/BAO/growth/lensing perturbation theory. |

## Item 62: time variation

The exact candidate contains no explicit time, age, epoch, clock, or evolving coupling.
That protects it from making a currently false time-variation prediction, but does not
validate the speculative time-varying families generated earlier. Any future descendant
must declare an evolution equation before facing clock, orbital-history, stellar, and
cosmological bounds.

## Item 63: massive modes

The survivor contains no propagating massive degree of freedom. There is therefore no
carrier mass, Compton range, gravitational-wave dispersion, or extra polarization to
test. Massive-mode ideas remain open search branches rather than evidence supporting this
formula.

## Item 64: distance running

The preregistered Item 59 distance-running scaffold used amplitude 4 and power 0.5. On
the X-COP development-selection rows its score was 216.72, versus 99.21 for the empirical
galaxy RAR and 9.62 for the selected boundary candidate. Lower is better.

More fundamentally, the curve ran with `r/R500`. `R500` is object-specific; it is not a
single derived universal length usable unchanged from a laboratory to cosmology. The
exact scaffold therefore cannot satisfy Item 64. This does not rule out distance-running
gravity; it defines the missing invention—a universal scale or a scale derived locally
from dynamical fields.

## Items 65–68: missing theory structure

The Item 60 result already showed that radial acceleration does not uniquely fix photon
motion. The same underdetermination propagates through the formal gates:

- lensing slip requires two potentials and a derived relationship between them;
- conservation requires an action or coherent field equations and matter coupling;
- stability requires dynamical degrees of freedom and their perturbations;
- causality requires time-evolution equations and their characteristic structure.

A good curve fit cannot establish any of these. These are blocked construction gates,
not empirical counterexamples.

## Item 69: strong-field/local-domain witness

For the frozen boundary formula,

```text
q = (gbar/a0) / [(gbar/a0) + 0.1]
g = gbar + beta [gbar K_in(q) + a0 K_sym(q)]
```

the Solar System has `gbar >> a0`, so `q` approaches 1. The normalized kernels applied
to this nearly constant occupancy also approach 1. With `beta=1.5`, the universal
high-acceleration limit is therefore

```text
g/gbar -> 1 + beta = 2.5.
```

The deterministic checks at Mercury, Earth, Jupiter, and Neptune-like radii all predict
roughly a 150% increase in acceleration. They fail even a deliberately coarse 1% local
tolerance. This is not one uncertain observational outlier; it is an analytic consequence
repeated across the formula's claimed universal local domain.

Under the counterexample policy, that hard witness vetoes the exact **unscreened universal
representation**. It does not erase the X-COP fit or prune the boundary/nonlocal family.
A descendant could remain viable if it derives a universal screening or high-acceleration
decoupling limit before seeing new targets. A per-object switch is not allowed.

## Item 70: cosmology

The formula has no homogeneous background, relativistic species, recombination physics,
perturbation-growth equations, or cosmological lensing potential. It cannot yet calculate
expansion, nucleosynthesis, the CMB, BAO, structure growth, or weak lensing. The gate is
blocked until an action-level descendant supplies those predictions without silently
adding a fitted dark component.

## What remains scientifically alive

The empirical finding remains: a nonlocal boundary-style acceleration term predicted
X-COP pressure and temperature shapes unusually well on held-out clusters. What has not
survived is the claim that the exact `beta=1.5` radial formula is already a universal law.

The most useful next mechanism search is therefore constrained but still creative:

1. derive a measurable disk-to-cluster transition variable;
2. derive high-acceleration screening;
3. embed both in an action or explicit two-potential field system;
4. derive matter, photon, background, and perturbation predictions from those same fields;
5. only then open the reserved direct-lensing and independent samples.

## Reproduction

```powershell
python -m sigma_theory_compiler.gravity_items62_70_formal_gates replay
python -m pytest tests/test_gravity_items62_70_formal_gates.py -q
```

Paid model calls: zero. GPU use: none.
