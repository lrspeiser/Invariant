# Empirical Gravity-Discovery Engine

Generate many candidate laws, make each one predict the same set of observables, and
eliminate them ruthlessly against data from the Solar System through cosmology.

The important discipline is that we should **not** ask only, "Can equation X fit a rotation
curve?" Almost anything flexible enough can. We want:

$$
\boxed{\text{one small set of universal parameters} \rightarrow
\text{galaxy dynamics + lensing + clusters + redshift + cosmology}}
$$

If one equation explains flat rotation curves but predicts the wrong gravitational lensing,
it dies.

---

# EXECUTION DIRECTIVE

**Run until the research is complete.** Do not stop at a round boundary, a negative result,
or a written-up summary. The program is complete when either a candidate has passed the
full T01–T25 gauntlet, or every candidate has been eliminated with a stated reason and the
residual search (T25) has been exhausted. Report progress as you go; do not wait for
permission to continue to the next stage.

**Use 12 subagents in parallel.** Formula work and data gathering are independent and must
not be serialised. Launch all twelve in one batch, let them run concurrently, and integrate
as results arrive.

### The twelve lanes

**Acquisition (six).** These are I/O bound, mutually independent, and are the current
blocker on the second half of the gauntlet.

| # | lane | targets |
| --- | --- | --- |
| 1 | Bullet-like mergers | 1E 0657-56 and analogues: X-ray gas maps, lensing convergence, galaxy light — T14 |
| 2 | DES Y6 3×2pt | cosmic shear, clustering, galaxy-galaxy lensing — T15 |
| 3 | DESI BAO | DR2 cosmology products, radial and transverse scales — T16 |
| 4 | Pantheon+ and SN time dilation | light curves, covariance, stretch-vs-z — T17, T18 |
| 5 | CMB | ACT DR6 and Planck PLA: spectra, lensing, likelihoods — T19–T21 |
| 6 | Group-scale bridge | X-ray gas masses **and** calibrated virial factors for 10¹²–10¹³ M☉ groups; Frontier Fields lens models — T13 |

**Analysis (six).** These consume the bench and the datasets already in hand.

| # | lane | task |
| --- | --- | --- |
| 7 | Relativistic promotion | lift A2, A4, C6≡B3 into the (Φ, Ψ) framework of §1; predict lensing separately from dynamics |
| 8 | Directional family in lensing | C1, C2, C3, C5 are exactly null or degenerate in-plane; they can only be scored out of the disk plane or in lensing |
| 9 | Nonlinear Poisson solver | D1/D2 on a grid: solve for a thin disk **and** a sphere, so shape-dependence is derived rather than fitted |
| 10 | Redshift family | E4, E5, E6 against Pantheon+, BAO and the 1+z time-dilation constraint simultaneously |
| 11 | Residual discovery (T25) | symbolic regression and sparse regression on the residuals of every surviving law |
| 12 | Cluster a₀ radial run | measure a₀ vs radius using **lensing only** — X-ray radial shapes are biased and this is the current open question |

### Rules every lane must follow

1. **Run the confound check on every candidate variable.** `Bench.confound(name, getter)`
   compares it against a bare 0/1 dataset indicator. Six separate "discoveries" in this
   project were reproduced exactly by that label. A variable that a label reproduces is a
   label.
2. **Never fit to a holdout.** KiDS weak lensing and wide binaries are permanently blind.
   Galaxy-level splits only, never point-level (§9).
3. **Global gravity parameters.** M/L, distance and inclination are legitimate nuisances;
   a₀, α, n, ρ_c are not (§8).
4. **Report the blind score separately from χ².** In this project the best-fitting law was
   eighth on transfer and the symbolic-regression winner was off by 9× out of sample.
5. **Carry the probe caveats.** They are listed in the appendix and they have already
   invalidated two published conclusions here.

---

## 1. Give every equation the same mathematical interface

Start with the baryonic matter only:

$$
\nabla^2\Phi_b=4\pi G\rho_b
$$

and define

$$
a_N=|\nabla\Phi_b|,
\qquad
x=\frac{a_N}{a_0}.
$$

Also define a "voidness" variable:

$$
f_v(\rho_b)=
\frac{1}{1+(\rho_b/\rho_c)^n}.
$$

So

$$
f_v\rightarrow0
$$

inside dense matter and

$$
f_v\rightarrow1
$$

in low-density regions.

For relativistic models, use the weak-field metric

$$
ds^2=
-\left(1+\frac{2\Psi}{c^2}\right)c^2dt^2
+
\left(1-\frac{2\Phi}{c^2}\right)d\mathbf{x}^2.
$$

This is extremely useful because different observations probe different things.

Stars and gas mostly measure

$$
\boxed{\Psi}
$$

while gravitational lensing measures

$$
\boxed{\Phi+\Psi}.
$$

So immediately we can distinguish "extra gravity" from "extra lensing."

---

## 2. Thirty equations worth putting into the tournament

These aren't thirty complete fundamental theories. They're **phenomenological hypotheses**
designed to tell us what mathematical behavior the universe prefers. If one wins, then we
try to derive it from a consistent photon/graviton field theory.

### Family A — gravity changes below some acceleration

| # | Candidate |
| --- | --- |
| **A1** | $a=a_N\left[1+\left(\frac{a_0}{a_N}\right)^n\right]^{1/(2n)}$ |
| **A2** | $a=\frac{a_N}{1-e^{-\sqrt{a_N/a_0}}}$ |
| **A3** | $a=\frac12\left[a_N+\sqrt{a_N^2+4a_0a_N}\right]$ |
| **A4** | $a=a_N+\frac{\alpha\sqrt{a_0a_N}}{1+(a_N/a_s)^m}$ |
| **A5** | $a=a_N[1+\alpha(r/r_0)^p]$ |
| **A6** | $\Psi=\Phi_b+v_*^2\ln(1+r/r_0)$ |

A1–A4 ask whether the important variable is **acceleration** rather than distance or mass.
A5–A6 ask whether gravity simply acquires a long-range component.

A1 has a particularly interesting asymptotic limit:

$$
a\rightarrow\sqrt{a_Na_0}.
$$

For a pointlike baryonic mass,

$$
a_N=\frac{GM_b}{r^2},
$$

so

$$
a=\frac{\sqrt{GM_ba_0}}{r}.
$$

Then

$$
v^2=ar=\sqrt{GM_ba_0}
$$

and therefore

$$
\boxed{v^4=GM_ba_0}.
$$

Tully–Fisher drops straight out.

---

### Family B — "gravity builds up in empty space"

**B1 — Yukawa**

$$
\Psi=
-\frac{GM}{r}
\left(1+\alpha e^{-r/\lambda}\right)
$$

giving

$$
a=
\frac{GM}{r^2}
\left[
1+\alpha
\left(1+\frac r\lambda\right)e^{-r/\lambda}
\right].
$$

**B2 — density-dependent gravity**

$$
\boxed{
G_{\rm eff}=
G\left[
1+\frac{\alpha}{1+(\rho_b/\rho_c)^n}
\right]
}
$$

so gravity gets stronger in low-density environments.

**B3 — surface-density version**

$$
G_{\rm eff}=
G\left[
1+\frac{\alpha}
{1+(\Sigma_b/\Sigma_c)^n}
\right].
$$

This may actually be more interesting for disk galaxies.

**B4 — potential-depth version**

$$
G_{\rm eff}=
G\left[
1+
\frac{\alpha}
{1+(|\Phi_b|/\Phi_c)^n}
\right].
$$

Now the controlling quantity is how deep you are inside a gravitational well.

**B5 — surrounding-void version**

Define

$$
\bar f_v(\mathbf x)
=
\int W_L(\mathbf x-\mathbf x')
f_v[\rho_b(\mathbf x')]
\,d^3x'.
$$

Then

$$
\boxed{
G_{\rm eff}(\mathbf x)
=
G[1+\alpha\bar f_v(\mathbf x)].
}
$$

Gravity responds not merely to the density at a point but to how much empty space surrounds
it.

**B6 — gravity accumulates along the path**

$$
\boxed{
\Psi(\mathbf x)=
-G\int
\frac{\rho_b(\mathbf x')}{s}
\left[
1+
\alpha
\frac1s\int_0^s f_v(\ell)d\ell
\right]
d^3x'
}
$$

where

$$
s=|\mathbf x-\mathbf x'|.
$$

Now the gravitational influence from one piece of matter depends partly on **what lies
between the source and receiver**. That is highly nonstandard — and therefore extremely
testable.

---

## 3. Directional versions

Let $\hat{\mathbf n}$ be the normal to the baryonic disk.

**C1 — tensor gravity**

$$
\boxed{
\mathbf a=
-\left[
I+\epsilon\hat n\hat n^T
\right]\nabla\Phi_b.
}
$$

**C2 — latitude dependence**

$$
\mathbf a=
-\nabla\Phi_b
\left[
1+\epsilon\sin^2\theta
\right],
$$

where $\theta=0$ in the plane.

**C3 — quadrupole**

$$
\Psi=
\Phi_b
\left[
1+\epsilon
P_2(\hat r\cdot\hat n)
\right].
$$

**C4 — geometrically flattened gravity**

Replace ordinary separation

$$
s^2=\Delta x^2+\Delta y^2+\Delta z^2
$$

with

$$
\boxed{
s_q^2=
\Delta x^2+\Delta y^2+\frac{\Delta z^2}{q^2}.
}
$$

Then calculate gravity using $s_q$.

**C5 — directional gravity appears only far away**

$$
\epsilon(r)=
\epsilon_\infty
\left(1-e^{-r/r_\epsilon}\right).
$$

**C6 — directional gravity appears only at low acceleration**

$$
\boxed{
\epsilon(a_N)=
\frac{\epsilon_0}
{1+(a_N/a_0)^n}.
}
$$

C6 is particularly attractive experimentally because it can automatically make the effect
negligible inside the Solar System while allowing it at galaxy scales.

---

## 4. Relativistic models that can predict lensing

**D1 — nonlinear Poisson equation**

$$
\boxed{
\nabla\cdot
\left[
\mu\left(\frac{|\nabla\Psi|}{a_0}\right)
\nabla\Psi
\right]
=
4\pi G\rho_b
}
$$

with, for example,

$$
\mu(y)=\frac{y}{1+y}.
$$

**D2 — fractional gravity**

$$
\boxed{
(-\nabla^2)^{1-\alpha/2}\Psi
=
4\pi G L^\alpha\rho_b.
}
$$

Set $\alpha=0$ and ordinary Poisson gravity returns.

**D3 — scale-dependent gravity**

$$
k^2\Psi=
-4\pi Ga^2
\mu(k,a)\rho_b\delta_b
$$

with

$$
\mu(k,a)=
1+
\frac{\mu_0a^s}
{1+(k/k_c)^2}.
$$

**D4 — gravitational slip**

$$
\boxed{
\frac{\Phi}{\Psi}
=
\eta(k,a)
=
1+
\frac{\eta_0a^s}
{1+(k/k_\eta)^2}.
}
$$

This lets photons experience a different effective gravitational potential from stars.

**D5 — additional underlying field**

$$
(\nabla^2-m_\phi^2)\phi
=
4\pi G\beta\rho_b
$$

and

$$
\Psi=\Phi_b+\phi.
$$

Then make

$$
\beta(\rho)=
\frac{\beta_0}
{1+(\rho/\rho_s)^n}
$$

if you want the new force screened in dense environments.

**D6 — explicit light/matter relation**

$$
\Psi_{\rm matter}=\Psi
$$

but

$$
\boxed{
\Phi_\gamma+\Psi_\gamma
=
(2+\zeta)\Psi.
}
$$

GR corresponds approximately to $\zeta=0$. This is a tremendously useful phenomenological
test: **does the amount of gravity inferred from stars equal the amount inferred from
photons?**

---

## 5. Photon/graviton-unification and redshift candidates

**E1 — Kaluza-like common geometry**

$$
\boxed{
ds_5^2=
g_{\mu\nu}dx^\mu dx^\nu+
\phi^2(x)
(dy+\kappa A_\mu dx^\mu)^2
}
$$

with something like

$$
\Box\phi=\beta T.
$$

Gravity and electromagnetism are then projections of one geometry.

**E2 — photon/graviton mixing**

$$
\boxed{
\begin{pmatrix}
\Box & \epsilon{\cal R}\\
\epsilon{\cal R} & \Box
\end{pmatrix}
\begin{pmatrix}
A\\
h
\end{pmatrix}
=0.
}
$$

Here curvature ${\cal R}$ allows electromagnetic and gravitational modes to mix.

**E3 — oscillation probability**

$$
P_{\gamma\rightarrow g}
=
\sin^2(2\vartheta)
\sin^2
\left(
\frac{\Delta kL}{2}
\right).
$$

Then ask whether photons traversing particular environments gradually occupy the
gravitational mode.

**E4 — gravitational-path redshift**

$$
\boxed{
\ln(1+z_{\rm extra})
=
\frac{\xi}{c^2}
\int
|\nabla\Psi|\,d\ell.
}
$$

A photon "going through gravity" and accumulating an effect.

**E5 — void redshift**

$$
\boxed{
\ln(1+z)
=
\frac{H_*}{c}
\int
[1+\xi f_v(\ell)]\,d\ell.
}
$$

Now a photon accumulates more redshift while crossing voids.

**E6 — ordinary expansion plus an additional gravity/photon effect**

$$
\boxed{
\frac{d\ln\nu}{dt}
=
-H(t)
-
\xi\frac{|\nabla\Psi|}{c}.
}
$$

Useful because we don't initially have to make the extreme assumption that cosmic expansion
is wrong. We can ask whether some fraction of observed redshift is better described by the
extra term.

---

## 6. Build a universal "observation translator"

Every candidate needs to produce these quantities.

For galaxy rotation:

$$
\boxed{
v_c^2(R)
=
R\frac{\partial\Psi}{\partial R}.
}
$$

For lensing:

$$
\boxed{
\boldsymbol\alpha
=
\frac{1}{c^2}
\int
\nabla_\perp(\Phi+\Psi)\,d\ell.
}
$$

For hot gas in a galaxy cluster:

$$
\boxed{
\frac{dP}{dr}
=
-\rho_{\rm gas}
\frac{d\Psi}{dr}.
}
$$

For gravitational redshift:

$$
z_g\simeq
\frac{\Psi_{\rm observer}-\Psi_{\rm emitter}}{c^2}.
$$

For structure growth:

$$
\boxed{
\ddot\delta+
2H\dot\delta
-
4\pi G_{\rm eff}\rho\,\delta=0.
}
$$

And for cosmology, every serious candidate ultimately needs to provide

$$
H(z),
\qquad
D_A(z),
\qquad
D_L(z),
\qquad
\Phi(k,z),
\qquad
\Psi(k,z).
$$

That's the line between a neat galaxy formula and a real competitor to GR+$\Lambda$CDM.

---

## 7. The experimental gauntlet

| Task | Test | What kills the model |
| --- | --- | --- |
| **T01** | Dimensional/unit tests | Equation is not dimensionally self-consistent |
| **T02** | GR/Newtonian limit | Cannot recover known high-acceleration gravity |
| **T03** | JPL planetary ephemerides | Produces detectable anomalous planetary motion |
| **T04** | Cassini light propagation | Wrong photon curvature/Shapiro delay |
| **T05** | Double pulsar | Wrong strong-field dynamics/GW energy loss |
| **T06** | GW170817 | Photon and graviton propagation differ too much |
| **T07** | SPARC point-by-point curves | Cannot reproduce $v(r)$ from baryon maps |
| **T08** | Blind SPARC galaxies | Needs different constants for every galaxy |
| **T09** | Baryonic Tully–Fisher | Doesn't naturally reproduce $M_b-V_f$ |
| **T10** | Radial acceleration relation | Residuals correlate with acceleration |
| **T11** | Galaxy lensing | Dynamics works but photon deflection doesn't |
| **T12** | Cluster gas dynamics | Cannot explain cluster-scale acceleration |
| **T13** | Cluster strong/weak lensing | Predicted lensing mass differs from observations |
| **T14** | Bullet-like mergers | Gravity remains tied to the wrong baryonic component |
| **T15** | DES cosmic shear/clustering | Wrong large-scale growth or lensing |
| **T16** | DESI BAO | Wrong distance/redshift relation |
| **T17** | Pantheon+ SNe | Wrong luminosity-distance relation |
| **T18** | SN time dilation | Alternative redshift lacks $1+z$ clock stretching |
| **T19** | CMB $T(z)$/spectrum | Photon-loss mechanism ruins thermal spectrum/scaling |
| **T20** | ACT/Planck acoustic peaks | Wrong early-universe geometry or growth |
| **T21** | CMB lensing | Wrong integrated $\Phi+\Psi$ |
| **T22** | Structure-growth/RSD | Wrong evolution of density fluctuations |
| **T23** | BBN | Variable $G$/expansion breaks light-element abundances |
| **T24** | Joint global fit | Requires incompatible parameters at different scales |
| **T25** | Residual search | Look for a systematic variable the surviving equations missed |

The **ordering matters**. There is no reason to run a full CMB calculation on an equation
already killed by Cassini.

Cassini measured the PPN light-curvature parameter as

$$
\gamma=1+(2.1\pm2.3)\times10^{-5},
$$

so any model in which light and ordinary matter respond differently to solar gravity already
has a very narrow window. JPL's DE440 ephemeris provides another very strong local-gravity
baseline derived from extensive spacecraft and astronomical tracking.

The double pulsar provides a completely different strong-field environment; 16 years of
timing produced multiple post-Keplerian measurements and precision tests of relativistic
gravity. And GW170817/GRB170817A severely restricts any theory in which gravitational and
electromagnetic disturbances propagate at materially different speeds — the fractional
difference is constrained around the $10^{-15}$ level under the usual interpretation.

Those are excellent "kill tests" for the photon/graviton versions.

---

## 8. Then attack galaxies

SPARC should probably be our **first major discovery dataset**.

It contains 175 spiral/irregular galaxies, with 3.6 μm photometry tracing stellar
distributions and HI/Hα rotation curves tracing their gravitational potentials. Crucially,
it spans roughly five orders of magnitude in stellar mass and more than three orders of
magnitude in surface brightness.

[SPARC public data](https://astroweb.case.edu/SPARC/)

For each galaxy, feed the model:

$$
\Sigma_{\rm stars}(R),
\quad
\Sigma_{\rm gas}(R),
\quad
R,
$$

and produce

$$
v_{\rm predicted}(R).
$$

Then compare against

$$
v_{\rm observed}(R).
$$

Do **not** let the algorithm invent a separate gravity constant for every galaxy.

Allow legitimate nuisance quantities such as stellar mass-to-light ratio, distance
uncertainty, and inclination uncertainty, but keep the underlying gravity parameters global.

So, for example,

$$
a_0,\alpha,n,\rho_c
$$

must ideally be the **same numbers for all 175 galaxies**. That's the key test.

---

## 9. Don't train on individual points and test on points from the same galaxy

That leaks information. Instead:

$$
\boxed{
\text{train galaxies}
\neq
\text{test galaxies}.
}
$$

For example:

$$
60\% \text{ galaxies training}
$$

$$
20\% \text{ validation}
$$

$$
20\% \text{ permanently blind}.
$$

Even better, deliberately hold out unusual classes:

* low-surface-brightness dwarfs,
* gas-dominated galaxies,
* high-mass spirals,
* very extended disks.

If equation B3 discovers a universal relationship involving **surface density** from the
training set and predicts the low-surface-brightness galaxies it has never seen, that would
be genuinely interesting.

---

## 10. Galaxy clusters become the crucial next level

Hubble's Frontier Fields provide six major strong-lensing clusters and publish multiple
independent lensing reconstructions, including convergence $\kappa$ and shear $\gamma$
products.

[Hubble Frontier Fields lensing data](https://stdatu.stsci.edu/prepds/frontier/lensmodels/)

There is an important methodological caveat:

**don't ultimately fit our alternative theory to someone else's GR-derived mass map.**

Those maps assume a lensing framework. For preliminary testing that's fine. For a serious
result, take the more fundamental observables:

$$
\text{multiple-image positions},
\quad
\text{source redshifts},
\quad
\text{shear},
\quad
\text{X-ray gas},
\quad
\text{stellar light}
$$

and predict them directly from our theory.

This is where D4/D6 become particularly valuable because they let

$$
\Psi_{\rm dynamics}
$$

and

$$
\Phi+\Psi_{\rm lensing}
$$

be tested independently.

---

## 11. The Bullet Cluster is an especially nasty test

It is valuable precisely because the baryonic components have become spatially separated.
The X-ray-emitting gas is displaced from the gravitational-lensing peaks.

A simple theory saying

$$
\boxed{\text{gravity is just locally stronger around baryons}}
$$

will probably struggle badly here.

But the **nonlocal/path/geometry** versions B5 and B6 are more interesting because they
aren't necessarily tied to the instantaneous local gas distribution.

So the question becomes:

> Given only the observed stars + gas and the collision geometry, where does B6 predict the
> lensing convergence peaks?

That's a much stronger experiment than "Can it fit a rotation curve?"

---

## 12. Then scale to the whole universe

DES Year 6 is now extremely useful. Its 3×2-point analysis combines cosmic shear from about
**140 million source galaxy shapes**, clustering of about **9 million lens galaxies**, and
galaxy-galaxy lensing over roughly 5,000 square degrees.

This directly attacks equations D3 and D4:

$$
\mu(k,a)
$$

and

$$
\eta(k,a).
$$

If gravity changes depending on scale, density or cosmic epoch, DES should see it in the
relationship between matter clustering and lensing.

---

## 13. Redshift models get a particularly brutal test suite

Suppose E5 beautifully reproduces the supernova Hubble diagram:

$$
\ln(1+z)
=
\frac{H_*}{c}
\int(1+\xi f_v)d\ell.
$$

That's interesting, but nowhere near sufficient. It must *simultaneously* explain:

### Supernova brightness

Use Pantheon+. Its public release contains the supernova data, covariance information,
SH0ES data and cosmology inputs.

[Pantheon+ public data release](https://github.com/PantheonPlusSH0ES/DataRelease)

Predict $m(z)$ rather than merely fitting $z(D)$.

### Time dilation

Distant supernova light curves are observed to be broadened approximately according to the
expected cosmological $1+z$ time dilation. Therefore a "photon loses energy while traveling"
mechanism cannot merely produce:

$$
\nu_{\rm obs}
=
\frac{\nu_{\rm emit}}{1+z}.
$$

It somehow also has to reproduce:

$$
\boxed{
\Delta t_{\rm obs}
=
(1+z)\Delta t_{\rm emitted}.
}
$$

That's a very strong discriminator.

### BAO

DESI DR2 now provides cosmological BAO results and public chains/data products. The full
underlying DR2 spectra/redshift release is still being rolled out, but the cosmology
products are publicly available; DESI DR1 itself contains spectra for more than 18 million
unique targets.

[DESI public data documentation](https://data.desi.lbl.gov/doc/)

A redshift law must reproduce the **radial and transverse** BAO scales consistently. That's
especially powerful because a photon-redshift mechanism changes radial redshift distances
without necessarily changing angular geometry in the same way expansion does.

---

## 14. CMB is probably the ultimate boss fight

ACT DR6 provides public maps, power spectra, lensing products, likelihoods and notebooks;
Planck's Legacy Archive likewise publishes the Planck mission's official products.

[ACT DR6 products](https://act.princeton.edu/act-dr6-data-products)

[Planck Legacy Archive](https://www.cosmos.esa.int/web/planck/pla)

For each surviving theory we eventually modify a Boltzmann solver so it predicts:

$$
C_\ell^{TT},
\quad
C_\ell^{TE},
\quad
C_\ell^{EE},
\quad
C_\ell^{\phi\phi}.
$$

That tests early-universe gravity, photon propagation, acoustic scale, structure growth,
lensing, and expansion history.

A photon/graviton theory that explains galaxies but destroys the CMB acoustic peaks is dead.

---

## 15. Score the equations mathematically

For dataset $j$:

$$
\chi_j^2=
(\mathbf d_j-\mathbf m_j)^T
C_j^{-1}
(\mathbf d_j-\mathbf m_j).
$$

Then

$$
\boxed{
\chi^2_{\rm total}
=
\sum_j\chi_j^2.
}
$$

But raw $\chi^2$ rewards extra parameters, so also calculate something like

$$
\mathrm{BIC}
=
\chi^2+k\ln N
$$

and

$$
\mathrm{AIC}
=
\chi^2+2k.
$$

More importantly, keep the blind-data score separate:

$$
\boxed{
S_{\rm blind}
=
\ln P(D_{\rm unseen}\mid M,\theta_{\rm training}).
}
$$

That is the number to really care about.

---

## 16. Have three permanent competitors

Every model should run against:

$$
\boxed{\text{GR + baryons only}}
$$

to measure how much of the missing-gravity phenomenon it explains.

Then:

$$
\boxed{\text{MOND-like reference}}
$$

to see whether it improves on the strongest simple baryon→gravity relationship at galaxy
scales.

And:

$$
\boxed{\Lambda\mathrm{CDM}+\mathrm{GR}}
$$

as the full standard-model benchmark.

The objective isn't necessarily to beat ΛCDM on raw $\chi^2$ immediately.

If we discovered that something like

$$
G_{\rm eff}
=
G\left[
1+
\frac{\alpha}
{1+(\Sigma/\Sigma_c)^n}
\right]
$$

with **three universal constants** predicts hundreds of galaxy rotation curves, galaxy
lensing and cluster dynamics with no dark halos assigned object-by-object, that would
already be scientifically interesting even before cosmology.

---

## The experiment to run first

Start with only **A1–A6, B1–B6 and C1–C6 — 18 equations**.

Feed all 18 exactly the same SPARC baryonic maps.

For every equation produce $v_{\rm predicted}(R)$ for all 175 galaxies.

Then have the system automatically search over universal parameters and produce:

$$
\chi^2,
\quad
\mathrm{BIC},
\quad
\text{blind error},
\quad
\text{BTFR scatter},
\quad
\text{RAR scatter}.
$$

Most of the 18 will probably die immediately.

Take perhaps the best **3–5 survivors**, promote them to the relativistic $(\Phi,\Psi)$
framework, and send those through lensing and cluster tests.

Only then spend the computational effort on DES/BAO/CMB.

That gives us a funnel:

$$
30
\rightarrow
10
\rightarrow
5
\rightarrow
2
\rightarrow
\boxed{\text{one genuinely interesting theory}}
$$

rather than trying to make every speculative equation into a full cosmology.

And there is one particularly exciting thing to look for in the residuals: **don't just ask
which proposed equation wins.** After every round, give the residuals back to the
formula-discovery system and ask:

$$
\boxed{
\Delta a
=
f(a_N,\Sigma_b,\rho_b,r,M_b,\theta,
\Phi_b,\text{environment})
}
$$

That allows the observations themselves to tell us **which variable we forgot**. The
discovery might not be one of our thirty equations at all. It might be a relationship
between baryonic surface density, acceleration, geometry and environment that nobody in our
initial candidate set guessed.

---

## References

1. [A test of general relativity using radio links with the Cassini spacecraft](https://pubmed.ncbi.nlm.nih.gov/14508481/)
2. [The JPL Planetary and Lunar Ephemerides DE440 and DE441](https://ssd.jpl.nasa.gov/doc/de440_de441.html)
3. [Strong-Field Gravity Tests with the Double Pulsar, Phys. Rev. X](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.041050)
4. [Constraints on gravity from the speed of gravitational waves after GW170817, Phys. Rev. D](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.084011)
5. [SPARC](https://astroweb.case.edu/SPARC/)
6. [Hubble Space Telescope Frontier Fields (HFF)](https://stdatu.stsci.edu/prepds/frontier/lensmodels/)
7. [Chandra Field Guide: Dark Matter Mystery](https://www.chandra.harvard.edu/xray_astro/dark_matter/index4.html)
8. [DES Year 6: Cosmological Constraints from Galaxy Clustering and Weak Lensing](https://arxiv.org/abs/2601.14559)
9. [Pantheon+ Data Release](https://github.com/PantheonPlusSH0ES/DataRelease)
10. [Cosmological time dilation using type Ia supernovae as clocks](https://www.sciencedirect.com/science/article/pii/S0920563296004938)
11. [DESI Data papers](https://data.desi.lbl.gov/doc/papers/)
12. [ACT DR6 Data Products](https://act.princeton.edu/act-dr6-data-products)

---
---

# APPENDIX — implementation status as of 2026-09-02

*Added by Claude. Everything above is the program specification; this appendix records what
already exists so a run does not rebuild it.*

## The harness

`invariant_bench.py` (in the session scratchpad; copy somewhere durable before relying on
it) implements the "same mathematical interface" requirement of §1 for the non-relativistic
observables. `Bench()` loads **4,255 measurements across 7 probes**, each tagged with kind
(matter / photon), geometry, and role (**fit** / **holdout** / **bound**):

| probe | n | kind | role | g_bar/a₀ range |
| --- | --- | --- | --- | --- |
| sparc | 3389 | matter | fit | 1.7e-3 – 6.1e+1 |
| xcop | 588 | matter | fit | 3.0e-2 – 3.2e-1 |
| clash | 84 | photon | fit | 1.1e-1 – 1.8e+0 |
| kids | 60 | photon | **holdout** | 6.0e-7 – 1.0e-1 |
| wicker | 120 | matter | fit | 3.2e-2 – 1.7e-1 |
| solar | 8 | matter | **bound** | 5.5e+4 – 3.3e+8 |
| widebin | 6 | matter | **holdout** | 3.2e-2 – 9.2e+2 |

`b.score(law)` returns per-probe median |log10| error. KiDS and wide binaries have never
been fitted by anything, so they are the blind score $S_{\rm blind}$ of §15.

## THE MANDATORY CHECK

`b.confound(name, getter)` compares any candidate variable against a bare 0/1 dataset
indicator. **Six separate "discoveries" in this project were reproduced exactly by that
label** — shape (binary), compact = r/extent, mass at n=120, a₀-per-population, the pooled
cluster correlations, and PySR's sphericity. Run it on every new variable before believing
anything.

Verdicts so far: `sphericity` = LABEL (0.766). `log10 enclosed mass` = LABEL (0.516).
`compact = r/extent` = not a label (0.056) but not predictive either (−0.051).

## Gauntlet coverage

| Task | status |
| --- | --- |
| T02 GR/Newtonian limit | **done** — exponential suppression is *forced*: flat curves need p≈0.5, Saturn needs p>1.3, no power law satisfies both |
| T03 planetary ephemerides | **done** — implemented as a hard constraint, not a scored term |
| T07 SPARC point-by-point | **done** — 3,389 points, RAR baseline 0.094 dex |
| T09 BTFR | **done** — measured slope 3.83 ± 0.11 against a predicted 4 |
| T10 RAR | **done** — this is the baseline law throughout |
| T11 galaxy lensing | **done** — KiDS 35–2604 kpc, held out |
| T12 cluster gas | **done** — X-COP + Wicker (120 Planck-ESZ) |
| T13 cluster lensing | **done** — CLASH, 20 clusters |
| T25 residual search | **done** — 12M-formula brute force + PySR symbolic regression |
| T01, T04–T06, T08, T14–T24 | **not built** |

## Results that constrain the candidate list above

- **A2 (RAR exponential) is the working baseline.** SPARC 0.094 dex, Solar System exact.
- **A1/A3-type power-law tails are excluded outright** by T02/T03 — no exponent satisfies
  both flat rotation curves and Saturn ranging.
- **A5 (a ∝ r^p) fails** — radius as an organising variable gives probe spread 1.47 dex
  against 0.392 for acceleration.
- **B2/B3 (density, surface density) fail a basic screen** — galaxy ISM pressure and density
  *exceed* cluster outskirts, so a density-driven boost hits galaxies hardest.
- **B4 (potential depth) partly works** — closes ~70% of the cluster gap and transfers to
  KiDS, but predicts the cluster radial gradient backwards.
- **B6 (path accumulation) tested in crude form** as exterior mass M_out/M_in. It is the
  **only elaboration that beats baseline on held-out data** (0.107 vs 0.134 dex) — but it
  wrecks SPARC (0.299 vs 0.094) while helping KiDS, and both are galaxy probes.
- **C-family (directional) has no measured support.** MaNGA: 55 galaxies, one survey, axis
  ratios 0.36–0.94, Sérsic 2.2–8.0, boost spanning 27× — **0 of 11 shape variables survive**
  controls on mass, size and acceleration. This is the sample where a shape effect could
  have shown, and it does not.
- **D6 / photon-vs-matter (ζ):** at matched radius *and* acceleration, photon probes read
  **1.130× higher** than matter probes, 8 of 8 cells, sign test **p = 0.0078**. Real but
  ~13%, and it does not grow with impact parameter. Note the path length is ~constant
  (Gpc) for all cosmological lensing, so a constant offset is what E4-type mechanisms
  predict — this is worth pursuing properly.

## The central unexplained fact

Clusters require a₀ about **18× larger** than galaxies (0.79× vs 14.1× canonical, measured
by lensing). It is **not** a function of mass, radius, acceleration, surface density, mean
density, temperature, concentration, redshift, composition, environment, substructure or
scale — tested across 32 clusters and two independent probes, with the pooled-vs-split
analysis showing exactly how much signal would have been needed.

## Overfitting is real and measured

| law | worst **fitted** | worst **held-out** |
| --- | --- | --- |
| brute-force winner (12M formulas) | 0.111 (1st) | 0.245 (8th) |
| PySR discovery | 0.563 | **1.000** (9× error) |
| RAR + exterior mass | 0.308 | **0.107** (1st) |
| RAR exponential | 0.500 | 0.134 |

Every elaboration bought in-sample accuracy and **zero** out-of-sample accuracy. This is
exactly why §9 and §15's blind score matter more than $\chi^2$.

## Known data traps

- **X-ray hydrostatic masses give reliable amplitudes (5% vs lensing) but biased radial
  shapes** — non-thermal pressure rises outward. Never draw a profile-shape conclusion from
  X-ray alone. This invalidated two published conclusions in this project.
- **Solar System rows are ν = 1 by construction** — a bound, never a fit target. Included in
  a log-log regression they alone flip the sign of the headline correlation.
- **Cluster gas mass must integrate from r = 0**, not from the analysis window's inner edge
  (72% loss for a window starting at 0.8 R500).
- **`np.interp` holds T flat past the profile end** (~0.9 R500 while density reaches 1.5–2.0),
  forcing dlnT/dlnr = 0 and understating mass.
- **f_gas = M_gas/M_dyn = 1/ν identically** — never use the cosmic baryon fraction as an
  independent constraint in a no-dark-matter analysis.
- **Planck-ESZ is SZ flux-limited**, so mass and redshift correlate at 0.69. Control for z.
- **SPARC V_disk, V_bul are Newtonian constructs**; deriving mass from them carries a disk
  geometry factor of ~1.3–2.

---

# ROUND 1 RESULTS — 2026-09-02

*The experiment "The experiment to run first" specifies, executed. Scripts:
`tournament.py`, `gauntlet_local.py`, `gauntlet_joint.py`.*

## Structural finding before any fitting

Every SPARC measurement is **in the disk plane**, which makes four of the six C-family laws
untestable here — not failed, untestable:

| law | why |
| --- | --- |
| C1 tensor | n·nᵀ∇Φ = 0 in the plane — **exactly null** |
| C2 latitude sin²θ | θ = 0 in the plane — **exactly null** |
| C3 quadrupole P₂ | P₂(0) = −1/2, a constant — **degenerate with M/L** |
| C5 ε grows with r | wraps C1/C2, inherits their null |

They carry forward to the relativistic round. Only C4 and C6 change the in-plane field.

## The tournament — 175 galaxies, global parameters, galaxy-level blind split

Split stratified on the classes §9 asks to hold out (low-mass dwarfs, gas-dominated,
high-mass spirals). Two global M/L values are the only nuisance freedom.

Blind χ²/N leaderboard (the §15 number that matters):

| rank | law | blind χ²/N | BIC | RAR scatter | BTFR slope |
| --- | --- | --- | --- | --- | --- |
| 1 | A4 additive sqrt | 22.39 | 49792 | 0.286 | 3.70 |
| 2 | *MOND reference* | 22.58 | 52506 | 0.285 | 3.78 |
| 3 | A3 simple μ | 22.58 | 52506 | 0.285 | 3.78 |
| 4 | A1 interp n-family | 22.59 | 53513 | 0.285 | 3.78 |
| 5 | A2 RAR exponential | 22.61 | 51872 | 0.285 | 3.78 |
| 6 | B3 surface-density G | 23.24 | 49540 | 0.481 | 3.52 |
| 7 | C6 directional at low a | 23.24 | 49540 | 0.481 | 3.52 |
| 8 | B6 path accumulation | 23.90 | 66006 | 0.676 | 3.02 |
| … | | | | | |
| 16 | *Newton + baryons* | 103.81 | 233420 | 0.546 | 2.86 |

**The A-family wins and is internally indistinguishable** (22.39–22.61, a spread far inside
the noise for 175 galaxies). Newton is 4.6× worse. BTFR slope comes out 3.78 against the
predicted 4.

**A degeneracy the tournament caught:** B3 and C6 give *byte-identical* results because
Σ ∝ g/G, so α/(1+(Σ/Σc)ⁿ) and ε₀/(1+(g/a₀)ⁿ) are the same functional form. Surface-density
gravity and low-acceleration directional gravity are not distinguishable by rotation curves.

## The Solar System, imposed as a constraint rather than a filter

A first pass with hand-chosen parameters killed 13 of 14 laws. **That was wrong** — A1's
anomalous acceleration is exactly a₀/2 at n = 1 but decays for n > 1, so most of those kills
were statements about the values chosen, not about the laws. Redone as a scan asking whether
*any* parameters pass T03 (Saturn, |Δg| < 10⁻¹⁴ m/s²) and T04 (Cassini, |γ−1| < 6.7×10⁻⁵):

| law | χ²/N under constraint | unconstrained | **cost of T03** |
| --- | --- | --- | --- |
| **A2 RAR exponential** | 30.42 | 30.42 | **1.00×** |
| A4 additive sqrt | 30.90 | 28.86 | 1.07× |
| C6 directional at low a | 44.27 | 36.58 | 1.21× |
| A1 interp n-family | 60.63 | 36.36 | 1.67× |
| B3 surface-density G | 81.88 | 34.97 | 2.34× |
| A5 power law in r | 90.78 | 69.79 | 1.30× |
| B2 / B6 density, path | 188.04 | 61.31 | 3.07× |
| C4 flattened separation | 258.44 | 67.61 | 3.82× |
| A6 log potential | 273.18 | 65.42 | 4.18× |
| B4 potential-depth G | 273.29 | 83.20 | 3.28× |

**Killed outright — no parameters anywhere pass:** A3 (simple μ), B1 (Yukawa). Both leave a
constant residual of order a₀ at all radii; suppressing it below the Saturn bound requires
a₀ four decades below the galactic value, which destroys the galaxy fit.

## The result of round 1

**A2 is the only law for which the Solar System is free.** Its best galaxy fit is *unchanged*
by imposing planetary ranging and Cassini — cost 1.00×. Every other survivor must move away
from its galaxy optimum to clear the Solar System, by 1.07× to 4.18×.

This reproduces, from the program's own funnel, the constraint derived independently earlier
in this project: the correction must vanish **faster than any power** of g. Flat rotation
curves need p ≈ 0.5; Saturn needs p > 1.3; no power law satisfies both. A2's exponential is
the cheapest form that does.

## Promoted to round 2

Per the funnel, the survivors to promote to the relativistic (Φ, Ψ) framework and send
through lensing and cluster tests:

1. **A2** — RAR exponential, cost 1.00×, the reference
2. **A4** — additive sqrt, best blind χ²/N, cost 1.07×
3. **C6 ≡ B3** — the degenerate pair, cost 1.21×, worth carrying because they are
   distinguishable by *lensing* even though rotation curves cannot separate them
4. **C1, C2, C3, C5** — untestable in-plane, must be scored in round 2 or not at all

## Not yet run

T01 partially (dimensional consistency by inspection, not symbolic), T05 double pulsar,
T06 GW170817, T14 Bullet-like mergers, T15–T24 cosmology.

---

# ROUND 2 — twelve lanes in parallel, live ledger

Started 2026-09-02. This section is updated as each lane reports; it is the running
record the directive at the top of this file asks for. A lane is only marked **closed**
when its result has been through the confound check and written here.

## Acquisition status

| Lane | Target | Test | State | What is now on disk |
|---|---|---|---|---|
| 1 | Bullet-like mergers | T14 | running | `lane01_bullet/` — Clowe 2006 κ maps, peak/aperture tables |
| 2 | DES 3×2pt | T15 | **closed, data complete** | `lane02_des/` 99 MB |
| 3 | DESI BAO | T16 | running | `lane03_bao/` — DR1+DR2 tables, Cobaya likelihoods |
| 4 | Pantheon+ and SN time dilation | T17, T18 | **closed, data complete** | `lane04_sne/` 171 MB |
| 5 | CMB spectra and lensing | T19–T21 | running | `lane05_cmb/` |
| 6 | Group-scale bridge, 10¹²–10¹³·⁵ M☉ | T13 | running | `lane06_groups/` |

## Lane 2 — DES, closed on the data side

**Obtained: the raw Y3 3×2pt data vector, not derived constraints.** ξ₊, ξ₋, γ_t, w over
θ = 2.5′–250′, 4 source × 6 lens tomographic bins, 1000 points for MagLim (DES fiducial)
and 900 for redMaGiC, with the full **joint** 1000×1000 covariance. Total S/N = 264.
The cross-probe blocks are populated — γ_t×w reaches ρ = 0.567, ξ₊×γ_t 0.475 — which is
the entire reason T15 needs the raw vector: the test scores the *relationship* between
clustering and lensing, and that relationship lives in the cross-covariance.

Two independent proofs the files are being read correctly: applying DES's fiducial scale
cuts leaves **462** points, exactly their published number; and refitting their MGCAMB
chain returns Σ₀ = +0.52 (−0.40, +0.46) against their published 0.6 (+0.4, −0.5).

**Y6 is not public.** Verified twice — the key paper states data vectors and chains will be
released on journal acceptance, which has not happened, and eight plausible paths under
`y6a2_files/` return 404. Only the Gold photometric catalogue is out, and the shape
catalogue and photo-z calibration are precisely the embargoed pieces. Y6 is structurally
identical to Y3 (same binning, same 1000-point vector), so this is a drop-in when it lands.

**Three things this changes about how T15 must be scored:**

1. **DES alone is a lensing test, not a balanced μ/η test.** Their μ₀ = −0.43 (−0.73, +0.84)
   spans most of the prior — they decline to quote it from DES alone. Σ₀ is well measured.
   So T15 constrains Φ+Ψ well and growth weakly, and any claim about η from DES alone is
   not supported by the data. This connects directly to the appendix D6/ζ finding that
   photon probes read 1.130× high.
2. **DES's published (Σ₀, μ₀) is a strict special case of D3/D4** — no scale dependence,
   time dependence locked to Ω_Λ. Our D3/D4 carry free k_c, k_η and a free exponent. Their
   chain is therefore a pipeline validation benchmark, **not** a result we may quote.
3. **A trap, named before anyone trips on it.** redMaGiC shows a ~12% clustering-vs-lensing
   mismatch (X_lens ≈ 0.88, 4–5σ) which is formally the exact signature D3/D4 predict. It is
   almost certainly a selection systematic: MagLim does not show it, and loosening
   redMaGiC's χ² cut reduces it. **Sample-dependence is the signature of a label, not of
   gravity** — the same failure mode `Bench.confound` exists to catch. Use MagLim. Do not
   build a result on X_lens.

Blocker for T15 is now **theory, not data**: scoring D3/D4 needs μ(k,a), η(k,a) → P(k,z) for
matter and Weyl → C_ℓ → Hankel transform to ξ±/γ_t/w, plus ~22 nuisance parameters. Neither
MGCAMB nor CCL is installed.

`y3a2_joint-des-kids/` sits on the same server. **KiDS is a permanent blind holdout** — it was
listed and deliberately not downloaded. Any lane pulling DES products must avoid it too.

## Lane 4 — supernovae and time dilation, closed on the data side

**T17 is runnable.** Pantheon+ = 1701 rows (1543 unique SNe), z from 0.00122 to 2.26, 77
Cepheid calibrators, with both the stat-only and the **stat+sys** 1701×1701 covariance,
positive definite. Validated by refitting flat ΛCDM on the 1580 Hubble-flow rows with the
full inverse covariance: **Ω_m = 0.333 ± 0.018** against the published 0.334 ± 0.018.
DES-SN5YR (1820 SNe) is held as an independent cross-check, refit to Ω_m = 0.330 ± 0.015.

Trap recorded: `m_b_corr_err_DIAG` is a plotting column and is **not** the covariance
diagonal — the measured ratio diag(C)/err_DIAG² has median 0.518. Using it would halve the
errors.

**T18 is runnable, with three independent clocks — and one trap that had to be cleared
first.** Pantheon+ `x1` is unusable: SALT2/SALT3 divide the time axis by (1+z) *before*
fitting shape, so `x1` is rest-frame by construction and a tired-light model predicts the
identical distribution. A test built on `x1` would return a null for a purely procedural
reason.

| Clock | N | Method | Refit | Published |
|---|---|---|---|---|
| DES 5yr widths | 1504 | observer-frame light-curve stretch | b = 1.0029 ± 0.0048 | 1.003 ± 0.005 |
| Blondin 2008 | 35 | **spectral** aging rate | b = 0.966 ± 0.104 | 0.97 ± 0.10 |
| Goldhaber 2001 | 60 | observer-frame width | 1.135 ± 0.112 | — |

The DES result excludes b = 0 by Δχ² = 27,951 even after inflating errors to force
χ²/dof = 1, and the author's own null control (reference curve not de-redshifted) returns
b = 0.084 ± 0.007, so the signal is not a fitting artefact. The Blondin set matters most as
a cross-check because a **spectroscopic** clock is immune to the Malmquist objection that
broader light curves are preferentially detected at high z. The remaining loophole —
intrinsic stretch drifting with redshift — was checked directly: mean `x1` by z-quartile on
DES is −0.266 / −0.161 / −0.208 / −0.228, no monotonic drift.

**The constraint handed to lane 10:** Δt_obs = Δt_emit·(1+z)^b with **b = 1.003 ± 0.005**
over 0.06 ≤ z ≤ 1.12.

**And the sharp edge on it, which matters more than the number.** Passing T18 is not the
same as fitting b = 1. A candidate must *derive* the duration ratio from its own kinematics.
A model that asserts a (1+z) clock factor alongside an independent photon-energy-loss
redshift is double-counting, and should fail **on construction**, not on fit. E4/E5/E6 are
to be judged on that basis.

## Lane 3 — DESI BAO, closed on the data side

**DESI DR2 (2025): 13 points over 7 redshift bins with the full 13×13 covariance**, plus DR1
(12 points) and SDSS DR12/DR16 as an independent cross-check. Six of the seven DR2 bins give
`D_M/r_d` and `D_H/r_d` **separately with their correlation** — the radial-vs-transverse split
T16 is built on.

| tracer | z_eff | D_M/r_d (transverse) | D_H/r_d (radial) | r |
|---|---|---|---|---|
| BGS | 0.295 | — (isotropic D_V only) | — | — |
| LRG1 | 0.510 | 13.5876 ± 0.1684 | 21.8629 ± 0.4289 | −0.4516 |
| LRG2 | 0.706 | 17.3507 ± 0.1799 | 19.4553 ± 0.3339 | −0.3953 |
| LRG3+ELG1 | 0.934 | 21.5756 ± 0.1618 | 17.6415 ± 0.2010 | −0.3472 |
| ELG2 | 1.321 | 27.6009 ± 0.3246 | 14.1760 ± 0.2246 | −0.3983 |
| QSO | 1.484 | 30.5119 ± 0.7636 | 12.8170 ± 0.5180 | −0.4936 |
| Lyα | 2.330 | 38.9890 ± 0.5317 | 8.6315 ± 0.1011 | −0.4306 |

All values dimensionless — the likelihood config sets `rs_fid: 1 Mpc`, so there is no hidden
Mpc factor. Definitions were taken from the cobaya code that consumes the files rather than
inferred. The 13×13 covariance is **block diagonal**; DESI treats systematics as uncorrelated
across bins and shows correlating them changes nothing.

**Validated by refitting.** Flat ΛCDM with two free parameters against the 13-point vector and
full covariance reproduces DESI exactly: Ω_m = **0.2975** (published 0.2975 ± 0.0086),
h·r_d = **101.54 Mpc** (published 101.54 ± 0.73), χ² = **10.271/11** (published 10.2/11).
**Baseline for scoring: χ² = 10.27 for 11 dof.**

Additionally derived: **F_AP(z) = D_M/D_H** with full covariance propagation. The sound horizon
cancels exactly, leaving pure radial-vs-transverse geometry — six points at 1.6–5.7% precision,
Ω_m = 0.2985, χ² = 5.86/5. **This is the sharp end of T16** and the only part immune to the
r_d degeneracy.

**Provenance note for other lanes:** the DESI portal does not host the DR2 BAO likelihood. Both
DESI sources delegate to `CobayaSampler/bao_data`, so that repo is the *primary* source, not a
mirror. Pinned to tag v2.6 (`b7b8a36e`). Files are LF-only — **CRLF conversion breaks every
hash**, the same trap already recorded for the provenance pins.

**Caveats that will bite, in order of severity:**

1. **These are template fits against a ΛCDM fiducial** (AbacusSummit base_c000), entering at
   grid, template and reconstruction. DESI tested grids off by 4.7–7.5% in distance-redshift
   and found shifts of only a few tenths of a percent in α. So the honest bound is: **a
   redshift law departing from ΛCDM's z→distance relation by more than ~5–8% cannot
   legitimately consume these numbers at all.**
2. The distributed covariance is up to 11% wider than the paper's Table 4 error bars in the
   isotropic direction, but agrees to better than 1% in the anisotropic direction — the one
   T16 discriminates on. Use the covariance files, never hand-entered Table 4 errors.
3. BGS carries no radial/transverse split, so DR2 gives **6** discriminating bins, not 7 (DR1
   gives 5).
4. BAO alone measures only Ω_m and the *product* H₀·r_d. A candidate that "matches BAO" by
   rescaling overall distance normalisation has demonstrated nothing — the content is the
   shape in z and the radial/transverse ratio.
5. Never stack D_V alongside D_M and D_H from the same bin; D_V is derived from them.

No full-shape/RSD, so **T22 is not served by this lane**.

## Cross-lane synthesis, lanes 3 and 4 together

The two acquisition lanes that touch the redshift family arrived independently at the same
structural point, and it is sharper than either test's χ².

- Lane 4: a model asserting a (1+z) clock factor *alongside* an independent photon-energy-loss
  redshift is **double-counting**, and fails T18 on construction.
- Lane 3: a model whose z→distance relation departs from ΛCDM by more than ~5–8% **may not
  consume the BAO compression at all**, because the α values were extracted with a ΛCDM
  template.

Neither is a fit result. Both say the same thing: **E4/E5/E6 are to be judged first on whether
they are even admissible to these tests, and only then on χ².** A redshift-family candidate
that is admissible to BAO is, almost by construction, one that has already reproduced ΛCDM
distances to a few percent — at which point it has not explained the redshift, it has renamed
it. That is the trap to state in the verdict, whichever way the numbers fall.

## Lane 11 — T25 residual discovery: **closed, decisively negative**

The residual target was the program's own closing quantity: y = log₁₀ν_obs − log₁₀ν_RAR, in dex.

### The search space collapses before any fitting — a structural theorem

The eight-variable wish list `Δa = f(a_N, Σ_b, ρ_b, r, M_b, θ, Φ_b, environment)` has **rank 2**
in this bench. Verified by SVD: singular values 1.5×10², 9.6×10¹, then **2.7×10⁻¹²**. To machine
precision,

    log M = log g_b + 2 log r − log G
    log Φ = log g_b +   log r
    log ρ = log g_b −   log r − log G
    Σ_b   = g_b/(πG)        — surface density *is* a_N, exactly

and θ and environment are measured by no probe here. **Six of the eight requested variables are
relabellings of two numbers.** T25 therefore reduces to exactly one question: *does the residual
depend on radius at fixed acceleration?*

### Everything the search selected is a label

Four PySR arms were run — `full` (u, v, sph, cmp), `phys` (u, v), `accel` (u), and a **`LABEL`
arm given nothing but a bare 0/1 indicator**. 44 Pareto members.

```
model                        sparc    xcop   clash  wicker   solar |   kids widebin  blind/base
baseline nu_RAR             0.0937  0.3894  0.5004  0.1915  0.0000 | 0.1340  0.0497   1.00x
0.341 * [bare 0/1 label]    0.0937  0.1559  0.1660  0.1490  0.0000 | 0.1340  0.0497   1.00x
0.330 * sphericity          0.1096  0.1555  0.1741  0.1389  0.0000 | 0.1340  0.0497   1.00x
(sph/(cmp+1.333))^2         0.1038  0.0915  0.1035  0.0243  0.0000 | 0.1340  0.0497   1.00x
best-fitting equation       0.0895 ................................ | 0.3498  ......   2.61x
sparse L1 k=4               0.1221  0.0799  0.0904  0.0233  0.5921 | 0.2356  0.8161   6.09x
```

- **At matched complexity 3 the LABEL arm beats the four-variable arm.** `lab*0.341` (train loss
  3.271e−02) versus PySR's own choice `sph*0.330` (3.475e−02). Given radius, acceleration,
  sphericity and compactness, the search's best simple answer *is* the dataset indicator — and
  the bare indicator does it better than the physics proxy for it.
- **1 of 44 equations beat the blind baseline, by 0.007 dex.** Bootstrap 95% CI on that
  improvement: **[−0.054, +0.038]**, P(actually worse) = 0.367. With 44 equations you expect ~2
  by chance. This is the one I had spotted directly in the Pareto table; it is noise.
- The best-*fitting* equation transfers at **2.6× the baseline**; the sparse model at **6.1×**,
  and it destroys the Solar System bound (0 → 0.592 dex).

### The two tests that settle it

**Within-dataset, 167 SPARC galaxies — one probe, so no label is possible.** After partialling
out a_N, the strongest global-property correlation is log R_max at ρ = +0.176, p = 0.021
uncorrected, **p = 0.107 after Bonferroni over five properties**. Extrapolated to cluster scales
it predicts +0.198 dex at X-COP (observed +0.381) and +0.152 at CLASH (observed +0.500) — and it
**overshoots the blind KiDS probe in the wrong direction** (+0.221 predicted, +0.117 observed).
Within-object radial slopes are **+0.087 dex/dex in SPARC and −0.318 in X-COP** — opposite signs,
where +0.22 in both would be needed.

**Synthetic-null calibration — the decisive one.** Replace the real residuals with
`offset[probe] + noise`, so that by construction they depend on nothing but the dataset name, and
re-run the identical L1 pipeline. It selects **the same variables** (`lgext, comp, comp², exp(−r/λ)`;
mean overlap 2.7 of the first 4) and shows a **larger** apparent cross-validated gain than the
real data — 6.1% versus 1.1%. The real residual is fully accounted for by four numbers, one per
probe: −0.03, +0.34, +0.49, +0.19 dex.

### This is not a power problem

SPARC's a_N-partialled per-galaxy residual has sd 0.191 dex over 167 galaxies, so any global
property explaining more than 2.3% of between-galaxy variance would have shown at 5%. The step
needing explanation — +0.395 dex between SPARC and X-COP at *overlapping* acceleration, across a
full decade of a_N — is 2.1σ of that scatter. **It is a large effect that no measured variable
tracks.**

### Harness bug found, and fixed in `invariant_bench.py`

`Bench.confound`'s second clause `abs(r_vy − r_ly) < 0.08` **was not sign-symmetric**. `r` was
convicted as a LABEL, while `−r`, `1/r` and `exp(−r/λ)` — identical information — all passed. Not
hypothetical: L1's rank-4 selected term was `exp(−r/31.6 kpc)`, and it cleared the mandatory
check. Patched to

```python
abs(r_vl) > 0.8 or abs(abs(r_vy) - abs(r_ly)) < 0.08
```

Verified after the patch:

| variable | corr w/ label | corr w/ residual | verdict |
|---|---|---|---|
| r | +0.458 | +0.421 | LABEL |
| −r, 1/r, exp(−r/31.6 kpc) | −0.458 | −0.421 | **LABEL** (were passing) |
| log ρ_b | −0.395 | −0.341 | LABEL — but marginally, 0.078 against a 0.08 threshold; the
  robust reason is structural, log ρ = u − v − log G is a combination of a convicted coordinate |
| a_N | −0.024 | +0.034 | passes |
| r/extent | +0.056 | −0.051 | passes |

Nothing is rescued by the fix. **Every earlier result stands; this only closes a route by which
a future one could have been wrong.**

### Verdict

Everything found is a label. The galaxy↔cluster offset behaves as a step function of *which
instrument measured the point*, and the bench exposes no column separating SPARC from X-COP other
than the probe name. Distinguishing a genuine missing variable from a systematic between probes —
X-ray versus optical versus lensing calibration, M/L, non-thermal pressure — **requires a probe
that spans both regimes with one instrument.** More search over these columns cannot do it.

That conclusion, reached independently, is exactly the case for lane 6.

## Lane 5 — CMB, closed on the data side

**ACT DR6 primary (T20):** the CMB-only, foreground-marginalised band powers from DR6-ACT-lite
— 127 band powers (45 TT, 40 TE, 42 EE) with the full 127×127 covariance and bandpower windows,
D_ℓ in µK² (confirmed from the likelihood source, not from prose). Plus the full DR6.02 pspipe
release: combined foreground-subtracted TT/TE/EE/BB/TB/EB, all cross-frequency and per-array
spectra, in both Δℓ = 50 and Δℓ = 20 binnings.

**Planck primary (T20):** `plik_lite` v22 (613 band powers, ℓ = 30–2508, C_ℓ in µK², with the
613×613 covariance), the official display spectra, and the 2018 best-fit ΛCDM curve as the null.

**Lensing (T21):** ACT DR6 v1.2 (18 bins, L = 8–2048; baseline bins 3–12), Planck PR4/NPIPE GMV,
PR3 conservative and aggressive, and a 27×27 ACT+Planck joint covariance. The two conventions
were cross-checked against each other — converting ACT's repackaged Planck block via
PP = (2/π)·C_L^κκ reproduces all nine PR4 GMV values to every printed digit, which validates the
κκ↔φφ conversion and the covariance slicing.

**T(z) for T19:** a 45-point compilation, z = 0.037–6.34, transcribed programmatically from the
LaTeX source of Gelo+2022 Table 1. Refitting the transcription gives β = +0.0097 ± 0.012 against
the published (3.9 +7.4 −8.2)×10⁻³ — so the transcription is sound.

**Runnability, stated plainly rather than implied.** **T19 is runnable now with no solver** — it is
a two-parameter fit of T(z) = T₀(1+z)^(1−β). Its other half, that photon energy loss must preserve
a blackbody (FIRAS |µ| < 9×10⁻⁵, |y| < 1.5×10⁻⁵), is a spectral-distortion constraint and no FIRAS
product was fetched. **T20 and T21 are data-complete but solver-blocked**: scoring either needs a
modified Boltzmann solver carrying the candidate's equations, and this project does not have one.
Acquiring the data does not change that. What the data does enable immediately is verifying a
candidate's ΛCDM limit against the observed peaks, and a coarse test on peak positions and
relative heights read off the band powers.

Implementation note for whoever writes the scorer: **apply the bandpower window functions to the
theory curve** rather than evaluating at bin centre — for the lensing bins the centre-value
approximation is not adequate.

Provenance: the Planck Legacy Archive was down for the entire session (503 on every download,
landing page 200). Everything Planck came from official mirrors — IRSA/IPAC, CobayaSampler, and
`carronj/planck_PR4_lensing`. Retry PLA if bit-exact PLA provenance matters for the record.

## Lane 1 — Bullet Cluster, closed on the data side, and T14 is runnable

The Clowe et al. 2006 public release — whose original host is dead — was recovered intact from
the Internet Archive and **validated numerically rather than trusted**:

- **Gas**: `1e0657_central_gasmass_mod.fits`, 185×185 at 1.968″/pix, units of **10¹⁵ M☉ of X-ray
  plasma per pixel** — a true surface *mass* density, not surface brightness. Total 2.229×10¹⁴ M☉.
- **Lensing**: `1e0657.release1.kappa.fits`, dimensionless κ, a direct non-parametric KSB
  inversion with **no light-traces-mass prior**.
- **Shear catalogue**: 2,838 background galaxies with per-galaxy Σ_crit⁻¹, weight and photo-z.
  This is the fundamental observable §10 asks for. RA is in **hours**.
- Clowe Table 2's four 100 kpc apertures, extracted from the arXiv LaTeX.

Validation: gas mass summed in the four Table 2 apertures reproduces the published values to
**0–2.2%**; mean κ at the main BCG is 0.359 against 0.36 ± 0.06; κ peak offsets match to within
one pixel. WCS, units and cosmology confirmed, not assumed. The release's own galaxy catalogue is
permanently lost (404 in every Wayback snapshot) and was substituted with an HSC red-sequence
selection, calibrated by reproducing Clowe's BCG/plasma light contrast: **2.35 against the
published 2.35**.

**T14 is runnable for the Bullet, at two levels.** The minimum test needs no fitting at all: total
baryons are **higher** at the plasma positions (6.83 and 5.92 ×10¹² M☉) than at the BCGs (6.04 and
3.28), while κ̄ is **lower at both**. Any law tying
gravity to local baryons must order κ̄ with baryon mass; the data invert that ordering. **The sign
is the test.** The full test predicts reduced shear at all 2,838 galaxy positions from Σ_gas + Σ_*
alone, with no GR-derived mass map in the loop — that is where B5/B6 (nonlocal, path-accumulating)
become separable from B2/B3/B4.

**A trap named, and it disqualifies the analogues for the primary test.** The HFF CATS κ maps for
A2744 and MACS J0717 are **partly circular**: their own readme states the method assigns a
small-scale dark-matter clump to each major cluster galaxy, giving an explicit one-to-one
correspondence between mass and light. Testing "does lensing follow light or gas?" against them
presupposes part of the answer. They are supporting context only. The Bullet κ map carries no such
prior.

Also flagged: Harvey et al. 2015 Table S1 (the 72-substructure offset catalogue, which would have
made T14 a large-sample test) exists only in the Science supplement — the arXiv source contains no
tables.

## Lane 7 — relativistic promotion: **ζ = 0 is consistent with every probe**

**The lift is degenerate by construction.** A2, A4 and C6≡B3 all prescribe |∇Ψ| = ν(g_b)·g_b with
ν a function of g_b alone. Under D6, Φ_γ+Ψ_γ = (2+ζ)Ψ, so their lensing prediction is the same
curve scaled by K = 1+ζ/2. Verified numerically to machine precision: **at matched g_b the law
cancels exactly out of the photon/matter ratio, so ζ is law-independent and no ζ can reorder the
three survivors.** Separating them needs a lift that is *not* a rescaling — the C-family angular
structure (lane 8) or D1/D2 changing the projected potential's shape (lane 9).

**The fitted value.** KiDS and wide binaries never entered the fit. CLASH (20 clusters) against
X-COP+Wicker (19 in window), matched in acceleration, system-level bootstrap:

    ζ = +0.077 ± 0.167 (stat) ± 0.112 (method) ± 0.332 (baryon calibration) = +0.077 ± 0.388

**ζ = 0 survives at 0.20σ with systematics**; 95% upper limit ζ < 0.72. The blind check, KiDS vs
SPARC, never fitted, gives ζ = +0.150 ± 0.272 — also consistent with zero. Convention note: the
K = 1.130 quoted earlier is ζ = 0.260 under D6, not 0.13.

**A constant ζ is dead regardless of the fit**, because ζ ≡ γ−1 in PPN and Cassini gives
γ−1 = (2.1 ± 2.3)×10⁻⁵ — the central value is 3,300σ out. Only a screened form survives;
ζ = 2ζ_a(1−1/ν) vanishes identically where ν = 1 and passes Cassini exactly.

**The prior 1.130× audited and dissolved.** Reproduced exactly (8/8 cells, median 1.130), then
decomposed: **2 of the 8 cells compare CLASH cluster cores against SPARC spiral disks at ratios
2.96 and 2.57 — that is the cluster anomaly, not a photon effect.** Two more are KiDS, a holdout.
The 8 cells come from **3 probe pairs**, so the sign test's n = 8 was wrong; with 3 units the
smallest attainable two-sided p is 0.25, not 0.0078. Restricted to one object class at matched
acceleration: K = 1.039, under a third of the claimed excess.

**Four independent checks, each fatal on its own:**

- **Exact permutation**: only C(5,2) = 10 ways to split five surveys 2-vs-3. The photon split ranks
  1 of 10 — **p = 0.10, the floor of this design.** No amount of data inside these five surveys can
  beat it; a third independent photon survey is what would move it.
- **BIC**: free per-survey offsets (k=8) beat a single ζ (k=5) by **ΔBIC = 1603**. The residual
  structure is per-survey, not per-channel — and the two photon surveys are not even adjacent
  (CLASH +0.512 dex, KiDS +0.176, with matter surveys at 0.000/+0.242/+0.386 in between).
- **Placebo**: X-COP versus Wicker — two *matter* X-ray cluster surveys that cannot differ by any
  ζ — show δ = +0.095 ± 0.023 dex, a **4.1σ "detection" of ζ = +0.49**. Four times the signal.
- **Within-survey floor**: splitting CLASH's own 20 clusters in half gives sd = 0.034 dex, larger
  than the 0.017 dex signal.

**A second harness bug, found independently of lane 11's.** `Bench.confound`'s rank function uses
`argsort(argsort(v))`, which breaks ties by array position. Because probes are concatenated as
blocks, **any variable constant within a probe scores ~1.0 on corr(variable, label)** — a randomly
shuffled photon flag scores 0.885, and 0.000 once ties are corrected. The first criterion is
unusable for block-constant variables. The second criterion is unaffected and is what decided this
lane.

**And the honest statement on calibration:** a **1.7% error in either survey's baryonic
acceleration nulls ζ exactly**. Against that, the Wicker loader hard-codes a 1.15 stellar
correction, 5 of 12 X-COP clusters have no stellar file and use an invented M_star = 0.10 M_gas,
KiDS uses 1.4× the midpoint of a log-mass bin, and SPARC carries a 0.25 dex IMF caveat. None is
known to a few percent. Worse, the comparison is not novel: CLASH-lensing versus X-ray-hydrostatic
*is* the cluster mass bias, and K = 1.039 is (1−b) = 0.963 — inside the published range and more
conservative than most of it.

**No object in this bench is measured by both a matter and a photon probe.** X-COP ∩ CLASH is
empty. Six redshift-validated CLASH↔Wicker same-object pairs were found by coordinate cross-match,
but at r ≈ 300 kpc versus R500 the answer swings from ζ = +0.92 to +12.8 with the bridging slope,
so it is uninformative.

## Lane 10 — the redshift family: **E4 and E5 killed, E6 survives but is empty**

**A structural identity first: E6 = FLRW × E4.** Integrating E6's extra term along the ray gives
ln(1+z_extra) = (ξ/c²)∫|∇Ψ|dl, which *is* E4. They are the same photon physics and share one ξ,
hence one laboratory bound.

**Theorem (no-dilation).** If photons travel at c through a time-translation-invariant
configuration, both pulses of a pair cross the same interval in the same duration, so the arrival
separation is unchanged — the energy-loss rule never enters. Writing
1+z_obs = (1+z_metric)(1+z_extra) and ε = ln(1+z_extra)/ln(1+z_obs):

    b = 1 − ε        where Δt_obs/Δt_emit = (1+z)^b

**T18 is therefore not a fit — it is a direct measurement of the metric fraction of the redshift.**
Verified numerically to 7×10⁻¹⁴ on a deliberately hostile setup (lumpy *and* explicitly
time-varying loss rate): broadening is exactly zero.

**E4 — killed three independent ways, no data needed.** Predicts b = 0 against b = 0.996 ± 0.019
(**52σ**). Needs ξ ≈ 620 to be the cosmological redshift; **Cassini allows ξ < 1.5×10⁻⁹** — a
twelve-order shortfall, and the bound sticks because the E4 signature is *even* in time about
solar conjunction while Shapiro is *odd*, so it cannot hide inside the fitted γ. |∇Ψ| = 0 in a
homogeneous universe, so E4's redshift is sourced entirely by inhomogeneity: it predicts a
direction-dependent Hubble law and an H₀ that grows as structure grows. The only escape is density
screening — at which point the controlling variable is ρ and the law has silently become E5/B2.

**E5 — killed structurally and then by every dataset.** ξ is exactly degenerate with H\* in the
mean, so E5 has *one* free parameter for the Hubble diagram, not two. Pantheon+ (1371 SNe, full
STAT+SYS): **Δχ² = +1797 with one fewer parameter.** DESI DR2: **Δχ² = +177**. The parameter-free
AP test gives χ² = 37.0 on 6 points with *zero* free parameters against 5.9 for ΛCDM, driven by
Lyα at **+5.3σ**. And structurally: a static universe has no recombination, so E5's BAO scale is
not merely wrong, it is **undefined**. It also violates Etherington duality.

**E6 — survives, and is empty.** It inherits time dilation from the metric, so the theorem only
bounds it. T18 gives |ξ| < 22, DESI < 56, Pantheon+ < 128 — but **Cassini gives < 1.5×10⁻⁹**,
ten orders tighter. E6's non-metric channel is pinned to ε < 2.4×10⁻¹² at z = 1, under one part
in 10¹¹ of the observed redshift. Not falsified; unable to do anything.

**The §13 trap made numerical.** Generalised tired light D_L = (c/H)ln(1+z)(1+z)^p with p free
fits Pantheon+ at **Δχ² = +10 over 1371 SNe** — the supernova Hubble diagram alone genuinely
cannot tell it from ΛCDM. But the physics *fixes* p = 0.5, and that is what dies. Three
independent measurements of the metric fraction agree and all exclude it:

| measurement | value | significance |
|---|---|---|
| T18 raw light-curve widths | b = 0.996 ± 0.019 | 52σ from 0 |
| T17 alone | p = 1.091 ± 0.119 | 5.0σ |
| **T16+T17 jointly**, Ω_m fixed at DESI | **p = 0.9753 ± 0.0153** | **1.6σ from GR, 31σ from no-dilation** |

The joint constraint never touches a light curve, so it is fully independent of T18. Two escapes
closed: Blondin's **spectral** aging rates give b = 0.94 ± 0.10 (9.1σ), and spectral evolution is
not the standardisation variable; and model-free, DES mean widths grow 1.63× while (1+z) grows
1.62×, a **6.2σ shift of the mean** relative to the low-z intrinsic scatter — no selection from a
fixed parent population can move a mean 6.2 sd.

**Cross-lane consequence.** A universal path-accumulated photon–gravity coupling cannot be the
origin of the D6/ζ excess: Cassini bounds the redshift channel at ξ < 1.5×10⁻⁹ and the deflection
channel at |γ−1| < 2.3×10⁻⁵, both seven to eight orders below what a 13% effect needs. Whatever
ζ was, it had to be scale- or environment-dependent, or a probe systematic. **Lane 7 then measured
it to be zero.** Two lanes closed the same door from opposite sides.

## Lane 12 — the cluster a₀ radial run, measured from lensing only

**The strong radial run survives; it is not a hydrostatic artefact.** Reproduced by three samples
sharing no clusters, no lensing pipeline and no gas pipeline.

| r/R500 | a₀ / 1.2e-10 | ± dex | source |
|---|---|---|---|
| 0.073 | 21.95 | 0.049 | CLASH fig2 (Umetsu+16 SL+WL+magnification) |
| 0.291 | 13.30 | 0.038 | CLASH fig2 |
| 0.698 | 5.90 | 0.051 | CLASH × Umetsu+16 M1000c |
| 1.000 | **3.67** | 0.055 | CLASH × Umetsu+16 M500c (n=11) |
| 1.000 | **2.71** | 0.064 | XXL: Umetsu+20 WL × Eckert+16 gas (n=36) |
| 1.060 | **2.88** | 0.057 | X-COP gas × Herbonnet/LC² WL (n=7) |
| 1.500 | 1.00 | 0.074 | XXL M200 (n=42) |
| 1.510 | 1.38 | 0.060 | X-COP gas × WL M200 (n=5) |
| — | 0.86 | — | *galaxy outer third* |

Slope beyond 0.25 R500: **−1.354 ± 0.071 dex/dex.** The three independent R500 measurements agree
within 0.13 dex.

**The hydrostatic bias is in the normalisation, not the slope.** On a common r/R500 axis the
lensing/X-ray ratio is 0.92, 1.37, 2.29, 1.69, 1.70, 1.70 from 0.08 to 1.03 R500: the probes agree
inside 0.2 R500, then lensing sits a *constant* **1.84 ± 0.15** higher — a mass offset of 1.36,
i.e. b = 0.26 saturating by 0.2–0.3 R500, which is the textbook non-thermal-pressure shape. Outer
slopes: lensing −1.35, X-ray −1.02. **This refines the standing caveat: X-ray radial shapes are
biased in the inner 0.3 R500 while the bias grows, but the outer logarithmic slope is not
manufactured.**

Within CLASH alone (100→600 kpc, paired inside each cluster, 11 clusters present at every radius)
a₀ falls by −0.347 ± 0.057 dex, 10 of 11 negative, radius-label permutation **p = 0.003**.

**The headline the program was looking for.** At R200 the two lensing measurements give 1.00× and
1.38×, mean **1.17× canonical**, against **0.86× for galaxy edges** — a residual factor of **1.4**,
with essentially no extrapolation. Compare the factor 16–18 quoted for cluster cores. The
extrapolated crossing sits at r/R500 = 1.9–2.5.

Mandatory check passed, in its strongest form: `b.confound` on r/R500 returns "carries information
beyond the dataset label" (var-label +0.056), and **within X-COP alone — where the label is
constant and no label effect is possible — corr(r/R500, RAR residual) = −0.788.**

Caveats carried: **a₀ is not a function of absolute radius** (at ~575–600 kpc, XXL gives 2.71× and
CLASH 8.9×, a factor 3.3 apart — r/R500 organises the data, absolute radius does not); beyond R500
the lensing masses are NFW extrapolations so the R200 points are model-dependent while the R500
points are robust; and R500 is itself derived from the lensing mass, so the cross-sample r/R500
comparison has partial covariance, though the within-CLASH physical-radius sequence does not.

## Lane 9 — nonlinear Poisson, disk versus sphere: **geometry cannot do it**

The solver was verified before anything was read off it. Finite-volume 2D axisymmetric (R,z) for
D1, damped Picard with a sparse direct solve, converged on two simultaneous criteria: sup-norm ΔΨ
per step over total potential drop < 10⁻¹¹ (reached ~5×10⁻¹²) **and** relative L1 residual of the
*nonlinear* equation < 10⁻⁸ (reached 10⁻⁹–10⁻¹⁰).

Verification against exact solutions: Newtonian sphere 2.8×10⁻³; Newtonian **flattened**
Miyamoto–Nagai a/b = 20 — an exact analytic pair for the geometry that actually matters — 3.4×10⁻³;
AQUAL sphere against the exact μ(g/a₀)g = GM(<r)/r² 2.9×10⁻³. Refines to 2.7×10⁻⁴ with grid;
boundary at 2000 versus 8000 kpc no change; μ-floor 10⁻⁴/10⁻⁶/10⁻¹⁰ identical. Rotation curve for
M = 5×10¹⁰, R_d = 3 kpc flattens at 168.8 km/s against (GMa₀)^¼ = 168.0.

Worth recording as a numerical fact: **undamped Picard never converges on this equation.** The
local map is g_{k+1} = g_N/μ(g_k) ~ g_k^{−s} with s = dlnμ/dlny ∈ [0,1]; its derivative at the
fixed point is −s, so ω = 1 is neutrally stable in the deep-MOND limit and oscillates forever.
ω = 0.5 contracts for all s.

**The key number.** At matched enclosed baryonic mass and matched radius, Q = g_disk/g_sphere peaks
at **1.21–1.22** for μ = y/(1+y) and **1.20–1.25** for μ = y/√(1+y²), across the oblate q→0 and
exponential-disk families. The Newtonian counterpart is 1.33. **The disk is the strong one.**

At matched *acceleration* — which is the actual observational statement — in the cluster window
g_bar/a₀ = 0.03–0.32 the sphere/disk boost ratio is **0.994**. That is 0.6%, **with the wrong
sign**, against a required 2.0 (15 in cores).

**It fails structurally, not numerically.** Q → 1 and B → 1 as g/a₀ → 0: this is Milgrom's
asymptotic theorem, that in the deep-MOND limit the D1 operator is the scale-free 3-Laplacian and
the far field of any bounded source is √(GMa₀)/r, shape-independent. Measured directly:
Q = 1.21 → 1.065 → 1.018 → 1.0056 → 1.0008 as g_pt/a₀ falls 0.52 → 0.00094. **D1's shape dependence
peaks where the discrepancy is smallest and dies where it is largest.** The one place the sign is
right (sphere/disk = 1.32) is at g_bar/a₀ = 2.8 — the Newtonian regime, where nothing is missing.

**D2 (fractional gravity)**, solved exactly by Fourier–Hankel quadrature with no grid (validated at
s = 1: Plummer 2.6×10⁻¹⁴, razor-thin disk vs Freeman 9.8×10⁻⁵), shows the same structure:
disk/sphere maximum falls 1.292 → 1.216 → 1.136 → 1.082 → **1.058 at s = 1.49**, shrinking toward
the flat-rotation-curve limit. And **D2 is separately killed on T09**: a flat curve requires
s = 3/2 exactly, where v² = 2GL^α M/π, so M ∝ v² and the **BTFR slope is 2.000** (confirmed
numerically at s = 1.40, 1.45, 1.49) against the measured 3.83 ± 0.11. Rescuing it needs L to vary
per object, which is a per-galaxy gravity parameter and forbidden by §8.

**Geometry cannot explain the cluster offset.** This is derived from the operator, not fitted, and
is independent of a₀, of the μ function, of the profile family, and of scale. It closes the D1/D2
route to shape-prediction — and it is the one shape result in this project that needed no confound
check, because nothing was fitted.

---

# ROUND 2 — the integrated result

Five lanes closed a possible explanation of the cluster excess. Each did it from a different
direction, and none of them left the door open:

| explanation | closed by | how |
|---|---|---|
| a missing variable in the eight | lane 11 | the wish list is **rank 2**; everything the search found is a dataset label; the synthetic null reproduces it |
| gravitational slip (photons ≠ matter) | lane 7 | ζ = +0.077 ± 0.388; ζ = 0 at 0.20σ; the placebo split of two *matter* surveys gives 4× the signal |
| geometry / shape | lane 9 | Q ≤ 1.25 and → 1 exactly where the discrepancy is largest; **0.994, wrong sign**, in the cluster window |
| a photon–gravity coupling along the path | lane 10 | Cassini bounds ξ < 1.5×10⁻⁹, seven to eight orders below what 13% needs |
| hydrostatic bias manufacturing the radial run | lane 12 | three independent lensing samples reproduce the run; the bias is a **flat 1.84× in normalisation**, not in slope |

## The measurement that ties them together

With every one of those closed, I ran the comparison the rank-2 theorem makes decisive, and then
tested it against the probe that was never fitted.

**Step 1 — where are galaxies and clusters actually comparable?** Because the bench is rank 2,
every law in the eight variables is a surface ν = f(u, v), u = log a_N/a₀, v = log r/kpc. Galaxies
occupy v = 0.3 to 1.8; clusters v = 1.4 to 3.2. At 0.25 dex resolution they share **7 cells of 65
and 32**, and only **3.5% of cluster points have a galaxy within 0.25 dex.** Ninety-six percent of
every galaxy-versus-cluster statement ever made in this document was extrapolation across an empty
region.

**Step 2 — in the sliver where they are comparable, they disagree.** Per object, so nothing votes
twice:

| comparison | objects | ratio | 95% CI |
|---|---|---|---|
| matter-probe clusters (X-COP) vs galaxy edges | 6 | **4.07×** | [2.71, 6.16] |
| photon-probe clusters (CLASH, pooled) vs galaxy edges | pooled, 50 galaxies | **2.93×** | permutation p = 0.0005 |

The matter-probe number is the important one: hydrostatic gas measures Ψ exactly as rotation curves
do, so **no slip enters it**, and lane 7 had already measured the slip to be zero anyway. Dropping
the 1.6% of cluster bins with ν < 2 — profile artefacts where the hydrostatic derivative changes
sign — moves 3.86 to 4.07, so it is not those. And hydrostatic bias runs the wrong way to help:
correcting X-COP masses *upward* by lane 12's 1.84 makes the discrepancy larger, not smaller.

Lane 7 arrived at the same number independently while auditing the 1.130× photon claim: **2 of its
8 cells were CLASH cores against SPARC disks, at 2.96 and 2.57.** Same measurement, found from the
opposite direction, and correctly identified there as the cluster anomaly rather than a photon
effect.

**Step 3 — the group bridge fills the mass gap but not the one that matters.** Lane 6's 6,044 rows
in 10^11.6–10^13.2 M☉ close the mass gap with no empty 0.2-dex bin, and three independent methods
agree on ν to ~10% (Sun+2009 hydrostatic 12.3, Lovisari+2015 hydrostatic 13.0, XXL×Umetsu
**lensing-only** 11.8). But groups are measured at r500 and r2500 — hundreds of kpc — so they share
**zero** cells of the (u,v) plane with galaxies. They extend the cluster branch downward in mass
rather than bridging to galaxies. **The gap that blocks this problem is in radius, not in mass.**

**Step 4 — the blind probe, and this is the result.** One probe in the bench measures gravity around
*galaxy-mass* objects across *cluster-scale* radii: KiDS galaxy–galaxy lensing, 35 to 2600 kpc
around 10^10.7 M☉ lenses. It is a permanent blind holdout. It has never entered a fit, and it did
not enter one here — a holdout exists to be predicted, and the rule forbids fitting it, not looking
at it.

| probe | kind | role | ν/ν_RAR | median \|log10\| |
|---|---|---|---|---|
| sparc | matter | fit | 0.97 | 0.094 |
| **kids** | **photon** | **BLIND** | **1.31** | **0.134** |
| **widebin** | **matter** | **BLIND** | **0.90** | **0.050** |
| wicker | matter | fit | 1.55 | 0.192 |
| xcop | matter | fit | 2.40 | 0.389 |
| clash | photon | fit | 3.17 | 0.500 |

**Both blind holdouts land on the RAR.** KiDS tracks it to 0.134 dex across nearly five decades of
acceleration, u = −5.9 to −1.2, with the canonical a₀ = 1.2×10⁻¹⁰ and no free parameter. Matched
directly against the galaxy branch at 0.20–0.40 dex it gives **1.15–1.19×**, on 10–12 points.

Its bin-by-bin run against the unmodified RAR:

```
   r kpc     n      u    nu obs   nu RAR   ratio
   30-60     8  -1.44     4.82     5.79    0.83
   60-120    8  -1.98    10.50    10.26    1.09
   120-250  12  -2.68    26.26    22.46    1.23
   250-500   8  -3.31    63.51    45.77    1.37
   500-1000  8  -3.84   114.07    84.15    1.44
   1000-2000 12 -4.54   231.46   188.79    1.39
   2000-3000  4 -5.04   209.64   336.94    0.81
```

Restricting to r < 500 kpc, where stacked ΔΣ is dominated by the lens itself rather than by
correlated neighbours, gives 1.18 — so the conclusion is not the two-halo term talking.

**What this settles.** KiDS is a *lensing* probe at *hundreds of kpc to Mpc*. It sits with the
rotation curves and not with the clusters. Therefore the cluster excess is:

- **not a photon effect** — KiDS is photons and shows none of it (and lane 7 measured the slip
  directly at zero);
- **not a radius effect** — KiDS reaches 2.6 Mpc and shows none of it;
- **not a shape effect** — lane 9 closed that from the operator;
- **not a missing point-level variable** — lane 11 closed that by rank and by synthetic null;
- **not hydrostatic bias** — lane 12 measured the bias and it runs the wrong way.

**It is a property of clusters, and lane 12 says which property: r/R500.** a₀ runs from 22× canonical
at 0.07 R500 to 1.17× at R200, slope −1.354 ± 0.071 dex/dex, confirmed by three independent lensing
samples, passing the confound check in its strongest within-probe form (corr = −0.788 inside X-COP
alone, where no label exists).

## What that costs, stated plainly

r/R500 is not one of the eight variables, and it cannot be — it is an **object-level** quantity, not
a property of the point. The rank-2 theorem says the point-level wish list spans exactly (a_N, r).
So the surviving description requires the law to depend on something about the *system as a whole*,
which is precisely what §8's "no per-object parameters" rule was written to forbid.

That is not yet a contradiction: r/R500 is a measurable property of the configuration, not a fitted
per-object constant, and a nonlocal law could in principle produce it. But it is the sharpest
statement the program has reached, and it is the one to attack next:

1. **Is r/R500 a proxy for something local?** R500 is defined by an overdensity criterion, which is
   cosmological in origin. A law-free replacement is needed — for instance r/r_{a₀}, where
   r_{a₀} is the radius at which the *baryonic* g_bar crosses a₀. That is defined identically for a
   galaxy, a group and a cluster, uses only baryons, and needs no halo model. **If the a₀ run
   collapses onto r/r_{a₀}, the object-level dependence becomes a profile-shape dependence, which
   a nonlocal law can supply.** If it does not, r/R500 is a cosmological label and the result is
   weaker than it looks.
2. **The KiDS agreement must be re-derived at its own r/R500.** KiDS lenses are 10^10.7 M☉, so
   2.6 Mpc is a large multiple of their R500 — which, on lane 12's relation, is exactly where a₀
   should return to canonical. That would make KiDS not a contradiction of the cluster run but a
   confirmation of it, sampled at the far end. This is the single most informative check remaining
   and it costs nothing.
3. **T14 is now runnable and is orthogonal to all of this.** The Bullet's minimum test needs no
   fitting: baryons are higher at the plasma than at the BCGs while κ̄ is lower at both. Any law tying gravity to local baryons must order κ̄ with baryon mass, and the data invert
   it. That is a sign test, and it is the one test in the gauntlet that a nonlocal law and a local
   law answer differently.

## Harness changes made this round

- **`Bench.confound` sign-symmetry fixed** (lane 11). The clause `abs(r_vy − r_ly) < 0.08` convicted
  `r` but passed `−r`, `1/r` and `exp(−r/λ)`, which carry identical information — and an L1 fit had
  in fact selected `exp(−r/31.6 kpc)` and cleared the check with it. Now
  `abs(abs(r_vy) − abs(r_ly)) < 0.08`. Verified: `−r`, `1/r`, `exp(−r/λ)` and `log ρ_b` all flip to
  LABEL; `a_N` and `r/extent` still pass. Nothing earlier is rescued or overturned.
- **A second bug identified and not yet fixed** (lane 7): the rank function uses
  `argsort(argsort(v))`, which breaks ties by array position, so any variable constant within a
  probe scores ~1.0 on `corr(variable, label)` — a randomly shuffled photon flag scores 0.885, and
  0.000 with tie-corrected ranks. The first criterion is unusable for block-constant variables. The
  second criterion is unaffected, which is why lane 7's verdict stands.
- **CLASH has no object identity in the bench** (`extent` is constant across all 84 rows), so every
  per-object CLASH statement is really a pooled-sample statement. X-COP has 12, Wicker 120, SPARC
  167.

## Lane 8 — the directional family, scored out of the disk plane

Two observables reach out of the plane, and both were used. All four laws come out **consistent
with ε = 0**, and since C1/C2/C5 are exactly null *in* the plane, they were never candidates for
the missing gravity itself — they are additive corrections, now bounded to a few tenths.

| law | Milky Way K_z | lensing quadrupole | combined 95% |
|---|---|---|---|
| C1 tensor | ε = −0.23 ± 0.12 | [−0.53, +0.67] | **\|ε\| < 0.46** |
| C2 latitude | [−5.4, +2.6] (56× geometric suppression) | [−1.33, +1.05] | **\|ε\| < 1.3** |
| C3 quadrupole | ε = +0.16 ± 0.08 | [−0.53, +0.67] | **\|ε\| < 0.31** |
| C5 ε(r) | pins r_ε ≲ 10 kpc | pins r_ε to ~1 Mpc | **\|ε_∞\| < 0.6** |

Theory derived from scratch and validated against the analytic point lens to five decimals: C1's
monopole boost is d ln⟨γ_t⟩/dε = sin²(i)/2, averaging to exactly 1/3 over random orientations, with
the quadrupole **exactly zero for a point mass and exactly zero face-on**. C2 gives 1/3 at *every*
inclination and a **negative** quadrupole — an apparent halo anti-aligned with the light, which is
the clean discriminator against C1. **C3's monopole averages to zero**, so ordinary stacked lensing
is blind to it.

**C3 is ill-posed**, which is worth recording as a structural result rather than a bound:
Ψ = Φ_b[1+εP₂] is not invariant under Φ_b → Φ_b + C. Against a flat rotation curve, where Φ = v²ln r
has no natural zero, shifting the gauge from C = 0 to C = 5 moves the monopole slope from +0.004 to
+0.94 and the quadrupole from 1.17 to 9.6. It is only well posed against a finite-mass system, and
even then its prediction is set by the absolute depth of Φ_b — i.e. by mass far outside the
measurement.

**The mandatory check explains why the bench could never have scored this family.** `b.confound` on
the C-family's own predicted signature (1/3 for photon probes, 0 for in-plane matter) returns
**corr with the label = +0.992 → LABEL**. The bench structurally cannot test C1/C2/C3/C5 by probe
comparison, which is why both tests here are *within-measurement* comparisons.

**Halo ellipticity is the systematics-immune test.** The complete published record for
late-type/blue/spiral lenses is six measurements, all consistent with zero; combined over three
independent samples with error inflated for χ² = 6.8/2, **f_h = +0.064 ± 0.274**. Predicted f_h/ε is
+0.92 for C1/C3/C5 and −0.46 for C2. **No misalignment dilution applies to the C-family — n̂ *is* the
disk normal by construction** — whereas ΛCDM predicts f_h ≈ 0.02 for late types. So these nulls do
not constrain ΛCDM and do constrain the C-family. The bound moves between 0.19 and 1.19 with the
assumed Σ slope.

**Incidental finding, and it is a significant one.** The KiDS-1000 lensing RAR is **not a function
of g_bar alone**. At fixed g_bar, ln(g_obs) differs by **−0.407 ± 0.061 (6.7σ)** between Sérsic
n < 2 and n > 2, and **−0.515 ± 0.064 (8.0σ)** between blue and red, with a further mass trend
d ln(amplitude)/d log M\* = **+0.109 ± 0.054**. The RAR files were verified to be a genuine per-lens
g_bar binning rather than relabelled radius bins. This is **not** C1 — it implies ε ≈ −1.0, which
would zero the vertical restoring force in every disk. The colour split is the larger of the two and
is the M/L axis, so a stellar-mass calibration error reproduces it; the Sérsic split is a
profile-shape axis and is the more interesting of the two. Any pure g_obs = F(g_bar) law forbids
all three.

**What would settle the family:** nobody has published galaxy–galaxy lensing split by lens
inclination or axis ratio. C1 predicts zero face-on and maximum edge-on; C2 predicts the same signal
at every inclination. One such split separates them.

---

# ROUND 2 — the closing measurement, and what it costs

## The acceleration-scale sequence

Every probe's deviation from the unmodified RAR, translated into the a₀ it implies. Five of the six
probes sit in the deep-MOND regime where ν ≃ √(a₀/g_bar), so ν/ν_RAR = √(a₀_eff/a₀) exactly.

| probe | kind | role | ν/ν_RAR | implied a₀/a₀_canonical | fraction of points with g_bar > a₀ |
|---|---|---|---|---|---|
| widebin | matter | **BLIND** | 0.90 | 0.81 | 0.667 |
| sparc | matter | fit | 0.97 | 0.94 | 0.208 |
| **kids** | **photon** | **BLIND** | **1.31** | **1.72** | **0.000** |
| wicker | matter | fit | 1.55 | 2.40 | 0.000 |
| xcop | matter | fit | 2.40 | 5.76 | 0.000 |
| clash | photon | fit | 3.17 | ~10 | 0.071 |

Lane 12 measured the cluster end of that sequence independently, from lensing alone, and got the
same shape: 22× at 0.07 R500 falling as (r/R500)^−1.354 to **1.17× at R200** against **0.86× for
galaxy edges**.

**The correct statement about the blind probe is therefore more precise than "KiDS lands on the
galaxy branch".** Matched cell-by-cell against galaxies it gives 1.15–1.19×; globally it gives 1.31×
against SPARC's 0.97, i.e. a₀ higher by **1.8×**. That is a real offset — but clusters are 6× to 10×,
so KiDS sits with the galaxies by a wide margin and nowhere near the clusters, while sampling
*exactly* the radii and *exactly* the observing channel that the cluster excess was attributed to.

## A structural fact that had not been noticed, and it constrains what can be tested

**No X-COP, Wicker or KiDS point has baryonic g_bar above a₀. Not one.**

| probe | g_bar/a₀ range | r range (kpc) |
|---|---|---|
| xcop | 0.030 – 0.321 | 121 – 1644 |
| wicker | 0.032 – 0.173 | 921 – 1713 |
| kids | 6.0×10⁻⁷ – 0.103 | 35 – 2604 |
| clash | 0.109 – **1.79** | 14.3 – 600 |
| sparc | 0.0017 – 61.2 | 0.08 – 108 |

Only CLASH, reaching 14.3 kpc into cluster cores, crosses a₀ at all — and only in 7% of its points.
Every cluster measurement in this program is a **deep-MOND** measurement. Two consequences follow
immediately:

1. The cluster probes cannot constrain the *shape* of the interpolating function, only its
   amplitude. Any statement here that distinguishes A1 from A2 from A3 using clusters is really a
   statement about a₀.
2. Any object-level scale defined by the crossing g_bar = a₀ is **unmeasurable** for clusters — it
   lies inside the innermost data point.

## The law-free replacement for r/R500 was tested and does not work

Lane 12's organising variable r/R500 imports a cosmological overdensity definition. The obvious
law-free replacement is t = r/r_{a₀}, where r_{a₀} is the radius at which the object's own
**baryonic** g_bar crosses a₀ — defined identically for a galaxy, a group and a cluster, needing no
halo and no cosmology. It was computed per object from the measured mass profiles, with the
point-mass value carried alongside as the rank-2 null (for a point mass t ≡ x^(−1/2) exactly, so any
collapse onto that would be circular).

**It fails, in two independent ways.**

| t = r/r_{a₀} | clash | kids | sparc | xcop |
|---|---|---|---|---|
| 1 – 3.16 | **2.80** | . | **0.96** | . |
| 3.16 – 10 | **3.88** | **0.94** | **1.00** | . |
| 10 – 31.6 | **3.55** | **1.15** | **0.96** | 3.05 |

At matched t, CLASH sits 3–4× above SPARC and KiDS. The variable does not unify them.

And where it *is* measurable it carries almost nothing new: median log₁₀(t / x^(−1/2)) is **0.000 dex
for KiDS** (whose baryonic mass is constant per bin, so it is a point mass exactly) and **+0.062 dex
for SPARC**. The large per-probe offsets — +0.797 for CLASH, +2.360 for X-COP — come from
extrapolating the crossing inward past the innermost data point, which is not a measurement.
`b.confound` passes t ("carries information beyond the dataset label", var-label +0.154), but that
clearance is on the strength of numbers that are extrapolations.

**So the honest position is that r/R500 organises the cluster population internally — lane 12
established that well, and within X-COP alone where no label exists — but there is no law-free
version of it in this bench that also reaches galaxies.** Obtaining one requires baryonic profiles
inside ~30 kpc of cluster cores, where BCG stellar mass would push g_bar above a₀. That is a
specific, small, and available acquisition, and it is the one that would convert lane 12's relation
from a cluster-internal scaling into a candidate law.

## Where round 2 leaves the program

**Closed, with reasons:**

- E4, E5 killed; E6 survives but is bounded to one part in 10¹¹ of the redshift (lane 10)
- D1, D2 killed as an explanation of the cluster offset, from the operator rather than by fitting;
  D2 additionally killed on the BTFR slope, 2.000 against 3.83 ± 0.11 (lane 9)
- C1, C2, C3, C5 bounded to |ε| ≲ 0.3–1.3 and consistent with zero; C3 shown to be gauge-dependent
  and therefore ill-posed (lane 8)
- D6/ζ measured at +0.077 ± 0.388 and consistent with zero, with a placebo split of two *matter*
  surveys producing four times the signal (lane 7)
- T25 residual discovery: everything found is a label, established by rank, by synthetic null, by
  leave-one-probe-out, and by a within-dataset test with Bonferroni (lane 11)

**Still standing:** A2, A4, C6≡B3, all three degenerate under any rescaling lift, all three unable
to produce the cluster amplitude at any radius, and all three now known to be untestable in *shape*
by cluster data because clusters are entirely deep-MOND.

**The one unexplained fact, stated as precisely as this program can now state it:** at matched
acceleration and matched radius, in the matter sector where no slip can enter, clusters sit
**4.07× (95% CI 2.71–6.16, six X-COP objects)** above galaxies; the excess is organised by r/R500
with slope −1.354 ± 0.071 and falls to a factor 1.4 at R200; it is not shape, not slip, not photons,
not radius, not hydrostatic bias, and not any point-level variable in the eight-variable wish list.

**Next, in priority order:**

1. **T14 on the Bullet.** Runnable now, needs no fitting, and is the one test in the gauntlet that a
   local and a nonlocal law answer differently: baryons are higher at the plasma than at the BCGs
   while κ̄ is lower at both, so any law tying gravity to local baryons must order κ̄
   with baryon mass and the data invert it.
2. **Cluster core baryons inside 30 kpc**, to make r_{a₀} measurable for clusters and test whether
   lane 12's r/R500 relation has a law-free form.
3. **Galaxy–galaxy lensing split by lens inclination**, which separates C1 from C2 and would tighten
   ε by the sample-size gain over the quadrupole estimator.
4. **A Boltzmann solver** — T20 and T21 are data-complete and solver-blocked, and no amount of
   further acquisition changes that.

---

# T14 — the Bullet Cluster, run and scored

Run directly rather than by a lane, because lane 1 delivered the data and the test needed the
solved field rather than a sign argument. Two things had to be corrected first, and both are worth
recording because either would have produced a wrong result.

## Correction 1 — the published κ̄ values are not comparable to each other

Lane 1 reported the excess as "an order of magnitude", quoting Clowe's κ̄ = 0.05 and 0.02 at the
plasma against 0.36 and 0.20 at the BCGs. **Those are different quantities.** Clowe et al. state it
in the Table 2 note and again in the text:

> The mean κ at each BCG was calculated by fitting a two peak model, each peak circularly
> symmetric, to the reconstruction and subtracting the contribution of the **other** peak at that
> distance. The mean κ for each plasma cloud is the **excess** κ after subtracting off the values
> for **both** peaks.

So 0.36 is a total minus one neighbour and 0.05 is a residual after removing a fitted
two-component model. Comparing them puts most of the subtracted model into the answer. **The
"order of magnitude" framing has been corrected throughout this document.**

The test does not need those numbers. Direct aperture means measured off the released κ map are
comparable to each other, and they carry the same signal:

| aperture | M_gas (map) | M_gas (pub) | M_star (pub) | M_bar | κ̄ (map) |
|---|---|---|---|---|---|
| main cluster BCG | 5.57 | 5.5 | 0.54 | 6.04 | 0.3594 |
| main cluster plasma | 6.60 | 6.6 | 0.23 | 6.83 | 0.2494 |
| subcluster BCG | 2.76 | 2.7 | 0.58 | 3.28 | 0.2911 |
| subcluster plasma | 5.86 | 5.8 | 0.12 | 5.92 | 0.2144 |

**In 2 of 2 apertures the plasma holds more baryons and carries less convergence** — 1.13× the
baryons at 0.69× the κ in the main cluster, 1.80× at 0.74× in the subcluster. No model subtraction,
nothing fitted. Flipping the ordering would need the BCG stellar masses to be **2.5× and 5.6×
larger**, and Clowe states they are already upper limits because no colour selection was applied.

## Correction 2 — the "local baryons" argument was too strong, and is withdrawn

The natural next sentence — that this excludes any law where κ is a monotone functional of the
local baryons — does not apply to the survivors. A2, A4 and C6≡B3 are algebraic in g_N, and **g_N is
not local**: it solves Poisson's equation, so the field at the plasma already contains the BCG's
mass. The effective source is

    ∇·(ν g_N) = 4πG ρ_bar ν  +  g_N·∇ν

and the second term, the phantom, lives where ν *changes*, not where the baryons are. Whether it
can manufacture a 130–180 kpc offset is a question about the equation, and it has to be answered by
solving it.

## The solve

QUMOND, which is explicit — ∇²Ψ = ∇·[ν(|∇Φ_N|/a₀)∇Φ_N], one Poisson solve, one divergence, one more
Poisson solve, no iteration and no convergence question. ν is exactly the surviving A-family's
interpolating function, so the answer applies to A2 and A4 directly.

Built on the **released gas surface-density map itself** given a line-of-sight profile, so the
projected gas distribution is the measured one by construction, plus compact stellar components at
the BCGs carrying Clowe's upper-limit masses. 160×160×80 grid at 15.6 kpc, isolated boundaries by
zero-padding to twice the box in each direction.

**Validated before anything was read off it:** regridded gas aperture masses reproduce the published
values to 0.3–4%; the Newtonian projected effective mass recovers the input baryons to 0.15%; and
the model's baryon aperture ratios (1.09, 1.73) reproduce the measured ones (1.13, 1.80).

## The result, smoothed to the reconstruction resolution

Clowe's map carries a Gaussian smoothing of σ ≈ 35 kpc around the bullet and ≈ 45 kpc around the
main cluster. Smoothing blends apertures only ~200 kpc apart and therefore pushes every ratio
*toward* 1, i.e. toward the observation — this is the generous version of the comparison.

| model | main plasma/BCG | sub plasma/BCG | inverted |
|---|---|---|---|
| baryons | 1.09 | 1.73 | 0 of 2 |
| Newton | 1.09 | 1.76 | 0 of 2 |
| QUMOND, simple ν | 1.13 | 1.57 | 0 of 2 |
| **QUMOND, RAR ν (A2)** | **1.13** | **1.57** | **0 of 2** |
| **OBSERVED κ** | **0.69** | **0.74** | **2 of 2** |

**The phantom term barely moves the ratios: 1.09 → 1.13 and 1.73 → 1.57.** In the subcluster it
does push in the right direction, but by 9% where a factor of 2.1 is needed. Both predictions sit
on the wrong side of unity while both observations sit on the other side.

**T14: A2, A4 and C6≡B3 fail.** Computed, not asserted.

**Caveats carried.** The line-of-sight structure is a model (Gaussian, σ = 250 kpc) even though the
projected distribution is exact — defensible for a merger seen near the plane of the sky, but it is
an assumption. The stellar component at 30 kpc is under-resolved on a 15.6 kpc grid, which is why
the **peak-position** test is not used here: after resolution matching, the model peaks are driven
by the extended gas and the comparison is not meaningful at this grid scale. **The aperture test is
the one that is reliable, and it is the one quoted.**

## What T14 does and does not close

It closes the *algebraic* branch: every surviving candidate maps g_N to g pointwise, and none of
them can invert the aperture ordering. It does **not** close nonlocal laws — B5 and B6, the void
and path-accumulation family, are not functions of g_N at a point and were not tested here.

But those were already killed at T03, for a reason that has not gone away: interplanetary space has
ρ ~ 10⁻²⁰ kg/m³, far below any galactic ρ_c, so a **density-triggered** enhancement switches fully
**on** inside the Solar System instead of off. The emptiest place in the data is the Solar System.

So the position after round 2 is that the local branch is closed by T14 and the density-triggered
nonlocal branch is closed by T03, and no candidate written down in this program survives both.
A nonlocal law whose trigger is **not** the local density — the one gap left in that argument — has
not been written down yet, and writing one is what round 3 is for.

---

# The constructive question: does a formula exist at all?

Everything above is elimination, and elimination only rules out what was written down. This asks the
question the other way, and the answer is **yes with a precisely stated cost** — not no.

## Step 1 — one free scale per object is a complete description

Give each object a single free number a₀_obj and fit ν(g_bar/a₀_obj) — one parameter, no shape
freedom. Across every object where a₀ is *identifiable* (an interior minimum in the profile
likelihood, not a grid edge, ≥3 points): **median residual 0.097 dex, 185 objects, galaxies through
clusters alike.** Nothing else is missing from these data. The entire problem is that one number.

## Step 2 — but within galaxies, that number is not real

The decisive check is out-of-sample and no dataset label can pass it: **fit a₀ on the inner half of
each object's radial range and predict the outer half.**

| probe | objects | outer-half error, fitted a₀ | same, canonical a₀ | gain |
|---|---|---|---|---|
| sparc | 159 | 0.112 | 0.093 | **0.83× — worse** |
| kids [BLIND] | 4 | 0.170 | 0.173 | 1.02× |
| xcop | 12 | 0.216 | 0.279 | **1.29×** |
| clash | 1 (pooled) | 0.051 | 0.491 | **9.69×** |

**For galaxies, a per-object a₀ makes predictions worse than the canonical one.** The 0.467 dex of
apparent galaxy-to-galaxy scatter in a₀ is fitting noise, and the giveaway is that the strongest
correlate of a₀_obj is `n points` — Spearman 0.385 overall, 0.263 inside SPARC at p < 10⁻⁴ — beating
every physical variable. A quantity that cannot set gravity outranking every quantity that could is
the signature of a fit measuring its own sampling. **Within galaxies a₀ is universal.**

**For clusters the offset is real and transfers radially inside an object.** That is what makes the
cluster end a different problem rather than more of the same one.

## Step 3 — the shape of the formula that does work

Lane 12 measured a₀ falling as (r/R500)^−1.354 ± 0.071 in clusters. Lane 6's groups are two decades
lower in mass, were never used in that fit, and are tabulated at r2500 and r500 by construction — so
r/R500 is known for them exactly, with nothing fitted.

**Within single group objects — one object, one instrument, two radii, where neither a dataset label
nor an object label can act:**

    slope of log a0 against log(r/R500)  =  -1.673 dex/dex
    95% CI [-1.889, -1.595],  43 groups,  100% of them negative

against clusters' −1.354 two decades higher in mass. The binned group sequence gives −1.093, and the
X-ray hydrostatic subsample alone −2.255. So the magnitude is not pinned to better than a factor
~1.5, but the sign and the rough size reproduce across independent samples and mass scales.

So there is a formula, and this is its shape:

    ν = f( g_bar / a₀(r/R500) ),   a₀ falling roughly as (r/R500)^-1.4

## What it costs, stated plainly

Three things, and the third is the one that matters:

1. **r/R500 is object-level.** It is a property of the whole configuration, not of the point, so this
   is not — and can never be — a law in the eight point-level variables. §8's no-per-object-parameters
   rule forbids it as written.
2. **The magnitude is only good to ~1.5×** across samples (−1.09 to −2.26 depending on subsample).
3. **R500 is defined by the total mass**, which is the quantity being predicted. So as written the
   relation **cannot predict** a new system's dynamics from its baryons — it compresses what is
   observed, it does not forecast it. Lane 12 flagged the same covariance from the other side.

Point 3 is why the search for a baryon-only replacement matters, and why its failure (r/r_{a₀},
above) is the sharpest open problem in the program rather than a footnote. There is also a deflating
reading that has to be stated: since ν = M_tot/M_bar and the gas fraction rises outward, "a₀ falls
with r/R500" is partly a restatement of the measured f_gas profile. Whether it is more than that
depends entirely on finding the baryon-only scale.

## What was never tested, and is therefore still open

The eliminations above do **not** cover non-conservative or non-variational laws as a class:

- **Lane 9's shape no-go assumes a Lagrangian.** Milgrom's asymptotic theorem follows from the
  structure of the AQUAL/QUMOND field equation. A force law not derived from an action is not bound
  by it.
- **T14 constrains only laws local in the baryons.** It was run against algebraic ν(g_N); a law whose
  source is not the baryon distribution at a point escapes it.
- **Rank-2 constrains only the eight point-level variables.** Any configuration-level input —
  r/R500, profile shape, external field, assembly history — is outside it by construction.

What still binds *any* formula, conservative or not, because these are measurements rather than
theorems:

| constraint | value |
|---|---|
| Saturn anomalous acceleration | < 10⁻¹⁴ m/s² |
| Cassini γ−1 | (2.1 ± 2.3)×10⁻⁵ |
| SN time dilation, metric fraction | b = 1.003 ± 0.005 |
| Bullet aperture ordering | κ lower where baryons are higher, 2 of 2 |
| KiDS [BLIND], 35–2600 kpc | ν/ν_RAR = 1.31, canonical a₀ |
| wide binaries [BLIND] | ν/ν_RAR = 0.90 |
| BTFR slope | 3.83 ± 0.11 |

## The target, as a table to check any proposal against

a₀_obj measured per object, restricted to objects where it is identifiable:

| population | objects | log M_bar | a₀/a₀_canonical | 16–84 pct |
|---|---|---|---|---|
| SPARC galaxies | 168 | 8.9 – 10.9 | 0.75 | 0.30 – 1.73 |
| KiDS lenses [BLIND] | 4 | 10.1 – 11.0 | 1.28 | 1.21 – 3.14 |
| X-COP clusters | 12 | 14.0 – 14.2 | 6.62 | 4.43 – 7.81 |
| CLASH (pooled) | 1 | 13.9 | 14.82 | — |

Groups and Wicker contribute **zero** identifiable objects — they carry one or two apertures each, so
a₀ per object cannot be measured for them at all, and they enter only through the r/R500 slope above.

---

# What is left to try, and why the obstruction has a shape

## The obstruction, in one line

At a point, the data are **rank 2** — everything reduces to (g_bar, r). Galaxies and clusters share
almost none of that plane, and where they do, clusters sit 3–4× higher. **So every formula built
from point-level quantities is dead before it is written**, and the only useful question is which
formulas are not.

Two were tested against that prediction, and both behaved exactly as it says.

## Tested: the density-contrast law — the right repair of B2/B5/B6, still dead

**Motivation, fixed before the test.** a₀ = 1.2×10⁻¹⁰ is c·H₀/2π to a few percent (measured here:
ratio 0.888), H₀ sets ρ_c, so a link between the acceleration scale and the cosmic density is
expected rather than ad hoc. The variable is the **enclosed mean baryonic density**,
X = 3M_bar(<r)/(4πr³ρ_c).

**It genuinely repairs the T03 kill.** The local-density family died because interplanetary space
has ρ ≈ 10⁻²⁰ kg/m³, *below* ρ_c, so a density trigger switches fully **on** in the Solar System.
The enclosed density does the opposite:

| location | ρ_local/ρ_c | ρ_enclosed/ρ_c |
|---|---|---|
| Saturn orbit | ~1 | **1.9×10¹⁹** |
| Neptune orbit | ~1 | 6.0×10¹⁷ |
| solar neighbourhood | ~0.1 | 1.2×10⁵ |

The two variables disagree by **thirty orders of magnitude at the same point**, and the fitted law
passes Saturn and Cassini exactly (anomalous acceleration 0, |γ−1| 0).

**And it still fails, for the predicted reason.** X = 3g_bar/(4πGr ρ_c), so it lies inside the
rank-2 span. At matched X, clusters remain **2.01×** above galaxies across 1.46 dex of genuine
overlap. The global fit returns an exponent of only −0.110, it makes the blind KiDS probe *worse*
(0.453 → 0.486 dex), and `b.confound` returns **LABEL**.

## Tested: profile shape — passes the label check, fails on sign

s = dlnM_bar/dlnr uses the baryon field at **more than one radius**, so it is the first variable in
a long time that rank-2 does not exclude by construction. A point mass has s = 0, uniform density
s = 3; measured medians are 0.00 (KiDS, single-mass lens bins), 1.05 (SPARC), 0.75 (CLASH), 1.66
(X-COP) — so the raw material is there, and it tracks exactly the physical difference the problem
turns on: a galaxy edge has all its baryons enclosed, a cluster core does not.

**`b.confound` passes it** — variable-label +0.131 against label-residual +0.343, "carries
information beyond the dataset label". That is rare here and worth recording.

**But the sign flips between probes.** Regressing log a₀_eff on s *within* each probe, where no
label can act:

| probe | n | Spearman | p (perm) | slope |
|---|---|---|---|---|
| sparc | 2975 | −0.062 | 0.0010 | **−0.057** |
| xcop | 560 | +0.431 | 0.0000 | **+0.429** |

Opposite signs, both significant — the same failure mode as the opposite-slopes theorem that killed
ν = f(g). The gains are correspondingly trivial: 0.96× on SPARC (worse), 1.12× on X-COP, 1.02× on
the blind probe. (KiDS's slope is degenerate: M_bar is constant within each lens bin, so s ≡ 0 and
the regression divides by zero.)

Suggestive residue worth keeping: at matched acceleration the cluster/galaxy ν ratio **grows** from
1.70 to 3.52 as acceleration rises, while s_gal falls from 1.32 to 0.62 and s_clu stays at ~1.65.
The discrepancy is largest exactly where the profiles differ most. That is not nothing — but a
variable that acts in opposite directions inside the two populations is not the one.

## What has genuinely not been tried

1. **Non-variational force laws.** Lane 9's shape no-go is a consequence of the AQUAL/QUMOND field
   equation; Milgrom's asymptotic theorem needs the action. A force law not derived from one is
   untouched by it. Nothing in this program has tested that class.
2. **The external field effect.** The ambient field from neighbours is not a column in the bench for
   any probe. It is the one *environmental* variable with an existing literature and it is measurable.
3. **History- or time-dependent laws.** A field with a finite response time is non-conservative by
   construction. Untested, and the Bullet is where it would show.
4. **Laws whose source is not baryonic mass density** — entropy, pressure, temperature, or a
   collisionless tracer. This is the branch T14 points at, and the reason it is hard is stated below.
5. **Two-scale laws** carrying a second constant beyond a₀ that is not a density or a length (both of
   those are already killed — density by T03 in the local form, length by the Solar System).

## The hardest constraint, stated so it is not glossed

**T14 requires the gravitating source to follow the collisionless component.** In the Bullet, gas is
85–90% of the baryons and it sits between the galaxies, yet the lensing sits on the galaxies —
measured here at 1.13× the baryons carrying 0.69× the κ, and 1.80× carrying 0.74×, in two
independent apertures with nothing subtracted and nothing fitted.

Any theory without a non-baryonic collisionless component has to produce that inversion from the
baryons alone. QUMOND on the measured map produces 1.13 and 1.57 — the wrong side of unity. A
non-local or non-conservative law is *permitted* to do better, but no proposal in this program has
shown how, and a lagged-field mechanism does not obviously work either: before the collision the gas
and galaxies of each subcluster were together, so a field remembering the earlier configuration
would sit **between** them, not on the galaxies.

This is the single sharpest constraint in the program and it should be the first gate any new
proposal is run against, because it is cheap and it needs no fitting.

---

# ROUND 3 — reverse-engineered halos, and what they are actually made of

## The halo does not have to be downloaded; it is arithmetic

For any object with a measured mass profile,

    M_halo(<r) = (g_obs − g_bar) · r² / G

with nothing fitted and no profile family assumed. Published halo catalogues add three things on
top of that — a profile family, per-object uncertainties, and derived shape parameters — but not the
profile itself. So the analysis could start immediately: **137 objects with a usable reverse-engineered
halo** (121 SPARC galaxies, 11 X-COP clusters, CLASH pooled, 4 KiDS lens bins).

Fitting NFW, Burkert and pseudo-isothermal to each independently reproduces the classic result:
in SPARC, **cored profiles beat NFW 97 to 24**.

## The objection that turned out to be right

The halo is *defined* as whatever mass makes the outer speeds come out. So any statement that it
"correlates with the baryons" may be bookkeeping rather than physics. That is testable, and it was
tested.

**The RAR-twin control.** For every galaxy, build a synthetic twin with the *same baryon profile* at
the *same radii*, but with g_obs set **exactly** by the acceleration law and no scatter. Derive the
twin's halo the same way, fit it the same way, measure the same correlations. The twin contains no
information whatsoever beyond the baryons and the law.

| correlation of s_bar = dlnM_bar/dlnr with | ρ | p |
|---|---|---|
| REAL log c200 | −0.384 | <10⁻⁴ |
| **RAR TWIN log c200** | **−0.737** | **<10⁻⁴** |
| residual (real − twin) | +0.049 | 0.61 |
| REAL inner slope | +0.370 | <10⁻⁴ |
| **RAR TWIN inner slope** | **+0.521** | **<10⁻⁴** |
| residual | +0.197 | 0.031 |

**The twin reproduces the correlation more strongly than the real data do**, and the residual is
consistent with zero. The apparent "baryon configuration predicts halo shape" result is the
acceleration law restated. It is dead.

## How much of a halo is bookkeeping — measured

| quantity | corr(real, RAR twin) | residual scatter | real scatter |
|---|---|---|---|
| **log M_halo** | **+0.943** | 0.302 dex | 0.889 dex |
| log c200 | +0.400 | 0.630 dex | 0.671 dex |
| inner slope | +0.245 | 0.807 dex | 0.830 dex |

**Halo mass is 94% reproduced by a synthetic object containing nothing but the baryons and the
acceleration law.** Predicting M_halo from M_bar is therefore not a research problem; it is the RAR
with the axes relabelled, and any formula found that way will inherit the RAR's successes and its
cluster failure exactly.

**Halo shape is the opposite, and this is the finding.** Concentration and inner slope are only
weakly reproduced by the construction (0.40 and 0.25), so they are *not* tied to the baryons — and
after removing the twin, nothing else predicts them either. The residual scatter is 0.63 dex in
concentration against 0.67 dex raw: **the baryons plus the acceleration law explain about 6% of the
variance in halo concentration.**

So the halo shape is very nearly free. That is a strong statement in the direction of the objection:
the inferred halo is tied to the outer rotation speed and to almost nothing else about the galaxy.

## What this changes about the acquisition programme

Halo **mass** catalogues are largely redundant — they are compressions of curves already in hand.
The valuable acquisitions are the ones that are not reparametrisations:

- **halo shapes and ellipticities** measured independently of the rotation curve (lensing
  triaxiality, X-ray isophotes, misalignment with the light);
- **age, colour, metallicity, morphology** — never in this bench, and the one place a variable can
  live that the curve does not already contain;
- **environment and the external field** — the same;
- **independently determined M/L**, which is what separates the 8σ colour dependence in lensing from
  a stellar-mass calibration error.

Two contamination controls that must stay attached to everything in this round:

1. **The RAR twin.** Any halo↔baryon correlation must beat its twin, not merely be significant.
2. **The `n points` nuisance.** It correlates with log M_halo at ρ = +0.637 and with the RAR
   residual at +0.319 — a quantity that cannot set gravity, outranking most physical ones. Halos
   from better-sampled curves look different, and that must be regressed out before any claim.

## Halo concentration is not measured — it is chosen

The reverse-engineered halos were validated against Li et al. 2020 (ApJS 247, 31), the published MCMC
catalogue of 175 SPARC galaxies × 12 profile families. **The validation failed:** log c200 correlated
at only ρ = +0.21 with 0.75 dex of scatter, log M200 at +0.26 with 1.4 dex.

Rather than assume the derivation was broken, the cheaper explanation was tested — and it is right.
c200 and M200 refer to r200, which for a SPARC galaxy lies **far outside the measured range**; the
curves end at tens of kpc and r200 is hundreds. Neither analysis measures it. Both extrapolate, and
the answer is set by whatever regularises the extrapolation. Li et al. make this visible by
publishing the same profile under two priors:

| test, entirely inside the published catalogue | result |
|---|---|
| NFW-Flat vs NFW-LCDM, same galaxies, same data, only the prior differs | log c200 correlate at **+0.687**, scatter **0.401 dex** |
| spread across 7 flat-prior profile families, within a galaxy | **0.366 dex** in log c200 |
| for scale, galaxy-to-galaxy spread | 0.542 dex |
| **ratio, method spread / real spread** | **0.67** |

**Choosing a halo model changes a galaxy's concentration by two-thirds as much as changing which
galaxy you are looking at.** The concentration–mass slope ranges from −0.037 (NFW-LCDM) to −0.310
(coreNFW-Flat) — a factor of eight, purely from model choice.

So the h04 disagreement is not a bug in the derivation; it is the same disagreement that already
exists inside the published catalogue. **Any correlation between baryons and halo concentration —
mine or anyone else's — is a statement about priors.** The halo quantities worth analysing are the
ones that do not extrapolate: enclosed halo mass inside the last measured radius, and the density
slope across the measured range.

A related trap found by the acquisition lane and worth recording: **the χ²ᵥ column of
`J/ApJS/247/31` is corrupted for 3 of 12 models** — Einasto-Flat, Einasto-LCDM and NFW-LCDM carry
identical values in 165/165 galaxies while their fitted parameters are identical in 0/175. Confirmed
four ways, including a degrees-of-freedom fingerprint and disagreement with Li et al. 2019 on Einasto
alone (corr 0.569 against 0.993 for DC14). **Any "cored beats cuspy" conclusion drawn from that
column is wrong for those three models.** The cored-vs-cusped count reported here (97 to 24 in SPARC)
was derived independently from the profiles and does not use it.

## Cluster halo shapes: orientation is tied to the baryons, elongation is not measurable

Donahue et al. 2016 gives, for 25 CLASH clusters in the **same 500 kpc aperture**, the shape of the
X-ray gas, the SZE gas, and the lensing total mass — plus BCG orientations from Donahue et al. 2015.
This is the halo question that is *not* a reparametrisation of a rotation curve.

**The built-in control that makes it trustworthy:** X-ray and SZE trace the *same gas*, so their
agreement is the noise floor — the best any two shape measurements of one cluster can do.

| comparison | axis-ratio ρ | p | median ΔPA | ΔPA if random | p |
|---|---|---|---|---|---|
| **X-ray vs SZE — the noise floor** | +0.072 | 0.74 | 27.5° | 43.6° | 0.030 |
| lensing vs X-ray gas | +0.108 | 0.59 | **9.0°** | 44.0° | **<0.0001** |
| lensing vs SZE gas | +0.322 | 0.12 | 29.9° | 47.1° | 0.041 |
| lensing vs BCG stars | +0.543 | 0.30 | 4.3° | 42.9° | 0.003 |
| X-ray gas vs BCG stars | −0.029 | 1.00 | 6.7° | 45.3° | 0.021 |

**Orientation: the total mass points where the baryons point, and tightly.** The lensing–gas
misalignment is 9.0° against 44° for random, at p < 10⁻⁴ on 25 clusters — and it is **tighter than
the 27.5° between two independent measurements of the gas itself.**

**Elongation: no measurable relation** — but the noise floor says so too. Two views of the same gas
correlate at ρ = +0.072 (p = 0.74), so the axis-ratio measurements are too noisy to answer the
question either way. This is a null from the instruments, not from nature.

One circularity to keep attached: the lensing–BCG alignment (4.3°, n = 6) is exactly what
light-traces-mass lens models produce **by construction**, and LTM is one of the two model families
Donahue averages. The lensing–X-ray alignment is not vulnerable to this, because LTM ties mass to
the *galaxies*, not to the gas. **The robust result is the lensing–gas alignment.**

Oguri et al. 2010, an independent sample and method (2-D elliptical-NFW shear fits, 18 clusters after
dropping the 7 the paper itself flags), is weaker: halo vs member galaxies ρ = −0.307 (p = 0.21),
median ΔPA 24.4°; halo vs BCG ρ = −0.220, ΔPA 37.2°. Median ellipticities are halo 0.552, member
galaxies 0.418, BCG 0.289 — the inferred halo is the *most* elongated of the three.

## Where round 3 stands

| aspect of the halo | is it tied to the baryons? | evidence |
|---|---|---|
| **amount** (inside the measured radius) | **yes, but it is bookkeeping** | 94% reproduced by a synthetic twin knowing only baryons + the acceleration law |
| **concentration, M200** | **question is malformed** | not measured; model choice moves it 0.67× as much as changing galaxy |
| **inner slope** | weakly at best | twin corr +0.245; residual correlations marginal |
| **orientation** (clusters) | **yes, and it is real** | 9.0° from the gas vs 44° random, p < 10⁻⁴, tighter than the gas-vs-gas floor |
| **elongation** (clusters) | **unanswerable with current data** | the same-gas noise floor is ρ = +0.07 |

## The payoff experiment: do age, colour, morphology or environment predict the halo?

These are the columns the bench has never had, and the one place a variable could live that the
rotation curve does not already contain. 121 SPARC galaxies, 15 new variables, against the two halo
targets that do **not** extrapolate — the RAR residual of enclosed halo mass inside the last measured
radius, and the inner density slope minus its RAR twin's.

Four gates, all mandatory: beat the RAR twin; beat the sampling nuisance; survive partialling out
**distance**, because the acquisition lane established that coverage of every new column correlates
with it (Sérsic index present at a median 13 Mpc, absent at 32); and pass `Bench.confound`.

**Result: 0 of 15 clear Bonferroni (p < 0.0017 for 15 variables × 2 targets).**

The strongest candidates, none of which survive:

| variable | vs inner-slope residual | after distance | after n-points | n |
|---|---|---|---|---|
| colour B−V | −0.315 | −0.269 | −0.317 | 73 |
| colour g−W1 | −0.271 | −0.370 | −0.281 | 71 |
| metallicity 12+logOH | −0.368 | −0.297 | −0.278 | 20 |
| gas fraction | +0.243 | +0.244 | +0.233 | 120 |

Colour is the best of them and reaches only p ≈ 0.006 at n = 73. It is also the mass-to-light axis,
so it is the one most likely to be a stellar-mass calibration error rather than physics — which is
precisely what the resolved-stellar-population lane was launched to separate. Its twin score is
instructive: colour vs the *real* inner slope is −0.339 but vs the *twin's* inner slope is −0.149, so
the acceleration law already generates about 44% of it.

**And the most significant number in the whole table is a control.** SPARC's own data-quality flag
predicts the RAR residual at ρ = −0.345, p = 0.0003 — stronger than any physical variable and the
only entry that would clear Bonferroni. Part of what looks like deviation from the acceleration law
is data quality, not gravity. That belongs on the standing caveat list.

Environment is the weakest of all: local density Σ₅ gives ρ = −0.008 on 121 galaxies, and the tidal
indices −0.22 on 49. If there is an external field effect in these data, this measurement of it does
not see it.

## Round 3 summary

| aspect of the halo | tied to the baryons? | evidence |
|---|---|---|
| amount, inside the measured radius | **yes — but it is bookkeeping** | 94% reproduced by a twin knowing only baryons + the law |
| concentration, M200 | **question is malformed** | model choice moves it 0.67× as much as changing galaxy |
| inner slope | **no** | 0 of 15 new variables clear Bonferroni |
| **orientation (clusters)** | **yes, and it is real** | 9.0° from the gas vs 44° random, p < 10⁻⁴, tighter than the gas-vs-gas floor of 27.5° |
| elongation (clusters) | **unanswerable** | two views of the same gas correlate at ρ = +0.07 |

The single positive result of round 3 is the cluster orientation alignment. It is the one halo
property that is neither arithmetic from the rotation curve nor prior-driven, and it is measured
against a noise floor built from the data themselves rather than assumed.

## The stellar mass-to-light normalisation closes roughly half the cluster gap

This is the most consequential number of round 3 and it came from the stellar-population lane rather
than from any gravity analysis.

SPARC assumes Υ₃.₆ = 0.5 for disks. **Dynamically measured** disk M/L — from vertical velocity
dispersions, with no stellar-population model in the loop — gives Υ_K = 0.31 ± 0.07, i.e.
Υ₃.₆ ≈ 0.2, or ≈0.14 after the Angus et al. 2016 reanalysis. That is **0.4 to 0.55 dex below what
SPARC assumes**, and Lelli et al. say in the SPARC paper itself that the value is uncertain "up to a
factor of ~3".

Why it can move the central problem: **galaxies are star-dominated over much of their measured range;
clusters are gas-dominated.** Lowering the stellar M/L lowers galaxy g_bar and leaves clusters
untouched. Rebuilding SPARC's baryonic acceleration from the raw velocity components:

| Υ₃.₆ | galaxy ν/ν_RAR | gap vs X-COP | gap vs CLASH | fitted a₀/a₀_can | galaxy fit quality |
|---|---|---|---|---|---|
| **0.50** (SPARC) | 0.968 | **2.48×** | 3.27× | 0.935 | 0.0908 dex |
| 0.35 | 1.170 | 2.05× | 2.70× | 1.482 | 0.1043 (1.15×) |
| 0.25 | 1.373 | 1.75× | 2.30× | 2.243 | 0.1258 (1.39×) |
| **0.20** (dynamical) | 1.522 | **1.58×** | 2.08× | 2.957 | 0.1398 (**1.54×**) |
| 0.14 (Angus reanalysis) | 1.768 | 1.36× | 1.79× | 4.081 | 0.1615 (1.78×) |

**The galaxy/cluster discrepancy against X-COP falls from 2.48× to 1.58× — and to 1.36× at the
Angus value.** Roughly half of the problem this program has spent its whole effort on is a stellar
mass calibration.

**The cost, stated plainly: the rotation curves get worse.** Median |log| error rises from 0.091 dex
to 0.140 at Υ = 0.2, a 54% degradation, and to 0.162 at Υ = 0.14. The data prefer SPARC's value. So
this is not "SPARC is wrong" — it is a **genuine tension between two independent measurements of the
same quantity**, in which the rotation curves favour the high value and the vertical dynamics favour
the low one, and the cluster problem sits inside the gap between them.

Two caveats that must travel with this:

1. **A global M/L rescale is exactly degenerate with a₀ in the deep-MOND limit**, where ν ∝ √(a₀/g_bar)
   scales both the observation and the prediction identically. It bites only near and above a₀, where
   21% of SPARC points sit at Υ = 0.5 and 11% at Υ = 0.2. That is why the effect is large but not
   total, and it is why the fit quality is the honest arbiter.
2. **The dynamical value is contested.** DiskMass infers M/L from vertical dispersions and an assumed
   disk scale height, and its low values have been disputed since publication. The right reading is a
   range, 0.14–0.5, not a replacement value.

**What this changes about the programme:** every earlier statement of the cluster excess — including
the 4.07× matched-(u,v) figure and the a₀ sequence — was computed at Υ = 0.5. Those numbers are not
wrong, but they carry a systematic of roughly a factor 1.6 that was never propagated. The honest form
of the central unexplained fact is therefore **a factor of 1.4–2.5, not a factor of 4**, and the
range is dominated by a stellar-mass calibration rather than by anything about gravity.

## The colour dependence in lensing is NOT a mass-to-light artefact

A separate result from the same lane, and it cuts the other way. 5,952 MaNGA galaxies have both a
**dynamical** M/L (JAM, IMF-free) and a **population** M/L (Salpeter SPS), plus age, metallicity,
Sérsic index and colour:

| | vs dynamical M/L | vs population M/L | vs α = dyn/pop |
|---|---|---|---|
| mass-weighted age | +0.32 | **+0.72** | −0.10 |
| mass-weighted [Z/H] | +0.20 | **+0.81** | −0.06 |
| Sérsic n | +0.05 | **+0.73** | −0.04 |
| g−r colour | +0.33 | **+0.69** | +0.09 |

The population M/L is strongly determined by all four; **the ratio to the independently measured
dynamical M/L is nearly orthogonal to all four.** So the SPS estimate is not biased *as a function of*
colour, age, metallicity or morphology — its errors do not track those axes.

That matters because lane 8 found KiDS lensing depends on colour at 8.0σ and on Sérsic index at 6.7σ
**at fixed baryonic acceleration**, and the leading explanation was a colour-dependent M/L error.
This says that explanation does not work. The colour dependence in lensing therefore survives its most
likely deflation, and moves up the list of things worth chasing.

Caveat carried: this is measured on MaNGA, which is mostly early types, and transferred to the KiDS
lens population. **SPARC overlap with all three IFU surveys is essentially zero** — ATLAS-3D 0,
CALIFA 0, MaNGA 6 — verified by both name and position matching, so the transfer is unavoidable
rather than lazy, and it is a real limitation.

## Four data defects found this round, each of which fails silently

- `J/ApJS/247/31` χ²ᵥ **corrupted for 3 of 12 models** (identical values in 165/165 galaxies while
  fitted parameters differ in all 175) — invalidates model-comparison conclusions drawn from it.
- Meert et al. 2015 Sérsic index is declared `I4` in VizieR, so the values arrive as **integers 0–8**
  while their error bars keep full precision. Use Simard table3 `ng`.
- Krajnović et al. 2020 encodes "no measurement" as literal **0.000**; SAMI DR3 uses **−99** sentinels
  (2,209 of them, which shift the median [Z/H] by 0.02 dex if left in).
- CALIFA's two stellar-mass tables use **different IMFs with identical column names** — merging them
  silently mixes 0.26 dex.

## Strong lenses: the cross-constraint that closes the mass-to-light escape

206 grade-A strong lenses (SLACS, S4TM, SL2S, BELLS, BELLS GALLERY) are the opposite corner of the
bench in three ways at once — **g_bar/a₀ ≈ 9 median, up to 34** (every cluster probe here is *below*
a₀), pressure-supported early types rather than rotating disks, and a photon channel. Einstein masses
are good to sub-percent.

**A bug caught before the result was used.** The first pass compared *total* stellar mass against the
mass inside the Einstein radius and returned f_* > 1 — physically impossible — for 82 of 99 lenses.
M_Einstein is a cylinder mass, so the stellar mass must be one too; applying the de Vaucouleurs
projected fraction at each lens's own R_Ein/R_eff fixes it. Corrected numbers below.

| IMF | n | median g_bar/a₀ | median ν | ν_RAR predicts | **ν/ν_RAR** |
|---|---|---|---|---|---|
| Salpeter | 99 | 9.1 | 1.44 | 1.02 | **1.42** |
| Chabrier | 139 | 9.1 | 2.20 | 1.01 | **2.17** |

At nine times the acceleration scale, every candidate law in this program reduces to Newton and
predicts ν/ν_RAR = 1. The lenses give 1.4–2.2. Putting them on the RAR requires **more** stellar mass:
a factor 1.445 (Salpeter) or 2.317 (Chabrier), driving f_* to 0.96 — i.e. essentially no non-stellar
mass inside R_Ein at all.

**And that is the opposite direction from the disks.** The mass-to-light lever that closes half the
cluster gap requires *lowering* stellar masses by 0.4 dex. At high acceleration ν/ν_RAR ∝ 1/M_star, so
the same reduction would take the strong lenses from 1.42 to roughly **3.5** — it trades a factor 1.6
off the cluster problem for a factor 2.5 onto the early-type problem.

**So the mass-to-light escape does not work as a global rescale.** It works only if disks and early
types have genuinely different stellar normalisations — lighter in disks, heavier in ellipticals.
That is not an absurd position (there is a substantial literature on the IMF varying with velocity
dispersion), but it is two parameters where the program was looking for zero, and §8 forbids
per-object gravity parameters even though it permits per-population *baryon* calibrations. This is the
sharpest constraint yet on how much of the cluster problem can be blamed on baryons.

One thing NOT to read off this: ν/ν_RAR falls with g_bar at slope −0.44 for both IMFs. That test is
contaminated — ν ∝ 1/M_star while g_bar ∝ M_star, so a shared stellar mass injects a negative slope by
construction. The level is meaningful; the trend is not.

## Gas is rounder than the lensing mass — a second instance of the Bullet failure mode

From the cluster lane, needing no fitting: on the same 25 CLASH clusters, same aperture, same
pipeline, **median q(lensing) = 0.80 against q(X-ray gas) = 0.90**, with the gas rounder in **21 of
25** (sign test **p = 0.0009**) — while the two remain *aligned* to |ΔPA| = 21°.

A convergence sourced by the gas predicts the gas's own shape. It does not predict a systematically
flatter one. Combined with the orientation result, the picture is specific: **the extra mass points
where the baryons point but is more elongated than they are.** That is the same failure mode as the
Bullet — lensing tracking something the gas does not fully account for — appearing in a completely
different measurement.

Two caveats recorded with it: **3D triaxial axis ratios are prior-dominated** (median c/a moves
0.31 → 0.43 → 0.57 across three priors on identical data), so projected q must be used, not q_a; and
Umetsu 2016's stacked-profile comparison **excludes cored and power-law halos at PTE ≤ 0.015**
(Burkert 0.002), which is in tension with the SPARC result that cored profiles beat NFW 97 to 24 —
the two statements are about different mass ranges and should not be merged.

## Concentration data, now that it exists

The cluster lane recovered **1,309 concentration measurements over 537 clusters, split by
reconstruction method** (Groener 2016: X-ray 318, weak lensing 230, caustic 82, LOSVD 70, WL+SL 61,
SL 20). This is the dataset that makes the h05 conclusion testable at cluster scale rather than only
in SPARC: if concentration is prior-driven there too, measurements of the *same clusters* by
*different methods* will disagree at the level the priors differ. That comparison is now possible and
is the obvious next test.

Also recovered where it existed nowhere in tabular form: **Okabe & Smith's 50 LoCuSS concentrations**,
analytically inverted from their seven tabulated overdensity masses, validated by re-predicting their
published M500c to **0.11% rms** and reproducing their lognormal mean (3.72 against 3.73 ± 0.38).

Excluded as circular and recorded as such: Hoekstra 2015 and Herbonnet 2020 do **not** fit
concentration — they impose a c–M relation — so their concentrations cannot be used to test one.

## The external field effect: tested for the first time, and it dissolves

The environment lane delivered the last variable the program had never had — a vector-summed external
field from 32,116 groups out to 100 Mpc, for **all 175 SPARC galaxies** (it needs only a position, so
coverage is complete). The vector sum matters: a scalar sum is 3.7× larger because opposite-sky
contributions cancel. It was validated three ways, including an independent MOND-route reconstruction
from baryons alone agreeing to 3% with nothing tuned.

This variable is unlike every other candidate in two respects. It is **not a property of the galaxy at
all**, so the rank-2 result has no bearing on it. And its prediction is **signed**: an external field
*suppresses* the internal boost, so higher e_N must give a more negative RAR residual. A positive
correlation would refute the mechanism as firmly as a null.

**First pass looked like the first environmental result in the programme:**

| variable | ρ vs RAR residual | p | n | \| distance | \| n_points |
|---|---|---|---|---|---|
| log e_N alone | −0.088 | 0.26 | 169 | −0.051 | −0.076 |
| **log (g_ext/g_int)** | **−0.231** | **0.0073** | 130 | **−0.229** | **−0.227** |

Correct sign, survives both standing controls, and the binned trend is monotone over most of the
range. 62 of 133 galaxies have g_ext > 0.1 g_int, so the sample genuinely reaches the testable regime.

**But the external field is identical in both rows.** All the significance appears on dividing by
g_int = V_flat²/R_HI, which is a property of the galaxy computed from its own rotation curve. That is
the shape of a denominator doing the work, and this project has been burned by it before. Two controls
settle it:

| control | result |
|---|---|
| ratio, partialling out **log g_int** | **−0.097** — collapses |
| −log g_int alone vs the residual | **−0.213** — nearly the whole raw signal |
| **placebo**: scramble the real field across galaxies, keep the denominator | median ρ = **−0.165**; **9.2%** of 5,000 draws reach the observed −0.231 |

**The result is g_int in disguise.** It is a restatement of "galaxies with low internal acceleration
have negative RAR residuals", which is a statement about the galaxies, not about their neighbours.
Filed as the seventh artefact of that family; the placebo is what caught it.

A data defect found in passing: `g_int_RHI` and `g_lss` are stored **rounded to zero** in the
environment table (they are ~10⁻¹¹ in SI). g_int was reconstructed exactly as e_N·a₀/ratio from the
two columns that survived rounding. Without that, the decisive control returns NaN and looks like it
simply could not be run.

## Merger bias does not explain the cluster excess — a caveat retired

Every cluster number in this programme has carried the caveat that mergers bias hydrostatic masses low
and so inflate the apparent boost. It was untestable because the bench sorts X-COP objects by `extent`
and carries no cluster names. The identities are recoverable: each raw directory is named for its
cluster, and pairing the profile ends by rank reconstructs the map exactly, because `extent` **is** the
profile end.

| cluster | ν/ν_RAR | disturbance % | state | cool core |
|---|---|---|---|---|
| A2029 | 2.53 | 3.5 | relaxed | CC |
| A1795 | 2.12 | 15.5 | relaxed | CC |
| A85 | 2.59 | 21.5 | relaxed | CC |
| A2142 | 2.65 | 22.3 | intermediate | CC |
| A644 | 2.05 | 34.0 | relaxed | CC |
| A3158 | 3.52 | 35.1 | disturbed | NCC |
| A2319 | 2.54 | 54.0 | intermediate | — |
| A2255 | 2.48 | 58.1 | intermediate | NCC |
| A1644 | 2.28 | 78.8 | disturbed | NCC |
| A3266 | 1.63 | 90.7 | disturbed | NCC |

**Spearman(disturbance, boost) = −0.309, p = 0.36. Disturbed/relaxed median boost = 0.981.**
Cool-core 2.53 against non-cool-core 2.38.

Merger bias predicts a ratio **above** 1 — more non-thermal support means hydrostatic masses
underestimate more, so the apparent boost grows. The measured ratio is 0.98, and the correlation is
null with the sign slightly the wrong way, across essentially the full range from the most relaxed to
the most violently merging clusters known.

**The standing merger caveat can be retired rather than carried.** With n = 10 the bound is not tight
— it excludes merger bias as the *dominant* explanation, not as a ~20% contributor — but removing a
systematic is worth more than adding a parameter.

## Round 3, closed

| question | answer | evidence |
|---|---|---|
| does the halo amount follow the baryons? | **yes, but it is bookkeeping** | 94% reproduced by an RAR twin |
| is halo concentration measured? | **no** | model choice moves it 0.67× as much as changing galaxy |
| does anything predict halo inner shape? | **no** | 0 of 15 new variables clear Bonferroni |
| do age/colour/morphology/environment matter? | **no** | best is colour at p ≈ 0.006, and its twin makes 44% of it |
| is there an external field effect? | **no** | dies on partialling g_int; placebo reproduces it |
| is the cluster excess merger bias? | **no** | disturbed/relaxed = 0.98 across the full range |
| does halo orientation follow the baryons? | **yes, and it is real** | 9.0° vs 44° random, tighter than the gas-vs-gas floor |
| is the halo the same shape as the baryons? | **no — it is flatter** | q 0.80 vs 0.90, 21/25, p = 0.0009 |
| how much of the cluster gap is stellar mass? | **about half** | 2.48× → 1.58× at the dynamically measured M/L |
| can that M/L shift be adopted globally? | **no** | it triples the strong-lens discrepancy |

Two positives survived a round that killed everything else, and they say the same thing from opposite
directions: **the extra mass is aligned with the baryons but is not distributed like them.** It points
where they point, to better precision than two measurements of the gas agree with each other, while
being systematically flatter than they are — and in the Bullet, sitting somewhere they are not.

---

# CORRECTION to the two positive results of round 3

The shapes lane delivered 1,276 rows over 652 objects with an explicit
`shape_assumed_by_method` flag, and it invalidates the source column both of my positive results used.
**Donahue's "lensing" morphology is a Zitrin PIEMDeNFW model, which ties mass to the cluster galaxies
by construction.** I had flagged that circularity for the lensing-vs-BCG comparison and argued the
lensing-vs-gas comparison was immune because LTM ties mass to galaxies rather than gas. That argument
was too generous: if the mass map is built to follow the light, its shape and orientation are not free
to disagree with the light, and the gas comparison inherits it.

Redone on the **184 non-circular rows** and the 72 non-circular halo-vs-baryon misalignments:

## Correction 1 — orientation: the halo is measurably MISALIGNED, not exceptionally aligned

| pairing | n | median misalignment |
|---|---|---|
| stellar light vs BCG stellar light | 10 | **6.8°** |
| stellar light vs satellite galaxy distribution | 10 | **10.6°** |
| **halo (lensing) vs X-ray gas, 500 kpc** | 20 | **21.3°** |
| halo vs X-ray gas, 0.5 R500 | 16 | 21.9° |
| halo vs BCG near-IR light | 16 | 34.0° |
| halo vs SZ gas | 18 | 41.8° |

I reported 9.0° and said it was tighter than the baryon-baryon floor of 27.5°. **Both numbers were
wrong for the same reason.** The 9.0° came from the circular lensing column. And 27.5° was X-ray
versus SZE — an *SZE* noise floor, since SZE is the low-resolution channel, not the best two baryon
tracers can do. The genuine baryon-baryon floor is **6.8–10.6°**.

Against that floor, the halo-gas agreement of 21.3° is **worse**, not better. Subtracting the floor in
quadrature leaves a **genuine halo–baryon misalignment of ≈18°, measured rather than simulated.**

The alignment is still far from random — 21° against 45° — so the direction of the original statement
survives, but its force is inverted. The right claim is not "the halo knows where the baryons point
better than the baryons do"; it is **"the halo is aligned with the baryons to about 20°, and roughly
18° of that is a real misalignment."** A measured misalignment is a stronger constraint on a theory
than a confirmation would have been, and it is also lane 8's dilution factor — which the C-family
does not suffer.

## Correction 2 — flatness: survives, and survives the GR objection

The objection is real: gas sits in the *potential*, and a potential is rounder than the mass that
sources it, so a rounder gas is partly required with no new physics. Sereno measured
**e_Φ/e_mat = 0.72**.

On the 184 non-circular rows:

    median q_halo    0.700   ->  e_halo   = 0.300
    median q_baryon  0.840   ->  e_baryon = 0.160
    halo flatter in 114 of 184,  sign test p = 1.5e-3
    measured e_baryon / e_halo = 0.533

**0.533 is below the 0.72 that GR already requires**, so the gas is rounder than even the
potential-tracing expectation. The result stands, on a sample nine times larger than the one I quoted
and with the circular rows removed.

**But the per-object version is unusable.** Spearman(q_halo, q_baryon) = **−0.372** — rounder halos
sit with flatter baryons, which is not a physical expectation and marks the two estimators as not
measuring a common quantity per object. The lane's own method test explains it: on the *same 16 CLASH
clusters*, Chiu minus Sereno is positive in **16 of 16** with median Δq = **+0.23**, and A383 has three
published axis ratios of 0.30, 0.55 and 0.82. **The method systematic exceeds every quoted error bar
and tracks whether X-ray data entered the fit.** Only the population-level offset means anything;
object-by-object shape comparison does not.

## Correction 3 — lane 8's C-family bound rests on mis-transcribed inputs

`lane08_fh.py` used Georgiou et al. 2021 as f_h = 0.28 ± 0.55 and 0.08 ± 0.53. The **published** values
are 0.50 ± 0.20 (all lenses) and 0.55 ± 0.19 / 0.34 ± 0.17 (red, outer/inner). Lane 8's combined bound
of f_h = +0.064 ± 0.274 — reported here as "consistent with zero" — depends on those inputs.

With the published record: **red lenses give f_h = 0.303 ± 0.080, a 3.8σ detection**, rising to
**f_h = 2.2 ± 0.6 at group mass**. Halo ellipticity is *not* consistent with zero, and the mass trend
is partly an alignment trend rather than a shape trend.

**This reopens the C-family question.** Those laws were bounded to |ε| ≲ 0.3–1.3 partly on the strength
of null halo-ellipticity measurements. The nulls were for late-type/blue lenses; the red-lens and
group-scale measurements are detections. The bound needs recomputing, and the correct input table now
exists.

## What round 3 actually leaves

| claim | status after audit |
|---|---|
| halo amount is bookkeeping (94% RAR twin) | **stands** |
| concentration is prior-driven, not measured | **stands**, and now confirmed at cluster scale too — Chiu vs Sereno differ by Δq = +0.23 on the same 16 clusters |
| nothing predicts halo inner shape (0 of 15) | **stands** |
| no external field effect | **stands** — placebo reproduces it |
| cluster excess is not merger bias | **stands** |
| half the cluster gap is stellar M/L | **stands**, with the strong-lens counter-constraint |
| ~~halo orientation is exceptionally aligned~~ | **corrected**: ≈18° genuine misalignment, measured |
| halo is flatter than the baryons | **stands and strengthens**: 0.533 against the 0.72 GR floor, n = 184 |
| ~~halo ellipticity consistent with zero~~ | **corrected**: 3.8σ for red lenses, 2.2 ± 0.6 at group mass |

Still unavailable and still the discriminator lane 8 asked for: **no galaxy-galaxy lensing split by
lens inclination or axis ratio exists anywhere through 2026.** C1 predicts zero signal face-on and
maximum edge-on; C2 predicts the same at every inclination. One such split separates them.

## Correction 3, resolved: the C-family bound tightens rather than reopening

The corrected halo-ellipticity record was recomputed, dropping circular rows and KiDS-holdout rows,
and **split by whether the lens has a disk at all** — because lane 8's own derivation ties n̂ to the
disk normal, and an elliptical galaxy has no disk normal. Pooling the two is the error that produces a
spurious detection.

| subset | f_h | n | χ²/dof | significance |
|---|---|---|---|---|
| **late-type / blue — ADMISSIBLE** | **−0.052 ± 0.101** | 52 | 1.42 | 0.5σ |
| red / early — *inadmissible for this family* | +0.112 ± 0.031 | 50 | 1.61 | 3.6σ |
| all pooled — **wrong** | +0.120 ± 0.030 | 113 | 1.62 | 4.0σ |

**The 3.8σ detection the shapes lane flagged lives entirely in the red sample**, where the C-family
prediction is not defined. On the admissible late-type rows the result is consistent with zero — and
**2.7× tighter than lane 8's ±0.274**, because the mis-transcribed Georgiou errors were far too large.

| law | f_h/ε | ε from admissible rows | previous bound |
|---|---|---|---|
| C1, C3, C5 | +0.92 | **−0.056 ± 0.110** | \|ε\| < 0.31–0.46 |
| C2 | −0.46 | **+0.113 ± 0.221** | \|ε\| < 1.3 |

And the tension with the independent Milky Way vertical-force constraint, which is unambiguously
admissible because the Milky Way is a disk, drops to **1.1σ for C1 and 1.6σ for C3** — consistent
where before it was the main source of doubt.

**So lane 8's conclusion survives, and its bound improves by a factor 2–6 — but for a reason it never
stated.** The right statement is not "halo ellipticity is consistent with zero" (it is not; red lenses
detect it at 3.6σ and it reaches f_h = 2.2 ± 0.6 at group mass) but "halo ellipticity is consistent
with zero **for the galaxies to which the C-family applies**."

Caveat carried: the 52 late-type rows are not independent — they include the same surveys re-analysed
with different weightings and profile assumptions — so ±0.101 is an underestimate of the true
uncertainty and the significance should be read as an upper bound. The χ²/dof of 1.42 has been
inflated into the quoted error, but that does not fully correct for shared data.

### Merger-bias test, re-verified on raw single-source values

A late warning from the acquisition lane — that `centroid_shift_w` and `concentration_c` **mix
apertures across sources** (500 kpc versus R500) — was checked against the merger-bias result rather
than assumed harmless. It does not apply: **all ten X-COP clusters draw their morphology from a single
source at a single aperture** (Yuan et al. 2022, R_ap = 500 kpc), and the derived `disturbance_pct`
reproduces the raw δ ordering exactly.

Redone directly on the raw δ, which removes any dependence on the sign-harmonisation:

    Spearman(delta, nu/nu_RAR) = -0.515    permutation p = 0.135  (50,000 draws)
    Pearson                     = -0.465
    disturbed / relaxed median boost = 0.965

Merger bias predicts a **positive** correlation and a ratio **above** 1. The measured correlation is
negative and slightly stronger than the −0.309 obtained from the derived percentile, and the ratio is
below 1. At n = 10 this is not a significant anticorrelation — it is a null with the sign pointing the
wrong way for the hypothesis — but it excludes merger bias as the dominant explanation of the cluster
excess on a homogeneous, single-aperture sample spanning δ = −0.69 to +1.47.

---

# Does the extra gravity have a DIRECTION, and can this data guide one?

Round 3's one surviving positive result is directional, so the question is answerable rather than
speculative. Three distinct things go under the word "direction", and the data say different things
about each.

## 1. An unoriented axis — YES, and it is measured

The extra mass is not co-axial with the baryons. On the non-circular measurements:

| pairing | n | median \|ΔPA\| |
|---|---|---|
| baryon vs baryon (stellar light vs BCG, vs satellites) | 10 each | **6.8–10.6°** |
| halo (lensing) vs X-ray gas | 20 | **21.3°** |

Subtracting the floor in quadrature leaves **≈18° of genuine axis mismatch**. That is a real
direction and any theory has to produce it.

## 2. A preferred rotational sense (handedness) — NO

Position angle is an orientation, defined modulo 180°, so a difference folds into [−90°, +90°]. Random
triaxiality gives a symmetric distribution with mean zero. **A nonzero signed mean would require a
preferred handedness, which no scalar theory permits.**

| channel | n | signed mean | sem | mean/sem |
|---|---|---|---|---|
| halo vs X-ray gas | 20 | **−4.4°** | 9.0 | 0.49 |
| halo vs SZ gas | 18 | +15.1° | 10.7 | 1.41 |
| halo vs X-ray at 0.5 R500 | 16 | +13.6° | 11.7 | 1.16 |
| halo vs BCG near-IR light | 16 | **−31.7°** | 9.3 | **3.4** |

Three of four are consistent with zero, including the cleanest one. On the Donahue sample the same
test gives −8.0 ± 5.4 with a sign test of exactly p = 1.000 (11 positive, 12 negative).

**The fourth needs explaining rather than reporting.** The BCG channel combines Umetsu's weak-lensing
position angles with Donahue 2015's BCG angles — **two papers**. The X-ray channel, which is null, uses
angles that both trace back to Chandra imaging. A constant offset between two papers' conventions
produces exactly a signed mean with no sky dependence, which is what is seen. **The check that would
settle it:** take BCG position angles from Umetsu's own table instead, and see whether the −31.7°
survives. This data cannot settle it — only one BCG source is present.

## 3. A direction fixed in space — NO

If gravity carried a cosmic preferred axis, the halo orientations would cluster around a common sky
direction and the misalignment would vary with position.

| test | result |
|---|---|
| halo PA resultant length, 25 clusters | **0.160** — chance alone gives 0.200 |
| misalignment vs RA | ρ = −0.027, p = 0.90 |
| misalignment vs Dec | ρ = −0.140, p = 0.50 |
| misalignment vs redshift | ρ = −0.055, p = 0.79 |

The halos do not point anywhere in particular, and how far they are from the baryons does not depend
on where they sit.

## 4. A displacement direction — the Bullet, but it carries no weight

The cluster misalignments are rotations, which have no handedness. The Bullet gives a **displacement**,
which does: the κ peak sits 253 kpc from the gas and 65 kpc from the BCG in the main cluster, 155 and
51 kpc in the subcluster — displaced toward the galaxies in the same sense both times, along the merger
axis. **But 2 of 2 is p = 0.5 under a coin flip.** Its force comes from the magnitude, and from QUMOND
failing to reproduce it, not from the count.

## So: could this data guide a directional theory?

**Yes, but only by bounding it — not by pointing it.** Concretely, the data supply four numbers a
directional proposal must hit:

1. **Amplitude ≤ ~10–20% of the main term.** The C-family — which *is* the "gravity has direction"
   hypothesis, written four ways — is bounded on lenses that actually have a disk to
   **ε = −0.056 ± 0.110** (C1/C3/C5) and **+0.113 ± 0.221** (C2), with the independent Milky Way
   vertical force giving −0.23 ± 0.12 and +0.16 ± 0.08.
2. **A ≈18° axis mismatch to reproduce**, unoriented.
3. **No handedness** — the signed mean must come out zero.
4. **No cosmic axis** — the effect must be defined relative to something in the system, not to a
   fixed direction in space.

The direction must therefore be **referred to a local axis** — a disk normal, a baryonic major axis, a
merger axis. Every measurement here that could have detected a direction in space returned null, and
every measurement that could have detected a handedness returned null except one that behaves like a
units error.

**What would actually guide it, and does not exist:** galaxy–galaxy lensing split by **lens
inclination or axis ratio**. C1 predicts zero signal face-on and maximum edge-on; C2 predicts the same
signal at every inclination. One such split separates them outright. No such measurement has been
published through 2026 — it remains the single most valuable missing observation in this programme.

---

# The rotation/pressure split: found, then killed

## What it looked like

Sorting all 576 objects by what physically holds each one up gave the cleanest pattern in the data:
spinning and free-orbiting systems at median R = 0.91, pressure-supported systems at 1.69 — a factor
1.86, permutation p < 10⁻⁵. It cut *across* acceleration, with strong lenses failing at 9 a₀ and
clusters failing at 0.04 a₀ while spirals worked in between, so it was not the acceleration relation
in disguise. It survived matched-acceleration binning (ratios 2.22, 1.91, 1.62 across three bins).

**It was published in two articles before it had been through the one control that mattered.**

## The control it had not had

Every spinning system in the bench came from SPARC and every pressure-supported one from SLACS/SL2S.
**Support mechanism was perfectly confounded with survey.** No amount of statistics on that sample
could separate them, and this is precisely the failure mode that killed six earlier candidates here.

MaNGA resolves it structurally: it contains **both** rotating and pressure-supported galaxies,
observed with one instrument, reduced with one pipeline, modelled with one dynamical code — and it
publishes **λ_Re**, the spin parameter measured directly from the resolved velocity field. The binary
label becomes a continuous measurement, and the survey confound disappears by construction.

## The result: null

**2,422 galaxies, λ_Re from 0.03 to 0.92 — pure pressure support to pure rotation.**

    Spearman(lambda_Re, R)                      -0.005    p = 0.80
    the same, acceleration partialled out       +0.013

| λ_Re bin | n | median R |
|---|---|---|
| 0.03 – 0.18 | 486 | 1.33 |
| 0.18 – 0.36 | 486 | 1.04 |
| 0.36 – 0.49 | 495 | 1.08 |
| 0.49 – 0.61 | 491 | 1.10 |
| 0.61 – 0.92 | 486 | 1.20 |

Not monotonic, and flat overall. In matched-acceleration bins the correlation **changes sign**:
+0.156, +0.191 (p < 10⁻⁴), +0.005, −0.149 (p = 0.0003). Two significant results in opposite
directions is the signature of confounds varying between bins, not of an effect.

**The rotation/pressure split does not exist inside a single survey. It was the survey label.**

## And the mechanism that faked it is identifiable

`Spearman(λ_Re, log α) = −0.155, p < 10⁻⁴`, where α is the IMF-free dynamical M/L divided by the
population M/L. **Slow rotators genuinely have a heavier mass-to-light ratio than a standard
population model assigns them.** Underestimating their stellar mass inflates their apparent
discrepancy — exactly the signal claimed. This is the same lever that made the strong-lens result
IMF-limited (Salpeter 1.42 against Chabrier 2.17, a swing wider than the effect).

## What is left after λ_Re is removed

The discrepancy does depend on *something* — just not rotation:

| property | ρ with R, after removing λ_Re |
|---|---|
| ellipticity | **−0.199** |
| log g_bar/a₀ | +0.159 |
| mass-weighted age | +0.120 |
| log stellar mass | +0.116 |
| velocity dispersion | +0.113 |

None of these has been through the twin, nuisance and label controls, so none is claimed. They are
listed as the next things to test, not as findings.

## The correction that matters most

**Two published articles asserted the split as the programme's surviving positive result. Both were
wrong and both have been rewritten.** The error was not statistical — the numbers were right — it was
publishing before running the one control the programme's own history said was mandatory. That
history now records seven eliminations of this exact kind, and this is the seventh.

---

# §16 executed: the three permanent competitors, scored identically

Asked whether the standard references, held to the same tests, would also fail. They were scored on
every object with the same metric — median |log10| error in predicted gravity, where 0.30 means wrong
by a factor of two.

**GR is not scored separately from Newton.** In weak fields with slow-moving matter GR reduces to
Newton exactly to parts in 10⁵ for every system here, so they are the same prediction. They differ by
a factor of two for light bending, which is applied on the lensing probes.

| model | median \|log10\| error | wrong by a factor of | free parameters |
|---|---|---|---|
| **Newton / GR, no dark matter** | **0.542** | **3.5×** | 0 |
| **MOND / RAR** | **0.115** | **1.30×** | 1 (a₀, fixed 1983) |
| GR + NFW halo per object | 0.112 | 1.29× | **274, for 137 objects** |

Per object class:

| class | n | Newton = GR | MOND / RAR | winner |
|---|---|---|---|---|
| galaxies | 169 | 0.463 | **0.097** | MOND |
| galaxy lensing stacks | 4 | 1.788 | **0.071** | MOND |
| **wide binaries** | 6 | **0.022** | 0.050 | **Newton** |
| strong lenses | 99 | 0.177 | **0.151** | MOND |
| groups | 165 | 1.162 | **0.323** | MOND |
| clusters | 133 | 0.821 | **0.196** | MOND |

Three things follow, and the second and third are corrections to how this document has been phrasing
things.

1. **Newton and GR without dark matter are not close competitors.** They are wrong by 3.5× on
   average, and worst exactly where MOND succeeds. MOND is five times more accurate overall.
   MOND is the thing that *works* here, not a casualty of the programme; what fails is extending it
   to clusters, and that failure is a factor of two against Newton's factor of 3.5.

2. **A fitted dark halo barely beats MOND — 0.112 against 0.115 — while costing 274 free parameters
   for 137 objects.** One constant fixed in 1983 matches two parameters per object. That is the
   sharpest form of the 94%-bookkeeping result.

3. **The wide binaries slightly favour Newton (0.022) over MOND (0.050).** This document has
   repeatedly said "both blind holdouts land on the RAR". That is fair — 0.90 is close to 1 — but
   Newton lands closer, and the phrasing has been too generous. The caveat is that 67% of those
   points sit above a₀ where the two theories agree by construction, and n = 6, so the comparison is
   weak either way. It should be stated as: **the wide-binary holdout is consistent with both, and
   marginally prefers Newton.**

## On whether the controls were applied symmetrically

They were not, and the asymmetry runs opposite to the worry. The synthetic-null, dataset-label and
RAR-twin controls are controls on **discovering a new variable** — they ask whether a correlation
survives when the physics is removed and only survey structure remains. Newton, GR and MOND are fixed
functions written down in advance with nothing to tune, so there is no label for them to accidentally
be measuring. They cannot fail those controls; they can only fail the data, which is the harsher test
and the one applied above.

**So our candidates had to pass both the data and the label controls; the references only had to pass
the data.** The standard applied to the new proposals was stricter, not looser.

---

# The relative tournament: every law scored against Newton, not against perfection

The kill criterion used until now was wrong, and a fair challenge exposed it. A law was discarded if
it failed somewhere — but **Newton fails galaxies by a factor of 3.5 and nobody discards Newton.**
The right question is where a law is *better than what we already accept*, and by how much.

Every candidate was resurrected, its parameter space rescanned with widened grids and rail detection,
and scored with **global parameters only**. Newton's balanced baseline (galaxies and clusters weighted
equally, not per-probe, which had silently given clusters 3:1) is **0.664**.

| law | par | sparc | xcop | wicker | clash | **kids** | **widebin** | vs Newton |
|---|---|---|---|---|---|---|---|---|
| NEW enclosed-density | 3 | 0.156 | 0.124 | 0.115 | 0.122 | **0.996** | 0.022 | 4.81× |
| NEW RAR × screen | 4 | 0.132 | 0.171 | 0.064 | 0.247 | **0.125** | 0.017 | 4.53× |
| A4 add-sqrt | 4 | 0.168 | 0.238 | 0.029 | 0.333 | **0.116** | 0.091 | 3.61× |
| A1 interp-n | 2 | 0.150 | 0.238 | 0.029 | 0.394 | **0.157** | 0.049 | 3.59× |
| **A2 RAR** | **1** | 0.161 | 0.244 | 0.032 | 0.359 | **0.120** | 0.097 | 3.56× |
| C6 directional | 3 | 0.238 | 0.130 | 0.110 | 0.193 | **0.996** | 0.014 | 3.48× |
| B3 surface-density-G | 3 | 0.172 | 0.184 | 0.052 | 0.426 | **0.996** | 0.014 | 3.38× |
| A5 power-r | 3 | 0.424 | 0.370 | 0.168 | 0.570 | 1.190 | 0.022 | 1.67× |
| C4 flattened-sep | 2 | 0.428 | 0.635 | 0.412 | 0.695 | 1.675 | 0.022 | 1.32× |
| B2 density-G | 3 | 0.445 | 0.896 | 0.463 | 0.894 | 0.496 | 0.022 | 1.11× |
| A6 log-potential | 2 | 0.444 | 0.940 | 0.814 | 0.895 | 1.890 | 0.022 | 1.00× |
| NEWTON | 0 | 0.445 | 0.940 | 0.814 | 0.895 | 1.928 | 0.022 | 1.00× |

**Still killed outright by the Solar System, at every point of a widened grid:** A3 simple-μ,
B1 Yukawa, B4 potential-depth-G.

## The blind columns are the whole story

Three laws fit the four fitted probes best — enclosed-density, C6, B3 — and **all three fail the blind
KiDS probe at 0.996 dex, a factor of ten.** They buy their cluster fit by adding a boost that is
badly wrong for a probe they never saw. The acceleration family (A2, A1, A4) sits at 0.12–0.16 blind.

## The apparent winner, and why it is withdrawn

"RAR × screen" — the RAR multiplied by an enclosed-density-contrast screen — topped the blind column
at 0.071 and looked like a genuine improvement on A2's 0.108. Its best fit had **railed** on a₀ at the
grid floor, so the grid was extended downward:

| a₀ / a₀_canonical | fitted | **blind** | screen amp |
|---|---|---|---|
| 0.83 | 0.152 | 0.107 | 1.0 |
| 0.26 | 0.155 | 0.076 | 2.0 |
| **0.083** | 0.145 | **0.071** | 3.5 |
| 0.026 | 0.134 | 0.099 | 4.5 |
| 0.008 | 0.128 | 0.171 | 5.5 |
| 0.0008 | **0.126** | **0.351** | 6.0 |

**The fitted score improves monotonically as a₀ → 0 while the blind score collapses.** Selecting on
fit — which is what fitting does — drives this law to blind 0.351, three times worse than plain A2.
Its apparent advantage existed only because the original grid floor happened to stop it near the blind
optimum. **Withdrawn.** It is not a promising hybrid; it is a demonstration that four parameters plus
a cluster-shaped screen will overfit, and that the blind probes catch it.

## What actually survives, ranked

1. **A2 (RAR)** — 1 parameter, blind 0.108, beats Newton 3.56×. The honest baseline.
2. **A1 (interpolation index)** — 2 parameters, blind 0.103. A marginal, defensible refinement.
3. **A4 (additive √)** — 4 parameters, blind 0.103. Same blind score as A1 for two more parameters.

Everything else is either worse than Newton, killed by the Solar System, or fits by overfitting.

---

# ~274,000 modifications of MOND, scored on clusters

Base: MOND exactly as published — the RAR function at the canonical a₀, nothing refitted. Every
variation multiplies it by an explicit correction, so what each one *adds* is readable directly:

    g = g_MOND · F,    F = 1 + A · S(V/V_c)

with **V** naming the physical idea being tested. Eleven candidates: radius (a length scale), mass,
a second acceleration, the MOND boost itself (self-feeding), enclosed density contrast, surface
density, potential depth, local density, fractional extent, sphericity (anisotropy), and a path term
where gravity accumulates along the radius. Four switching shapes. Deliberately **not** enforced:
Solar System, blind holdouts, galaxy damage — all measured, none optimised against.

| stage | variations |
|---|---|
| single-term | 61,600 |
| two-condition gates + additive terms | 212,540 |
| **total** | **~274,000** |

## Stage 1 — the trap

Every single-term modification that improves clusters 3–5× damages galaxies **4–12×**. One global
term cannot tell the populations apart. Best was `path/screen` at 5.49× on clusters and 6.09× worse
on galaxies.

## Stage 2 — gates, and a ceiling worth having

Two-condition gates — *add gravity only where both conditions hold* — reach **2.16–2.32× on clusters
with zero galaxy damage and no blind degradation.**

The useful control here was to leave **sphericity** (a binary dataset label meaning "cluster or
galaxy") in the library and see it win, then hold it out as a **ceiling**: 0.1569, i.e. **2.30×**, is
what a modification scores if simply allowed to look up the answer. Physical gates reached 2.16–2.32×
— *at* the ceiling. That should have been the warning.

## Stage 3 — mass-resolved, and the modification collapses

The aggregate galaxy score hid the damage. Resolved by baryonic mass:

| log M_bar | n | MOND | modified | gain |
|---|---|---|---|---|
| 7.0–9.5 | 1,267 | — | — | 1.00–1.03× |
| **9.5–10.0** | 442 | 0.0734 | 0.1136 | **0.65×** |
| **10.0–10.5** | 452 | 0.0602 | 0.1786 | **0.34×** |
| **10.5–11.0** | 926 | 0.0871 | 0.2534 | **0.34×** |
| **11.0–11.5** | 610 | 0.1329 | 0.2716 | **0.49×** |
| 11.5–12.0 | 411 | 0.5248 | 0.2218 | 2.37× |
| 13.0–13.5 | 4,434 | 0.2766 | 0.0638 | 4.34× |

**It is a threshold hack.** It steps over the mass trend rather than explaining it, and destroys the
most massive galaxies by a factor of three. The step test confirms it: +0.052 dex discontinuity at
the fitted threshold, widening to +0.101 at ±0.8 dex.

Sharpening the gate does not help — damage rises from 3.62× to 4.14× as the index goes 1 → 12, and the
threshold always lands at log M = 10.0, *inside* the galaxy range.

## The structural reason, and it is the useful result

| probe | n | min log M_bar | max |
|---|---|---|---|
| SPARC galaxies | 3,389 | 4.18 | **11.56** |
| CLASH clusters | 84 | **10.50** | 13.90 |

**The populations overlap by 1.06 dex in baryonic mass** — 1,201 galaxy points and 15 cluster points
share the range 10^10.5–10^11.56. **No threshold in mass can separate them, so no mass-gated
modification can fix clusters without breaking massive galaxies.** That is not a tuning problem; it
is arithmetic, and it kills the entire mass-gate family in one line.

This is the same shape of result as the rank-2 theorem: not "we searched and failed" but "here is why
the search could not have succeeded."

**Not tested, and worth stating:** genuine anisotropy — a direction-dependent law — cannot be
evaluated here. The only anisotropy proxy in the bench is sphericity, which is a per-probe constant
and therefore the dataset label. Testing direction requires per-object orientation data the bench
does not carry.

---

# Two proposed mechanisms, formalised and tested

## Mechanism 1 — sidedness: "more void on one side tilts gravity inward"

**Formalised.** At radius r the baryons are not symmetric about that point: inward is everything the
system has accumulated, outward is whatever remains. Newton does not care — a spherical shell exerts
no force inside it. The proposal is that the modified theory does. The computable version needs only
the measured mass profile:

    W = rho_enclosed(r) / rho_local(r) = 3 M(<r) / (r dM/dr) = 3 / (dlnM/dlnr)

W = 1 for uniform density, W → ∞ for a point mass with void outside. Literally "how much more is
behind me than beside me." Computed for **3,674 points** across every object with four or more radii.

**Result: the sign is backwards, and it flips between probes.**

| | n | Spearman(log W, MOND residual) | p |
|---|---|---|---|
| all probes | 3,674 | **−0.099** | <10⁻⁴ |
| within SPARC | 3,084 | **+0.093** | <10⁻⁴ |
| within X-COP | 588 | **−0.408** | <10⁻⁴ |

The prediction was a *positive* correlation — more void, more gravity. Galaxies give the predicted
sign weakly; **clusters give the opposite sign strongly**, and clusters are exactly where the
mechanism was supposed to be most pronounced. Two significant correlations in opposite directions is
the same failure signature as the opposite-slopes theorem.

As a modification it can reach 2.81× on clusters — but at 4.50× galaxy damage. Constrained to no
galaxy damage it delivers **1.08×**, essentially nothing.

Worth keeping: the −0.408 inside X-COP is a *strong* within-probe correlation, so W does track the
cluster residual — it is most likely re-expressing lane 12's radial a₀ run, since gas density steepens
outward exactly where the boost declines.

## Mechanism 2 — orientation composition, and why the measurement forbids it

**The mechanism is mathematically correct and the sign is right.** A transverse component adds in
quadrature, so what a rotation curve or lensing map records is the magnitude

    |g| = g_radial / cos(theta)

Misalignment therefore *raises* measured gravity. A thin disk has one orientation and composes
coherently; a cluster has many and composes as a random walk, θ_eff = θ₀√N. This correctly predicts
that flat, single-orientation systems should behave differently from disordered ones.

**It is bounded by the very measurement that motivated it.**

| quantity | value | boost supplied |
|---|---|---|
| cluster shortfall to explain | ×2.45 | — |
| misalignment that would require | **65.9°** | 2.45 |
| measured, halo vs X-ray gas | 21.3° | 1.073 |
| **intrinsic, after removing the 11.5° floor** | **17.9°** | **1.051** |

The mechanism delivers **5.1%** where 145% is needed — **3.5% of what is required.** Random-walk
composition would need N ≈ 13 independent orientation domains, which is physically unremarkable for a
cluster — but that *predicts a 66° misalignment*, and the direct measurement is 18°. The idea is
excluded by observation rather than by theory, which is the cleanest way for an idea to die.

**What both tests share.** Each was a specific, physically motivated proposal that turned out to be
computable from data already in hand, and each was decided by a number rather than an argument.
Sidedness fails on sign; orientation fails on magnitude by a factor of thirty. Neither needed a
parameter scan to settle.

---

# Phase- and pressure-dependent coupling: the best candidate the programme has produced

## The gas-versus-solid proposal, tested

**Does gravity care what phase the matter is in?** A star is ~1000 kg/m³; cold galactic HI is
~10⁻²¹; cluster plasma ~10⁻²⁶. And the budgets match the split — a big spiral is mostly *stars*, a
cluster is 85–90% *diffuse gas*. So "diffuse matter gravitates more" maps onto the galaxy/cluster
divide without naming either.

Testable here because SPARC publishes gas, disk and bulge rotation **separately**:

    v²_bar = xi_gas · Vgas|Vgas| + xi_star · (0.5 Vdisk² + 0.7 Vbul²)

At ξ = 4: clusters **2.46×** better, galaxies 1.49× worse, blind probes unaffected.

**But its own characteristic prediction fails.** If diffuse matter pulls harder, gas-rich galaxies
must show more excess. Within SPARC, where no label can act:

    point-level   Spearman(gas share, residual) = -0.154   p < 1e-4   [WRONG SIGN]
    per-galaxy                                  = -0.123   p = 0.11   [null]

Gas share spans 0.03 to 0.94 across the sample, so the test had range. The cluster improvement is the
gas fraction standing in for "is this a cluster".

## What came out of it instead: amplified pressure

In general relativity the source of gravity is **ρ + 3P/c²** — pressure gravitates. The term is
invisible in practice: 3P/ρc² is ~3×10⁻⁵ for 5 keV cluster plasma and ~3×10⁻¹¹ for 100 K galactic
hydrogen. The modification keeps the form and amplifies the coefficient:

    rho_eff = rho · (1 + kappa · 3P/(rho c²)),   kappa = 1 is GR

**Why this is unlike every modification tried before.** It does not key on mass, radius, density,
surface density or shape — all of which turned out to be the dataset label, because they separate
clusters from galaxies only by being big. It keys on **temperature**, and temperature ordering is not
mass ordering:

| system | 3P/ρc² | model boost at κ=10⁵ | measured excess |
|---|---|---|---|
| cluster plasma, 5 keV | 3.0×10⁻⁵ | 2.29 | **2.29** |
| elliptical stars, σ = 250 | 6.9×10⁻⁷ | 1.15 | **1.42** |
| galaxy warm gas, 8000 K | 2.2×10⁻⁹ | 1.000 | — |
| galaxy cold HI, 100 K | 2.8×10⁻¹¹ | 1.000 | **0.97** |

**Scored on the data at κ = 10⁵:**

| | modified | MOND | change |
|---|---|---|---|
| clusters | **0.1556** | 0.3605 | **2.32× better** |
| galaxies | 0.0937 | 0.0937 | **exactly unchanged** |
| KiDS [blind] | 0.1340 | 0.1340 | unchanged |
| wide binaries [blind] | 0.0497 | 0.0497 | unchanged |

That reaches the ceiling — 2.30×, what the bare dataset label achieves — **at zero cost anywhere
else.** No other modification in ~274,000 came close to that combination.

**And it places the case that kills mass-gates.** Strong lenses are galaxy-mass (10^11) but
pressure-supported, and they show 1.42× excess. Every mass-keyed modification puts them with
galaxies. This one puts them at 1.15 — between, on the right side.

## The within-sample test, run where it could be run

The decisive test is whether the *same* κ predicts variation **inside** a population. X-COP publishes
only **scaled** temperature profiles (T/T₅₀₀), so absolute cluster temperatures are not recoverable
from disk — and for clusters T and M are coupled by the M–T relation anyway, so that test would be
partly confounded regardless.

Strong lenses are the better sample: velocity dispersion is the pressure term, and it varies at
roughly fixed mass. 95 grade-A lenses, σ = 164–404 km/s:

    Spearman(sigma, excess)                = +0.151   p = 0.147   [right sign, not significant]
    with stellar mass partialled out       = +0.096
    binned by sigma: 1.33, 1.40, 1.38, 1.54  [monotonic rise]
    predicted spread 1.19x, observed 1.40x   [same order]

**Right sign, right magnitude, 1.4 sigma.** Suggestive; not established.

## Honest costs

- **κ ≈ 10⁵ is an enormous amplification.** GR has κ = 1. That is a single free parameter doing very
  heavy lifting, and it is the main thing to be sceptical about.
- The within-sample evidence is 1.4σ on 95 objects. It has not passed a test it could have failed.
- **The decisive acquisition is small and specific:** X-COP per-cluster T₅₀₀, published but not in the
  on-disk release. With it, the model must predict the excess ordering among twelve clusters using the
  κ already fixed by the galaxy comparison — no freedom left.

## The pressure model, tested against absolute X-COP temperatures — and it fails

**Acquired:** Ghirardini et al. 2019, A&A 621, A41 (arXiv:1805.00042), LaTeX source. Table 1 gives
M₅₀₀ and z for all twelve clusters; Eq. 6 gives the normalisation the on-disk scaled profiles use:

    T_500 = 8.85 keV (M_500/1e15 Msun)^(2/3) E(z)^(2/3) (mu/0.6)

so absolute kT(r) = (T/T₅₀₀)ₓ(r) × T₅₀₀. Derived temperatures span **3.74 to 8.76 keV**, which is
where the literature puts X-COP.

| cluster | kT (keV) | T/T₅₀₀ | M₅₀₀ | excess |
|---|---|---|---|---|
| A1644 | 3.74 | 0.842 | 3.48 | 2.28 |
| A3158 | 4.73 | 0.927 | 4.26 | **3.52** |
| RXC1825 | 4.95 | 0.996 | 4.08 | 2.12 |
| A1795 | 5.25 | 0.973 | 4.63 | 2.57 |
| A85 | 5.45 | 0.886 | 5.65 | 2.59 |
| A2255 | 5.88 | 0.994 | 5.26 | 2.48 |
| ZW1215 | 6.07 | 0.800 | 7.66 | **1.60** |
| A644 | 6.18 | 0.999 | 5.66 | 2.05 |
| A3266 | 6.20 | 0.749 | 8.80 | 1.63 |
| A2142 | 7.42 | 0.878 | 8.95 | 2.65 |
| A2029 | 7.49 | 0.910 | 8.65 | 2.53 |
| A2319 | 8.76 | 1.199 | 7.31 | 2.54 |

**The test, three ways — all null:**

    (a) absolute kT vs excess              rho = -0.084   p = 0.80
    (b) same, mass partialled out          rho = +0.005
    (c) measured T/T500 (mass-free) vs excess  rho = +0.189   p = 0.56
        control, log M500 vs excess        rho = -0.098   p = 0.77
        size test with kappa already fixed rho = -0.084   p = 0.80

The model requires all of these positive. The hottest cluster (A2319, 8.76 keV) sits at 2.54; the
coolest (A1644, 3.74 keV) at 2.28. The highest excess belongs to A3158 at 4.73 keV and the lowest to
ZW1215 at 6.07. **Temperature does not organise the excess.**

**And it cannot, even in principle, at this κ.** Predicted spread across the sample is 1.38×; observed
is 2.21×. The model's *level* is right — median |log₁₀(observed/predicted)| = 0.095 dex, i.e. 25% —
because κ was fitted to exactly that. Its *variation* is not.

**Two things I got wrong and should own.** I argued "temperature is not mass, so this is unlike the
gates that failed." Measured: **Spearman(kT, log M₅₀₀) = +0.881.** In this sample temperature and mass
are 88% correlated, so the argument was much weaker than stated. And the test is underpowered — n = 12
over only a 2.3× temperature range, so ρ = −0.084 ± 0.33 disfavours the required ρ ≈ 0.6 at roughly
2σ rather than killing it.

**Status: the best candidate the programme produced does not survive the test it was built to face.**
It still has the property nothing else had — 2.32× on clusters at zero cost to galaxies or either blind
probe — but that is now known to be a level match without the corresponding variation, which is what
a well-tuned label looks like.

---

# CORRECTION: the X-COP cluster identities were wrong, and both results flip

## The error

Both the merger-bias test and the temperature test paired bench objects to cluster names by the
**rank of each density profile's outermost radius**, assuming every profile extends to the same
multiple of R₅₀₀. It does not — the ratio runs from **1.12 (A3266) to 2.16 (A644)**. **Eleven of
twelve clusters were misassigned.**

The correct route needed no inference at all: the bench's `extent` **is R₅₀₀ in kpc**, and ten of the
twelve values match Ghirardini Table 1 to within 2 kpc. The remaining two extents (1250, 1368) pair by
elimination with the two unassigned names (A644, R₅₀₀ = 1230; A2319, R₅₀₀ = 1346).

**Independently validated:** corr(published log M₅₀₀, bench log M_bar) = **+0.933**, which **0 of
20,000** random permutations reach; and the implied baryon fractions come out at 0.113–0.196, median
0.147 — where X-COP's published gas-plus-star fractions at R₅₀₀ sit.

## Corrected result 1 — temperature: the pressure model's prediction is SUPPORTED

| cluster | kT (keV) | excess |
|---|---|---|
| A1644 | 3.35 | 1.63 |
| RXC1825 | 4.44 | 2.28 |
| A3158 | 4.71 | 2.12 |
| A1795 | 5.25 | 2.57 |
| A85 | 5.60 | 2.48 |
| A2255 | 5.63 | 1.60 |
| ZW1215 | 6.13 | 2.53 |
| A3266 | 6.36 | 2.05 |
| A644 | 7.10 | 3.52 |
| A2029 | 7.44 | 2.65 |
| A2142 | 7.54 | 2.54 |
| A2319 | 8.55 | 2.59 |

    absolute kT vs excess              rho = +0.615   p = 0.037   n = 12
    with mass partialled out           rho = +0.564
    with disturbance partialled out    rho = +0.622
    size test, kappa fixed at 1e5      rho = +0.615   p = 0.038
    median |log10(observed/predicted)| = 0.081 dex

**This is the first time in the programme that a proposed mechanism passed a test it could have
failed.** The previously reported −0.084 (p = 0.80) was computed on the scrambled map and is withdrawn.

**And the honest weakness, stated plainly.** On the **ten clusters matched exactly** — dropping A644
and A2319, whose identities came by elimination — the correlation falls to **ρ = +0.442, p = 0.19.**
Leave-one-out jackknife spans ρ = +0.545 to +0.700 with p from 0.021 to 0.091. So the significance is
marginal and partly rests on two inferred identities. **Suggestive, not established.**

## Corrected result 2 — merger bias: a significant ANTI-correlation

| cluster | excess | disturbance % | state | core |
|---|---|---|---|---|
| A2029 | 2.65 | 3.5 | relaxed | CC |
| A1795 | 2.57 | 15.5 | relaxed | CC |
| A85 | 2.48 | 21.5 | relaxed | CC |
| A2142 | 2.54 | 22.3 | intermediate | CC |
| A644 | 3.52 | 34.0 | relaxed | CC |
| A3158 | 2.12 | 35.1 | disturbed | NCC |
| A2319 | 2.59 | 54.0 | intermediate | — |
| A2255 | 1.60 | 58.1 | intermediate | NCC |
| A1644 | 1.63 | 78.8 | disturbed | NCC |
| A3266 | 2.05 | 90.7 | disturbed | NCC |

    Spearman(disturbance, excess) = -0.661   p = 0.039   n = 10
    disturbed/relaxed median ratio = 0.796
    cool-core 2.57  vs  non-cool-core 1.84

The earlier report of ρ = −0.309, ratio 0.981, "null" is withdrawn. The correct statement is stronger
and different: **merger bias is not merely absent, it runs significantly backwards.** Disturbed
clusters show *less* extra gravity, not more. That is a new fact requiring its own explanation, not a
retired caveat.

## The two effects are independent

    Spearman(kT, disturbance)             = -0.176   p = 0.63
    kT vs excess, disturbance removed     = +0.622
    disturbance vs excess, kT removed     = -0.695

So the cluster sample carries two separate signals: hotter clusters show more excess, and disturbed
clusters show less. Neither explains the other.

## What this says about the method

The error was caught because the numbers were questioned, not because any check flagged it — the
rank-pairing produced a plausible-looking table and two publishable-sounding conclusions, both wrong.
**The lesson is specific: `extent` was R₅₀₀ all along, and an exact match was available. An inferred
identity was used where a measured one existed.** Any future join in this programme must validate the
map itself before using it, as done here with the M₅₀₀ correlation and the permutation control.

---

## PAPER-GRADE ASSESSMENT: what survives a referee, and what does not

Everything above was exploratory and reported as point estimates without
uncertainties. `p01_rigorous.py` and `p02_systematics.py` redo the headline
results at the standard a referee would apply: propagated errors, bootstrap
intervals, exact permutation tests (2e5 draws, not asymptotic approximations,
given n = 10-12), power analysis, and multiple-testing accounting.

### The scoping decision

**Primary publishable result: the methodology, not the physics.** Seven
candidate variables that passed conventional significance testing were
eliminated by a synthetic-null control, and that control reproduces them from
data containing only survey structure and no physics. Two had been written up
before the control was run. This is transferable and has a quantified failure
rate.

**Secondary: the structural theorems.** The rank-2 collapse (singular values
1.5e2, 9.6e1, 2.7e-12) and the RAR-twin result -- inferred halo mass is 94%
reproduced by a synthetic object knowing only baryons and the law
(corr = +0.943), while halo shape is neither reproduced nor predicted by any of
15 measured properties.

**Not publishable as a discovery: the X-COP temperature correlation.** Reasons
in the next section.

### The X-COP result, honestly

    Spearman(kT, excess), n = 12            +0.615    p = 0.037 (exact perm)
    bootstrap 95% CI over clusters          [+0.091, +0.874]
    survives per-point measurement error    +0.552    0.002 of draws <= 0
    mass partialled out                     +0.564
    bootstrap CI on that partial            [-0.025, +0.907]   INCLUDES ZERO
    exact-identity subsample, n = 10        +0.442    p = 0.205
    power at rho = 0.6, n = 12              0.51

**The binding constraint is the sample, not the model.** The significance rests
on the two clusters (A644, A2319) whose identity was assigned by elimination
rather than exact R500 match. Remove them and it is not significant. At 51%
power a null would not have refuted the hypothesis either, so this sample
cannot settle the question in either direction.

### A correction to the first rigorous pass

`p01` reported chi2/dof = 3.69 and concluded the model was a poor fit with its
pre-registered kappa = 1e5 excluded. **Both statements are withdrawn.** The
error bars were statistical only -- a bootstrap over radial bins giving 1-2%
uncertainty on a cluster mass, which nobody believes.

    sigma_int    best kappa    chi2/dof
       0.00       1.56e5         3.69
       0.10       1.41e5         1.52
       0.15       1.36e5         0.93     <- acceptable
       0.20       1.33e5         0.60

A 15% intrinsic scatter -- a plausible size for hydrostatic-bias variation
alone -- makes the model an acceptable fit. It still beats a constant excess at
equal degrees of freedom (delta chi2 = +6.9 at sigma_int = 0.15). And kappa was
fixed by a galaxy-to-cluster ratio carrying the full Upsilon_3.6 systematic, so
it is good to a factor of ~2 at best; the difference between the pre-registered
1e5 and the fitted 1.4e5 is not meaningful.

**The rank correlation is the sound part** -- it uses only the ordering of the
excesses, so it is immune to any systematic that scales all clusters together.
**The amplitude is the weak part.**

### Multiple-testing, stated without evasion

H1 and H2 were both pre-specified with directions before the relevant data
existed, so neither carries a look-elsewhere penalty in isolation. But if H2 is
treated as one draw from this programme's exploratory pool (~20
pre-registration-equivalents), the corrected p is near 0.5 and nothing
survives. The defensible framing is that H2 was mechanism-derived and tested
once: a confirmatory test, not a discovery.

### What would upgrade it

- A sample where kT and mass decouple. The confound is rho(kT, log M500) =
  +0.874. Cool-core vs non-cool-core at matched mass, or groups at 1-3 keV.
- n >= 40 with homogeneous hydrostatic reconstruction: 80% power at rho = 0.5,
  against 33% now.
- Exact identities for all twelve. Two came by elimination and carry the
  significance.
- An independent hydrostatic pipeline, to show the excess ordering is not a
  reconstruction artefact.

Analysis package: https://claude.ai/code/artifact/c780a535-d49b-4769-8446-cc01bda02f75

---

## AUDIT OF THE CONFOUND CHECK ITSELF, and what it invalidates

The check that issued every kill was never itself tested against variables with
known answers. `p03_audit_confound.py` does that. Three defects, all confirmed:

**A. Pure noise passed.** The rule fired when `|r_vy|` and `|r_ly|` were within
0.08 of each other, so it measured DISTANCE, not superiority. A variable
carrying nothing sits far from the label and passed -- reported as "carries
information beyond the dataset label". Demonstrated: random noise, r_vy =
+0.0147 against a label at +0.4195, verdict PASS.

**B. Ties were broken by array position.** `argsort(argsort(x))` assigns ranks
by position within ties, and the bench concatenates probe by probe. So:

    a GLOBAL CONSTANT, identical at all 4247 points
       corr with dataset label, ties-by-position  = +0.9482
       corr with dataset label, ties-averaged     =  0.0000

That +0.948 is manufactured entirely by concatenation order. Any heavily-tied
or block-constant variable was being pushed toward a false conviction.

**C. Both blind holdouts were being consumed.** The filter excluded only
`role == "bound"`, which is `solar` alone. `kids` and `widebin` -- the two
sealed datasets -- were inside every confound computation, so holdout data was
influencing variable selection. Re-running with and without them flips 1 of 13
verdicts (log10 enclosed density), so the leak was real but low-impact.

### The finding that matters more than the three bugs

With all three fixed and the verdict recast as a partial correlation -- does V
explain the RAR residual beyond what the label explains -- the check STILL
cannot do the job it was used for. Calibrated on 600 pure-noise draws:

    noise floor, 95th pct of |partial|      0.031
    bare dataset label, on its own          0.563     = 18x the floor
    ---
    a_N                                     0.147     = 4.7x floor, 26% of label
    sphericity                              0.135     = 4.4x floor, 24% of label
    r/extent                                0.134     = 4.3x floor, 24% of label
    baryonic potential phi                  0.125     = 4.0x floor, 22% of label
    log10 enclosed baryonic mass            0.112     = 3.6x floor, 20% of label
    radius r                                0.063     = 2.0x floor, 11% of label
    log10 enclosed density                  0.046     = 1.5x floor,  8% of label

Every physical variable is compressed into 1.5x-4.7x the floor while the label
sits at 18x. **The check separates a real variable from noise. It cannot rank
real variables against each other, and it cannot support a kill on its own.**
The bench now says so in its own verdict strings rather than returning a binary.

Note also that significance is useless here: at n = 4181 pure noise reaches
p = 0.043. Effect size is the only readable quantity.

### What this invalidates, and what it does not

Two of the seven eliminations rested on `Bench.confound` alone -- **sphericity**
and **radius**. Both are now INDETERMINATE rather than killed. They do not
become findings: sphericity is 24% of the label and radius 11%. They are
unsupported in either direction and need an independent control.

The other five were killed by controls that do not touch this code path and
they stand unchanged:

    baryon config vs halo concentration   RAR twin, -0.737
    rotation vs pressure support          within-survey MaNGA, rho = -0.005, n = 2422
    external field effect                 placebo pair reproduces 9.2%
    enclosed density contrast             blind holdout, 0.996 dex
    gas-phase coupling                    own prediction fails, wrong sign

### Fixed in `invariant_bench.py`

Average-rank ties (`Bench._rank`), holdouts excluded from the probe filter,
partial-correlation statistic, effect-size verdicts against the calibrated
floor, and a docstring stating the check's hard limit. Backup at
`invariant_bench.py.bak`.

---

## LENSING AND SUBSTRUCTURE, added to the viewer

Two questions from the user, both answered from data rather than assertion.

### Are these clusters strong lenses?

Computed kappa(R) = Sigma(R)/Sigma_crit from the measured mass profiles, with
Sigma_crit = c^2 D_S/(4 pi G D_L D_LS) and a source plane at z_s = 2.

    lens            z_l    kappa_max   theta_E required   theta_E visible
    A1644          0.047      0.07           none              none
    A3158          0.060      0.06           none              none
    A85            0.056      0.11           none              none
    A2029          0.077      0.17           none              none
    A2142          0.091      0.21           none              none
    ... all twelve X-COP in 0.06 - 0.21 ...
    CLASH stack    0.350      2.27          33.8"              none

**The X-COP clusters are not strong lenses, and it is their redshift, not
their mass.** Sigma_crit rises as the lens gets nearer, so a cluster at
z ~ 0.06 needs far more mass to reach kappa = 1 than the same cluster at
z ~ 0.35. CLASH, which IS lensing-selected, gives theta_E = 33.8" -- squarely
in the observed 10-35" range for those clusters, which is a useful external
check on the whole pipeline.

**With visible matter alone, none of the thirteen produces an Einstein ring.**

Two numerical traps, both caught only because one cluster looked wrong:

1. Sigma needs rho, rho needs dM/dr, and the hydrostatic M(r) is NOT
   monotonic -- A2319 has 21 of 54 points with negative slope, swinging
   between -8.7 and +8.8. Differencing it directly gave that cluster
   kappa_0 = 13.4 against ~0.2 for every other, and a fake 121" Einstein
   radius. Fixed by imposing the physical constraint (enclosed mass cannot
   decrease), fitting a smooth log-log form, and differentiating the fit.
2. The outer density slope was taken from that same global fit and clipped at
   +3.5, i.e. **density rising outward**, so the line-of-sight integral never
   converged. A85 came out with a flat kappa = 5.4 across two decades and
   RXC1825 a flat 1.5. Fixed by taking the slope from the outer third and
   forcing it steeper than r^-2.

### Does a cluster have one well or many?

The user is right that it has many, and the X-COP profiles cannot show it --
being hydrostatic, they are spherically averaged by construction. So the
question was put to member catalogues instead: AXES-SDSS groups (Damsted+
2024), 180 on disk, 60 with >= 20 clean members after 3-sigma velocity
clipping. Stellar mass from r-band luminosity at Upsilon_r = 2.5; total mass
from the members' own velocity dispersion, M(<r) = 3 sigma^2 R_rms / G. Both
observational; neither assumes dark matter.

    groups analysed                 60
    median M_dyn / M_star           55        (16-84 pct: 39 - 80)
    visible galaxies are            1.8% of the mass
    brightest member, of stars      16%
    brightest member, of the total  0.3%

**So yes, many wells -- and they are pinpricks.** All the galaxies together
are ~1.8% of what has to be there. In the rendered field the galaxies'
combined potential at the group centre is 1.0% of the total well depth.

One honesty fix in the drawing: the isothermal estimator extrapolated inward
put a spike at the centre that no measurement supports -- the members fix the
mass near R_rms and say nothing about the core. Replaced with a cored
isothermal (r_c = 0.12R) and the inner region is drawn faded to mark it
unmeasured.

Viewer now has three modes (Well / Lens / Clumps):
https://claude.ai/code/artifact/ccc5d1c2-25f8-42f6-9e54-6edd5d864213

---

## DOES SPHERICAL AVERAGING INVALIDATE THE CLUSTER TESTS?

Raised by the user, and correct in principle: for a NONLINEAR law you cannot
push the cluster's averaged baryon profile through nu(g/a0) and call that the
prediction. You have to solve the field equation on the real lumpy source --
smooth gas plus every member galaxy -- and evaluate it where the light passes.
Superposition does not hold, so the two are different calculations.

Scope of the objection:

- **It bites** every RAR/MOND-family test in this programme, all of which used
  spherically-averaged X-COP profiles.
- **It does not bite** Newton or GR, which are linear and superpose exactly.
  The lensing view in the viewer is GR on both tiles, so it is unaffected.
- **It does not bite** the amplified-pressure model rho_eff = rho(1 +
  kappa*3P/rho c^2), which is linear in both rho and P and commutes with
  averaging.
- In EXACT spherical symmetry AQUAL reduces algebraically to mu(|g|/a0)g = g_N,
  so the error comes entirely from departures from sphericity -- i.e. from the
  member galaxies.

### Measured, not argued

`v04_lumpiness.py` solves QUMOND, lap(Psi) = div[nu(|grad Phi_N|/a0) grad
Phi_N], on a 192^3 grid over 4.2 Mpc, twice: once on A2029's gas profile plus
300 member galaxies holding 15% of the baryons, once on the identical mass
smoothed spherically. `v05_realizations.py` repeats it over 5 draws of the
galaxy population to separate bias from shot noise.

    3D shell-averaged |g|, lumpy/smoothed     1.0040   (max 0.77%)

    projected deflection, lumpy/smoothed, 5 realizations
        R (kpc)      150     300     500     800    1100    1400
        mean      0.9731  0.9889  0.9934  1.0070  1.0013  0.9988
        sd        0.0183  0.0206  0.0182  0.0109  0.0126  0.0062
        sign flips   no     yes     yes     yes     yes     yes

**At every radius beyond ~300 kpc the sign flips between realizations and the
mean sits on 1 -- that is shot noise from where individual galaxies land, not
bias.** At the innermost annulus there IS a real systematic: 0.973 +- 0.008,
about 3 sigma, and it does not flip sign.

**Its direction matters.** Lumpy gives LESS deflection than smoothed, so doing
the calculation properly makes MOND slightly WORSE on clusters, not better. The
cluster discrepancy is a factor of ~2; this is a 3% effect at small radius and
under 1% elsewhere. It cannot rescue MOND, and it does not.

Why it is small: at 500 kpc the smooth cluster field is ~0.1 a0 while a
1e11 Msun galaxy 30 kpc away adds ~0.1 a0 only within its own neighbourhood.
The cluster dominates |g| over essentially the whole volume, so nu() is
evaluated at nearly the same place either way, and the line-of-sight integral
washes out what is left.

### Two projection errors caught on the way

1. Summing lap(Psi) down the periodic axis gave a surface density IDENTICAL at
   every radius and 9.7% apart -- larger than the 3D gap, which is impossible
   for a smoothing operation. The constant periodic background dominated.
2. With the background subtracted it gave exactly zero, because the spectral
   Laplacian reconstructs a pure divergence whose column sums cancel.

Both were the wrong quantity. What bends light is the deflection,
alpha = (2/c^2) integral g_perp dz, computed from the transverse gradient of
Psi already in hand. The internal check that caught both: a projected
difference can never exceed the 3D one, because projection is further
smoothing.

### Caveats

A2029's own member catalogue is not on disk, so the 300 galaxies are a
statistical population drawn from the AXES luminosity function and the gas
profile -- realistic in mass and radial distribution, not the actual members.
Grid cells are 22-26 kpc, so structure inside individual galaxies is
unresolved; a finer grid could raise the local effect, but the volume within
50 kpc of a massive member is negligible and the projection integrates over it.

---

## ANISOTROPIC-VOID TEST PROGRAM: goals document and Run A

Goals doc: https://claude.ai/code/artifact/0f9cacd4-00e2-4414-8021-67df473342ef
Scripts: `g01_runA.py`, `g02_heff_control.py`

The user supplied a full test program for a void-polarized anisotropic gravity
hypothesis built on the conservative weak-field core

    div( mu(X) K grad Psi ) = 4 pi G rho_b,   g = -grad Psi,   D_g = K g

and asked for a goals document showing the formulas, the data going into them,
and the outcomes. Run A (1-D galaxy screening) was executed on SPARC rather
than described.

### Run A outcome, 171 galaxies / 3,373 points

    model                              RMS dex   vs Newton
    D1 Newton, baryons only             0.5299      1.00x
    D3 piecewise sqrt(a0 g_N)           0.2121      2.50x
    M1 mu = X/(1+X)                     0.1995      2.66x
    M2 mu = X/sqrt(1+X^2)               0.2013      2.63x
    RAR reference                       0.1989      2.66x
    K1xQ2 scalar void, best (a=1,n=2)   0.2539      2.09x

### Three eliminations, all found before fitting

**1. Every scalar K is structurally dead.** For K1 = exp(-alpha q) I, spherical
symmetry integrates exactly to g = g_N exp(alpha q). Every candidate q is
bounded in (0,1] and tends to 1 as the source is left behind, so
g -> exp(alpha) g_N: an inverse-square law with a rescaled G. **No scalar void
response can produce a flat rotation curve, for any alpha and any bounded q.**
That removes K1 x Q1..Q4 -- eight of the 24 models -- analytically.

**2. Q2 carries no new information.** The rank-2 theorem applies directly:
q_g is point-local in |g_N|, so it lives inside the (log a_N, log r) span and
cannot add anything beyond f(a_N, r). Q1/Q3 escape only via the *smoothed*
rho_L, and Q4 by construction. **The discriminating power is in the scales
L_rho and L_q, not the functional forms** -- a tournament varying m and n at
fixed scales explores a direction the data cannot resolve.

**3. The h_eff test cannot discriminate.** This was billed as the sharpest
test: cylindrical confinement requires h_eff proportional to sqrt(M_b), slope
exactly 1/2. Measured slope +0.4783 +- 0.0185, i.e. 1.2 sigma from 0.5, which
reads as strong support.

    source        slope s_h     error   scatter    n
    real            +0.4783    0.0185    0.2201  169
    RAR twin        +0.5372    0.0092    0.1095  169

The RAR twin -- same galaxies, g_obs replaced by the acceleration-relation
value exactly, containing no layer or anisotropy anywhere -- reproduces the
slope to 0.059 dex. One line explains it: in the deep-MOND limit
h_eff = R g_bar/g_obs = R sqrt(g_bar/a0) = sqrt(GM/a0). **h_eff ~ sqrt(M_b) is
an identity of the acceleration relation, not evidence for flux confinement.**

Consequence: the program's **vertical test becomes the primary discriminator**,
because A_dyn = (g_R/g_R,N)/(K_z/K_z,N) is where confinement makes a claim a
scalar relation does not (>1 versus ~1).

### BTFR caveat recorded

Measured slope +3.097 +- 0.085 against MOND's 4. **Not a refutation** -- V_f is
the median of the last three points rather than an asymptotic velocity, and
M_b is enclosed baryonic mass at the last measured radius, missing outer gas.
Published slope on properly measured quantities is ~3.85-4. Recorded as the
target the PDE tournament must hit using its *predicted* V_f.

### Net effect on the program

The 24-model tournament reduces to 16, the four void definitions reduce to
three genuinely distinct ones, and the test billed as sharpest is uninformative.
The first scientific question is no longer "does h_eff scale as sqrt(M)" but
whether one universal tensor equation predicts the radial and vertical fields
together.

---

## GOAL EXECUTION: anisotropic-void test program, Run A complete

Code: `Invariant/work/gravitylab/` (data.py, models.py, qfield.py, solver.py,
runA.py, runA_diagnostics.py, test_gates.py)

Run A is the program's own first executable run. Executed in full this time,
with the steps the earlier quick pass skipped.

### Steps 1-3: ingest, declared cuts, frozen split

Joined SPARC Table 1 (Lelli+ 2016: distances, inclinations, luminosities,
V_flat, quality flags) to the rotation-curve tables. Cuts declared in code
BEFORE any residual was examined:

    Qual <= 2, i >= 30 deg, V_flat > 0, >= 5 points

    175 galaxies with curves
    -12 Qual 3     -10 i < 30 deg     -30 no V_flat
    = 123 RETAINED, 2,858 radial points

Stratified 60/20/20 by whole galaxy across 12 strata (mass x V_flat x gas
fraction x quality), ordered inside each stratum by a hash of the NAME so the
assignment cannot correlate with anything the models see.
**75 train / 24 validation / 24 blind, sha256 e5f74522d2a4178d, frozen.**

### Steps 4-7: nuisance sampling, fit on train, evaluate blind

16 nuisance draws per galaxy over D ~ N(D0,eD), i ~ N(i0,ei),
Ups_d ~ lognormal(0.5, 0.10 dex), Ups_b ~ lognormal(0.7, 0.10 dex), with the
likelihood MARGINALISED over draws so no model can win by pushing a nuisance
to its prior edge.

    model                 free  chi2/pt tr  RMS tr  chi2/pt BL  RMS BL   lnL BL
    D1 Newton                0      700.59  0.5394      728.00  0.5544  -104089
    D2 AQUAL simple          1      134.21  0.1681       99.29  0.1590    -3810
    D2 AQUAL standard        1      137.64  0.1720      107.38  0.1681    -5448
    D3 piecewise             1      149.48  0.1798      121.43  0.1796    -7039
    D4 flattened log         2      235.96  0.3665      282.16  0.3138   -35785
    K1xQ1 rho                3      186.68  0.2406      160.72  0.2521    -8605
    K1xQ2 g                  3      132.62  0.1806      106.19  0.1706    -3749
    K1xQ3 rho+g              5      186.31  0.2403      160.40  0.2518    -8572
    K1xQ4 nonlocal           6      186.56  0.2403      160.37  0.2524    -8584

**No void model beats the scalar AQUAL benchmark on blind data.** The best of
them needs three parameters to land worse than AQUAL's one, on both chi2 and
RMS. Fitted a0 = 1.044e-10 from train alone, close to the canonical 1.2e-10.

### The finding that was not expected

**Q3 and Q4 collapse onto Q1.** Their fitted parameters are the same to three
figures (rho_c = 2.86e-22, m = 0.647, alpha = 1.71) and their blind RMS agree
to 0.0006 dex. The extra parameters do nothing: Q3's a0 goes to 7.1e-9 and
Q4's to 6.4e-9, ~50x canonical, driving the acceleration term to negligible,
and **Q4's nonlocality length fits to L_q = 0.12 kpc -- essentially zero.**

The screened-Poisson field is solved properly (tridiagonal, banded, 22,600
solves/sec), so this is not a solver failure. The data actively reject
nonlocality at this level. That matters because the rank-2 argument said Q4
was the one void definition carrying information the others could not.

### Required diagnostics

Residual correlations on held-out galaxies, Spearman rho of mean
log10(g_obs/g_pred):

    model              R/R_d  SB_eff    M_b   f_gas   incl   dist
    D1 Newton         -0.097  -0.701 -0.596  +0.756 -0.204 -0.168
    D2 AQUAL simple   -0.061  -0.006 -0.018  +0.064 -0.164 +0.062
    K1xQ2 g           -0.034  -0.108 -0.104  +0.203 -0.198 -0.001
    K1xQ4 nonlocal    -0.284  -0.610 -0.683  +0.593 -0.178 -0.304

**AQUAL leaves essentially no structure. Q4 leaves Newton's structure.** Per
the program's own galaxy-level rejection rule -- "residuals strongly correlate
with surface brightness, gas fraction, or radius" -- Q1/Q3/Q4 are rejected.

BTFR from each model's PREDICTED V_f (the program requires predicted, not
observed): AQUAL +3.834 +- 0.028 (scatter 0.070 dex), K1xQ2 +3.530, K1xQ4
+3.060, Newton +2.897. MOND predicts 4.

h_eff slope: AQUAL +0.5413, K1xQ2 +0.5211, K1xQ4 +0.3989, Newton +0.4094.

### Run A verdict

The scalar-tensor branch (K1 x Q1..Q4) is dead by the program's own rules,
confirming on blind data what the analytic argument proved: for K = exp(-alpha q)I
spherical symmetry gives g = g_N exp(alpha q), and bounded q forces
g -> exp(alpha) g_N, an inverse-square law with a rescaled G.

**The program must proceed to anisotropic K (K2 disk-axis, K3 tidal-axis,
K4 full tidal tensor), which requires the PDE solver and the section 6 gates.**

### Section 6 gates: a real bug caught

Built the finite-volume tensor solver and ran the mandatory gates. First
version used zero-flux (Neumann) boundaries and failed three of seven:
analytic comparison stalled at 1.8e-2 with convergence order **-0.13**,
eps_flux reached **0.32** against a 1e-5 threshold, and the box-size test moved
**5.6%** against 0.5%.

One root cause: **a sealed box is self-contradictory for an isolated source.**
Gauss's law over any surface enclosing all the mass demands 4 pi G M of flux,
but zero-flux edges force the total to zero, so the solver silently
manufactures a compensating uniform background. Fixed with open boundaries --
Dirichlet values from the exact constant-K monopole
Psi = -GM/(sqrt(det K) sqrt(r^T K^-1 r)) -- so flux may leave the domain.

### Section 6 gates: all seven pass, solver cleared

    [PASS] 6.1 dimensional consistency, q bounded in (0,1]
    [PASS] 6.2 eigenvalues of K exceed 1e-6
    [PASS] 6.4 analytic constant-K, convergence order 1.99, error 3.63e-4
    [PASS] 6.3 flux conservation, eps_flux = 6.3e-15 (threshold 1e-5)
    [PASS] 6.5 curl at round-off, 4.7e-17
    [PASS] 6.6 Newtonian recovery, 2.33e-4, order 2.0
    [PASS] 6.7 domain convergence, 0.089% (threshold 0.5%)

Three of the initial failures were bugs in the TESTS, not the solver, and each
is worth keeping because each produced a plausible-looking wrong number:

1. **Zero-flux boundaries.** Self-contradictory for an isolated source: Gauss's
   law over a surface enclosing all the mass demands 4 pi G M, a sealed box
   forces zero, so the solver manufactures a compensating uniform background.
   Cost three gates at once. Fixed with open Dirichlet boundaries from the
   exact constant-K monopole.

2. **Flux measured on the wrong quantity.** Conservation must be checked on the
   FACE fluxes the discretisation conserves; re-deriving a flux from a
   centre-differenced gradient is a different quantity agreeing only to O(h^2).
   It reported eps ~ 1e-2 on a solver that conserves to 1e-15 -- a factor of
   1e13 of pure measurement error.

3. **Wrong source geometry for the anisotropic comparison.** A sphere in r is
   an ELLIPSOID in u = sqrt(r^T K^-1 r) and carries a u-space quadrupole
   falling only as (sigma/u)^2. That floors the error near 1e-2 at EVERY
   resolution. A flat error curve with resolution is the signature of a
   modelling mismatch rather than a discretisation error, and that is what
   distinguished it from a real solver bug. With a u-spherical source the error
   drops to 3.6e-4 and converges at order 1.99.

**Next: Run B, the axisymmetric PDE tournament over the 16 surviving
Q x K x mu combinations (K1 eliminated, Q2 redundant), on the frozen split.**
