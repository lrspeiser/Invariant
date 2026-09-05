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

> **CORRECTED IN RUN AT.** "Organised by r/R500" is dead as a distinct claim:
> `log(r/R500_i) = log r - log R500_i`, and `log R500_i` is already in the span of
> the cluster indicators, so given per-cluster levels r/R500 and r are the SAME
> regressor (rank 13 = 13; slopes equal to ten decimals). The normalisation is
> worth 0.0095 in correlation and the well-posed comparison separates by 0.68
> sigma. Read "organised by RADIUS". The trend itself survives — 16 sigma against
> a forward null, and unchanged under baryon-only radii.
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

---

# Run J — GPU-scale law search against real data, with physics-free twins

The instruction was that a few hundred tests of one or two formulas is not
sufficient, that the RTX 5090 permits billions of combinations, and that the
work should run as a reinforcement-learning loop that tries, edits, prunes and
keeps going. That is what this run is. It also closes the gap left in the
well-mirror sweep, where Option 1 was never tested.

## J.1 Why the existing billion-candidate screen did not count

`runs/gpu-baryonic-screen/billion-v1.json` records in its own manifest
`observational_data_opened = False` and `synthetic_analytic_controls_only =
True`. A billion laws were screened for flatness, BTFR slope and Newtonian
limits on idealised analytic curves, and not one was ever scored against an
observation. The number was real; the contact with data was not.

## J.2 The architecture that makes billions mean something

A candidate law is a functional FORM, not a coefficient vector. For any fixed
set of basis functions the optimal coefficients solve a linear system, so
sampling coefficients at random is a slow, bad way to rediscover what linear
algebra gives exactly. The expensive axis is WHICH basis functions appear.

So the search is an ATOM BANK plus SUBSET SELECTION:

    1. An atom is one parameterised basis function -- e.g. 1/(1 + x/s) at one
       particular s -- evaluated on every point-draw of the bench. Ten
       functional families over eight physical variables on a 25-point
       logarithmic scale grid give 1,898 atoms.
    2. The atom Gram G = A A^T and the projections v = A y are computed ONCE,
       on the training points only, in float64.
    3. A candidate law is a SUBSET of K atoms. Its optimal coefficients and its
       exact training RMS come from a K x K sub-matrix GATHERED out of G. The
       data is never touched again.

Scoring one candidate with its optimal coefficients therefore costs O(K^2)
memory traffic and a K x K Cholesky -- about 1e2 flops -- instead of
O(K n_points), about 1e5. That is the entire trick, and it is what makes the
throughput real rather than rhetorical.

    measured: 4.55 billion exact optimally-fitted evaluations in 59 s
              = 77 million candidate laws per second, against real SPARC
              = 2.8e11 per hour

The intercept is carried in every model and is not counted toward K. That
detail mattered: because atoms are standardised to zero mean, genomes that
happened to contain no intercept could not reach the data's mean at all and
scored ~0.47 dex, which made every k=6 result look catastrophic. It was the
encoding, not the solver.

## J.3 What the rank-2 theorem actually licenses

The theorem is that the eight-variable set (a_N, Sigma_b, rho_b, r, M_b, theta,
Phi_b, environment) has numerical rank 2 on this bench. It does NOT say only
functions of a_N are available. It says there are exactly TWO independent
directions and that a_N and r span them. Atoms are therefore tagged by which
direction they use -- [g] the RAR's own direction, [r] the second one, [gal]
galaxy-level constants, [nl] nonlocal functionals -- so the report can state
what a winning law needed instead of asserting it. An earlier draft of this
module tagged r as "escaping the rank-2 span", which was wrong.

## J.4 The controls, which are the point

Every candidate is scored against four targets that differ only in v and y.y,
so all four cost the same gather:

    real     measured log10(g_obs / g_bar)
    null     RAR + Gaussian noise matched to the real residual scatter
    perm     RAR + the real residuals permuted across ALL points
    perm_g   RAR + the real residuals permuted WITHIN each galaxy -- keeps
             every galaxy's mean offset, since distance, inclination and M/L
             errors are real and are not new physics, and destroys only the
             radial dependence. This is the sharp control.

Each target is scored against ITS OWN baseline. Comparing a control to the real
target's baseline is a baseline mismatch, not a control.

## J.5 Result 1 — the evolutionary run, three seeds

4,000,000 population x 300 generations x 4 targets = 4.8e9 evaluations per
seed, 1,898 atoms, K = 8, ~4 minutes per seed.

    seed 0    train real +5.15%   perm_g +3.20%    blind real -4.07%   0/20 win
    seed 1    train real +5.20%   perm_g +3.20%    blind real -3.92%   0/20 win
    seed 2    train real +4.93%   perm_g +3.07%    blind real -4.22%   0/20 win

Three independent seeds, 14.4e9 evaluations in total. The spread across seeds
is 0.27 points on train and 0.30 on blind, and not one of the sixty shortlisted
laws beat the RAR out of sample.

Read the middle column first. A target with the radial physics deliberately
destroyed recovers +3.1% of the +5.1% the real data gives. Roughly five eighths
of the apparent training gain is manufactured by the search itself.

And the remainder does not survive. On the blind galaxies, with the
coefficients frozen at their trained values, the best laws are 4% WORSE than
the RAR, while the physics-free controls put through the identical
shortlist-then-blind procedure come out slightly ahead (perm_g +1.31%).

A bug found on the way to that number is worth recording, because it produced a
false positive: the first blind evaluation called the scorer with the blind
Gram, which REFITS the coefficients on the blind set. That reported +2.17%. The
correct procedure -- fit on train, freeze, evaluate -- reports -3.73% on the
same law. A held-out set that the fit is allowed to touch is not held out.

## J.6 Result 2 — the exhaustive enumeration, which is a proof

At low complexity the space is small enough to enumerate completely, and then
the answer is not "the best we found" but "the best that exists".

    k = 1           1,897 laws
    k = 2       1,798,356 laws
    k = 3   1,135,961,540 laws
    total x 4 targets = 4.55 billion, in 59 s

                real gain    best control    margin      BLIND real   margin
    k = 1        +0.23%        +0.42%       -0.19 pp      -0.07%     -0.78 pp
    k = 2        +2.29%        +1.28%       +1.01 pp      -2.14%     -2.23 pp
    k = 3        +3.61%        +2.13%       +1.47 pp      -3.66%     -3.70 pp

The monotonic pattern is the finding. The more complexity the search is
allowed, the better it fits the training set and the worse it does out of
sample -- and this is now demonstrated by enumeration over every possibility,
not inferred from a search that might have looked in the wrong place.

At k = 1 the real data gain less than the physics-free control does. The RAR's
own interpolating function, with a0 refitted, is already essentially optimal in
its own direction; there is nothing left to win there.

Two further facts fall out:

  * The exhaustive k = 3 winner is EXACTLY the law the evolutionary search
    found at k = 3 -- nu_std(gbar/2.15) + nu_simp(r/0.841) + expneg(r/1.36),
    0.18792 dex. The evolutionary loop located the true global optimum. That
    validates the loop as an instrument.
  * The perm_g winner at k = 3 is built from two galaxy-level M_b atoms. The
    control is exploiting exactly the galaxy-level offsets it was constructed
    to preserve, which is the behaviour it was designed to expose.

Every reported RMS is verified against a direct recomputation from the data,
bypassing the Gram algebra entirely: max |difference| = 4.5e-8 dex.

## J.7 Result 3 — Option 1 of the well-mirror model

Option 1 was specified as a perpendicular push projected on a curved well,

    g_mirror(r) = A_u(r) h'(r) / (1 + h'(r)^2)
    g(r)        = (1 - eta) g_N(r) + eta g_mirror(r)

and was omitted from the earlier sweep. It has two readings.

**Option 1a**, the ansatz as written, closes the system by DEFINING
g_mirror = g_N + V_W^2/(r + r_t), so that the mirror restores the eta g_N
removed from the direct well and adds an edge term. Substituting gives
g = g_N + eta V_W^2/(r + r_t): eta appears only in the product eta V_W^2.
Refitting at fixed eta = 0.10, 0.25, 0.50, 0.75, 0.90 with everything else free
gives train RMS 0.224752 at every value, spread 0.00e+00 dex -- bit-identical.
eta is not a measurable push/pull ratio in Option 1a; it is a relabelling of the
edge-term amplitude. That is the same degeneracy already found in Options 2
and 3, now confirmed for Option 1.

The form also fits worse than the RAR: train 0.2240 / blind 0.2093 against
0.1970 / 0.1768, i.e. -18.4% on blind. The structural reason is visible in the
formula: as r -> 0 the edge term tends to V_W^2/(lambda r_M), a CONSTANT, not
zero, so the model fails to reduce to Newton in the inner region.

**Option 1b**, the literal point mirror, is the one that answers the question
actually asked. An attractive source at u = -d with an equal repulsive image at
u = +d gives, on the middle surface, A_u(r) = 2 G M_b d/(r^2 + d^2)^{3/2}. Not
inserting g_N into the mirror by hand:

    h'(r) = chi r_M/(r + lambda r_M),  d = delta r_M,  r_M = sqrt(G M_b/a0)
    g     = (1 - eta) g_N + eta A_u h'/(1 + h'^2)

Here eta IS identifiable, because it sets the coefficient of the direct
Newtonian term independently of the mirror amplitude, and at small r the data
pin it. Profiling eta from 0 to 0.95 with all other globals refitted at each
step:

    eta = 0.00   0.53446        eta = 0.50   0.65552
    eta = 0.05   0.53266        eta = 0.75   0.71949
    eta = 0.25   0.55720        eta = 0.95   0.98941

    best eta on train: 0.05, spread across eta 0.4567 dex

So the push fraction is driven to essentially zero, and eta = 1/2 -- the "half
pull, half push" concept -- is excluded by 0.12 dex of RMS. The model at its
best eta is 0.5327 train / 0.5427 blind against the RAR's 0.1970 / 0.1768, i.e.
-207% on blind: at eta -> 0 it degenerates to pure Newtonian gravity, which is
what 0.53 dex is.

This is the outcome the proposal's own text anticipated: a point-like mirror
gives A_u ~ 1/r^3 at large r and cannot sustain the 1/r field a flat rotation
curve requires. The data agree, quantitatively, and they do so in the one
formulation where the push/pull ratio is a measurable quantity rather than a
reparameterisation.

## J.8 Where this leaves the register

Nothing was discovered. What was established is stronger than another
falsification of one formula:

  1. **A proof at low complexity.** Over this atom bank, NO law of complexity
     <= 3 beats the RAR out of sample, and none beats the physics-free controls
     by more than the controls beat themselves. This is enumeration, not
     sampling.
  2. **A calibrated overfitting scale for this bench.** A physics-free twin
     with only per-galaxy offsets preserved yields +3.2% of training gain at
     K = 8 and +2.1% at k = 3. Any future claim on SPARC of a few per cent
     improvement over the RAR now has a number to clear.
  3. **An instrument that is validated.** The evolutionary loop recovered the
     exhaustive global optimum at k = 3, so its results at K = 8, where
     enumeration is impossible, can be trusted as searches rather than
     accidents.
  4. **The push/pull ratio is settled.** It is unidentifiable in Options 1a, 2
     and 3 by construction, and in Option 1b, where it is identifiable, the
     data set it to ~0.05 and exclude 0.5.

The honest reading of the negative result is that this bench is exhausted for
point-local laws. Two independent directions, one already used optimally by the
RAR, and the second one buys training fit that does not generalise. Progress
requires a probe that adds a direction -- the vertical/anisotropy channel, or
cluster scales where the RAR is known to fail -- not more search in this space.

**Artifacts:** `work/gravitylab/hypersearch.py`, `evolve.py`, `exhaustive.py`;
`work/gravity-cluster-audit-2026-09/mirror/mirror_option1.py`;
`runs/evolve-sparc/{evolve-seed0,evolve-seed1,evolve-seed2,exhaustive-k3}.json`;
`work/gravity-cluster-audit-2026-09/mirror/option1_results.json`.

---

# Run K — LoCuSS, the exact forward chain, and one retracted result

The first critique demanded the amplified-pressure law be evaluated through its
exact forward chain rather than the compressed `E^2 - 1 = kappa t` proxy, with
the `M_gas/M_b` factor restored and the full RAR interpolation on both sides.
Done, on 40 of the 41 LoCuSS clusters (Mulroy 2019), against Subaru weak-lensing
masses — so this test is free of hydrostatic circularity. Artifacts in
`work/gravity-cluster-audit-2026-09/locuss2/`.

    1  aperture      r500 = [3 M_WL / (4 pi 500 rho_c(z))]^(1/3)   <- LENSING mass
    2  stars         M_star  = 0.73 L_K,tot
    3  baryons       M_b     = M_gas + M_star
    4  thermal       t       = 3 kT / (mu m_p c^2),  mu = 0.6
    5  pressure mass DM_P    = (3 kappa/c^2) Int 4 pi r'^2 P dr' = kappa t M_gas
    6  Newtonian     g_N_eff = G (M_b + DM_P)/r500^2
    7  acceleration  g       = nu(g_N/a0) g_N
    8  PREDICTION    E_pred  = F(g_N_eff)/F(g_N_b)
    9  OBSERVATION   E_obs   = M_WL / (nu(x_b) M_b)

Step 5 is the only algebraic substitution and it is exact for isothermal gas
independently of the density profile — verified numerically on a beta-model to
1.000000000000. Pipeline validated by reproducing the previous run's compressed
form to the last digit.

## K.1 The exact chain does not rescue the model; it hurts it

The reason only appears once the chain is actually computed. At kappa = 1.36e5
the pressure term multiplies the source by 3.3-9.7, which lifts g_N_eff/a0 from
0.042-0.119 to 0.175-0.914 — out of the deep-MOND regime and into the RAR
transition, where F is steeper than sqrt. The prediction is therefore 19% HIGHER
than the deep-MOND formula implies. The restored M_gas/M_b factor pulls the
other way but is worth only 1/0.895 = 1.107 in kappa. They do not cancel.

    stated amplitude kappa = 1.36e5     E_pred/E_obs median 1.756, all 40
                                        over-predicted, weighted mean ln
                                        residual -0.544 at -13.0 sigma
    free amplitude                      4.08e4 = 0.300 x X-COP, and 1.36e5 lies
                                        far outside the bootstrap interval
    universal kappa?                    per-cluster kappa_i spans a factor 8.45,
                                        running down with kT (-0.342, p = 0.031)
    amplitude vs shape                  amplitude wants 4.08e4; the temperature
                                        shape excludes anything above ~1.7e4

That last line is an internal contradiction, not merely a poor fit: a
one-parameter source term cannot satisfy both channels at once.

The null is informative. Size-corrected to a true 5% false-positive rate, this
sample has power 1.000 at kappa = 1.36e5 and an 80% detection floor of
4e3-1.6e4, a factor 9-35 below the tested value. A coupling of the claimed size
would have been seen every time.

## K.2 RETRACTION — the negative correlation was an artefact

Recorded previously as evidence against the model: `rho_p = -0.304`, the wrong
sign for the pressure hypothesis. **That result is withdrawn.**

`ln E_obs` and `ln M_WL` carry measurement errors correlated at +0.96, because
M_WL appears in both. The naive partial estimator therefore has expectation
**-0.12 under a true null** — attenuation 0.66 — so the observed naive
`p = -0.155` sits at **p = 0.563 against its own biased null**. An
errors-in-variables MLE, validated as unbiased across the whole p range by
simulation, returns

    p_EIV = -0.166  [-0.356, +0.228]    p = 0.143   (null median +0.010)
    leave-one-out across all 40:  -0.205 to -0.093

There is no anti-correlation with temperature. There is no correlation at all.
The model is still refuted — the amplitude channel does that at -13 sigma, and
the model's own predictive distribution for this same statistic is +0.592
[+0.347, +0.889], which sits 5.4 sigma from the measurement — but the specific
claim that the correlation ran the wrong way was a statistical artefact of a
shared noisy denominator, and it should not be repeated.

This is the second time in this programme that a shared-denominator or
shared-label effect has manufactured a result. It is now the standing first
suspicion for any correlation involving a quantity that appears on both axes.

## K.3 What is real and survives

A large gravity excess over baryons-plus-RAR is present and is measured with
weak lensing, so it is not hydrostatic bias: E median 1.62, range 1.22-2.34. It
behaves as a roughly constant multiplicative offset (free constant +0.603 in
ln E, a factor 1.83) with a mild MASS dependence (d ln E/d ln M_WL = +0.31) and
NO temperature dependence. That is the classic MOND cluster missing-mass
problem, independently confirmed with lensing, and it is simply not shaped like
3 kappa P/c^2.

Declared limits: isothermality assumed (a realistic declining T(r) moves the
amplitude by 0.88-0.94, reconciliation needs 0.30, so the assumption is not
load-bearing); a0 and nu held fixed; no gravitational slip; non-thermal pressure
unmeasured and would make the over-prediction worse; the radially resolved test
still needs joint X-ray/SZ pressure profiles with independent lensing masses.

One monotone-invariance check was reproduced deliberately: two rank statistics
are exactly constant across three decades of kappa, spread 0.000000 — the same
trap that made the earlier X-COP radial test vacuous. Every statistic used in
Run K moves by orders of magnitude across that range.

---

# Run L — A_dyn from DiskMass, and a second blind statistic caught

The second critique's objection was accepted: h_sigma_z/h_R cannot see a constant
boost, so the earlier "the vertical field is near-Newtonian" claim did not follow
from it. This run replaces that statistic with the ABSOLUTE dispersion amplitude
and builds the full forward chain. Artifacts in
`work/gravity-cluster-audit-2026-09/adyn/`.

## L.1 The old statistic was blind, now demonstrated numerically

Holding the baryon model fixed, multiplying K_z by a constant B_0, and re-fitting
the exponential exactly as DiskMass did:

    B_z = 0.25   sigma_z,0 =  25.87 km/s   h_sigma_z = 31.64452 arcsec
    B_z = 1.00   sigma_z,0 =  51.73 km/s   h_sigma_z = 31.64452 arcsec
    B_z = 8.00   sigma_z,0 = 146.32 km/s   h_sigma_z = 31.64452 arcsec

The amplitude spans 0.753 dex; the scale length moves 1.6e-15 dex across a factor
of 32 in the very parameter it was supposed to measure. That is the SECOND
monotone-blind statistic caught in this programme, after the X-COP rank test.
The parameter-responsiveness check dS/dtheta != 0 is now mandatory.

## L.2 What the amplitude can and cannot decide

    B_z observed   0.715   68% [0.468, 1.079]   95% [0.301, 1.670]
    statistical only              [0.768, 0.853]

The systematic floor is 0.191 dex, **8.4x the statistical part**. It is a
degeneracy, not a noise problem, and the dominant terms are common-mode, so more
galaxies do not help.

    Upsilon_K zero point   0.154 dex        alpha = sigma_z/sigma_R  0.078
    h_z relation           0.072            k, vertical profile      0.033

Every law sits within 2 sigma: Newton 0.76, RAR 1.70, AQUAL 1.71, anisotropic
tensor 0.81, isotropic tensor 0.76. The largest law-to-Newton separation is
0.190 dex against a width of 0.192 dex, i.e. **0.99 sigma**. So:

    A_dyn = 1 and A_dyn > 1 are NOT separable with this measurement.

What IS decided: the isotropic tensor is rejected from the SCALE LENGTH, at
chi^2/dof ~ 133 against observed h_sigma_LOS = 28.65 arcsec versus its predicted
48.16. The scale length is blind to amplitude but not to shape.

## L.3 A factor of two that was hiding in a symbol

DiskMass write Sigma_dyn = sigma_z^2/(pi G k h_z) and adopt k = 1.5. k is fixed
by the SHAPE of the vertical profile at fixed exponential scale height, and for
the van der Kruit family rho ~ sech^(2/n)(n z/2 h_z) the z-Jeans equation gives
exactly

    k = Int_0^inf sech^(2/n)(n u/2) du = (2/n) (sqrt(pi)/2) Gamma(1/n)/Gamma(1/n + 1/2)

so k = 2 for sech^2, pi/2 = 1.5708 for sech, 1 for exponential. k enters
sigma_z^2 linearly. Feeding the tabulated h_z into a sech^2(z/h_z) layer — the
naive reading — is a factor-2 error in B_z. It is drawn from a prior and is never
fixed by the data.

## L.4 The one live signal, and why it is not promoted

A common-mode error in Upsilon_K, h_z or k moves the intercept, not the slope,
and the sample spans a factor of 35 in central K-band surface density. The
between-galaxy trend is

    d log B_z / d log Sigma_0 = -0.346 +- 0.173   (2.0 sigma)
    galaxy bootstrap           -0.349 [-0.561, -0.071],  p(slope >= 0) = 0.0095

against predictions of 0.000 (Newton), -0.291 (RAR), -0.264 (AQUAL), -0.020 and
-0.055 (the tensors). It survives every audit run against it: a label-control
shuffle null at p = 0.000-0.008; errors inflated by the chi^2/dof = 1 factor
before any significance is quoted; partial slopes holding colour, central surface
brightness and the h_z-h_R relation; inclination uncorrelated with Sigma_0
(r = -0.057) so an alpha error cannot fake it; a 20 km/s instrumental floor
weakening it only to -0.260; a factor-6 gas-fraction trend weakening it only to
-0.304; and the Bershady relation would have to be wrong in SLOPE by 1.99 against
a published 0.643 to remove it.

**And the same data contradict it radially.** The within-galaxy radial dependence
of B_z that RAR and AQUAL require is not seen — observed h_sigma_LOS = 28.65
arcsec sits essentially at the Newtonian 30.80, against RAR's 35.20. The
between-galaxy trend and the within-galaxy trend disagree, so the economical
reading is a Sigma_0-correlated systematic not yet identified. It is NOT promoted
to evidence for MOND.

## L.5 What a decisive test would require

log B_z = 2 log sigma_z - log Upsilon_K - log h_z - log k + const, so a 3 sigma
separation of RAR from Newton needs the total budget below 0.063 dex and each
dominant term near 0.032 dex (8%):

    Upsilon_K to 8% ABSOLUTE     4.9x better than today (an IMF zero point)
    alpha measured not adopted   2.5x
    h_z MEASURED not inferred    2.3x
    k pinned by resolving the tracer population mix   1.1x

None of that is a sample-size problem. The differential route needs no zero
points and would need N ~ 220 for sd(slope) = 0.15 — but for the tensor laws the
predicted B_z varies galaxy to galaxy by only 0.027 dex, so no sample size works
and a wider RANGE IN SURFACE DENSITY is required instead.

## L.6 An endogeneity that bounds the headline

V_c sin i / sin(i_TF) = V_flat(TF) to 0.4% for all 28 galaxies: the deprojected
circular speed IS the Tully-Fisher prediction from M_K. No law is fitted to
DiskMass rotation (SPARC only) and the rotation-curve shape is real and is used,
but the Upsilon-free A_dyn = 3.75 [2.76, 5.15] is TF-conditional and is not the
headline. The Upsilon-independence itself was verified rather than asserted:
scanning Upsilon_K over 16x moves B_R by 1.20 dex and A_dyn by 0.00 dex, while
the same code shows A_dyn responding properly to h_z, k and alpha.

---

# Run M — the spherical blindness theorem

Before spending compute on well-alignment and pair-channel tensors, one
structural question decides the experimental design: which observations can see
them at all? Code and JSON in `work/wellnet-2026-09/spherical_blindness.py`.

**THEOREM.** For div_i[mu(X) K^ij grad_j Phi] = 4 pi G rho with rho = rho(r)
spherically symmetric and the most general spherically symmetric response tensor
K = kappa_r(r) rhat rhat^T + kappa_t(r)(I - rhat rhat^T), the solution Phi(r)
depends on kappa_r ALONE.

    Proof. Spherical symmetry gives grad Phi = Phi'(r) rhat, so K grad Phi =
    kappa_r Phi' rhat — the transverse block annihilates it — and
    X = |Phi'| sqrt(kappa_r)/a0. The divergence of a radial field F(r) rhat is
    (1/r^2) d/dr (r^2 F), so the equation integrates exactly over a sphere to
    r^2 mu(X) kappa_r(r) Phi'(r) = G M(<r). kappa_t never enters. QED

**Corollary 1.** For K = exp[s_0 I + s_T S] with S traceless, spherical symmetry
forces S = s(r)(rhat rhat^T - I/3), so kappa_r = exp(s_0 + (2/3) s_T s) and
kappa_t = exp(s_0 - (1/3) s_T s). Only the combination s_0 + (2/3) s_T s is
observable; the anisotropy parameter s_T is not separately measurable.

**Corollary 2.** A constant s_0 gives K = e^{s_0} I, which is Newtonian gravity
with G -> G e^{-s_0}. Only the VARIATION of s_0 is physical.

**Corollary 3.** With no gravitational slip, lensing is built from the same Phi,
so a spherically averaged lensing profile cannot separate kappa_t either.

## M.1 Numerical verification, and a bug it caught first

Varying kappa_t by a factor of 3 with kappa_r fixed, on the validated 3-D solver:

    n     h [kpc]   transverse max   transverse med   radial max   ratio
     32     18.75     2.051e-2         6.62e-3          3.369e-1     16.4
     48     12.50     7.093e-3         4.01e-3          3.344e-1     47.1
     64      9.38     4.765e-3         2.61e-3          3.341e-1     70.1
     80      7.50     3.122e-3         2.02e-3          3.339e-1    107.0

Convergence order 2.02; refining the grid 2.50x drops the transverse sensitivity
to 0.152 of its coarsest value, where an exact order-2.02 artefact predicts
0.157. The radial sensitivity is flat at 0.334 throughout, as it must be. The
ratio DIVERGES with refinement, which is the signature of one quantity converging
to zero while the other does not.

The first attempt reported 2.2e-2 flattening toward a non-zero floor and looked
like a modelling mismatch. It was a bug in the COMPARISON, not the solver: the
radial bin edges were derived from the grid, so every resolution was compared at
different radii and the maximum was taken over a moving window. With fixed
physical shells the second-order convergence appears immediately. This is the
same class of error as the three test bugs in section 6 of the earlier record —
the measurement, not the thing measured.

## M.2 What this settles about experimental design

Any test of a well-network or pair-channel tensor performed on spherically
averaged cluster data is **vacuous with respect to the anisotropy it claims to
test**. It measures an effective radial rescaling, which is degenerate with a
modified mu and with a modified G. A resolved, non-spherical configuration is not
merely a better test; it is the only thing that can see the mechanism.

That is the formal justification for making the primary statistic the comparison
of the TRUE member configuration against mass-preserving angular scrambles,
rather than any fit to a radial profile.

---

# Run N — is potential depth a new direction on SPARC?

Before assembling a galaxies-to-clusters ladder to test
g_obs = F(g_bar, |Phi_b|), the cheap prior question: on SPARC alone, is |Phi_b| a
NEW direction, or is it already spanned by the two the rank-2 theorem found?
Code in `work/wellnet-2026-09/phi_rank.py`.

Phi_b(r) = -[Int_r^Rmax g_bar dr' + g_bar(Rmax) Rmax], i.e. a Newtonian
point-mass tail beyond the last measured point, for which the outer integral is
exactly G M/Rmax. A flat-ish tail with a 10x Rmax cutoff gives the same
conclusions, so nothing below depends on that choice.

    2,856 points, 123 galaxies, log|Phi_b| spanning 3.21 dex

    linear in (log g_N, log r)      R^2 = 0.9007   residual 0.218 dex
    quadratic in (log g_N, log r)   R^2 = 0.9322   residual 0.180 dex
    median spread of log|Phi_b| within narrow log g_N bins:  0.309 dex
      (falling to 0.19 dex in the high-acceleration bins)

    corr(RAR residual, log|Phi_b|)                      +0.105
    partial, controlling for log g_N and log r          +0.0175
    galaxy-level bootstrap CI                    [-0.118, +0.145]

So |Phi_b| is 93% predicted by a smooth function of the two known directions, the
independent residual is only 0.18-0.22 dex, and once the two known directions are
controlled for the RAR residual carries no potential-depth signal at all. SPARC
can exclude a partial correlation larger than 0.145 and no more.

**SPARC alone cannot decide the potential-depth hypothesis.** The
galaxies-to-groups-to-clusters ladder is not an improvement on this test; it is
the entire test. The number that ladder must beat is 0.309 dex of at-fixed-g_bar
leverage, and the honest measure of whether it has bought a new direction is the
R^2 of log|Phi_b| on a quadratic in (log g_bar, log r), not a correlation
coefficient.

---

# Run O — billions of FIELD laws, and the two theorems that shrink the space

Run J's engine reached 77 million laws per second because a candidate was a
subset of precomputed basis functions and its optimal coefficients came from a
Gram computed once. That works only when the map from coefficients to prediction
is linear, and a field law is not algebra: scoring one means solving a nonlinear
elliptic PDE, seconds per candidate on a 128^3 grid. At that rate a
billion-candidate search over field equations is about thirty years.

## O.1 The trick that collapses it

Write the law in QUMOND form, so the source is an explicit pointwise function of
the Newtonian field, computed once:

    laplacian Psi = div [ nu(|grad Phi_N|/a0) K(I) grad Phi_N ]

and expand the response tensor linearly in the grammar's own basis:

    K(I) = I + sum_alpha c_alpha f_alpha(I) B_alpha

Because div and the inverse Laplacian are both linear, the solution decomposes
EXACTLY:

    Psi      = Psi_0 + sum_alpha c_alpha R_alpha
    Psi_0    = laplacian^-1 div [ nu grad Phi_N ]
    R_alpha  = laplacian^-1 div [ nu f_alpha(I) B_alpha grad Phi_N ]

One Poisson solve per ATOM, not per candidate. Measured: **361 isolated-boundary
FFT Poisson solves in 1.2 s on a 64^3 grid**, after which every sparse subset
with its optimal coefficients is an instant linear combination and the Run J Gram
machinery applies verbatim. The exponential form is recovered to first order in
c, and the shortlist is re-solved exactly with the validated nonlinear
finite-volume solver — the linear expansion is a screen, never a verdict.

Code: `work/wellnet-2026-09/field_grammar.py`, `field_search.py`.

## O.2 Two theorems that shrink the search space before it is searched

**Spherical blindness** (Run M): for a spherically symmetric source the
transverse eigenvalue of K is exactly invisible. So this engine may only be run
on resolved, non-spherical configurations, and `field_grammar.py` computes the
source's axis ratio and REFUSES to proceed above 0.97 rather than returning a
degenerate answer silently.

**QUMOND tensor degeneracy** (new, and found by the engine itself when I and
ghat ghat^T returned identical amplitudes on every scalar shape). In QUMOND form
K appears only through the vector field K grad Phi_N, so any two tensors agreeing
on grad Phi_N are indistinguishable. In particular

    (ghat ghat^T) grad Phi_N = ghat (ghat . grad Phi_N) = ghat |grad Phi_N|
                             = grad Phi_N = I grad Phi_N

so the field-direction projector is EXACTLY the identity in disguise. Verified
numerically at 6.4e-16 relative — round-off. A tensor basis element is
independent only if it TURNS the flux away from the Newtonian direction; the
tidal tensor and a fixed structural axis do, the field-direction projector cannot.

Between them the two theorems removed **150 of 360 generated atoms as exact
duplicates**, 42% of the bank. Those duplicates were not merely wasteful: they
make the normal equations singular and the "selected" atom arbitrary among its
clones, which is precisely how a search reports a discovery it did not make.

    generated 360 -> independent 210 (140 anisotropic, 70 scalar)

## O.3 The engine is validated before it is used

Both halves of control 5 were run before any science.

**RECOVERY.** Inject a known two-term anisotropic law and search exhaustively:

    injected   log1p(x_g/0.3) x dd  +  inv1p(x_Phi/3) x That
    noise  0%   best k=2 residual 9.73e-08   recovered 2/2
    noise  2%   best k=2 residual 2.92e-02   recovered 2/2
    noise 10%   best k=2 residual 1.53e-01   recovered 2/2

The exact injected pair is recovered at every noise level, from 210 atoms.

**ABSTINENCE.** Inject a purely SCALAR law — no anisotropy anywhere — and count
how often the search selects an anisotropic atom anyway:

    1 of 24 scalar-only injections   =>  false-positive rate 4.2%

That number is the credibility of any future "anisotropy detected" claim from
this engine, and it is now measured rather than assumed.

**MANUFACTURED GAIN.** With no law injected at all, the best three atoms out of
210 reduce the residual by only **+0.2%**.

## O.4 The contrast that justifies the whole change of level

    manufactured gain at k=3, physics-free
       SPARC rotation curves (Run J)        +2.1%
       resolved field observable (Run O)    +0.2%

An order of magnitude less overfitting capacity, on a bench with FEWER atoms
(210 against 1,898). The reason is structural: the SPARC bench carries only two
independent directions, so a 1,898-atom bank is 1,898 ways of writing a function
of two numbers, and the search spends its freedom on noise. The field observable
is 3,072 numbers carrying genuine geometric information — a midplane
acceleration map and a projected deflection map on a triaxial source — and an
atom must reproduce that geometry to score, which it cannot do by accident.

This is the quantitative form of the argument for moving the search from
transformations of (g_N, r) to operators acting on resolved mass maps: the new
level is not merely more interesting, it is measurably harder to fool.

## O.5 What is NOT claimed

No law has been fitted to an observation here. This run establishes the engine,
its two exact degeneracies, its recovery rate, its false-positive rate for
anisotropy, and its manufactured-gain floor. The resolved cluster data is being
acquired in a parallel lane and the tournament runs on it, not on the synthetic
triaxial source used for validation.

The linear-in-c expansion also has a systematic error against the exact nonlinear
solve which is NOT yet measured; that measurement is a prerequisite before any
shortlist from this engine is reported, and the shortlist must be re-solved with
the finite-volume solver regardless.

---

# Run P — where the field-law screen stops working

Run O's engine screens candidate field laws by expanding K = I + sum c_a f_a B_a
so the QUMOND solution decomposes into precomputed atom responses. But the
grammar's actual tensor is the matrix exponential exp[sum c_a f_a B_a], which is
what guarantees symmetry and positive-definiteness. The linear form is its
first-order truncation, and a screen that is systematically wrong is a screen
that discards the right answer. In QUMOND form the exponential version is still
ONE Poisson solve, so the truncation error can be MEASURED rather than bounded.

Test law: inv1p(x_T/0.3) x That + id(q_L/0.3) x That + id(x_g/3) x I.

    |c|    ||M||max   min eig K   screen err / signal   verdict
    0.01     0.149    8.90e-01     7.67e-04 / 3.39e-02 = 0.023   ok
    0.03     0.448    7.05e-01     7.06e-03 / 1.06e-01 = 0.066   ok
    0.10     1.495    3.12e-01     8.57e-02 / 4.14e-01 = 0.207   MARGINAL
    0.20     2.989    9.73e-02     3.96e-01 / 1.05e+00 = 0.378   MARGINAL
    0.40     5.978    9.46e-03     2.23e+00 / 3.51e+00 = 0.633   UNUSABLE
    0.80    11.957    8.95e-05     2.14e+01 / 2.39e+01 = 0.895   UNUSABLE
    1.50    22.419    2.57e-08     6.19e+02 / 6.24e+02 = 0.993   UNUSABLE
    3.00    44.837    6.60e-16     1.19e+06 / 1.19e+06 = 1.000   UNUSABLE

**The linear screen is faithful to within 10% of the signal only for
|c| <= 0.03**, which corresponds to a ~10% change in the rotation observable.

That is a real limitation and it matters: **a factor-of-two cluster gap is far
outside the regime where the linear screen is trustworthy.** The screen cannot be
used to hunt for the cluster excess directly. What it can do is rank-order the
STRUCTURE — which atoms matter — in the weak-response regime, and supply a
shortlist for a tier that has no such restriction.

Note also the min-eigenvalue column. The exponential form keeps K positive
definite at every |c| by construction; the linear truncation stops guaranteeing
it as |c| grows, reaching 9.5e-3 at |c| = 0.4 and 2.6e-8 at 1.5. A screen that
silently admits a nearly singular response tensor is not merely inaccurate.

## P.1 The three-tier funnel, with measured throughput

    tier 1  linear atom decomposition   ~1e9 laws/hour   valid |c| <= 0.03
            one Poisson solve per ATOM; every subset then free
    tier 2  exact exponential QUMOND    2.4e5 - 3.1e5 laws/hour   any |c|
            one Poisson solve per CANDIDATE, still no nonlinear iteration
    tier 3  exact nonlinear finite-volume solve   shortlist only

Tier 2 measured: 15.3 ms/candidate at 48^3, 15.1 ms at 64^3, 11.6 ms at 96^3, so
235,000 to 311,000 optimally-posed field laws per hour with NO linearisation
error at all. The counter-intuitive speed-up with grid size is FFT
size-factorisation: 96 = 2^5 x 3 factorises better than 64's padded 128.

So the honest headline is not "billions of field laws". It is: **billions in the
weak-response regime where the decomposition is exact, and a quarter of a million
per hour at any response strength.** Both numbers are measured, and the second is
the one that applies to the cluster problem.

---

# Run Q — the redshift path-geometry test, and an artefact the size of the effect

The proposal was to test whether redshift depends on the geometry of the path a
photon actually took: ln(1+z) = c1 D + c2 I_q + c3 I_T + c4 I_g + c5 I_q^2 +
c6 I_q I_T, with the decisive question being whether two objects at the SAME
independently measured distance have systematically different redshifts. Full
record in `work/wellnet-2026-09/void-data/REPORT.md`; 22 files with manifests.

**No fit of the redshift law was performed.** Everything below is design,
leverage and power, so nothing is unblinded.

## Q.1 The artefact is as large as the hoped-for effect

`I_q` is built from a ray truncated at D_C(z), so it knows the TRUE distance,
while the regressor D is the NOISY independent distance — 6% for Pantheon+, 23.5%
for Cosmicflows-4. A regression therefore uses I_q to repair D's noise.
Simulating the exact null (truth is ln(1+z) = c1 D_true, NO path term, real
per-source errors, 2000 draws):

    estimator                       null mean c2   significance   in units of c1
    raw I_q, VoidFinder               1.32e-4        +38.0 sigma      0.397
    raw I_q, REVOLVER                 9.00e-5        +29.6 sigma      0.270
    transverse dI_q, VoidFinder       2.21e-6         +0.46 sigma     0.0066
    transverse dI_q, REVOLVER         1.15e-5         +2.95 sigma     0.034

A naive analysis would have "detected" a path effect of 27-40% of the Hubble
term at 30-38 sigma, from data containing no path effect whatsoever. The
transverse decomposition — subtracting the footprint-averaged void path length at
that radius, so the regressor is orthogonal to distance by construction — reduces
the bias 60x for VoidFinder and 24x for REVOLVER, but does NOT zero it. **The
null must be simulated per algorithm and subtracted, never assumed zero.**

This is the third shared-denominator artefact in this programme, after the
retracted rho_p = -0.304 and the label control that killed seven variables. It is
now unambiguous that this is the dominant failure mode of the whole approach.

## Q.2 Two corrected premises

**There are two independent void algorithms, not four.** VIDE, REVOLVER and
ZOBOV in DESIVAST are three PRUNINGS OF THE SAME ZOBOV watershed zones — all
three carry an identical ZONEVOID table of 2950 zones. VIDE vs REVOLVER correlate
at r = 0.92. Only VoidFinder (sphere-based) and the V2 watershed family are
genuinely independent, and they correlate at 0.74.

**The VIDE triangulation is incomplete in the released product**: the TRIANGLE
table is missing 199 of 1258 NGC voids and 22 of 220 SGC voids, verified by
setdiff in both directions. VIDE's I_q is a lower bound. REVOLVER is complete.

## Q.3 The tidal term is only separable on watershed geometry

VIFs over the full six-term design:

    VoidFinder  2.61 / 10.18 / 12.13 / 3.06 / 16.52 / 18.23   cond 212
    REVOLVER    2.45 /  4.34 /  3.97 / 3.16 /  3.47 /  3.92   cond 25.4
    corr(I_q, I_T):  -0.754 (VoidFinder)   vs   +0.051 (REVOLVER)

The physical cause is exact: inside a uniform sphere the potential is quadratic,
so T_ij is proportional to delta_ij, and T_ij k^i k^j loses all direction
dependence — the tidal term collapses onto a density-weighted copy of I_q.
**c3 and c6 may be fitted only on watershed voids.** On sphere-based voids any
fitted value is a parameterisation artefact, not a measurement.

c1 and c2 themselves ARE separable, at VIF 1.28-2.30.

## Q.4 Leverage and power

Dynamic range at fixed distance is 0.68 to 3.46 times the mean. In three of four
DESI radial bins the 5th percentile is literally zero — many sight lines cross no
catalogued void while others cross 124-145 Mpc/h. Matched pairs with |dD| < 20
and |dI_q| > 100 Mpc/h: 122,306 (VoidFinder), 270,407 (REVOLVER), 385,104 (SDSS).

    sample            n        3 sigma detectable c2/c1   equivalent dz
    SDSS arm       20,683              2.8%                253 km/s
    DESI REVOLVER   4,389              4.3%                632 km/s
    Pantheon+ only     73             17.2%                  --

**The systematic floor is much worse than the statistical one.** Two
independently built VoidFinder catalogues on the SAME 2,141 sight lines give mean
I_q of 52.2 against 112.1, a factor 2.15, with raw r = 0.462 and — decisively —
**transverse-residual r = 0.153**, which is the quantity the test actually uses.
That inflates the realistic threshold from 2.8-4.3% to tens of percent.

The cause was traced rather than guessed: DESI's void fraction rises from 0.104
to 0.290 across r = 100-300 Mpc/h while SDSS is flat near 0.5, and the deficit is
identical inside and outside the overlap (0.229 vs 0.230), so it is not a
localised mask hole. It is **footprint size** — DESI DR1 BGS covers 0.745 sr, so
at r = 125 Mpc/h the wedge is only ~108 Mpc/h across and a 10 Mpc/h void cannot
be inscribed without touching the edge. SDSS has 2.13 sr. **Use SDSS VAST below
z = 0.11 and DESIVAST above it. Do not average them.**

## Q.5 Circularity, stated in full

It enters in four places: void positions are mapped from (RA, Dec, z) through
D_C(z; Omega_m = 0.315), so the catalogue is a redshift-space product in
Cartesian clothing; the volume-limited sample definition uses a
cosmology-dependent luminosity distance, so WHICH GALAXIES EXIST depends on it;
voids are found in redshift space and are RSD-stretched along the line of sight,
uncorrected; and the source endpoints use the same law.

Relative to fiducial at z = 0.24 the radial scale moves by +6.2% for linear
cz/H0, -4.8% for Milne and -9.7% for Einstein-de Sitter — 1.5 to 4 void radii,
larger than a void. But the shift is SHARED by voids and sources, both placed by
redshift, so ordering along the ray survives and only the differential stretch of
5-10% matters. The half that does not cancel is the endpoint, measured directly
by recomputing every I_q with the ray truncated at the source's own independent
distance times a single fitted global h = 0.743: median |dI_q| = 6.20 Mpc/h =
0.177 sd(dI_q), an 18% perturbation on the leverage variable.

**A genuine no-expansion analysis cannot reuse this catalogue as-is.** It would
have to rerun the void finders under its own distance law, which also changes the
sample definition. Reuse costs ~18% on the leverage variable: tolerable for a
feasibility and power study, not for a claimed detection.

## Q.6 The Pantheon+ covariance trap, quantified

Confirmed in the official release, on a byte-identical copy of upstream:
sqrt(diag(STAT+SYS)) is not MU_SH0ES_ERR_DIAG. Median ratio 0.713; 1700 of 1701
rows differ by more than 1%; 0.564 for z < 0.01 and **0.449 for the 77 Cepheid
calibrators**, i.e. worst exactly where low-redshift work needs it. For
non-calibrators the ratio is a tight constant 1.4013. The quadrature excess is a
median 0.148 mag, about 5x the tabulated peculiar-velocity term, so it is NOT
the VPEC contribution. Documented, not diagnosed.

## Q.7 Validation that was run

Fiducial D_C reproduces DESIVAST's own header to 3.2e-8 Mpc/h; RA/Dec recovered
from catalogue Cartesian coordinates to 5.7e-14 deg. The ray tracer was validated
three ways: rays through VoidFinder maximal centres land inside 200/200; the V2
triangulated surfaces are watertight with odd-parity count 0 on all 5,631 sight
lines; and an independently reconstructed density field puts VoidFinder centres
at delta median -0.66 with 99.6% negative, against -0.22 and 62.6% for random
in-survey cells. Interior counts reproduce the publication (VIDE 297 against 295,
VoidFinder EDGE == 0 giving 1489, matching exactly). All 8 DESIVAST FITS files
verified byte-exact against DESI's published sha256sum manifest. The
monotone-invariance check gives sd(dI_q) = 35.00, 34.99, 34.55, 32.70, 25.96
Mpc/h across hole-radius cuts, range/median 0.262, so the statistic does respond
to its own parameter.

---

# Run R — the potential-depth ladder, and the identity that kills it

Full record in `work/wellnet-2026-09/potential-depth/REPORT.md`. 4,150 rows, 317
systems, six rungs from SPARC field galaxies through SDSS small groups, Sun+2009
and Lovisari+2015 X-ray groups, to Gonzalez+2013 and X-COP clusters.

## R.1 The ladder delivered exactly the leverage asked for

    median within-g_bar-bin spread of log|Phi_b|
       SPARC alone (Run N)                      0.309 dex
       full ladder                              0.766 dex     2.5x gain
    range inside one 0.25-dex bin holding all six rungs        3.63 dex
    matched pairs, |dlog g_bar| <= 0.1 and dlog|Phi_b| >= 1    12,224
    q detectable at 3 sigma, statistics only                   0.115
    q required to explain the 2.43x cluster excess             0.371

Statistically the experiment is easy. The fitted coefficient is
q = +0.337 +- 0.028, 9.3 sigma from its own shared-denominator null.

## R.2 And it still cannot decide the question — because of an identity

With the same boundary condition Run N used, define S(r) = |Phi_b(r)|/(g_bar(r) r).
Then for spherical M_b(<r),

    S(r) >= 1 exactly,  because M_b(<r') >= M_b(<r) for r' >= r, so
    |Phi_b(r)| = Int_r^inf G M_b(<r')/r'^2 dr' >= G M_b(<r)/r,
    with equality iff there is no baryonic mass outside r.

Therefore

    log|Phi_b| = log g_bar + log r + log S

and |Phi_b| carries NO information beyond (g_bar, r) except the bounded shape
factor S. Across the whole ladder sd(log S) = 0.387 dex. This generalises Run N
from SPARC to every system class at once, and explains structurally why no amount
of ladder-building can buy a new direction:

    R^2 of log|Phi_b| on a quadratic in (log g_bar, log r)
       SPARC alone      0.9322     residual 0.218 dex
       full ladder      0.9147     residual 0.247 dex
       X-ray only       0.9665     residual 0.155 dex

Four decades of system mass and three of radius bought nothing. The decisive
collinearity is with RADIUS: partial corr(log|Phi_b|, log r | log g_bar) =
**+0.9217**. At fixed acceleration, potential depth IS radius.

A useful corollary the lane established: every single-radius row is a strict
LOWER BOUND on |Phi_b|, not an estimate, and is labelled so. 316 SPARC rows have
S < 1 (min 0.877) — all disk rows, where g_bar is the razor-thin-disk field
rather than GM(<r)/r^2 and the signed V_gas from central HI holes makes V_b^2 r/G
non-monotone. No spherical row violates the theorem.

## R.3 86% of the leverage is the class label

    sd of log|Phi_b| after removing g_bar                      0.768 dex
    sd after removing g_bar AND the bare class label           0.286 dex

which is LESS than SPARC alone had. Within-class leverage, one instrument, one
pipeline, one systematic:

    field galaxies 0.356   small groups 0.098   poor groups 0.160
    rich groups    0.169   low-mass cl. 0.158   massive cl.  0.114

**The largest within-class leverage anywhere is SPARC's own.** The group regime
was prioritised as the place the variables should decouple, and it is where they
decouple LEAST — X-ray groups are observed at two overdensity radii a fixed ratio
apart, so r barely moves at fixed g_bar within the rung.

The label control is decisive. Replacing log|Phi_b| by a bare class index 1..6
reproduces 99% of the fit quality (R^2 0.8663 against 0.8748) and 91% of the
partial correlation (+0.4778 against +0.5270).

## R.4 A physics-free step beats the hypothesis

252 systems, every model carrying a free quadratic in log g_bar:

    model                          k   rms      dBIC   frozen transfer
    M0 RAR only                    3   0.2212  +143.9     0.2917
    M1 + beta log|Phi_b|           4   0.1703   +17.6     0.1066
    M2 + gamma log r               4   0.1735   +26.9     0.1252
    M3 + step: "is it a galaxy?"   4   0.1645     0.0     0.0954
    M4 + full class dummies        8   0.1638   +20.0       --

A one-parameter step knowing only whether the object is a galaxy beats the
potential-depth model by dBIC = 17.6 at equal parameter count AND wins the frozen
transfer test onto held-out clusters. beta was fitted on rungs 1-4, frozen, and
evaluated once on rungs 5-6.

The rungs are also **not monotone in |Phi_b|**: the boost jumps ~2x from galaxies
to the first group rung, FALLS through the group regime (rich groups 1.723,
low-mass clusters 1.721), then rises at the cluster end (2.381). A monotone
A(|Phi_b|) cannot produce that shape.

## R.5 The systematic budget forges most of the effect

Any class-level systematic in log nu forges the same signal, because the contrast
is carried by the class boundary:

    HSE bias, X-ray rungs vs galaxies                0.080 dex
    SPARC Upsilon* = 0.5 vs dynamical                0.150
    group stellar masses (relation extrapolated)     0.080
    gas clumping (measured P_X/P_SZ, median 6%)      0.030
    non-thermal pressure support                     0.060
    quadrature sum                                   0.199

    spurious q = 2 x 0.199 / 2.08 = 0.192   against q_required = 0.371

Ratio 1.93. **The experiment is systematics-limited at q ~ 0.19, not
statistics-limited at q ~ 0.12.**

## R.6 What is recorded in the hypothesis's favour

beta fitted on galaxies and groups alone is +0.1719; on everything, +0.1687. The
group rungs extrapolate to the clusters without adjustment, and the implied
q = 0.34 is within 10% of the q = 0.371 the cluster excess requires. **The
potential-depth law is not falsified.** It is not distinguishable from a step at
the dataset boundary, which is a different and weaker statement.

The cleanest constraint the programme holds remains Run N's within-galaxy bound:
partial correlation +0.018, CI [-0.118, +0.145], corresponding to |q| <= 0.29
with no class boundary anywhere in it — already consistent with zero.

## R.7 What would decide it

|Phi_b| varying by >= 1 dex at fixed g_bar WITHIN one class, one instrument, one
pipeline. Concretely: resolved X-ray profiles for groups rather than two
overdensity radii. Bahar+2022 eFEDS (J/A+A/661/A7) ships Vikhlinin-form n_e(r)
for 542 systems and is already on disk; its M_tot is a scaling-relation product
so it needs an independent g_obs, and pairing it with X-GAP or CLoGS hydrostatic
profiles is the obvious next lane. Alternatively, pin Upsilon* or the HSE bias to
0.03 dex and halve q_sys.

---

# Run S — the resolved cluster data, and a finding that reshapes the experiment

309 files, 158 manifests, 151 checksummed raw envelopes, 0 validation problems.
Record in `work/wellnet-2026-09/cluster-data/`.

## S.1 The binding constraint is weak lensing, and it is severe

**A public raw shear catalogue exists for exactly ONE of the seven target
clusters: Abell 370.** For the other six there is no per-source catalogue and —
not anticipated — **no public shear PROFILE table either.** The binned profiles
for A2744 (Medezinski 2016), AS1063 (Gruen 2013) and the CLASH clusters (Umetsu
2014/2016) exist only as figures. What those papers tabulate is NFW masses, which
presuppose a dark-matter halo and are therefore forbidden as observations by
section 2 of the brief. Nothing was substituted; every avenue is recorded in
`weaklensing/WEAK_LENSING_AVAILABILITY_AUDIT.json`.

A sharper corollary: **the one cluster with weak lensing is one of the two
WITHOUT resolved member structural parameters.** A370 has 18,556 shear
measurements to 6.2 Mpc but only SExtractor q and theta for its members — no R_e,
no Sersic n. A2744, MACS J0416, MACS J1149 and AS1063 have full seven-band Sersic
fits and no weak lensing. **No target has both.**

Quantitatively, for A370: strong lensing spans 12-332 kpc, members 21-2857 kpc,
weak lensing 282-6208 kpc. Without weak lensing the other clusters stop at the
X-ray gas edge, 0.71-1.07 R500. Since this programme's surviving signal is a
cluster-only excess organised by r/R500, weak lensing is the only probe reaching
the regime of interest, and one cluster is not a sample.

ICL was ruled out numerically as the rival binding constraint: ICL is 5-25% of
stellar light and stars are ~13% of baryons, so ICL profile ignorance is a 1-4%
error on rho_b. It does bind for any mass-follows-light family.

**The six-cluster BUFFALO release is announced (arXiv:2602.06904, "available upon
acceptance") but is not on the HLSP as of today. That gap has an expiry date and
should be re-checked before the test is finalised.**

## S.2 What was acquired that changes the design

  * **SN Refsdal time delays** (MACS J1149): SX-S1 = 376.02 d, 16-84th percentile
    370.50-381.65, i.e. **1.4% precision from pure light-curve fitting with no
    mass model at all**. The only cluster in the sample with delays, and the
    cleanest single observable in the whole set.
  * **213 measured internal stellar velocity dispersions** (Granata 2026,
    MUSE/pPXF) joinable to the Sersic fits — member mass components can be
    constrained by measured kinematics rather than a sigma-L scaling.
  * **Bolocam SZ for all six HFF clusters** (IRSA, not in the brief), centres
    0.00 arcmin from archive positions, decrements -522 to -1255 microK. This is
    what keeps MACS J0416 in the sample, since it has no X-ray profile at all
    (confirmed absent across five independent leads).
  * **A370 in ACCEPT**, 32 bins to 806 kpc, found by noticing that ACCEPT
    zero-pads Abell numbers below 1000.

## S.3 Line-of-sight depth is not merely unmeasured, it is unrecoverable

x and y are measured to 0.15-0.64 kpc. z_a is not: 1 Mpc of depth moves cz by
only 5-9% of sigma_v, reading sigma_v as Hubble flow implies a depth 3.2-4.7x the
cluster diameter, and Finger-of-God makes the inferred depth ANTI-correlate with
true 3-D radius. Downstream must sample and marginalise, which is what the
Bayesian formulation of section 1 requires anyway.

## S.4 Traps that fired, including a new VizieR failure mode

**For a bad `-source=`, VizieR can return HTTP 200 serving an UNRELATED REAL
CATALOGUE** (J/MNRAS/430/1125). This is worse than the generic-page trap already
recorded, because the response is a valid table with plausible columns. The only
reliable detector is echoing back the exact identifier.

LaTeX split-tables fired twice for real (DeMaio: 7 of 23 systems lost including
MACS J1149; Limousin: 71% of the catalogue). A novel variant: a `%` glued to a
data row ate 3 of 213 rows. VizieR fuses Braglia 2009's A2744 and A2537 tables
into one, so an unfiltered pull injects galaxies 530 Mpc away. The CLASH-VLT
AS1063 file has one ID containing a space, so whitespace parsing silently loses
exactly one row. **ACCEPT stores radial bins in DESCENDING order in every file.**

Every null in the lane was audited against the cone-search failure mode flagged
earlier; none rests on a cone search. No mass, convergence, magnification or
deflection map was downloaded at all — they were located and left alone.

---

# Run T — the control harness, and a certified false-positive rate

`work/wellnet-2026-09/controls/`, 11/11 worked examples pass in 288 s.

## T.1 The number that licenses any future tensor claim

600 scalar universes (200 each Newtonian, MOND, GR+NFW), 199 nulls each,
counting how often "a tensor effect is detected in scalar data":

    naive    (tensor atoms improved the blind fit at all)      28.7%
    threshold (> 1% blind improvement)                          0.0%
    calibrated (p <= 0.05 against a matched permutation null)   5.2% +- 0.9%

The calibrated p-values are uniform (KS D = 0.043, p = 0.205), so the test is
correctly SIZED rather than merely quiet, and the median tensor "gain" on
physics-free data is -0.06%. All five injected families were recovered; tensor
was detected only in tensor data (p = 0.005) and nonlocal only in nonlocal data
(p = 0.010).

**Stated plainly as a negative: the NONLOCAL detector is not certified.** Its
calibrated false-positive rate is 11.7% (n = 120), 2.3 sigma above nominal — its
covariate-permutation null is not exact. Any nonlocal claim needs p <= ~0.02
until that null is fixed. The tensor detector is certified; the nonlocal one is
not.

## T.2 A permutation null is not automatically the conservative choice

Three nulls for the SAME statistic on the LoCuSS case:

    permute-the-carrier null   mean -0.004   p = 0.023   "significant"
    structural null            mean -0.113   p = 0.525   nothing

Permuting kT destroys the real mass-temperature relation, which is the very
mechanism that biases the estimator. **Picking the wrong null manufactures the
detection.** The structural null — fit the errors-in-variables model, set the
coefficient of interest to zero, keep everything else — is the correct one.

## T.3 Nested families defeat a bare argmin

The tensor and nonlocal families NEST MOND, so the richer one can only tie or
win. A bare argmin over blind RMS identifies 4 of 5 injected families; a
one-standard-error parsimony rule gets 5 of 5.

## T.4 The retraction reproduced from raw tables

Rebuilt independently from the raw Mulroy tables: rho_p = -0.3042 (record
-0.304), error correlation +0.957 (+0.96), naive slope -0.1550 (-0.155), null
expectation -0.1132 (-0.12), p = 0.525 (0.563), EIV slope -0.1634 (-0.166). The
EIV estimator is validated unbiased to 0.021 across beta in [-0.6, +0.6], where
the naive bias reaches 0.32 and is **+0.19 at beta = 0**.

## T.5 Other catches

The frozen-coefficient guard: -0.73% frozen against **+4.07%** re-solved on
blind, a 4.80 percentage-point swing that flips a refutation into a discovery
(Run J's own instance was 5.90 pp). Seven attack routes blocked.

Control 4 caught a **cubic-lattice l = 4 floor** of 0.0887 against the source's
own 0.0973 — an artefact of the grid that would have read as residual anisotropy.
Replaced with a lattice-free shell statistic.

Control 6 found that **2 of 5 statistics on the real LoCuSS data are bit-identical
across three decades of kappa**, spread exactly 0.000000, and now raises rather
than warns.

Controls 1-4 are demonstrated on mocks, because no real resolved 3-D member map
exists yet — which Run S explains.

---

# Run U — the resolved-versus-scrambled test is not the test it looks like

`work/wellnet-2026-09/resolved_power.py`. Synthetic cluster, 200 members plus
smooth gas, angular scrambles preserving every clustercentric radius and every
member mass exactly.

## U.1 Two of my own errors, both instructive

**The first power run reported zero power everywhere.** With 12 scrambles the
minimum attainable permutation p-value is 1/13 = 0.077, above the alpha = 0.05
threshold, so the power was zero BY CONSTRUCTION rather than by physics. The
module now refuses to run unless 1/(1 + N_scramble) <= alpha.

**The second used the wrong statistic.** An RMS of the difference between the
true and smoothed deflection maps is blind to WHERE the lumps are, and it
reported a discriminating margin of only 3.4% of the signal. Replaced with a
chi-squared map likelihood, which penalises a scrambled configuration for putting
its lumps in the wrong places. The difference is enormous:

    sT      RMS statistic, power at S/N 100      likelihood, power at S/N 3
    0.30              0.12                              0.76
    1.00              0.43                              1.00

## U.2 The finding that matters: p_geometry alone is not a test of the mechanism

    sT = 0.00 (NO well-network tensor at all, plain QUMOND on a lumpy source)
       map S/N   3   power 0.29    median margin 0.95 scramble-sd
       map S/N  10   power 0.78    median margin 2.21
       map S/N  30   power 1.00    median margin 2.93
       map S/N 100   power 1.00    median margin 3.07

With no anisotropy whatsoever, the true configuration beats 95% of angular
scrambles essentially always. That is not a bug — it is correct, and it means
the proposed primary statistic answers the wrong question. p_geometry detects
"the member positions matter for the lensing map", which is TRIVIALLY true in
plain QUMOND: lumps in different places make different maps.

**Delta_resolved > 0 and p_geometry small are therefore necessary but NOT
sufficient.** They are satisfied by ordinary MOND on a resolved source. The
discriminating test has to be a model comparison AT FIXED TRUE CONFIGURATION:
does K != I improve on K = I? The scramble ensemble calibrates how much of any
improvement is available from geometry alone, but the comparison that carries the
hypothesis is between LAWS, not between configurations.

This is the same failure mode the control lane independently found in T.3 —
nested families, where the richer model can only tie or win, so the naive
selection statistic answers a question nobody asked.

## U.3 What the power table does say

Once the statistic is right, and given the true configuration, the test is not
noise-limited: power reaches 1.00 at map signal-to-noise of 3 for sT >= 1.0 and
at 10 for sT >= 0.1. The binding constraint on the resolved cluster tournament is
therefore NOT the lensing noise. It is data availability — Run S found raw shear
for one cluster of seven, and that one lacks the member structural parameters the
source model needs.

---

# Run V — environment and two-direction data, and two structural obstructions

Full record in `work/wellnet-2026-09/env-data/{REPORT,MATCHED_PAIRS,TWO_DIRECTION_INVENTORY}.md`.
330 manifests, 330 targets verified, zero mismatches.

## V.1 Matched field-versus-cluster pairs

10,071 unique MaNGA DR17 galaxies cross-matched against Tempel+2014/2017 SDSS
group catalogues and the MCXC X-ray cluster catalogue, plus the same build on
SAMI DR3 (3,068 galaxies). Resolved DAP MAPS kinematics for all 902 galaxies
involved, 5.3 GB. Seven MaNGA tiers and two SAMI tiers, matched by optimal
one-to-one assignment inside a hard tolerance box declared in advance.

    tier                                  pairs   3 sigma detectable offset
    MaNGA B1_primary (late, sigma_v>=400,
                      inside R_vir)          23        0.049 dex
    MaNGA C2_xray_disk (X-ray host)         218        0.016
    MaNGA B4_disk_wide                      281        0.014
    SAMI S1_latetype                        108        0.022
    SAMI S2_diskbearing                     364        0.012

MaNGA alone is not big enough — its clean tier detects only ~12% velocity shifts.
SAMI fixes it: 5.7x more clean pairs, at higher host dispersion (690 against
605 km/s) and deeper in (0.46 R200 against 0.77 R_vir), reaching ~5% in velocity.
Median |g_ext|/a0 = 0.17, so the sample sits where an external-field effect would
actually live.

## V.2 The environment under test destroyed one of the control variables

**The gas-matched tier returns ZERO pairs, and the reason is physical rather than
a sampling failure.** HI is stripped in clusters: 17 of 494 cluster galaxies are
detections against 572 of 1603 in the field. Matching on gas fraction therefore
cannot be done at all in the regime the test is about.

This is a nice example of a control variable being an outcome of the treatment.
Any matched-pair design here must either drop f_gas from the matching set and
declare it, or restrict to the outskirts where stripping has not yet acted — and
the second option removes exactly the deep-potential galaxies the test wants.

## V.3 Two of the five matching variables are one variable

Sigma_b and g_bar(2.2 R_d) correlate at 0.996 in MaNGA and **exactly 1** in SAMI.
For an exponential disk the Freeman formula makes g_bar equal to Sigma_b times a
constant, so this is an identity, not a coincidence. The five-variable matching
set has about **three** effective independent directions.

That is the same phenomenon as the rank-2 theorem and the
log|Phi_b| = log g_bar + log r + log S identity of Run R: a variable list that
looks rich collapses because the variables are definitionally related. It is now
worth checking the rank of any proposed matching or predictor set BEFORE building
a sample around it.

## V.4 Two silent bugs, both caught by validating against a known answer

**(i) The X-ray flag matched group CENTRES.** Coma's catalogued group centre sits
12.4 arcmin from its X-ray peak, so the flag failed precisely on the richest
systems — the ones the test most needs. Re-matching on the galaxy rather than the
group centre raised the flagged count from 222 to 978.

**(ii) An h convention mismatch worth 0.31 dex.** The 103 galaxies observed by
both MaNGA and SAMI disagreed by -0.308 dex in stellar mass. Cause: NSA
quantities assume h = 1 while the lane ran H0 = 70. Every baryonic quantity was
0.31 dex low. After correction the disagreement is +0.002 dex, and the median
V_obs/V_bar is 1.40 against an RAR prediction of 1.40 — which is the check that
would have caught it immediately had it been run first.

Both were found by comparing against an answer known in advance. Neither would
have raised an error.

## V.5 The most powerful configuration is the one with no numbers

**No polar-ring galaxy anywhere has a tabulated rotation curve in both planes.**
Nine have rotation measured independently in both planes, all as figures only.
The configuration named as the single most powerful test of a tensor law — one
baryonic system supplying tracers in two nearly perpendicular planes — is not
available in machine-readable form for even one object.

The category listed last turns out to be the largest by two orders of magnitude:

    near-orthogonal MaNGA systems, cubes on disk        105
    counter-rotating MaNGA systems, cubes on disk        38
    near-orthogonal / counter-rotating SAMI (PAs)   447 / 261
    Milky Way streams with measured 3-D tracks           60  (30 with 6-D,
                                                              16 and 8 near-polar)
    warped disks with i(R), PA(R) and V(R) tabulated     15
    SAGA hosts with orientation but NO in-plane
      rotation curve                                    101  (half-measurement)

## V.6 A resolved vertical-dispersion upgrade, with honest limits

240 near-face-on MaNGA disks now have resolved sigma_LOS(R), 4-9 points each,
median error 1.0 km/s — against DiskMass's 30-galaxy exponential FIT, which Run L
showed is blind to the amplitude it was supposed to measure. Limits stated:
it is sigma_LOS, not sigma_z; no scale height is produced; and only 45 galaxies
sit entirely above MaNGA's 70 km/s instrumental floor.

The NGC 628 profile previously reported unrecoverable was recovered from a
rotated PDF table and validated by refitting the paper's own published
parameters: 74.4 km/s and 92.7 arcsec against a stated 73.6 +- 9.8 and
92.7 +- 13.1.

## V.7 The obstruction behind the stuck A_dyn measurement

**No external galaxy has a MEASURED scale height alongside a resolved
sigma_z(R).** h_z requires an edge-on view; sigma_z requires a face-on one. The
two cannot be had for the same object.

That is geometric and inescapable, and it explains Run L's result rather than
merely accompanying it. DiskMass's h_z is inferred from h_R through the
Bershady+2010b relation, so the two columns are correlated by construction — not
because that survey was careless, but because the alternative does not exist.
Run L found the systematic floor on B_z to be 8.4x the statistical part, with the
h_z zero point contributing 0.072 dex of the 0.192 dex total; V.7 says that term
cannot be reduced by observing more external galaxies.

The routes that remain are therefore: the Milky Way, where h_z and sigma_z are
both measurable for the same tracer population; statistical deprojection over a
population with a modelled inclination distribution; or a system where the
vertical structure is constrained by something other than a scale height.

---

# Run W — polar rings: a bounded negative, and the object that nearly fooled me

Artifacts in `work/wellnet-2026-09/env-data/raw/polar-rings/` — 82 manifests,
435 MB, 59 arXiv e-print sources and 8 VizieR tables covering every polar-ring
kinematics, HI, CO, photometry and catalogue paper the searches surfaced from
1993 to 2026.

## W.1 The negative, and its bound

**Zero polar-ring galaxies have a numerically tabulated rotation curve in both
planes.** Nine systems genuinely have rotation measured independently in both
planes — NGC 4650A, NGC 4262, SPRC-7, SPRC-260, NGC 4632, NGC 6156, A0136-0801,
UGC 7576, UGC 9796 — and in every case at least one of the two curves exists only
as a published figure.

The negative is bounded rather than absolute: A&A returns HTTP 403 to
programmatic fetches and two key classics are pre-arXiv. It is a thorough
search, not a proof.

Tiering: 9 objects with rotation in both planes and at least one resolved in
radius; 9 more with a two-plane detection but one plane lacking usable V(r); 4
with one plane only. The confirmed census is 40 kinematically verified polar
rings (Yu et al. 2026), and 21 have a published dark-halo axis ratio.

Only two tabulated curves exist anywhere, both single-plane: NGC 4650A's host
disk (Sackett+1994, 23 points, stellar V and sigma) and NGC 2685 (Jozsa+2009,
21 tilted rings).

**Baryons for g_bar are complete in both planes for NGC 4650A only** — and there
is a trap in them. The two published decompositions are NOT independent
(Iodice+2015 reuse Combes & Arnaboldi's radii verbatim) and the polar STELLAR
disk mass differs by 58% between them, 9.5 against 15e9 Msun. That component
dominates polar-plane g_bar, so it must be carried as an explicit systematic
rather than adopted from whichever paper is cited.

Four premise corrections worth recording: Egorov & Moiseev 2019 is a
metallicity/ionization paper, not a kinematic compilation; "Arnaboldi 1997 A&A
325, 145" does not exist, the NGC 4650A ATCA HI paper being AJ 113, 585; "Iodice
2002 A&A 391, 103" is the NIR photometry paper, not HI; and Whitmore 1990, the
Polar Ring Catalogue itself, is not in VizieR.

## W.2 NGC 2685, and the check that stopped me running the wrong test

Jozsa et al. 2009 tabulate for each of 21 tilted rings not only V_rot and radius
but the ring's inclination, position angle and full **3-D spin normal**
(n_W, n_N, n_LOS) with errors. The position angle swings 126 degrees between 0.9
and 31 kpc. That looked like exactly the section 9 measurement — one baryonic
system, tracers in more than one plane, as numbers rather than a figure — and
better than a classical polar ring, because Jozsa et al. show NGC 2685 is a
single warped COHERENT disk rather than two dynamically independent components
whose relative mass would be a free parameter.

Before computing anything I checked whether ring orientation is separable from
radius, because this programme has now found five exact identities that collapsed
a variable list. Taking the angle between each ring's spin normal and the
innermost ring's:

    r [kpc]    0.88   1.77   3.54   7.08   10.32   14.75   20.64   30.96
    angle       0.0   36.3   37.4   60.3    88.4    84.4    84.7    90.0

    Spearman(r, angle) = +0.9038      Pearson = +0.7537
    sign changes in d(angle)/d(r): 6 of 18
    angle range 0 - 90 deg

**The warp opens monotonically.** Orientation on this object is radius wearing a
different label, and a direction test run on it would have measured the radial
dependence of the boost and reported it as an orientation dependence. The module
refuses to proceed, and records the refusal.

There is a second, weaker structure worth noting: beyond 8 kpc the angle is
nearly constant at 78-90 degrees while radius spans a factor of four. That is
leverage in the opposite direction — radius at fixed orientation — which
constrains a radial term but says nothing about a directional one.

## W.3 What the direction test actually needs

Not a better single object. **A POOLED SAMPLE of warped disks**, so that at a
given radius different galaxies present different orientations and the
degeneracy breaks between objects rather than within one. The environment lane
found 15 warped disks with i(R), PA(R) and V(R) all tabulated; that is the sample
to build the test on, and the pooling is what supplies the separation NGC 2685
cannot.

A second requirement that has to be stated before anyone attempts it: g_obs per
ring is immediate (V_rot^2/r, spanning g_obs/a0 = 0.197 to 7.297 here), but
**g_bar is not**. The source is a warped disk, so g_bar depends on direction as
well as radius, the axisymmetric solver does not apply, and obtaining it means a
3-D solve on the tabulated ring geometry. That is the reason this object has not
already been used this way, and it is work rather than an obstacle — the
normalisation is available (M_HI = 1.7e9 Msun, L_I = 15.2e9 Lsun, D = 15.2 Mpc).

## W.4 Trap that fired, and one that generalises

The split-table trap fired again: Smirnova & Moiseev 2013's geometry table spans
two `table*` environments and a naive parse returns 43 of 78 rows. Both were
parsed and the total asserted against the paper's stated 78. Sackett+1994's table
arrived from `pdftotext` as three separate column runs with the layout renderer
off by one, and was re-zipped under length, count and monotonicity assertions.

Two defects in published tables were **recorded rather than repaired**:
Khoperskov+2014 Table 2 has a dropped column label making labels and units
inconsistent, and Jozsa+2009 Table 5's caption enumerates 23 items for a
21-column table.

The generalising trap: **`-out.all` with an EMPTY value silently returns
VizieR's default column subset.** For van Driel 2002 that is 12 columns instead
of 34, losing W20, W50, M_HI, L_B and D25 — a valid-looking table with the
science columns missing, passing every structural check. `-out.all=1` is
required. Assert the COLUMN list against the ReadMe, not only the row count.

---

# Run X — streams, satellites and counter-rotators: the Milky Way is the answer

Artifacts in `work/wellnet-2026-09/env-data/raw/streams-satellites/`. 258 files,
253 manifests, 595.7 MB, zero integrity failures; every SHA-256 and byte size
re-verified, 145 row counts independently recounted, every file carrying a
`measurement_or_model` label.

## X.1 The pairing, stated plainly

**The Milky Way is the only system where both legs of the two-direction test are
fully measured and free of any assumed halo** — 69 usable 3-D stream tracks, 33
of them full 6-D, against a 38-point rotation curve (Eilers 2019). M31 is the
only external galaxy with both legs. NGC 4651 is the cleanest control, with 46
tracers, 30 disc and 15 halo, in-plane and out-of-plane **from the same Keck
dataset**. Everything else is missing a leg.

That converges with Run V's independent finding that no external galaxy can have
a measured scale height alongside a resolved sigma_z(R), because h_z needs
edge-on and sigma_z needs face-on. Two different obstructions, two different
lanes, the same answer: **the vertical-gravity channel has to be done in the
Milky Way.**

## X.2 What the Milky Way actually offers

From galstreams v1.2.1 (Mateu 2023; there is **no Zenodo DOI** — the Zenodo API
returns `total: 0`, distribution is GitHub/PyPI plus the MNRAS DOI), 217 tracks
over 147 distinct streams:

    sky track      192 empirical          25 great-circle ASSUMED
    distance        69 measured           68 placeholder, 6 interpolated,
                                           2 mean, 72 absent
    proper motion  165 measured            6 constant, 46 absent
    radial velocity 98 measured           15 UNPHYSICAL, 104 absent

    usable_3d = 69 tracks / 60 streams
    usable_6d = 33 tracks / 30 streams

Of the 69 usable 3-D tracks: **16 within 10 degrees of polar**, 26 reaching
|z| >= 10 kpc, 11 reaching 20 kpc, and **14 extending beyond 25 kpc — past the
outer edge of the rotation curve.** That last number is the leverage: those
tracks constrain the field in a regime the in-plane rotation curve does not
reach at all.

## X.3 Four silent defects in galstreams, affecting 102 of 217 tracks

The library's own flags advertise data that is not there. None of these raises
an error.

  * **68 `ibata2024` tracks claim a distance track while the column is
    identically 1.000 kpc** — GD-1 among them. And float round-trip noise
    (`0.9999999999999946`) means an exact `== 1.0` test finds NOTHING. A
    tolerance test is required.
  * **15 tracks advertise `InfoFlags=1111`, i.e. full 6-D, with unphysical
    velocities.** `Hydrus.ibata2024` reaches **9,561,412 km/s, 32 times the
    speed of light**; `NGC1261b` reaches -32,929,072 km/s.
  * `Pal5.pricewhelan2019` Vrad is the sentinel `999.0`, and 16 tracks have the
    Vrad flag clear but a populated junk column.
  * 3 summary files have no track file at all.

The rule that resolves it, adopted after an earlier pass got it backwards:
**the flag governs; data may only downgrade a track, never promote it.**

A physical-plausibility gate — reject anything exceeding a stated fraction of c —
would have caught the worst of these instantly and is cheap. It is now worth
running on every kinematic ingest in this programme.

## X.4 External galaxies: a clean negative

Across 146 hosts, **{SPARC} intersect {stream kinematics} is EMPTY**. The 5 SPARC
hosts with streams are all imaging-only; the 5 hosts with stream kinematics are
none of them in SPARC. About 150 external streams have no kinematics whatsoever.

**NGC 5907 is exactly half the prize** — it is in SPARC, has zero stream
kinematics, and its stream shape is directly contradicted in the literature
between Martinez-Delgado+2008 (multiple loops) and Dragonfly (a single stream).

M31 is the exception and is complete: 61 giant-stellar-stream stars, 115 stream
A-D stars, 3126 planetary nebulae, against a 100-ring HI curve, with streams A-D
at median dPA = 83.2 degrees from the disc major axis — essentially on the minor
axis, which is the geometry the test wants.

## X.5 Satellites, and the gap that blocks them

SAGA DR3 (not in VizieR; from the official release): 101 hosts, 378 satellites,
0 join orphans, **101 of 101 with axis ratio and position angle — and no in-plane
rotation for any host.** That is the blocking gap, and it is the same
half-measurement Run V flagged. ELVES is better proportioned: 31 hosts, 444
satellites, 24 with orientation and 27 with rotation. The Milky Way has 68 dwarfs
with full 6-D.

Corrected premise: Battaglia+2022 is not a systemic-proper-motion catalogue. It
holds 645,720 per-star membership probabilities.

## X.6 A seventh variable-list collapse, in the counter-rotator sample

Raimundo+2023 (SAMI) gives 1310 galaxies with both components measured, 47 polar
at 60-120 degrees, 19 within 10 degrees of 90. Moiseev 2012 gives 47 rows of
which 22 are polar by **deprojected** di rather than projected dPA, which is
geometrically the stronger criterion.

But: **ATLAS-3D's 2-sigma, kinematically-decoupled-core and counter-rotating-core
systems are about 180 degrees ANTI-PARALLEL — the SAME PLANE.** They supply NO
new direction at all. Only the ~90 degree polar sets do. A sample built on
"counter-rotating" without the angle cut would have been a sample of one
direction wearing the label of two.

That is the seventh time in this programme a variable or sample list has
collapsed on inspection, after the rank-2 theorem, the potential-depth identity,
Freeman's formula, the QUMOND projector, spherical blindness, and NGC 2685's
monotone warp.

Barrera-Ballesteros+2014 (80 systems, zero polar) is retained as a clean aligned
control.

## X.7 A new VizieR failure mode, worse than the previous ones

**A literal `+` in `J/A+A/...` is decoded as a space**, whereupon VizieR runs a
KEYWORD SEARCH and returns 468 KB of unrelated catalogues at HTTP 200 **with no
error line at all**. The generic-fallback trap and the unrelated-real-catalogue
trap both at least return one table; this returns a plausible bulk response to a
query that was never issued. The shared validator was hardened with a guard and
all six of the lane's own fetches re-verified against it.

The split-table trap fired four more times, caught each time by row-count
assertions — galstreams' own 63 + 63 = 126 against the paper's `\Ntracks`.

---

# Run Y — the nonlocal path kernel, and a correction to my own brief

Full record in `work/wellnet-2026-09/nonlocal/REPORT.md`. The family tested:

    Phi(x) = -G Int [ rho_b(x') / |x - x'| ] F[ qbar(x,x'), Tbar(x,x') ] d^3x'
    qbar(x,x') = Int_0^1 q[(1-s)x + s x'] ds

with q defined two ways (a clipped density contrast, and the screened response
(1 - L^2 laplacian)^-1 S), and F drawn from four families.

## Y.1 The headline is a theorem, not a fit

**Asymptotically flat rotation curves are impossible for the whole family.**
Outside a source Phi = -G M F / r is exact, so

    v_c^2 = (G M F / r) ( 1 - dlnF/dlnr )

and since qbar lies in [0,1), F is bounded, so r v_c^2 -> G M sup F. Verified
numerically at 30 Mpc: 3.9998 against sup F = 4.000, outer slope -0.5000.

## Y.2 But my stated reason was wrong, and the correction matters

The brief I wrote argued: "the surrounding q is roughly constant on galaxy
scales, so F is roughly constant, so Phi is roughly a rescaled Newtonian
potential, giving v^2 ~ 1/r." **That argument drops the -G M F' term.** The
measured dlnF/dlnr reaches **0.899** where the argument assumes 0.

Over the measured range the family genuinely DOES flatten rotation curves. On a
six-galaxy ladder the RMS outer slope falls from Newton's 0.190 to **0.056**, and
forward-modelled on **71 SPARC train galaxies with one global parameter set** it
reproduces the required modification factor to **0.156 dex rms against Newton's
0.646** — with a baryon-model control at 0.013 dex. (An earlier exponential-
sphere baryon model gave a 0.324 dex control and was discarded rather than
reported, which is the right call: a control that large means the comparison is
measuring the baryon model.)

So the right answer arrived for the wrong reason. The family fails
asymptotically, by the boundedness of F, not in the observed regime — and inside
the observed regime it is a much better approximation to the data than my
argument implied. That distinction changes what a modification of the family
would have to do: it must break the boundedness of F, not add r-dependence.

## Y.3 Reciprocity holds and does NOT conserve momentum

Reciprocity F(x,x') = F(x',x) is satisfied to 4.1e-16 by the path-averaged
construction. It is not sufficient. An isolated UNEQUAL-MASS pair self-
accelerates, with the residual known in closed form and verified to 4.9e-12:

    |f_1 + f_2| / f_N = F'(qbar) [ q_2 - q_1 ]

exactly zero for equal masses, and **11% of the binding force at mass ratio
100**. A deliberately non-reciprocal path weight adds a second, independent leak.
Any member of this family must declare an explicit momentum carrier.

This is worth keeping as a general lesson: symmetry of the interaction kernel
under exchange of the two endpoints does not imply Newton's third law when the
kernel depends on a field that differs at the two endpoints.

## Y.4 Four further failures, and the deepest one

    repulsive shells      42% of the allowed grid gives dlnF/dlnr > 1, hence
                          v_c^2 < 0  (70% for the clipped q, 15% for screened)
    galaxy scatter        0.160 dex irreducible against the RAR's 0.11
    BTFR                  slope 2.88, scatter 0.229, size residual +0.49
                          against the observed 3.85, 0.10, 0
    Oort limit            F_local = 2.2-5.3 against 1.36 +- 0.15
                          (47 of 396 parameter sets fit, best 1.666, at the
                          window edge)

**The deepest failure is a scale conflict: 0 of 108 parameter points keep the
nonlocal signature.** For voids and filaments to differ in q at all, rho_ref must
sit near the cosmic mean, about 6.2 Msun/kpc^3. For the rotation curves to work,
rho_ref must be 1e5-1e6. One global scale, four to five decades apart.

**The family can be a theory of rotation curves OR a theory of intervening
structure. It cannot be both.** That is a structural verdict rather than a poor
fit, and it applies to every member of the family regardless of F.

## Y.5 Solar-system safety, with a twist

The clipped-delta form is EXACTLY safe: q(Sun) = 0 identically, so F = 1 and the
kernel reduces to Newton with no residual at all. The smooth form needs
rho_ref < 3.0e5, and the density contrast rho(8.2 kpc)/rho(25 kpc) = 663 still
leaves q = 1 at 25 kpc, so safety and efficacy coexist with about one decade of
headroom.

The screened form passes the inverse-square-law test easily (epsilon ~ 5e-13) —
**the screening length protects it** — and fails the Oort limit for exactly the
same reason. The mechanism that makes it locally safe is the mechanism that stops
it doing vertical work.

## Y.6 Computation, and four test bugs

4.5e8 pair-samples/s on the RTX 5090 against 3.8e6/s on CPU, a factor 119. An
all-pairs n = 128 run is 44 hours. The **spherical reduction — taking D as the
inner integration variable — is exact, removes the 1/|x-x'| singularity
analytically, and costs 2.2e-6**, which is what made the screen tractable. The
FFT/low-rank route is cheap (F1 at p = 1 is exactly rank 2) but its midpoint
surrogate is wrong by 4.3% rms and 61% worst-case, so it was used only for
benchmarking and never for a result.

Gates: domain-size convergence 3.8e-10 (against the local solver's 0.089%),
label-permutation invariance 2.2e-16, CPU/GPU agreement 2.2e-16, Newtonian limit
exactly 0.0. Four test bugs were found and recorded, and all four were test bugs
rather than solver bugs — consistent with the pattern in section 6 of the earlier
record.

Nothing was fitted; only the SPARC train split was used.

## W.5 Correction after the NED pass landed

NED's TAP service was dead throughout the lane — every sync query timed out and
batched ones returned HTTP 202 — but the classic `objsearch` CGI works. All 37
inventory names now resolve, with independent NED positions, heliocentric
velocities, redshifts, types and magnitudes. Two cross-identifications change the
inventory and both are the kind of error that propagates silently.

**MCG-05-07-001 = ESO 415-G026 = PRC A-02 are one galaxy**, verified by two
independent single-name queries returning byte-identical rows at RA 37.08376,
Dec -31.88100. They were carried as two separate Tier C entries. The inventory is
**21 rows (A = 9, B = 9, C = 3)**, not 22. Knock-on, recorded as an inference
rather than a certainty: Khoperskov+2014's dark-halo-axis-ratio roster lists
"ESO 415-G26" and "MCG-5-7-1" as separate entries with different references, so
that roster most likely covers **20 distinct galaxies, not 21** — unless
"MCG-5-7-1" there is a mistyped designation for a third object.

**AM 1934-563 is a galaxy TRIPLE, not the polar ring.** The bare name resolves to
type `GTrpl` at 294.66646, -56.45439. The polar-ring galaxy is the member
**AM 1934-563 NED02 = PGC 089058 = PRC B-18** at 294.66002, -56.45796. Anyone
matching on the bare name gets the wrong object, at a separation small enough
that a cone search would not flag it. Both rows are retained with the distinction
recorded.

One quirk noted so it is not propagated: querying `NGC 5907` returns the compound
preferred name `NGC 5907:[IDD2022] X026`, but the row's position, magnitude
(11.12) and 600 references are the galaxy's own.

The headline is unchanged.

## W.6 The concrete blocker for the whole polar-ring channel

Two primary sources remain unreachable, and they are precisely the ones that
matter:

  * **Reshetnikov & Combes 1994, A&A 291, 57** — the canonical
    two-perpendicular-rotation-curves measurement for UGC 7576 and UGC 9796.
  * **Arnaboldi et al. 1997, AJ 113, 585** — the NGC 4650A ATCA HI paper.

Neither is on arXiv or in VizieR, and A&A blocks programmatic fetches. **These
two papers are the difference between nine Tier-A systems whose curves exist only
as figures and two systems with real numbers.** Obtaining them needs journal
access or an author request — it is not a search problem, and no amount of
further automated acquisition will produce them.

---

# Run Z — Lead 01 on eFEDS, and why the hydrostatic route is vacuous

The strongest live lead was potential depth, and Run R named the decisive test:
|Phi_b| varying by >= 1 dex at fixed g_bar WITHIN one class, using Bahar+2022's
resolved eFEDS electron-density fits. This run does it. Code and JSON in
`work/wellnet-2026-09/lead01/`.

## Z.1 The chain, validated against the paper's own gas masses

The model is taken from the paper's table caption rather than assumed:

    n_e^2(r) = n0^2 (r/rs)^-alpha [1+(r/rs)^2]^(-3beta+alpha/2) [1+(r/rs)^3]^(-eps/3)

One trap on the way in: **the VizieR column labelled `n0` is the paper's n0^2**,
in 10^-7 cm^-6. The units give it away, and reading it as n0 would put every
density off by its own square root. rs is in arcsec and needs the angular
diameter distance.

Two observables, neither assuming dark matter:

    g_bar(r) = G M_b(<r)/r^2                     from integrating the fitted n_e
    g_obs(r) = -(kT/mu m_p r) dln n_e/dln r      hydrostatic, isothermal

Cuts declared before residuals; the binding one is the temperature error, which
drops 425 of 542 systems at eT/T <= 0.5. **GATE: reproduce the published
M_gas,500 from the density parameters** — median mine/published = **1.0079**,
scatter **0.0476 dex**, n = 105. The chain is right.

## Z.2 Resolved profiles do NOT remove the leverage cap

    median within-g_bar-bin spread of log|Phi_b|, 117 systems, 1,638 points

       eFEDS groups, resolved, ONE class          0.185 dex
       SPARC alone                                0.309
       full six-rung ladder                       0.766  (86% the class label)
       ladder minus the class label               0.286
       two-overdensity-radius group rungs         0.10 - 0.17

The resolved profiles buy **1.1x** the two-radius cap and remain **below SPARC's
own within-class leverage**. Restricting to systems with beta measured to better
than 50% gives 0.194 dex, so it is not a precision effect. The quadratic R^2 of
log|Phi_b| on (log g_bar, log r) is 0.8774 with a 0.191 dex residual, against
SPARC's 0.9322 / 0.218 — marginally more independent structure, the same
absolute residual.

## Z.3 The residual IS the shape factor, exactly

    corr( residual log|Phi_b| , residual log S ) = +1.0000

after controlling for log g_bar, its square, and log r. That is not a strong
correlation, it is the identity log|Phi_b| = log g_bar + log r + log S seen
numerically. Whatever a fitted coefficient of log|Phi_b| means at fixed
(g_bar, r), it is a coefficient of **log S**, the shape factor, whose spread here
is 0.3385 dex.

## Z.4 And the shape factor is the observable — a fourth shared-quantity artefact

    corr( log S , log |dln n_e / dln r| ) = -0.8735

A shallower outer profile means MORE mass beyond r (larger S) and a SMALLER
logarithmic density slope. But the hydrostatic g_obs IS that slope, up to kT/r.
The test variable and the observable are two views of the same profile shape, so
the regression is guaranteed to find something.

The fitted coefficient, radius controlled, system-level bootstrap:

    beta = -0.400,  95% CI [-0.626, -0.226],  q = 2 beta = -0.813

which is the WRONG SIGN for the hypothesis and would have been reported at
z = -4.11 against a null that omitted the shape. Three checks were run before
believing it, and all three licensed it:

  * tightening the temperature cut STRENGTHENS it (-0.400 at eT/T <= 0.50,
    -0.430 at 0.35, -0.551 at 0.25) — so it is not a temperature artefact;
  * every radial sub-range keeps it negative, -0.38 to -0.70;
  * a null carrying coherent gas-mass AND published-magnitude temperature errors
    is centred on zero (+0.0094 +- 0.0591, 95% [-0.104, +0.125]).

Then the decisive one:

    beta with the density log-slope ITSELF controlled = +0.4626

**The sign flips.** Controlling the shared shape factor moves the coefficient
from -0.400 to +0.463. Neither number is a measurement of gravity; both are
views of the same profile fit. The responsiveness gate is perfect throughout
(d beta / d q = 0.5000 exactly), which is precisely why the artefact is
dangerous: the estimator is unbiased for the thing it estimates, and the thing
it estimates is not what the hypothesis is about.

## Z.5 Verdict, and the repair it names

**The hydrostatic route to the potential-depth hypothesis is structurally
vacuous.** Not underpowered, not systematics-limited — the test variable and the
observable are algebraically the same quantity. No sample size and no better
temperature fixes it.

That is a stronger statement than "the leverage is too small", and it points
directly at the repair: **the gravity side must come from an observable that does
not contain the gas density profile.** Weak-lensing shear is exactly that. The
eFEDS field has HSC weak-lensing coverage for several hundred of these same
systems, making the correct experiment a within-class, same-survey comparison of
raw shear against a continuous baryon profile, with the density profile entering
only the baryonic side where it belongs.

Fourth shared-quantity artefact in this programme, after the label control that
killed seven variables, the retracted rho_p = -0.304, and the redshift path
integral worth 0.27-0.40 x c1 at 38 sigma. In every case the naive analysis
produced a confident number of plausible size; in every case the check that
caught it was writing down the construction expression for both axes and looking
for a shared factor.

---

# Run AA — the well-network tensors, and a convergence on potential depth

Full record in `work/wellnet-2026-09/tensor/REPORT.md`, 677 lines, with
`gates.json`, `mechanism_map.json`, `calibration.json` and
`seed_robustness.json`. Synthetic A2029 = the real X-COP baryon profile plus 300
members. B is defined as |g|(tensor)/|g|(K = I) on the same source with the same
mu, so B is the EXTRA factor on top of what MOND already supplies.

## AA.1 Amplitude was never the difficulty

    requirement, applied in sequence          well network   pair channels
    parameter points scanned                        1920            288
    reach B = 2 at 1 Mpc                            1541            140
    + a field galaxy stays inside 0.04 dex           760            140
    + a CLUSTER MEMBER galaxy inside 0.04 dex        563              0
    + flat profile, 0.7 < B(1414)/B(300) < 1.4       291              0
    + whole B(r) inside 1.6 - 2.5                    127              0
    best RMS against the measured radial run       0.033 dex      0.077 dex
    + within 0.10 dex AND both galaxies safe           1              0

Both tensors reach the factor-of-two cluster amplitude easily. What separates
them is radial SHAPE, and one constraint the brief never named.

## AA.2 The constraint nobody wrote down: galaxies inside the cluster

A cluster MEMBER galaxy sits at |Phi_N| = 1.09e12 m^2/s^2, **deeper than the
cluster's own 1 Mpc shell at 7.22e11**, with all 44,850 member pairs threading
it. So anything that switches on with potential depth, or with pair density,
switches on HARDEST inside cluster member galaxies.

That kills the pair-channel tensor outright — 0 of 288 points survive it — and
leaves the well-network survivors marginal, at a member violation of
0.031 +- 0.023 dex against a 0.040 tolerance.

It also converts itself into an observation. If gravity is boosted by potential
depth, the internal dynamics of galaxies inside clusters must be modified, and
Run V assembled exactly that sample: 108 clean SAMI field/cluster matched pairs
sensitive to a 0.022 dex offset, at median |g_ext|/a0 = 0.17. **The tensor lane's
own survival constraint is measurable with data already on disk.**

## AA.3 The convergence: only potential depth can separate clusters from galaxies

    Every viable parameter point uses a POTENTIAL-DEPTH gate.
    Not one uses an acceleration gate.
    126 of the 127 survivors need gate exponent m >= 2 at Phi_0 = 1e12 m^2/s^2.

The reason is exact rather than empirical: **a cluster at 1 Mpc and a galaxy
outskirt sit at the same g_N/a0.** That IS the radial acceleration relation. No
function of g_N/a0 can separate them, so no acceleration-gated law can boost one
without boosting the other.

This is arrived at by NECESSITY, from 3-D tensor field solves on a synthetic
cluster, with no fitting to the potential-depth hypothesis anywhere in it. It is
independent of Run R's ladder and of Run Z's eFEDS work, and it reaches the same
variable. Three lanes, three methods, one conclusion: **if anything separates
clusters from galaxy outskirts at matched acceleration, potential depth is the
only candidate on the table.**

## AA.4 And the anisotropy is not the active ingredient

The lane's own verdict, which is a simplification rather than a rescue: what
makes the well-network model work is an environment-dependent switch on local
Newtonian potential depth. The traceless tensor (det K = 1 exactly when A_0 = 0)
merely converts that switch into a radial-conductivity change. **A scalar
a0 -> a0 f(|Phi_N|/Phi_0) would reproduce both the amplitude and the shape to the
accuracy of this map.**

So the tensor family is not eliminated — well-network tensor 1 beats plain MOND
on clusters by 2x in a region clearing every gate — but the discrimination should
be credited to the potential-depth screen, not to the anisotropy. That is the
same conclusion the spherical blindness theorem (Run M) predicted structurally:
in a near-spherical configuration only the radial eigenvalue is observable, so an
anisotropic law is doing scalar work.

## AA.5 The effect is geometry, not shot noise

The lumpiness contribution is 0.2-0.3%, against the 0.4% that was the ENTIRE
effect in the earlier QUMOND-on-lumpiness calculation. Realisation scatter over
five independent member draws is 5-11% against a factor-2 effect. Projected
deflection is boosted 1.5-3.5x, so lensing sees it too.

## AA.6 Sign convention, measured rather than asserted

K = exp[+alpha C] is the sign that makes the response stronger ALONG the channels
joining wells; the brief's exp[-alpha C] makes it stronger TRANSVERSE. Anchored
on the exact uniform-K monopole, which gives |g|_par/|g|_perp = sqrt(k_par/k_perp)
and is reproduced to 1.5%. But sign = -1 is nevertheless the only sign that
raises the SHELL-AVERAGED field: B <= 1 at all 144 sign = +1 settings. Both facts
are true and the report explains why they are compatible.

d_par defaults to mode "clip", d_par = max(0, |t| - d_ab/2) from the pair
midpoint — a capsule flat along the segment. The naive infinite-line form is
implemented as "line" and is the every-pair-everywhere failure mode.

## AA.7 Gates: 13 of 13

K positive definite, minimum eigenvalue 1.83e-2 across 540 tensor fields; flux
3.29e-14 on face fluxes; curl 4.82e-17; Newtonian recovery 2.30e-4 at order 1.99;
constant-K analytic 3.57e-4 at order 1.99; resolution 64 -> 96 moves 1.47%, i.e.
NOT flat; domain 4 -> 8 Mpc moves 0.357%; source-label permutation 3.3e-14;
closed-form matrix exponential against scipy 2.4e-12; the operator is
bit-identical to `solver.py`, which was imported unmodified. The GPU channel
kernel agrees to 2.1e-15, and a 6-sigma cutoff costs 9.8e-9 while doing 9.4e10
tube evaluations in 2.0 s at 128^3.

Gate 5 needed correcting: "S -> 0 for a spherical distribution" is true **only at
the centre** (5.9e-17 for an exact octahedral configuration). Off-centre S is
non-zero and must be, and was validated instead against a 2-D Gauss-Legendre
quadrature of the exact continuum integral, agreeing to 1.1 sigma_MC at every
radius.

## AA.8 Two methodology traps

**The shell average of k is physics, not bookkeeping.** The arithmetic mean is
qualitatively wrong: it turns over, fakes a saturation at B = 2.1, and reports
A_T = -12.8 for a B = 2 model. The harmonic mean, calibrated against six full 3-D
solves, is correct, and the true value is **A_T = -4.7 ungated**. Worst surrogate
error 20.4% harmonic against 46.9% arithmetic, and the arithmetic error is always
one-signed.

**Two NaN sources, both from the same corner.** mu(0) = 0 together with
grad Phi = 0 at the centre gives a singular operator; and 400 radial bins on
94 kpc cells leaves empty bins, hence k = 0, hence NaN in the boundary condition.

## AA.9 What this makes the next measurement

The decisive question is no longer "is the response anisotropic?" It is:

    Does the internal dynamics of a galaxy change when it sits inside a cluster?

Every surviving well-network point predicts that it does, because the member
galaxy is the deepest potential in the system. Plain MOND with an
acceleration-only gate predicts that it does not. Run V's 108 SAMI matched pairs
address exactly this, at a sensitivity of 0.022 dex against a predicted member
violation of 0.031 +- 0.023 dex — the same order, which means the measurement is
close to decisive rather than comfortably clear of it.

---

# Run AB — the Stage-1 screen, and boundedness as the recurring killer

Full record in `work/wellnet-2026-09/screen/REPORT.md`. `gravitylab/solver.py`
and `axisym.py` were imported and never modified, with mtimes and SHA-256
recorded to prove it.

**The funnel's front stage is validated as feasible: 158,406,840 settings swept
at 2.05 million settings per second, so the brief's 1e9 Stage 1 is about eight
minutes.** Of those, **450 survive the three decisive screens, and all 450 have
s_0 = s_T = 0** — Newton with the network switched off. Family D has zero
survivors at any parameter value.

## AB.1 The bounded-response no-go, and its reach

    |S|_2 < 2/3 identically     (the sweep measures max|lambda(S)| =
                                 0.666666666666, saturating the bound)
    |That|_2 < 1

So K's eigenvalues sit in a fixed band, and

    d ln g / d ln r = -2.0000   for every C, D and E candidate
                    = -1.00001  for AQUAL, QUMOND and family B

**A bounded K can only rescale G.** The gain at 20 Mpc is 0.717 for family C and
1.000 for D and E, against the 3719 needed — short by a factor 3719 to 5191, and
the shortfall grows linearly with r.

This is the SAME structural failure Run Y found in the nonlocal path kernel,
where Phi = -GMF/r with qbar in [0,1) bounds F, so r v_c^2 -> GM sup F and
asymptotically flat curves are impossible. Two entirely different families, two
different derivations, one theorem:

    **A response factor confined to a bounded range cannot change the asymptotic
    force law. It can only renormalise the constant in front of it.**

That is now the single most productive structural result in the programme, and
it names the repair in both cases: the response must be unbounded, or must be
sourced from something that is not a bounded ratio. Note that families C and E
also do NOT tend to a constant K at infinity — every n_a -> -xhat, so
S -> xhat xhat^T - I/3 and K keeps a fixed anisotropy locked to the radius. The
correct far-field shell is Psi = -GM/(k_r r), not the constant-K monopole, and
the lane had to add that (`radial_far_field`) before anything else was reliable.

## AB.2 Why the tensor lane's survivors all needed a potential-depth gate

Run AA found that every viable well-network point uses a potential-depth gate and
not one uses an acceleration gate. AB explains it: the anisotropy ALONE, being
bounded, can only rescale G. Without an unbounded external switch there is
nothing for it to do. The gate is not an optional refinement of the tensor
mechanism — it is the entire mechanism, and the tensor converts it into a radial
conductivity change.

The two lanes were run independently and neither saw the other's result.

## AB.3 Momentum is not conserved, and nobody declared a carrier

    F_net = -oint T.n dS - (1/8 pi G) Int (d_i K_jk) d_j Psi d_k Psi

Measured on an isolated two-body configuration, as a fraction of G M1 M2 / d^2:

    family C1   0.564      of which the grad-K term supplies 0.555
                           and the surface term 1.4e-4
    family E1   0.197
    family B1   0.688
    Newtonian null          0.0019

and the identity gap falls as h^2.03, so this is physics rather than
discretisation. **No family declares a momentum carrier.** Run Y found the same
thing for the nonlocal kernel by a different route — reciprocity held to 4.1e-16
and momentum still leaked, 11% of the binding force at mass ratio 100. Symmetry
of the interaction does not give Newton's third law when the coupling depends on
a field that differs across the configuration.

## AB.4 Family C's screening term is mathematically inert

`g_N` carries no index `a`, so it cancels exactly in the ratio of sums that
defines S. The measured effect on S is **3.89e-12**, and it scales exactly with
the regulator epsilon (3.89e-12 at eps = 1e-12, rising to 0.285 at eps = 1).
The acceleration screening in family C is degenerate with a rescaling of the
regulator and does nothing at all.

## AB.5 Family D has no continuum limit

    ||C|| ~ N^(2-2p)      measured slopes 0.0102 / 1.0101 / 0.1667
                          against predicted 0 / 1 / log

At p = 1/2 the minimum eigenvalue of K falls from 3.4e-1 to 8.3e-80 as N goes
from 10 to 800. Cost is O(P N^2), i.e. 5.0e11 pairs for 1e6 rows.

## AB.6 Coarse graining: the exponent cancels, so selective refinement is the test

Uniform refinement from N = 1 to 10^4:

    family C1-C5    dPhi 10.06-10.11%   dv_c 9.10-9.12%   convergent quadrature
    D1 (to N=256)   21.8%               30.1%             convergent quadrature
    D2, D3          25.9%, 44.0%        29.1%, 43.6%      D2 catalogue-artefactual
    E1, E2          9.7%, 37.6%         10.8%, 36.1%      CATALOGUE-ARTEFACTUAL
    Newton control  0                   0                 partition-independent
    counting control 118.9%             379%              catalogue-artefactual

**The mass exponent cancels exactly under uniform refinement** — p = 0.5, 1 and 2
all give drift 0.28013 to five figures. That is why uniform refinement is the
weaker test and SELECTIVE refinement is the one with teeth. Under selective
refinement the weight ratio moves as N^(1-p), measured 0.7507 / 0.5007 / 0.2507 /
0.00067 / -0.4993 / -0.9994 for p = 0.25 to 2 against the predicted 1 - p.
**Only p = 1 is admissible**, in every family that carries a mass exponent.

## AB.7 The discriminator between a physical scale and a catalogue row

Convergence alone does not separate them, which was the worry the brief raised.
The discriminator that does is the response of the drift to the coherence length:

    d ln(drift) / d ln L  =  -3.11   genuine smoothing kernel (control X4)
                             -0.55   family C1
                             +0.12   pure row-counting

Family C's directional factor turns over on the distance to the NEAREST ROW,
which no parameter controls. And K has **no value at a catalogue point at all** —
only a direction-dependent limit, of amplitude 0.389 for family C at an isolated
row and 0.449 for family E at any N.

Control X4 was built specifically so that the coherence test could return
"physical" rather than only ever rejecting, and it does.

## AB.8 Stage 2: no spurious forces, but representation dependence

**No family produces a spurious midpoint force** — |g(mid)| / (GM/(d/2)^2) is at
most 1.5e-12 against a 1e-3 tolerance. Every Stage-2 rejection is
representation-dependence instead, and the sharpest instance is worth quoting
directly:

    **Family C's inferred M_dyn for one and the same cluster moves 14%
    (6.16e13 -> 7.04e13 Msun) depending only on whether it is entered as one
    catalogue row or 10^4.**

Family D switches off entirely beyond about 32 kpc (4e-15 deviation) and has
exactly zero effect on an isolated object.

## AB.9 The repairs, named per family

  * **C**: p = 1 mandatory; replace the row sum by an integral over rho, which
    also removes the discontinuity; move the g_N screening OUTSIDE the
    normalisation so it stops cancelling; let s_0 diverge as g_N -> 0 to escape
    the bounded-response no-go; make the alignment field dynamical. The repaired
    object is "QUMOND plus density-functional anisotropy".
  * **D**: p = 1 and q < 3 mandatory, plus a double integral over rho tensor rho.
    The O(P N^2) cost problem remains.
  * **E** is closest to salvageable: **source the tidal tensor from the smooth
    density rather than the row list** and four screens plus two gates pass
    automatically, leaving only the S6 screening and the unbounded backbone.
  * **B**: |Phi| must be replaced by something built from derivatives, since a
    potential defined only up to a constant cannot be a physical argument.

## AB.10 Discipline

All seven headline statistics pass the monotone-invariance guard, and reciprocity
is exactly zero at s_T = 0 as it must be. Two of the lane's own test bugs were
caught and recorded: the discontinuity probe compared plus and minus e, but
n n^T is even in n so the comparison is identically zero — the real answer is
0.389, not 5e-5; and a two-body test divided an acceleration by a force. Box-size
convergence at FIXED grid spacing is 0.39%, where the 21% first seen at fixed n
was resolution rather than the boundary. No data file was opened.

Could not establish: whether any repaired candidate fits anything, which needs
data; a variational completion carrying the missing momentum; family D beyond
4e7 pairs directly; and hierarchical multi-scale partitions. The Stage-2 cluster
geometry does not exercise family D or the counting control, whose 10 and 8 kpc
scales sit far below 400 kpc — both are exercised and both fail at galaxy scale
in Stage 1b, and their cluster "pass" carries no information.

---

# Run AC — the field detector's real false-positive rate, and a correction to Run O

Run O reported "injection recovery 2/2 at 10% noise" and a 4.2% false-positive
rate for anisotropy from 24 scalar-only trials, and I described the machinery as
"good enough to trust a positive". **Both numbers were measured on a degenerate
null, and the conclusion was too strong.** Code in
`work/wellnet-2026-09/field_power.py`, results in `field_power.json`.

## AC.1 The degenerate null

Run O's scalar-only injections were built from the bank's OWN atoms. That makes
the truth exactly representable by the scalar basis, so admitting anisotropic
atoms improves nothing, the statistic

    D = RMS(scalar atoms only) - RMS(all atoms)

is identically zero, and the critical value collapses to zero — at which point
every trial "detects" and the power is 1.00 by construction. The first run of
this module reproduced exactly that: D median 0.000e+00 and power 1.00 at every
amplitude, including 0.05.

**A null in which the truth lies exactly in the model's basis is not a null, it
is a best case.** Real physics is never exactly in the basis. The repair is to
inject BETWEEN the bank's grid points — the bank uses scales 0.3, 1.0, 3.0, so
the injection uses 0.55 and 1.8 — leaving an irreducible misspecification that
both bases must compete over, which is the situation the detector actually faces.

## AC.2 The naive rule is far worse than reported

With a non-degenerate null, the rate at which "a tensor atom appears in the
winning pair" fires on scalar-only data:

    geometry            noise 2%   noise 10%   noise 30%
    strongly triaxial      47%        43%         50%
    mildly triaxial        63%        30%         63%
    near-spherical         63%        77%         70%
    triaxial, tilted       30%        27%         30%

**27% to 77%, against the 4.2% Run O reported.** That is the rate the nested
family guarantees: the anisotropic set contains the scalar set, so the richer
model can only tie or win, and on any misspecified target it wins by fitting the
misspecification. This is the same mechanism the control lane found when a bare
argmin identified 4 of 5 injected families where a one-standard-error parsimony
rule got 5 of 5.

## AC.3 The calibrated detector, and its actual power

Using the EMPIRICAL critical value D*, the 95th percentile of D under
scalar-only data at that noise — correctly sized by construction, which neither
a nominal p <= 0.05 nor a guessed p <= 0.02 would be:

    geometry (axis ratio)   noise   amp 0.05   amp 0.15   amp 0.35
    strongly triaxial 0.50    2%      0.00       0.25       0.81
                             10%      0.00       0.56       0.88
                             30%      0.00       0.31       0.75
    mildly triaxial   0.80    2%      0.00       0.06       0.44
                             10%      0.00       0.19       0.44
                             30%      0.00       0.12       0.31
    near-spherical    0.95    2%      0.00       0.12       0.31
                             10%      0.00       0.38       0.62
                             30%      0.00       0.00       0.31
    triaxial, tilted  0.50    2%      0.12       0.62       0.94
                             10%      0.38       0.88       0.88
                             30%      0.31       0.88       0.75

Three things follow.

**The detector is blind below amplitude ~0.1.** Power is 0.00 at amplitude 0.05
in eleven of twelve cells. Any future anisotropy claim must state the amplitude
it could have detected, and it is not small.

**Triaxiality matters more than noise.** Going from axis ratio 0.50 to 0.80 costs
more power than going from 2% to 30% noise. That is the spherical blindness
theorem showing up as a power surface rather than as a theorem.

**A tilted axis is EASIER to detect than an aligned one** — 0.62 against 0.25 at
amplitude 0.15 and 2% noise. When the injected preferred axis cuts across the
source's own principal axes the signature is more distinctive, which is worth
knowing when choosing which real clusters to test.

## AC.4 A control that was mis-specified, and what it teaches

The near-spherical row does NOT collapse to zero power, which the spherical
blindness theorem might seem to require. It does not, and the theorem is fine:
the theorem says the transverse eigenvalue of a SPHERICALLY SYMMETRIC K is
invisible in a spherically symmetric source. An injected `dhat dhat^T` tensor
imposes an EXTERNAL preferred axis, so K is not spherically symmetric and the
configuration is not covered. The control tested the wrong thing.

The corrected statement: sphericity of the SOURCE suppresses anisotropy that is
sourced BY the source (the well-alignment and tidal tensors), but not anisotropy
imposed by an external axis. Those are different claims and the power surface
separates them.

## AC.5 What this does to the register

The Run O line "the machinery is now good enough to trust a positive" is
withdrawn as stated. The accurate version:

  * the calibrated detector is correctly SIZED by construction, because D* is
    measured rather than assumed;
  * its POWER is modest — good only above amplitude ~0.15 on strongly
    non-spherical sources, and negligible below 0.05 anywhere;
  * the naive "a tensor atom appeared" rule has a false-positive rate of 27-77%
    and must never be used;
  * and the calibration remains conditional on these geometries, this noise
    model, this grid resolution and this candidate-selection procedure. It is not
    a general certificate.

This is the fourth time in this programme that a control turned out to be
measuring a best case rather than a null. The pattern is the same each time: the
null was constructed from the same machinery as the signal, so it inherited the
signal's advantages.

---

# Run AD — Lead 01 dismantled, and a fifth blind statistic (mine)

Full record in `work/wellnet-2026-09/lead01-ablation/`. Four things were tested
and all four went against the lead.

## AD.1 The headline number is invariant to the effect it measures

    d(beta)/dq                                  = 0.5000  (responsive, correct)
    d(RMS of M1 on the holdout)/dq              = 0.0000  EXACTLY

M1's coefficient absorbs the injection, so the transfer RMS cannot move. **"0.1066
dex, a 63% reduction" reads identically at q = 0, at q = 0.371 and at q = 0.8.**

That is the fifth monotone-blind statistic caught in this programme, after the
X-COP rank test, the DiskMass scale length, and two others — and this one is
mine. It is the number I put at the top of the promising register. Scaled against
the responsive statistic, the observed paired difference corresponds to
**q ~ 0.18**, not the 0.371 the cluster excess requires.

## AD.2 The potential law and the class step cannot be separated, ever

    paired dRMS = RMS(M1) - RMS(M3)   = +0.01128 dex
    frozen bootstrap                    +0.01115 +- 0.00575, 95% [-0.00007, +0.02246]
                                        P(M1 better) = 0.026
    nested, training resampled          +0.01241 +- 0.01347,  P = 0.145
    per-object sign test                M1 closer on 24 of 52, p = 0.678
    vs its own shared-denominator null  z = +1.76

The variance decomposition is the important part: 0.00575 dex comes from the
held-out set and **0.01217 dex from the 200-system TRAINING set, which does not
shrink with n_test.** The ceiling with infinite validation data is **0.93 sigma**.
A 3-sigma verdict needs about **2,100 training systems**.

And they barely differ by construction: the two frozen predictions correlate at
+0.9227, the rms prediction difference on the holdout is 0.0461 dex against a
0.0500 dex median measurement error, and only 20 of 52 clusters exceed their own
error.

## AD.3 The ablation: the transferable content is one constant

    A, galaxies only      beta = +0.0900, half the published value.
                          Recovers 33.5% of the cluster offset and
                          -0.4% of the GROUP offset.
    B, groups only        beta = -0.3189, WRONG SIGN, transfer 0.3331 dex --
                          worse than the RAR alone, worse than doing nothing.
    C, galaxies+groups    the published +0.17188.

So the groups cannot constrain beta at all, and the galaxies alone give half of
it while failing on the groups. What actually transfers is a single number: the
frozen mean group offset of **+0.2344 dex, with no parameter**, predicts the 52
clusters at **0.1033 dex** — the tightest interval in the table, and
indistinguishable from both M1 (dRMS -0.0033, P = 0.659, closer on 33 of 52) and
M3.

**Simpson's paradox, explicitly.** Every WITHIN-class beta is zero or negative:

    field galaxies    +0.090  [+0.003, +0.183]
    groups            -0.319
    clusters          -0.181
    groups+clusters   -0.199  [-0.298, -0.133]
    pooled, 252        +0.169   <- only here is it positive

Inside the holdout itself the observed gradient is q = +0.04 against M1's frozen
prediction of +0.12 dex. **M1 passes by getting the LEVEL right, not the
gradient**, which is what a constant offset does.

This is consistent with Run Z, where the within-class eFEDS fit also came out
negative and then flipped sign when the shared shape factor was controlled. The
within-class evidence is now consistently zero-or-negative in two independent
datasets.

One genuine point for M1: the class step is **not estimable at all** unless the
training set straddles the boundary — in arms A and B, M3 collapses onto M0
digit for digit.

## AD.4 The boundary rule matters twelve times more than the stability I quoted

Reconstruction gated at 3.6e-15 dex against the published column. Across four
declared rules — BARY (primary, declared in advance), PHYS, OVER and TAIL —

    beta spans          +0.1408 ... +0.1779     a 22.4% range
    M1 transfer spans    0.0990 ...  0.1614 dex

I offered the 1.9% refit shift as evidence of stability. The boundary rule moves
beta by **twelve times that**. And under all four rules the class step wins, on
error reduction (67.3-68.5% against 44.5-66.1%) and on BIC (+14.6 to +69.2).

Two corrections to the earlier record:
  * the corr = +1.0000 between the residual and log S is an **algebraic tautology
    under every boundary rule**, not a property of the point-mass-tail
    convention as Run Z implied;
  * Run R's **S >= 1 theorem holds only for the point-mass-tail convention**.
    Under finite reference rules log S reaches -1.66.

## AD.5 The fresh sealed sample says something new, and it is not what the lead wanted

Babyk+2018 (ApJ 857, 32): **94 Chandra early-type galaxies at 5 r_e**, plus
2MASS K-band photometry. Never used in this programme; acquired by a process that
computed no residual; the model was frozen and hashed at 12:18:16Z before the
sample was read. An ingest gate caught a 2-dex gas-mass error (r_c rounded to
0.01 kpc) and the fix was recorded at 12:27:35Z, still before any residual.

On 67 primary galaxies:

    observed deviation   +0.3418 dex  [+0.2766, +0.4089],  60 of 67 positive

**That is LARGER than the cluster offset (+0.263) at a potential depth 0.8 dex
SHALLOWER.** A monotone A(|Phi_b|) cannot produce that, which is the same
non-monotonicity Run R found across the rungs, now confirmed on a sealed sample.

All three frozen models fail: M1 recovers 12% of it, M3 recovers -8%, the RAR 4%.
But on the pre-declared comparison **M1 beats M3 with P = 1.0000, closer on 54 of
67** — the opposite of the ladder verdict. M3 only works if these galaxies are
relabelled as non-galaxies after the answer is known (0.4668 -> 0.2865 dex),
which is precisely the objection the step was introduced to answer.

Robust to the stellar mass-to-light ratio: the offset is +0.381 / +0.342 / +0.287
at Upsilon_K = 0.6 / 0.75 / 1.0, and erasing it needs 3.7x more baryons. The real
exposure is hydrostatic bias — erasing it needs M_tot lower by a factor 2.2, and
this lane cannot distinguish that from gravity.

## AD.6 Where this leaves the lead

Demoted, and the register entry rewritten. What survives:

  * a real, large, sealed-sample deviation in early-type galaxies, +0.342 dex,
    60 of 67 positive, LARGER than clusters at shallower potential;
  * the fact that a single constant group offset transfers as well as any fitted
    law, which is itself informative about what the cluster gap is;
  * the class step's own weakness, that it is inestimable without a boundary in
    the training set and needs post-hoc relabelling to survive the ETG sample.

What does not survive: the 63% figure as evidence for anything, the "predicts
clusters from galaxies and groups" framing, and the claim that beta is stable.

---

# Run AE — do galaxies change inside clusters? Measured, with the power stated first

Full record in `work/wellnet-2026-09/member-dynamics/`. `power_prestatement.json`
was written and closed BEFORE any offset was evaluated, which is possible here
because the blocked sign-flip null's width depends only on |dY| and the block
structure, not on the signs.

## AE.1 Power, stated in advance — and it is not decisive

Combined MaNGA `B4_disk_wide` and SAMI `S2_diskbearing`: 402 pairs, 92 host
systems.

    statistical only            sd 0.0048 dex in log V, 3-sigma reach 0.014 dex
                                power against the H1 prediction = 0.90
    with the measured
    environmental systematics   sd 0.0097 dex, 3-sigma reach 0.058 dex in log g
                                power = 0.36

**The sample cannot decisively separate the hypotheses.** Its 3-sigma reach in
log g, 0.058 dex, is larger than the tensor lane's own 0.040 dex tolerance.

## AE.2 The measurement

    Delta log g_internal = -0.019 +- 0.010 (stat) +- 0.017 (syst)
                         ( = Delta log V  -0.0095 +- 0.0048 +- 0.0084 )

    hypothesis                                 prediction      separation
    H1  potential-depth gate                 +0.031 +- 0.023   -1.7 sigma
                                                               (-2.6 naive,
                                                                -5.3 if nuisances
                                                                are regressed out)
    H2  acceleration-only / algebraic RAR         0.000        -1.0, CONSISTENT
    H3  MOND WITH the external-field effect,
        computed per pair from the sample's
        own g_ext and g_bar                  -0.024 to -0.009  +0.3 / -0.6,
                                                               CONSISTENT

One-sided 95% upper limit **Delta log g_int <= +0.013**, below H1's central value
and below the 0.040 tolerance: **78% of the H1 band is excluded, the lower 22% is
not.**

So: internal dynamics do not measurably change, and the central value is slightly
NEGATIVE — the MOND external-field-effect direction, opposite in sign to a
potential-depth gate. The observable is Y = log10 sigma_e,tot, the flux-weighted
stellar second moment <V^2 + sigma^2> inside 1 R_e, from 902 DAP MAPS cubes for
MaNGA and the algebraically identical MGE combination for SAMI.

## AE.3 Three things that change how this test must be run

**1. The naive tracer gives a big, confident, wrong answer.** Same pairs, same
machinery:

    rotation only      -0.069 +- 0.023   (-2.9 sigma)
    dispersion only    +0.003 +- 0.009

Cluster members rotate 15% slower at unchanged total kinetic energy. That is the
kinematic morphology-density relation, and it is four times H1's size and
opposite in sign. Anyone using rotation velocity as the tracer would report a
confident detection of the wrong thing.

**2. Cutting on the observable manufactures an H1-sized signal.** The lane's own
declared cut (`med_sigma_astro >= 40`) is a cut on the outcome. Scanning it, the
answer marches from -0.0048 to +0.0072 (MaNGA) and -0.0121 to +0.0009 (SAMI), and
a "high-purity" subsample returns **+0.027 +- 0.012, p = 0.03 — a spurious
2-sigma CONFIRMATION of H1.** The corrected cut set moves the primary answer by
0.0005 dex.

**3. Being inside the tolerance box is not being balanced.** Inclination is
imbalanced at 2.1 sigma in MaNGA and 2.3 sigma in SAMI, in opposite directions,
despite every pair sitting inside the declared matching box. The covariate-adjusted
estimator handles it, moving the answer 0.002-0.003 dex.

## AE.4 The contamination budget, and what dominates it

Reached **0.008 dex in log V, 0.017 in log g** — 1.9% in velocity, below the 5%
target set in the brief but NOT below the effect being tested. Dominated by g-i
colour at matched stellar mass (+0.0111 +- 0.0035), continuum signal-to-noise
(-0.0065 +- 0.0021) and the HI-detection term (+0.0038 +- 0.0019). **No term can
rescue H1 by a sign flip.** sigma_e and lambda_R were excluded from the total as
circular, being components of the observable.

f_gas was dropped as a matching variable and declared, per Run V's finding that
HI stripping makes it unavailable in this regime; it is priced instead as the
HI-detection budget term. `logSigma_b` and `log_gbar` are treated as one
direction, per the Freeman identity.

## AE.5 Validation

Null simulation with the real error covariance and shared inputs gives a bias of
**-0.0003 +- 0.0071**, i.e. unbiased. The injection test gives
d(estimate)/d(delta) = **1.000000 exactly** over a 0.100 dex injected range, so
the monotone-invariance gate passes. Leave-one-cluster-out moves the answer by at
most 0.0047 dex (SAMI's 8 clusters) and 0.0055 (MaNGA). Apertures of 1 R_e,
0.5 R_e, 3 kpc and 5 kpc span 0.009 dex.

## AE.6 The gradient test has no power, and that is quantified rather than asserted

No gradient was detected on log|Phi| or on clustercentric radius. But the
injection test shows the 3-sigma minimum detectable end-to-end offset is
**0.068 dex (MaNGA) and 0.062 (SAMI)** — four times H1's entire predicted offset.
**This must not be quoted as a null.** SAMI's radius estimator with sigma in the
denominator was replaced by the sigma-free one wherever sigma is the other axis.

## AE.7 The limitation that names the next experiment

**The sample sits just below the knee of the gate it is testing.** SAMI's median
member is at log|Phi| = 11.68 and its deepest hosts at 12.00, against the tensor
lane's member galaxy at 12.04 — and 126 of 127 tensor survivors use exponent
m >= 2 at Phi_0 = 1e12. A gate with m >= 2 is still nearly off at 11.68.

So the decisive measurement is **not more pairs** — the statistical error is
already 0.0048 dex — but **deeper potentials**: the fundamental plane of cluster
against field early-type galaxies near cluster cores, which needs no matched
pairs at all. That is the same population where Run AD's sealed Babyk+2018 sample
found a +0.342 dex deviation, larger than clusters at a shallower potential. Two
lanes, arriving from opposite directions, at the same next target.

Second priority: kill the colour systematic at source by matching on
spectroscopic stellar populations rather than photometric colour, which would
remove about 70% of the MaNGA budget.

---

# Run AF — the vertical channel is real, and it rejects Newton and the RAR

> **CORRECTED IN RUN AS (§AS.5).** The title's second clause is not supported by
> the slope. Observed -0.346 sits 0.32-0.82 sigma from the RAR's -0.291 and
> 0.47-1.22 sigma from AQUAL's -0.264 — it AGREES with them. The rejection comes
> from the joint amplitude-and-shape statistic in AF.3, and only under a
> constant-Upsilon_K model. Read this run as "the between-galaxy slope resembles
> the RAR/AQUAL prediction; the within-galaxy profile does not."

Full record in `work/wellnet-2026-09/vertical-audit/` — `REPORT.md`,
`bz_formula.md`, `vertical_audit.json`, code. All four pre-declared promotion
gates pass. This is the first result in the programme to survive a complete
shared-denominator audit, and it comes with a degeneracy that must be stated in
the same breath.

## AF.1 The tautology charge: structurally correct, quantitatively negligible

`Sigma_L0` IS in the denominator of `B_z`, with a measured coefficient of
**-0.994**, and it IS the abscissa. So the charge was right about the structure,
and a label shuffle genuinely cannot see it.

But pushing the COMPLETE observational covariance through the REAL pipeline at a
true Newtonian `B_z = 1` displaces the slope by only **-0.012 to -0.018**, i.e.
3.5 to 5.2% of the observed -0.346. Four hundred Newtonian universes per scenario
across five covariance scenarios — independent errors; the fixed-`L_tot`
degeneracy `d log Sigma_0 = -2 d log h_R`; plus inclination error; plus all
photometric errors tripled; plus galaxy resampling:

    the recovered slope NEVER reached -0.346 in 2,000 trials
    worst case -0.325,  P = 0,  95% bound 0.0075

**CORRECTED IN RUN AS (§AS.5): `P = 0` overstates a finite simulation.** With no
exceedances in 2,000 trials the Monte-Carlo p-value is `p <= 1/(N+1) = 5.0e-04`.
The `95% bound 0.0075` is the rule-of-three bound on the per-scenario rate (3/400)
and is correct as printed.

Producing -0.346 by the artefact requires `e_mu0_K,i` about **0.70 mag, ten
times the tabulated 0.070**.

The null is correctly sized: residual rms 0.1705 dex against 0.1669 observed, a
ratio of 0.98, and injected slopes from -0.60 to +0.30 are recovered with bias
below 0.012.

**The clean form agrees.** Fitting the latent-variable regression directly, with
errors on both axes and the measured per-galaxy covariance,

    s = 0.646 +- 0.064      against  s - 1 = -0.346  =>  s = 0.654

with the identity between the two forms verified numerically, and attenuation
moving `s` by +0.012 — the entire displacement. The closed-form DiskMass
estimator, which never touches the forward chain at all, gives `s = 0.668`, an
implied slope of -0.332. **The signal is in the raw catalogue numbers.**

## AF.2 The two numbers reconciled, and a broken error bar found

`+-0.173` is a raw standard deviation times 1.3485 of a distribution with
**skewness +26 and excess kurtosis +901**. It does not converge:

    draws   200    400    800   1,600   3,200   6,400
    sd     0.066  0.166  0.129  0.316   0.260   0.328
    robust 0.064 - 0.071 throughout

**Cause found and it is a code defect.** `newton_chain` floors `sigma_z^2` at
`1e-30`. For UGC 6918 and UGC 1862, an extreme `h_z` draw lets the leakage term
exceed the vertical gravity and `log B_z` diverges to **+41 dex**, inside the fit
window, in 11 of 1,600 draws.

`0.0095` is a one-sided galaxy-BOOTSTRAP tail from a different resampling
(reproduced at 0.0106 over 20,000). Both numbers were individually correct;
printing them side by side invited the meaningless quotient 0.346/0.173 = 2.00.

    Defensible headline:  -0.346 +- 0.117, one-sided p = 0.010, i.e. 2.3-3.0 sigma

## AF.3 Local against global, with the model comparison itself calibrated

M_global, `B_z ~ Sigma_0^p` with fitted `p = -0.350`, wins:

    dAIC   +8.8  over free-a0 RAR          +10.7 over frozen-a0 RAR
          +10.9 over a general local power law in |g|
          +16.8 to +19.5 over the three declared potential-depth rules
          +17.9 over Newton               +58 over the isotropic tensor

on 20 of 20 nuisance draws. The mechanism is legible: every model cuts the
amplitude scatter from 0.080 to about 0.053, but **only M_global leaves the
scale-length scatter at Newton's 0.149** — the local laws inflate it to
0.176-0.184.

Then the dAIC itself was calibrated, with 100 synthetic universes under EACH
hypothesis and a gate requiring the machinery to reproduce the real-data AICs to
0.000:

    statistic                  observed   Newton-truth null        RAR-truth null
    dAIC(Newton - global)       +17.90   -1.4 [-2.0,+3.0] P=0.000  +22.6 P=0.710
    dAIC(RAR - global)           +8.80   +31.2 P=0.820             -22.8 P=0.010

**Both Newton and the RAR are rejected, each by the statistic the other passes.**
That calibration also retires the naive reading of the raw +8.8: under
Newton-truth the median is already +31.2, so the unadjusted number was never
interpretable.

## AF.4 But the winner is not gravity

`degeneracy_check.py`: M_global and "Newton with `Upsilon_K ~ Sigma_0^p`" give
identical amplitudes and identical scale lengths to **5e-15 dex** — machine
precision — because `sigma_z^2`, `g_R` and the leakage term are all exactly
linear in `Sigma_*0`.

What it would take to be a systematic rather than gravity, quantified:

  * `Upsilon_K` anticorrelating with surface density at **-0.395 dex/dex**, a
    factor 4.1 across the sample — and **opposite in sign to the observed B-K
    colour trend**, which is evidence against the systematic reading;
  * or `h_z/h_R` varying by a factor **5.2**, where Bershady gives 1.22.

Neither is comfortable, and the sign of the first is wrong. But the degeneracy is
exact, so this cannot be settled inside DiskMass.

**An eighth variable-list collapse**, found here: `r(log g_b(2.2 h_R),
log Sigma_0) = +0.9907`. Local-in-`g_b` and global-in-`Sigma_0` are the SAME
between-galaxy predictor on this sample — the same identity the environment lane
measured at 0.9955. Only the RADIAL SHAPE separates them, and `g_b` varies by
just 0.05 dex across the fitting window.

## AF.5 The laws, through the identical pipeline

Verified rather than quoted: RAR **-0.288** against the published -0.291, AQUAL
**-0.260** against -0.264, isotropic tensor -0.048 against -0.055. And the median
per-galaxy `h_sigma/h_R`:

    observed 2.086     Newton 2.499     RAR 2.896

**The observed vertical profile is STEEPER than Newton, where the RAR needs it
flatter.** That is the within-galaxy half of the tension, now measured through
the same aperture, PSF and fit window as the data.

## AF.6 Corrections to the earlier record

Measured sensitivities: **`h_z` carries -0.667, not -1; `k` carries -0.797, not
-1.** Consequently Run L's "the Bershady relation would have to be wrong in slope
by 1.99" was inconsistent with its own numerical scan in the same block, which
implies **4.1**. The conclusion is unchanged and in fact strengthened.

**Independent check on MaNGA** (Run V's 240 face-on disks): the sign replicates
across five tiers at -0.69 to -0.83 with P(slope >= 0) = 0, stable all the way up
the instrumental-floor ladder to sigma > 110 km/s. But its own shared-denominator
floor is -0.15 to -0.60, `sigma_LOS` is not `sigma_z`, there is no scale height,
and its median sigma is 77-154 km/s against DiskMass's 24-45. It confirms the
SIGN, not the amplitude, with much weaker control.

Two more code defects reported: the `sigma_z^2 >= 1e-30` floor above, and grid
points with `h < 0` where `-1/b` is finite but negative and `log10` returns a
silent NaN.

## AF.7 What is promoted, and what is not

**Promoted:** a measurement of the vertical `Sigma_dyn`-`Sigma_b` relation,

    s = 0.65 +- 0.12

explicitly degenerate with `Upsilon_K(Sigma_0)`. It is not a tautology, not a
shared-denominator artefact, and it replicates in sign on an independent survey.

**Not promoted:** any claim that this is evidence for a modified force law. The
same analysis rejects Newton, the RAR, AQUAL, both tensor forms and all three
potential-depth rules. Something in the vertical direction is not described by
any law currently on the table — including the ones this programme has been
promoting — and the leading alternative explanation is a stellar mass-to-light
ratio that varies with surface density, which is exactly degenerate with it and
can only be broken from outside DiskMass.

---

# Run AG — the nonlocal kernel: numerics sound, physics dead, and two corrections to my own reporting

Full record in `work/wellnet-2026-09/nonlocal-repair/REPORT.md`.

## AG.1 The stability audit clears the numerics

`D = F - rF'` is converged at the previous lane's production settings: against an
all-fine reference it differs by **0.13% median, 0.97% p95, 5.6% worst** over the
1028 SPARC train points. On the configuration where `dlnF/dlnr = 0.8995` and the
margin `D/F = 0.0654`, every quadrature knob moves `D` by at most **0.22%**. The
10:1 cancellation costs about one decimal digit and float64 had fifteen spare.

The method matters: the previous lane obtained `D` by DIFFERENCING `F`, which
differences the same cancellation twice. This lane differentiates under the
integral sign, and gates it — the analytic `alpha = 0` limit to 1.3e-9, and the
gauge identity to 3.5e-13.

**A gauge fact that sharpens the boundedness theorem.** `F` is defined only up to
an additive multiple of `r`, since `F -> F + cr` shifts `Phi` by the constant
`-GMc`. **`F` is not an observable; `D` is.** So the correct statement of the
no-go is not "F must be unbounded" but

    |F| / r  ->  infinity

and even that is only necessary. `F = 1 + 3r` is **pure gauge** — `D = 1`
exactly, i.e. Newton. `F = 1 + 3r^1.1` has `F/r -> infinity` and reverses gravity
at `min D = -7535`. The unique exactly-flat form is `1 + 3r ln(r_0/r)`, which is
unbounded, grows faster than `r`, and eventually goes negative.

**And one of the previous lane's two headline configurations does not survive.**
The "best set passing both local tests", reported at 0.196 dex, produces
**`D <= 0` — repulsive gravity — at 260 of 1028 train points (25.3%) in 36 of 71
galaxies**, confirmed by three independent routes. The earlier screen tested
`F_eff <= 0`, the potential's sign, not `D <= 0`, the force's. That corner is
withdrawn.

## AG.2 The comparison I reported was apples to oranges

I wrote "0.156 dex against Newton's 0.646" in the promising register. That was a
POTENTIAL-space statistic beyond 2 R_disk placed beside the RAR's ~0.11, which is
an ACCELERATION-space scatter on all points. Different quantities, different
point sets. Corrected, on identical galaxies, points, nuisances and split:

    acceleration space, rms log10(g_pred/g_obs), R >= 2 R_disk
                                  train      validation    BLIND
      AQUAL simple mu             0.121        0.098       0.121
      RAR                         0.121        0.098       0.122
      nonlocal kernel             0.256        0.218       0.209
      Newton                      0.597        0.517       0.588

**The kernel loses to the RAR by roughly a factor two, on train, validation and
blind.** Its blind result was never previously quoted at all: 0.209 dex
acceleration, 0.135 potential, against the RAR's 0.122 / 0.120 on the same
points. Every choice in the comparison was made to favour the kernel — `a0`
fitted on train and frozen, exact AQUAL solutions, and points where the kernel
predicts `g <= 0` simply dropped.

So the register line "reproduces the required modification factor to 0.156 dex
against Newton's 0.646" should have read: **the kernel is twice as far from the
data as the RAR, in the currency that matters.**

## AG.3 The clipping audit: no shell, but a quadrature trap and a hard requirement

The clip is C0 with a kink of exactly 1.000 in `q'`. The C2 replacement is the
unique quintic `phi(u) = 2w(t^3 - t^4/2)`.

**In the kernel formulation the clipping surface adds no force, no flux and no
energy discontinuity** — `[Phi] = 0` and `[g] = 0` under shrinking-window
extrapolation. What survives is a step in `dg/dr`, i.e. in the inferred density,
at 79-265x the local baryon density for `p = 1` and **exactly zero for `p = 2`**.
The same conclusion follows in closed form for a field formulation, where
`rho_eff = rho/K - M K'/(4 pi r^2 K^2)` jumps but carries no delta. **`p >= 1` is
a requirement, not a preference**, and at `p = 0.5` the effective density
diverges.

**A quadrature trap worth propagating.** `D` contains `dqbar/dr`, and a hard clip
makes the integrand **discontinuous in the path variable**, so Gauss-Legendre
converges at first order. At the production `n_s = 12`, `D` near a clip surface
carries a **20-50% error** — a defect of the clip, not the solver. The C2 clip
buys two decades by `n_s = 16`, and this is why `n_s` was one of only two knobs
that amplify the cancellation.

**C2 and exact solar safety are compatible.** Compact-support rounding leaves
`q(Sun) = 0` identically for any `w <= 0.975`, so `eps(1 AU) = 0` exactly, just
as for the hard clip. **It is ANALYTICITY that costs the exact zero, not
smoothness** — `softplus` leaves `q(Sun) = w^2/(4|u|)` and forces `rho_ref` down.
That retires a trade I had assumed was real.

But smoothing does not repair the physics: across 40 configurations no variant is
free of repulsive points, and smoothing moves the rms by under 0.03 dex.

## AG.4 The family is not repairable, and the second no-go is the decisive one

**Search: 6720 settings, seven atoms.** `D > 0` everywhere AND flat:
**0 of 6720.** The Pareto frontiers do not meet — best flatness with `D > 0` is
`dlnD/dlnr = 0.915` where flat needs 1.000; best `min D` among flat settings is
`-0.973` where it needs to be positive.

The mechanism is a theorem, not a search outcome. For `F = 1 + a G(qbar) H(r)`,

    D = 1 + a G (H - r H') - a r G'(qbar) qbar'(r) H

and **the last term is strictly negative** whenever the modification grows in
voids and paths get more void-like outwards. Raising `a` to lengthen the flat
part raises the negative term by the same factor.

**The second no-go is independent and harder.** `Phi[rho]` is a LINEAR functional
of rho whenever `F` does not depend on rho, so `v_f^2 ~ M` and the **BTFR slope
is 2**. Verified: universal `c` gives 2.000, and MOND's `c ~ M^(-1/2)` gives
4.000. The only nonlinearity available is `q`, and a bounded `q` saturates in the
outskirts — exactly where the flat part lives — restoring linearity there.
**That is why the measured slope was 2.88: 2 and 4 bracket it, and 2.88 is the
transition, not a solution.** A repaired unbounded `F` inherits this unchanged,
because the repair is to `F`'s r-dependence and not to the kernel's linearity
in rho.

## AG.5 The tensor atom: one structural gain, one confirmed mechanism, still not a solution

**The gain is free and worth keeping.** `k_r = exp(...) > 0` identically, so
**the exponential tensor grammar cannot produce a repulsive shell.** The scalar
kernel produced them at 23-48% of train points. That failure mode is removed by
construction.

**Stage 1, 9600 settings: 0 pass all five metrics, and the first empty cumulative
cut is the BTFR** (best 3.066 anywhere, 2.50-2.92 among flat settings, against
3.85 +- 0.09) — exactly as the linearity no-go predicts.

**Nonlocality is the only thing that can change the asymptotics at all.** The
local-`q` control gives `dln g/dln r = -2.0000` exactly, Kepler, for all 4800 of
its settings. The path average escapes Kepler, but only into a runaway.

**A bug that changed a conclusion, and it is not this lane's alone.**
`families.tidal_hat` normalises by `sqrt(eps_T^2 + |T0|^2)`, and the first run
used `eps_T = 1e-30 s^-2`, which is **190 times larger** than the actual
`|T0| = 5.3e-33` at 50 kpc from a 6e10 Msun galaxy. Corrected:

    eps_T                          10-20 kpc   20-40 kpc   40-70 kpc
    3.9e-31 (the SCREEN's default)   -0.297      -0.051      -0.008
    1e-37 (unregularised)            -0.794      -0.814      -0.816

**Under the existing screen's own regulariser the anisotropy is switched off
throughout galaxy outskirts, a factor 97 suppression at 40-70 kpc.** With it
corrected, the directional term does **12% of independent work**: turning `f_T`
on raises the radial boost from 2.447 to 2.731 while leaving the vertical boost
at 0.958 against 0.961, and flattens the outer curve from -0.064 to -0.022.

So the brief's hypothesis — that a directional version could keep the radial
behaviour while avoiding the vertical boost — is **confirmed in direction**. But
the decoupling is overwhelmingly the GATE, not the direction: even at `f_T = 0`
the radial boost is 2.45 against a vertical 0.96, because the vertical force near
the midplane is sourced by dense material where `qbar ~ 0` and `K = I`.

**This partially undercuts Run AB.** The screen lane concluded that a bounded
anisotropy does no independent work, and it drew that at the default `eps_T`
which suppresses the tidal tensor by 97x in exactly the regime that matters.
Families C, D and E were NOT re-run at a corrected `eps_T`. The bounded-response
no-go itself is unaffected — it is about `|S|_2 < 2/3` and the asymptotic slope,
not about `That`'s normalisation — but **"the anisotropy is not the active
ingredient" now needs re-testing**, and that is the single most actionable
follow-up in the programme.

On SPARC the tensor atom scores 0.448 train / 0.443 blind with **zero repulsive
points**, against the scalar kernel's 0.256 / 0.209 and the RAR's 0.121 / 0.122.
Worse than both, but the equivalent-spherical treatment makes it a lower bound —
the 3-D disk solve gives a radial boost 50-67% larger at the same radii.

---

# Run AH — the joint tournament, and a scalar gated on the TIDAL invariant

Full record in `work/wellnet-2026-09/tournament/REPORT.md`, with `METHODS.md`,
`tournament.json` (3,123 records) and eight verification JSONs. Nothing was
rebuilt; every reused module imported unmodified with its SHA-256 recorded.

## AH.1 The result nobody proposed

3,123 candidates scored simultaneously on radial rotation, vertical amplitude,
vertical radial shape and cluster amplitude+shape. **18 survive all seven
screens**, and the discriminating one is the constraint the tensor lane
discovered rather than any of the four channels:

    screen                                kills alone   UNIQUE kills
    H3 member galaxy <= 0.040 dex            2850          105
    H1 cluster reach / H2 field galaxy    634 / 2848     14 / 14
    H7 asymptotic slope                       959            1
    H4 radial / H5 amplitude / H6 shape  2728/2291/900      0

The three channels that kill the most candidates kill **nothing uniquely**.

    probe                              median |T| (s^-2)    median |Phi_N|
    cluster shells 300-1414 kpc          3.66e-34            5.9-10.6e11
    isolated field galaxy 10-30 kpc      6.87e-32  (19x)     1.13e10
    CLUSTER MEMBER galaxy 10-30 kpc      5.54e-31 (151x)     1.09e12 (deepest)

**Potential depth orders the member galaxy ABOVE the cluster shell**, so every
depth gate fires hardest exactly where it must not. **The tidal invariant orders
it the other way round, by 151x.** An inverse-tidal gate therefore switches off
inside galaxies automatically, with no anisotropy and no weight-family choice:

    aqual | scalar_a0 | tidal | inv m=2 | T0 = 1e-33
    a0 = 1.002e-10, A = +16.0, radial 0.168 dex, B_z 1.515, h 34.82 arcsec,
    cluster 0.271 dex (0.118 against the flat target), field 0.0070,
    MEMBER 0.0001 dex, asymptotic slope -1.00002, k = 4

**A scalar law gated on the tidal invariant passes all seven screens with a
member violation of 0.0001 dex and no anisotropy at all.** 13 of the 18
survivors use it, and not one survivor of any kind uses an acceleration gate —
reproducing the tensor lane's finding from a much wider grammar. It pays with a
cluster profile that **RISES outward** where the lensing-derived shape falls: a
clean falsifiable prediction rather than a fitted freedom.

## AH.2 The tensor's advantage was a weight-family choice, not anisotropy

At a matched potential-depth gate the scalar competitor equals or beats every
tensor on radial rotation, on both vertical channels and on cluster shape,
losing only the member screen — by a factor of sixty.

The escape depends on `S` being a NORMALISED direction average, so whether the
host galaxy or the crowd of 300 members dominates is set by the mass exponent
`p`. At `p = 0` a 4e11 Msun host counts no more than a 1e9 dwarf, the crowd wins,
and `S` inside the member points along the CLUSTER radius. **Restore the brief's
literal p = 1 and the member violation goes 0.007 -> 0.528 dex, back to the
scalar's.**

Eight independent draws of the 300 members, amplitude frozen after one fit:

    tensor_S p=0, phi sat m=2     0.042 +- 0.028 dex     4 of 8 inside tolerance
    tensor_S literal p=1          0.486 +- 0.216         0 of 8
    scalar_a0 phi sat m=2         0.418 +- 0.011         0 of 8

Independently reproducing and extending the tensor lane's 0.031 +- 0.023 on five
draws: **marginal, not comfortable.**

## AH.3 The boundedness theorem, sharpened a third time

Of 130 measured combinations, 100 leave the asymptotic slope at the base law's
value. **Making f unbounded does not help, and the reason is not the form of f:
every invariant in the grammar DECAYS outward** — g_N ~ r^-2, |Phi_N| ~ r^-1,
rho = 0 outside, |T| ~ r^-3, qbar -> const — so an unbounded f is evaluated on a
vanishing argument. Only an invariant that GROWS outward could change the
asymptotics, and there is none; forcing one overshoots to a rising curve.

Third sharpening: first "F bounded", then "F is gauge, D is the observable, so
|F|/r -> infinity", now "the obstruction is the decay of the argument, not the
boundedness of the function".

## AH.4 Momentum, and what it is really a property of

    AQUAL / QUMOND base alone      0.000  (variational)
    scalar_a0 potential-depth      0.801 / 0.667 / 0.591
    scalar_a0 TIDAL                0.823
    tensor_T                       0.872 / 0.616 / 0.581
    tensor_d                       1.699 / 1.756 / 1.694
    iso_K                         16.53 / 15.57 / 14.93

as a fraction of GM1M2/d^2. **The third-law violation is a property of "the
response depends on position", not of anisotropy** — both scalars violate it as
badly as the tensors. No candidate has a declared carrier or a variational
completion.

## AH.5 Two things this settles about the vertical channel

Among the 58 candidates passing both galaxy screens, h_sigma,LOS spans 33.8-37.4
arcsec. **Every one is 5-9 arcsec above the observed 28.65 and worse than
Newton's 30.80. The vertical shape channel prefers Newton over every law in the
tournament, base laws included.** No gated law relieves Run L's tension.

And the vertical channels **can never discriminate a gate**: for any candidate
passing the galaxy screens the gate is off in galaxies by construction, so they
test only the base law. That is Run L's "the amplitude constrains rather than
discriminates", now quantified as 0 unique kills, and it extends to the radial
channel too.

## AH.6 A first-order systematic in the potential-depth mechanism

Of four boundary rules for |Phi_N|, two are global prescriptions and admissible.
**They differ by 0.87 dex in the median galaxy potential depth** (1.42e10 against
1.06e11), and the margin between the gate being off and on is only 0.9 dex.
**The whole potential-depth mechanism rests on a quantity uncertain by nearly a
decade.** The tidal-gated survivors have no such ambiguity — |T| is a local
second derivative with no boundary constant. A second structural reason to prefer
the tidal gate, independent of the member screen.

## AH.7 Verification, and how soft the survivor list is

The cluster channel reproduces the tensor lane's published survivor through an
independently written code path to **0.2%** (B = 2.509, 3.224, 2.576, 1.979
against 2.51, 3.22, 2.57, 1.98; RMS 0.0986 against 0.099; member 0.0151 against
0.015). Run L's SPARC and vertical numbers reproduce to the fourth decimal.

But four direct nonlinear 3-D solves show the spherical surrogate — calibrated at
|A_T| <= 8 — under-predicting by **up to 76%** at the survivors' amplitudes of
25-102. One survivor's true 3-D profile RISES outward. **The cluster RMS and the
ranking among the 18 are good to tens of per cent, not better.**

**Verdict: promote nothing.** Three things are robust because they are one to two
orders of magnitude larger than these uncertainties — the member screen's unique
power, the scalar's parity with the tensors on every other channel, and the tidal
gate's 0.0001 dex member violation.

## AH.8 The next measurements, now sharply specified

1. **The internal dynamics of cluster member galaxies.** The only screen with
   independent power, and it separates the two surviving gate families by three
   orders of magnitude in the invariant they respond to. Run AE measured
   Delta log g = -0.019 +- 0.010 +- 0.017; the tidal gate predicts essentially
   zero there, which is what was observed.
2. **The radial run of the cluster excess**, with the tidal gate's RISING profile
   as the explicit alternative hypothesis — where the two surviving families
   disagree most sharply, predicted before the data are consulted.
3. **A variational completion or a declared momentum carrier** for anything that
   gets that far, because nothing currently has one.

---

# Run AI — potential depth against raw shear: the test ran, and found nothing

Full record in `work/wellnet-2026-09/efeds-hsc/REPORT.md`. This is the test Run Z
said was necessary rather than merely better, and it has now been run for the
first time.

## AI.1 HSC is unreachable; DECADE is the answer

Every HSC route returns HTTP 401, Chiu's data-availability line is "shared upon a
reasonable request", and the only public products are NFW-fitted masses, which
hard-constraint 2 forbids as observables. The premise in the brief was wrong.

**DECADE** — the metacalibration shape catalogue in DELVE DR3, served
unauthenticated by the NOIRLab Astro Data Lab TAP endpoint — covers the eFEDS
field completely: **14,498,544 sources**, with the 1p/1m/2p/2m sheared copies so
the response matrix is RECOVERED rather than assumed, plus weights, tomographic
bins and per-source photo-z at ~6.8 galaxies per square arcmin.

**A new provenance trap, and a serious one.** DataCite and Zenodo return records
titled *"HSC Y3 Shape Catalog — GAMA09H Full Field"* (10.5281/zenodo.15482851 and
two siblings) describing precisely the eFEDS-overlapping field with e1/e2/RA/Dec
columns. **They must not be used**: the creator list includes an LLM simulation
assistant, and the records' own correction notice admits the depositor is "not
the creator or sole rights holder" and that one is "an algorithmically scaled or
pipeline-derived artifact". They are indistinguishable from real data by title,
and a search for a shape catalogue in this field returns them.

## AI.2 The measurement

496 systems, 3365 (system, bin) points, z = 0.017-0.855.

    tangential, inverse-variance mean   +0.00134 +- 0.00014  = +9.6 sigma
    cross (B-mode)                      -0.00004 +- 0.00014  = -0.3 sigma
    random-point null, 246 positions    -0.00032 +- 0.00021  = -1.5 sigma
                                        cluster signal 4.2x the residual
    responsiveness d(beta-hat)/d(beta)   0.9984, spread 1.000

**The sign convention was measured, not assumed.** All four axis-sign
combinations were run: `atan2(d_dec, +d_ra cos dec)` gives -0.00213, and
`atan2(d_dec, -d_ra cos dec)` gives +0.01082. **The DECADE/DES ellipticity basis
has its first axis pointing WEST.** The first pass used the wrong convention and
produced a null signal. The +0.0108 amplitude also matches Chiu's HSC stacked
profile in the same field (0.0117 at 0.73 Mpc), an independent check on the chain.

## AI.3 Potential depth adds nothing

| model | k | chi2 | dBIC |
|---|---|---|---|
| M3 + gamma log r | 2 | 1850.30 | **0.00** |
| M0 RAR only | 0 | 1870.72 | +5.54 |
| M1 + free amplitude (CLASS STEP) | 1 | 1870.22 | +12.48 |
| ... f_gas, redshift, free a0, log T, log M_b ... | | | +17 to +18.5 |
| **M2 + beta x_Phi (THE HYPOTHESIS)** | 2 | 1870.22 | **+19.92** |

**Last of ten on BIC, and it improves chi2 over the class step by exactly 0.00.**
On frozen transfer to the held-out half it gains +0.00, and the training winner
M3 does not transfer either (+0.22 of 1719). The literal differential slope is
consistent with zero in all four acceleration quartiles.

Null-calibrated: **beta = +0.072 +- 0.084**, which is 0.86 sigma from zero and
1.20 sigma from Run R's +0.17188. **The test does not exclude the hypothesis; it
fails to find it**, at a precision (0.084) just below the hypothesis's own
systematic floor (0.096). Systematics-limited, not statistics-limited, and the
Fisher forecast says the private HSC per-cluster profiles would give 0.075 — no
better.

## AI.4 The shared-quantity null fired again, on X-ray fit noise alone

Monte Carlo with the actual published errors on every density parameter, shear
redrawn independently:

    E[beta-hat | H0: beta = 0]  =  -0.0666 +- 0.0101   =  -6.6 sigma_MC from zero

**Noise in the X-ray density fit by itself drives the naive estimator to -0.067.**
A naive analysis would have reported a significant NEGATIVE potential-depth
effect that is pure fit noise. That is the fifth artefact of this family in the
programme.

But note what changed: **the shear route DOES break the Run Z identity.** The
construction expressions share no input — x_Phi depends only on the density-fit
parameters and z, while g_t depends only on galaxy shapes, weights and photo-z —
where in Run Z the hydrostatic g_obs literally WAS the density log-slope. The
residual bias is propagated fit noise, not an algebraic identity, and quoting the
estimate against its own null removes it.

## AI.5 Stacked profiles are structurally incapable of this test

On the one public HSC stacked profile, across the same five boundary rules:

    beta = +2.0 (grid edge), +2.0 (edge), +1.25, +0.50, +2.0 (edge)
    and a bare radius tilt beats the hypothesis by dBIC 27

On the per-cluster DECADE data, across the identical five rules, beta is pinned
at zero (+0.1, +0.1, +0.1, -0.1, -0.1 on a 0.2 grid), and is equally stable
across four radial ranges and three stellar-fraction choices.

**With only radial-shape information the boundary rule determines the answer.**
Cross-system leverage is what makes the variable identifiable at all. That is the
Run Z warning realised on a real lensing observable, and it retires stacked
profiles as a route to this question for good.

## AI.6 The leverage that exists, measured

    R^2 of x_Phi on a quadratic in (log g_b, log r), shear-measured radii, 248 systems
        0.9863,  residual 0.087 dex

**98.6% of potential depth in this sample is a function of acceleration and
radius.** The residual 0.087 dex is all the leverage there is — against a
within-class spread of 0.220 dex under the primary boundary rule and 0.115-0.232
across the five.

The Run Z identity survives on the difference: corr(residual x_Phi, residual
log S) = **+1.0000**, and corr(log S, log|dln n_e/dln r|) = **-0.9450**. On the
X-ray side potential depth is still exactly the shape factor; only the observable
changed.

## AI.7 One real bug, caught by the checklist

The first Abel projection was missing the cosh Jacobian and sat at exactly 2/pi
of truth **independently of every grid parameter** — the flat-error-curve
signature the programme's own checklist names, and it was pinned by the SIS
closed form. Gates after the fix: SIS 2.6e-4, Plummer 2.7e-3 / 9.5e-4, NFW
against Wright & Brainerd 5.7e-4, and the truncation error is NOT flat (moves
2.0e-2 over r_t = 25-200 Mpc), which is what it should do.

## AI.8 What this closes

Combined with Runs AD, AE and AH, the potential-depth hypothesis has now been
tested four independent ways and found nothing each time:

    Run AD  transfer statistic blind to its own parameter; within-class beta
            zero or negative in every class; a constant offset transfers as well
    Run AE  cluster-member internal dynamics -0.019 +- 0.010 +- 0.017, consistent
            with an acceleration-only law and with MOND's external-field effect,
            1.7 sigma from the potential-depth prediction
    Run AH  the member screen kills it; and the two admissible boundary rules
            differ by 0.87 dex against an off/on margin of 0.9 dex
    Run AI  raw shear, within one class, same survey: last of ten on BIC,
            chi2 improvement 0.00, beta = +0.072 +- 0.084

The hypothesis is not excluded — Run AI's precision sits just below its own
systematic floor — but nothing supports it, and the three structural objections
(the boundary rule defines the variable, 98.6% of it is (g_bar, r), and the
member galaxy is the deepest potential in a cluster) are unanswered.

---

# Run AJ — Stage 0: the detector is NOT correctly sized against an arbitrary scalar

Code `work/wellnet-2026-09/stage0_null.py`, results `stage0_null.json`, log
`stage0_full.log`. Grid 40^3, 10% noise, six-axis search grid, 120 calibration /
120 audit / 40 injection realisations, all three sets disjoint.

## AJ.1 What was changed, and why the previous null was still friendly

Run AC repaired a degenerate null — truth exactly in the basis, power 1.00 by
construction — by injecting between the bank's grid points. That still only asked
whether the tensor family could beat a slightly misspecified member of its OWN
grammar. Stage 0 replaces it with

    H0 : g = F(invariants) grad Phi_N     for ANY sufficiently smooth scalar F

realised as five families the bank cannot express: a different interpolating
function, a rational Pade with free poles, a piecewise spline with random knots,
a response built from a DIFFERENT invariant combination, and a
**Gaussian-process draw in log x** — an arbitrary smooth scalar drawn without
reference to the grammar. Six false-anisotropy generators are applied while the
truth stays strictly scalar: triaxial tilt, deprojection error, miscentring, a
radial mass-to-light gradient, unresolved substructure, and a multiplicative
shear-calibration error.

The statistic is the max improvement ANYWHERE in the search, taken over the
six-axis grid as well as over subsets, so the look-elsewhere effect of axis
choice sits inside the null. And the critical value is set on the calibration set
and then verified on an UNTOUCHED audit set, because a rate that is 5% by
construction is not a measurement.

## AJ.2 The verdict: 0.092 against a nominal 0.05

    realised false-positive rate at D*  =  0.092 +- 0.026     (nominal 0.05)

**Nearly double, at 1.6 sigma.** The detector over-rejects against an arbitrary
scalar response even with its critical value calibrated. Broken down by the
defect that produced it:

    deprojection error     0.15      <- three times nominal, the worst
    triaxial tilt          0.12
    substructure           0.09
    mass-to-light gradient 0.08
    none (clean source)    0.07
    miscentring            0.05

**Deprojection error is the single worst false-anisotropy generator**, which is
what a wrong assumed axis ratio does: it puts a genuine quadrupole into the
observable that no scalar law can absorb, and the tensor atoms take it. That was
predicted as a concern and is now measured.

## AJ.3 Why it fails: the tails, not the medians

    family      median D     MAXIMUM D
    nu_simple   8.30e-05     8.12e-04
    pade        1.03e-04     5.02e-04
    spline      2.58e-04     3.89e-02      <- 150x its own median
    other_inv   2.66e-04     3.52e-03
    gp          6.38e-05     1.33e-02      <- 208x its own median
    all         1.18e-04     D* = 6.53e-03 (95th percentile)

The medians are all within a factor four of each other and would suggest the
detector is fine. **The maxima are not.** The spline and Gaussian-process
families — the two that are genuinely outside the grammar — occasionally produce
an apparent tensor improvement 150 to 200 times their own median, and those tails
set the critical value.

A 95th percentile estimated from 120 draws of a distribution with tails that
heavy is itself poorly determined, which is exactly why the audit set overshoots.
**The correct reading is that D* is under-estimated, not that the detector is
mildly miscalibrated.**

## AJ.4 Power, which is genuinely better

    injected amplitude   0.15    0.35    0.60
    power                0.65    0.82    0.95

Better than Run AC's surface at comparable amplitudes, because maximising over
the axis grid is a more powerful statistic as well as a more honest one. But
power is only interpretable once the size is right, and it is not yet.

## AJ.5 What this means for every tensor result in the programme

Nothing in the programme currently rests on a tensor detection — the tournament's
verdict was "promote nothing", and the tidal-gated survivor is a SCALAR. So no
published number moves. What changes is the standard:

  * **The tensor detector's false-positive rate against an arbitrary scalar
    response is 0.092, not 0.05 and not Run AC's 0.042.** Any future anisotropy
    claim must clear a critical value estimated from far more than 120 draws,
    because the null distribution is heavy-tailed.
  * **Deprojection error must be controlled at source, not calibrated away.** At
    0.15 it is the dominant single contributor, and it is a modelling choice
    rather than a noise term.
  * Run AC's per-geometry surface remains valid for what it measured — an
    off-grid member of the bank's own grammar — but that is a weaker null than
    the one a real claim faces, and its 4.2% should not be quoted as the
    detector's size.

## AJ.6 A resource limit worth recording

Running six heavy lanes concurrently exhausted the machine's paging file, and for
a period numpy could not load its DLLs at all: *"ImportError: DLL load failed
while importing _multiarray_umath: The paging file is too small for this
operation to complete"*, with PowerShell itself failing to start the CLR. The
condition cleared once the lanes moved off their numerical phases, and the run
above completed normally at grid 40 in 147 seconds with 24 source banks.

This is a real constraint on the parallel-lane strategy rather than a code fault,
and it argues for staggering the heaviest numerical lanes rather than launching
every one at once.

---

# Run AK — the redshift branch: time dilation kills the mechanism in its natural form

Full record in `work/wellnet-2026-09/redshift/REPORT.md`. **This is a logically
independent hypothesis.** Nothing in the gravity lanes is evidence for it and
nothing here bears on that work; no data, fit, calibration or model-selection
step is shared.

## AK.1 The cheapest decisive test, run first, and it decides

Acquired White et al. 2024 (arXiv:2406.05050, MNRAS 533, 3365): **1504 DES
supernovae over 0.1 < z < 1.2**, fitting `dt_obs = dt_em (1+z)^b`, giving

    b = 1.003 +- 0.005 (stat) +- 0.010 (sys),   total sigma_b = 0.0112  (1.1%)

corroborated by Lewis & Brewer 2023 on quasars at n = 1.28 +- 0.29.

The decomposition `b = 1 - f(1 - eta)`, with `f = ln(1+z_path)/ln(1+z)` the
fraction of the redshift the mechanism carries and `eta` its time-stretch
efficiency, makes the test exact and splits the hypothesis in two:

    eta = 0   any energy-drain, tired-light or scattering mechanism.
              Carrying all of z predicts b = 0, which is 90 SIGMA away.
              Survives only at f < 1.9% (2 sigma), i.e. c2/c1 < 3.4%.

    eta = 1   a genuinely GEOMETRIC path stretch, acting on frequency and time
              together. Predicts b = 1 identically, so this test has ZERO
              POWER against it. The only surviving class.

So the branch is not merely constrained — it is **partitioned**, and one half is
dead at 90 sigma from a measurement that costs nothing to apply.

## AK.2 A second gate, derived in the lane, that does reach the survivor

An achromatic path redshift gives `dT/T = -c2 dI_q` on the cosmic microwave
background, so the observed anisotropy caps

    |c2| / c1  <  0.28 - 0.44%

and it closes the obvious escape — that the mechanism is confined to low
redshift — because it is entirely about LOCAL FOREGROUND structure. This is the
lane's own derivation and is labelled as such.

**Both external bounds beat this dataset's honest sensitivity (7-10%) by factors
of 2 to 30.** The fit below is therefore a bounded feasibility study, not a
measurement that could have decided anything.

## AK.3 The fit, null-subtracted

Primary arm, SDSS VAST VoidFinder at z < 0.11, n = 20,683:

    c2/c1 = -1.17% +- 1.45% (stat) +- 1.40% (sys)   = -0.58 sigma
    |c2/c1| < 5.11% at 95%
    matched-pair differenced estimator: +0.62 sigma on 20,547 pairs
    six-term law on watershed geometry: NO TERM above 1.8 sigma

Power: the 3-sigma minimum detectable c2/c1 is 3.9% statistical, 5.9% with
systematics. Audit false-positive rate 0.053-0.075 against a nominal 0.05 on
untouched simulations — the Stage 0 discipline applied and passed.

**Against ANALYTIC errors the same six-term table would have announced 6.1
sigma (c2), 5.4 sigma (c5) and 3.4 sigma (c4).** Against its own simulated null
nothing exceeds 1.8. That gap is the whole methodological point.

## AK.4 The arm-to-arm dispersion is the real error bar

Five estimators of the same coefficient span **-17.7% to -1.2%, sd 6.7%**,
against a median quoted sigma of 3.5%. But the RAW coefficients, before null
subtraction, span only -3.0% to +2.6%.

**The data agree; the corrections do not.** The dispersion is manufactured by the
choice of null and de-biasing scheme rather than by the measurements, and any
quoted error that does not include it is understated by about a factor two.

## AK.5 Two structural obstructions

**The tidal terms cannot be determined at all.** Run Q established that c3 and c6
are separable only on watershed geometry, because inside a uniform sphere the
potential is quadratic so `T_ij k^i k^j` loses direction dependence — that
restricts them to DESIVAST. But the footprint analysis restricts DESIVAST to
z > 0.11, where **n = 46**. **c3 and c6 have no footprint-safe determination.**

**The regressor is not measuring emptiness.** `corr(dI_q, mean line-of-sight
density) = +0.319` for REVOLVER, and **-0.190** against the true underdensity
path integral. The watershed TILES the volume rather than selecting voids, so its
"void path length" is closer to a density-weighted path than to an emptiness
measure. That is a second independent reason the pipelines cannot be averaged,
and it prompted a void-finder-free density-field arm.

## AK.6 A bug whose fix moved more than every statistical error

The Malmquist log-slope, measured over a fixed 30 Mpc/h window, reached **+-8**
and drove a spurious 10-30% null bias. Corrected to the distance-error window,
demeaned, with the residual amplitude still uncertain. Both treatments are
reported, and **the 15-point swing from that single choice exceeds every
statistical error in the lane.**

Lensing was quantified rather than assumed negligible: void-correlated
sd(kappa) = 1.4e-4, i.e. 0.014% in ln D. And the lane's independent null
reproduces Run Q's to 0.6 percentage points (+0.79 against +0.66% for
VoidFinder, +3.95 against +3.44% for REVOLVER).

## AK.7 Circularity, and the honest verdict

Four places, unsoftened, plus a fifth the lane quantified: **the source's own
peculiar velocity moves both its redshift and its ray's truncation point**, with
`dI_q = 1_void x v/H0` while `d ln(1+z) = v/c`. Endpoint reuse costs 18% on the
leverage variable and the radial metric 5-10%. A genuine no-expansion analysis
would have to rerun the void finders under its own distance law, which also
changes which galaxies exist.

**No detection is claimed, and none could have been.** The branch's status is:
the energy-drain half is excluded at 90 sigma by supernova time dilation; the
geometric half is invisible to that test but capped at 0.28-0.44% by the CMB,
which is 2 to 30 times tighter than this dataset can reach.

The highest-return next step is to cross-correlate the void path-length map with
Planck, which is the only cheap step that reaches the one surviving mechanism
class.

---

# Run AL — the lensing closure, and the contamination it caught upstream

Full record in `work/wellnet-2026-09/closure/REPORT.md`. The closure is now a
named parameter with a measured value in two regimes and a stated error budget,
rather than an assumption sitting silently under every cluster number this
programme has produced.

## AL.1 Why the ordering is the source of identification, not a preference

A constant slip applied to the 3-D lensing mass and the same constant applied to
the projected mass give chi2 agreeing to **0.0e+00 relative**. Within lensing
alone, slip and lens mass are EXACTLY degenerate — no shear profile, no image
configuration and no time delay can separate them. **The slip is identifiable
only because the dynamics law is frozen first.**

## AL.2 The free-closure control, and a small theorem

    RAR under no slip                     chi2 = 1869.7   (the reference)
    Newton + ONE closure parameter        1877.6   recovers 93.0% of the gap
    Newton + TWO                          1872.9   recovers 97.1%
    Newton + an unrestricted per-cluster
      closure (248 parameters)            1656.1   -- 214 chi2 PAST the RAR
    a deliberately WRONG law
      (Newton x (r/Mpc)^-1) + two         1872.9   IDENTICAL to Newton + two

That last row is not a coincidence, it is an identity:
`(r/Mpc)^s x (r/Mpc)^-1 = (r/Mpc)^(s-1)`. **A radial closure and a radial
modification of the force law are algebraically the same object.** So a free
closure cannot be a nuisance parameter alongside a radial gravity law; it is the
same degree of freedom twice.

And what the free closure learns is instructive: at 1 Mpc, where the shear
carries most of its weight, Newton's fitted "closure" reproduces the RAR's
dynamical boost **to 2%** (9.12 against 9.31), diverging by factors of 2-4
elsewhere where the data do not care.

## AL.3 A provenance failure in the tournament's headline survivor

    RAR, AQUAL                a0 frozen on SPARC rotation curves      PASSES
    tidal scalar, A = 7.5     from the X-COP flat target, which is
                              hydrostatic gas, i.e. slow matter       PASSES
    tidal scalar, A = 16.0    selected against the lane-12 radial
                              requirement, which is interpolated from
                              published lensing MASS profiles         **FAILS**

Those published masses assume Sigma_s = 1. **Scoring A = 16 against raw shear
under no slip and calling the agreement a success would be circular — the
amplitude was set by the answer.** This is the "never fit the law and the closure
simultaneously" rule caught one level upstream: the closure was not fitted, it
was INHERITED. The lane takes A = 7.5 as primary and carries A = 16 labelled.

## AL.4 eFEDS cannot test the tidal gate at all

    W = 1/(1+(|T|/T0)^2)  on the 3365 measured points:
        median 0.9998, only 0.45% of points below 0.90
    log10(B_tidal/B_RAR)  median +0.4390 dex, sd 0.0085 dex

**The gate is saturated.** |T| sits far below T0 everywhere the data are, so the
tidal law degenerates to AQUAL with a0 -> a0(1+A) — a CONSTANT rescaling by
sqrt(1+A) = 2.915 varying by 0.0085 dex across the entire data set. **A universal
slip absorbs a constant exactly**, so on eFEDS the tidal law and the RAR are the
same hypothesis up to a closure.

Quantified by embedding both in one family and profiling the slip out:
lambda_hat = -7.50 +- 1.25 against a lambda = 0-vs-1 separation of **0.80
sigma**. No power. **The programme's sharpest falsifiable claim — a boost that
rises outward — is not testable on eFEDS weak lensing**, and more groups would
not change it. |T| ~ T0 needs ~1e14 Msun inside a few hundred kpc: cluster cores.

## AL.5 Raw shear under no slip, and a genuinely positive result

All 496 systems, 3365 points, Sigma_s = 1 exactly, nothing fitted:

    law            chi2      Sigma_s it would need     eta
    Newton        3803.2            8.322            +15.64
    RAR           3588.4            0.981             +0.96
    AQUAL         3588.2            0.992             +0.98
    tidal A=7.5   4472.9            0.358             -0.28
    tidal A=16    5998.2            0.253             -0.49
    g_pred = 0    3865.0

**The RAR and AQUAL, with a0 frozen on galaxy rotation curves and no free
parameter of any kind, land on 3365 raw cluster shear points needing a lensing
response of 0.98 and 0.99 — no slip, to within 2%.** That is the cleanest
positive result the cluster work has produced.

> **CORRECTED IN RUN AS (§AS.2).** "No slip, to within 2%" conflates a central
> value with a precision. 0.981 is a central value 1.9% from unity; the
> uncertainty is a FACTOR OF TWO, as AL.7 says two sections below and as AL.9's
> null-corrected bracket of 1.06-2.17 shows. Also read "zero new fitted GRAVITY
> parameters" rather than "parameter-free" — the pipeline still carries X-ray
> profile fits, gas conversions, stellar masses, centring, source redshifts,
> shear calibration and distance factors. And RAR vs AQUAL here is ONE result
> reported twice, not two confirmations: in spherical symmetry AQUAL reduces to
> an algebraic mu-relation approximating the RAR by construction.

**Both tidal variants fit worse than predicting no lensing signal at all**, and
both require eta < 0, i.e. light bending the wrong way relative to matter.

## AL.6 A correction to Run AI

Stacking the SAME clusters against Chiu+2022's HSC profile:

    all 496 systems                    DECADE/HSC = 0.273  (-0.564 dex)
    top 50% by M_gas,500 (248)                      0.914  (-0.039 dex)
    top 20% (100)                                   0.875  (-0.058 dex)

**The 0.56 dex offset is sample composition, not shear calibration.**
Mass-matched, DECADE is good to about 0.05 dex. Run AI's inferred 0.2-0.4 dex
photo-z dilution came from comparing FITTED AMPLITUDES across two model setups
rather than comparing the data, and is withdrawn.

## AL.7 The shared-quantity null fires harder than anywhere yet

With the actual published errors, bracketed over three scalings because the
Vikhlinin parameters are strongly covariant and their covariance is not
published:

    error scale 0.25   E[est|H0] = -0.026 dex    -2.5 sigma_MC
    error scale 0.50               -0.125        -9.8
    error scale 1.00               -0.336       -17.0

**X-ray density-fit noise ALONE drags a fitted Sigma_s down by up to a factor
2.2, at 17 sigma.** Sixth artefact of this family, and the largest, because
Sigma_s is a pure amplitude with nothing to absorb it. Every Sigma_s must be read
against this null rather than against 1, and its factor-two width is the lane's
dominant uncertainty — a property of the published X-ray catalogue, not of the
shear. **Publishing the Vikhlinin covariance would take the slip measurement from
a factor of two to about 10%.**

## AL.8 Refsdal, and the question it actually answers

Required `Delta phi(SX-S1) = 3.5419 +- 0.0525 arcsec^2` from a 1.48% delay.

**The mass-sheet transform, measured:** image positions are moved by at most
1.4e-14 arcsec across a factor of four in lambda, while Delta phi scales exactly
linearly. **Image positions cannot see this closure change at all; delays see it
linearly.** Two genuinely independent handles, and they agree to 11-14% for every
law.

**One closure parameter is enough to bring EVERY law, including unmodified
Newton, onto 376.02 d.** A single time delay can never test a gravity law; it can
only measure the closure, and only with the law frozen and the lens model right.

The strong-lensing deficit is not baryon bookkeeping: closing the factor ~4 would
need about 8x the ACCEPT gas inside 80 kpc or 100x the catalogued stars.

## AL.9 The result that matters most

    law           Sigma_s WL (null-corrected)   Sigma_s SL (delay)   SL/WL
    Newton              9.04 - 18.45                 13.907        0.8 - 1.5
    RAR                 1.06 - 2.17                   4.113        1.9 - 3.9
    AQUAL               1.06 - 2.17                   4.136        1.9 - 3.9
    tidal A=7.5         0.39 - 0.81                   2.117        2.6 - 5.4
    tidal A=16          0.28 - 0.57                   1.574        2.8 - 5.6

**Newton is the only one of the five for which a single universal slip serves
both regimes** — its bracket contains 1. Every MOND-like law needs 2 to 6 times
more lensing response in the cluster core than in the group outskirts, which is a
new and specific statement of the cluster problem: not "MOND is short of mass"
but "MOND requires a regime-dependent photon coupling".

> **CORRECTED IN RUN AS (§AS.4).** Too strong. What is demonstrated is an
> effective CLOSURE MISMATCH, consistent with core-baryon modelling,
> line-of-sight structure, triaxial deprojection, mass-sheet degeneracy, the
> assumed closure, or a changing photon response. AL.8 in this same run already
> notes that one closure parameter brings every law including Newton onto
> Refsdal's 376.02 d, so the delay is not a model-free slip measurement.

**And the tidal gate makes it worse rather than better.** It multiplies the RAR
by 2.75 in the eFEDS groups where the shear already agreed without it, but by
only 1.88 at MACS J1149's 50-80 kpc where |T| is large and the gate is partly
off. **Its sign is backwards for the cluster problem.** Stated confound: the two
regimes differ in radius AND in host mass, so this is not yet a clean separation.

Two test bugs caught, one of which is worth propagating: D_dt was converted to
metres twice, giving Delta phi = 0.0000 exactly — **caught because the number was
impossible, not by any test.**

---

# Run AM — the pre-data admissibility compiler: 97.2% of the search was decidable without data

Full record in `work/wellnet-2026-09/compiler/REPORT.md`. Code
`gates.py`, `compile.py`, `retrospective.py`, `test_compiler.py`.
**No observational data of any kind is opened by this lane** — asserted
mechanically by intercepting `open`: 0 files opened. The only file read is the
tournament's own candidate list.

This answers the Stage 0 instruction to *"add automatic invariance gates for
potential zero-point, constant-K coordinate degeneracy, mass partition and
coarse-graining, reciprocal kernels, and variational consistency."*

## AM.1 Validation first: 35 tests, 35 passed, no disagreement with any verdict

The compiler was pointed at every family this programme has already decided.
AQUAL, QUMOND, QUMOND+RAR and Newton **ADMIT**; families B, C, D and E
**REJECT**, with the gates firing on the grounds the screen lane recorded.

Recorded numbers reproduced independently:

    Phi_K(x) = Phi_N(K^-1/2 x) residual        4.84e-16    (an identity)
    selective refinement slopes, p=0.25..2     0.7496/0.4996/0.2496/
                                               -0.00045/-0.5004/-1.0004
      recorded                                 0.7507/0.5007/0.2507/
                                               0.00067/-0.4993/-0.9994
    coherence slope, genuine kernel            -3.113   (recorded -3.11)
    coherence slope, family C p=1              -0.549   (recorded -0.55)
    coherence slope, pure row counting         +0.124   (recorded +0.12)
    family D lambda_min, N = 10 -> 800         3.397e-1 -> 8.301e-80
      recorded                                 3.4e-1   -> 8.3e-80
    family C cluster M_dyn, 1 row vs 10^4      +15.5%   (recorded +14%)

**8 of 8 recorded third-law violators fail Gate 4; 4 of 4 variational base laws
pass at round-off.** The gates are two-sided — Gate 4's admissions are the
non-trivial half, since in QUMOND form the law comes from an action iff
`K(u)u` is a gradient in `u = grad Phi_N`, which holds for `phi(|u|)I` and for
the field-direction structure and fails for anything reading Phi_N, the Hessian,
rho, a ball mass or a row list.

## AM.2 The retrospective: 3,036 of 3,123 rejected before any data

    3,123 candidates compiled in 31-47 s, one CPU core, caches cold

    REJECTED                       3,036 / 3,123  =  97.2%
      by gates 1 and 3 alone         1,701        =  54.5%
    FLAGGED convention-dependent       624        =  20.0%
    ADMITTED                            87        =   2.8%

The 87 admissions are 3 named base laws, 12 whose fitted amplitude is exactly
zero, and **72 with a live response, every one of them gn-gated** — QUMOND with a
redefined interpolating function. **Not one is a tournament survivor.** The
compiler and the tournament agree the grammar contains nothing new; they disagree
only about how much of that was knowable in advance.

| gate | kills alone | unique kills |
|---|---|---|
| 1 constant-K degeneracy | 149 | **52** |
| 2 potential gauge | 0 (flags 624) | 0 |
| 3 coarse graining | 1,560 | 0 |
| 4 reciprocity and action | 2,984 | **1,335** |

## AM.3 All 18 tournament survivors fail a gate — and the split is informative

**The seven `tensor_S` survivors carry p = 0, which selective refinement
forbids.** Measured `d ln(W1/W2)/d ln N = +1.000` against an admissible 0 — the
maximum possible representation dependence. At p = 0 a 4e11 Msun host counts no
more than a 1e9 dwarf, so how finely a *different* object is tabulated sets the
field near an untouched one. Run AH found this from the other side ("restore
p = 1 and the member violation goes 0.007 -> 0.528 dex") and called the escape a
coin flip at 4 of 8 realisations. **Gate 3 says it is not a coin flip; it is
inadmissible, and could have been known before a single member galaxy was
drawn.**

**The eleven tidal-gated survivors pass Gates 1, 2 and 3 and fail only Gate 4.**
|T| is a local second derivative with no boundary constant, a functional of the
smooth density with no row list, and it separates a galaxy from a cluster shell
by two orders of magnitude so no coordinate stretch can imitate it. **Run AH's
two structural arguments for preferring the tidal gate are independently
confirmed here, before any data.**

Gate 4's 2,984 rejections are a statement about the grammar, not about these
eighteen: every response except the gn-gated ones reads a functional of rho that
is not `grad Phi_N`. **Read Gate 4 as a to-do list, not an extermination** — a
variational completion or a declared momentum carrier is a prerequisite for the
whole grammar, not an optional extra for the winner. A symmetric Jacobian does
not prove a relativistic completion exists; it only fails to reject.

## AM.4 Gate 1 says what the constant-K degeneracy is degenerate WITH

    sqrt(det K) on GM          an Upsilon* offset of -0.0375 dex
                               against a measured Upsilon* uncertainty of 0.06
    eigenvalue ratios          an apparent axis ratio of 0.569, inside what
                               inclination, depth and deprojection supply
    isotropic K                axis ratio exactly 1 — NO shape signature at all,
                               a pure G rescale

And a sharp special case: the nonlocal invariant `qbar` is smoothed on the
declared global L_NL = 300 kpc, so across a galaxy's 10-30 kpc it does not move
at all — `spread(ln k_r) = 0.0` exactly, residual 4.6e-16 dex. **Any qbar-gated
response is a pure conductivity inside a galaxy, degenerate to round-off with the
mass-to-light ratio.**

## AM.5 The coarse-graining sign convention is settled — the two lanes swept different variables

The +1.0 to +1.5 discrepancy between the screen lane and the tournament is not an
error in either. The screen lane sweeps **the LAW's coherence length L at fixed
N**; the tournament's `coarse.py` sweeps **the PARTITION's nearest-neighbour
spacing**. Both computed here on the same three controls:

| law | screen-lane convention | tournament successive-step |
|---|---|---|
| genuine smoothing kernel | **-3.113** | **+2.387** |
| family C p = 1 | **-0.549** | **+2.140** |
| pure row counting | **+0.124** | **-4.069** |

**The two conventions order the controls in opposite directions.** A positive
value under one cannot be read as a positive value under the other, and every
future coarse-graining number in this programme must name its convention.

Catalogue perturbations, same lane: detection threshold 0.0927, mesh 4x 0.0133,
merge 0.00252, deblend 0.00134, **ICL transfer 8.4e-13**. Five of six exceed
tolerance and the exception is informative — S is a normalised average so a
uniform mass transfer divides out. **It is a cataloguer's geometric choices, not
their mass bookkeeping, that move this family.**

## AM.6 Gate 2 flags the tournament's own headline gate

Six boundary rules over 400 synthetic galaxies give a median |Phi_N| spanning
4.23e9 to 4.58e10 m^2 s^-2 — **1.035 dex of spread against a 0.90 dex off/on
margin**. For the tournament's depth gate the response spans **2.07 dex and the
on/off verdict itself changes**: it fires under `saddle`/`env_volume`/`inf`/
`flat_1Mpc` and not under `overdensity`/`scale_radius`. Gate 2 never eliminates;
it flags, and here it flags loudly.

## AM.7 A decade error in Run AH's probe table — mine to correct

**The ratios I recorded in Run AH do not follow from the medians I recorded
beside them.** 6.87e-32 / 3.66e-34 = **188, not 19**; 5.54e-31 / 3.66e-34 =
**1513, not 151**. Both annotations are exactly 10x too small, consistent with a
shell median of 3.66e**-33** rather than 3.66e-34. One of the two columns carries
a decade error and the table as printed is internally inconsistent.

The compiler's independent probes give shell 8.01e-34, field 9.89e-32, member
9.88e-32, so member/shell = **123** against the recorded annotation of 151 —
close. **The qualitative conclusion is unaffected and independently confirmed:**
the tidal invariant orders the member two orders of magnitude above the shell
while potential depth places them within a factor of **1.02** (9.04e11 against
8.89e11). The tidal gate's structural advantage stands; the printed ratio does
not.

Separately, the compiler's member galaxy is *not* 8x tidally louder than a field
galaxy — 9.88e-32 against 9.89e-32, factor 1.001. At 10-30 kpc from a 5e10 Msun
galaxy its own tide dominates whatever the environment. This is a limitation of
the compiler's probe geometry rather than a claim against Run AH, and it does not
touch the member/shell ordering the gates use.

## AM.8 Throughput — the gates are cheap enough to run in front of everything

    distinct Gate 3 families in 3,123 settings         8
    distinct Gate 4 structural families              253
    cost of one previously unseen family            0.075 s
    structural pass over all 3,123                   3.2 s = 983/s
    INHERITING a known family's verdict          4.2-6.9e6 settings/s
    settings surviving the structural gates          139
    Gate 1 on that residue                           1.9 s
    the Stage-1 screen it must front              2.05e6/s

**The family-verdict lookup runs at 2-3x the Stage-1 screen's own throughput**,
and Gate 1 then runs only on the 139 survivors. The family count does not grow
with the setting count, so on a 1e9-setting Stage 1 this is about 3.5 minutes
against the screen's 8. There is no reason not to run it in front of every future
search.

## AM.9 What this does NOT establish

Nothing here says an admitted candidate fits anything — 72 of the 87 admissions
are QUMOND with a redefined nu, which the tournament's data screens already
killed. Gate 4 shows a law AS WRITTEN is not the Euler-Lagrange system of an
action with Phi_N Newtonian; promoting the gating field to a dynamical one might
produce one, at the price of changing the law. Gates 1 and 4 use the spherical
reduction the tournament's channels score, so a candidate whose only signature is
the phase of a shear quadrupole would not be seen — deliberate, since a full
tensor solve costs a PDE per candidate. And the gate has no power below its
measured floor (2.10e-3 FD), which is reported rather than hidden.

---

# Run AN — the four environmental variables, and a control that beats all of them

Full record in `work/wellnet-2026-09/envvars/REPORT.md` (658 lines, every number
rendered from the JSONs by `report.py`, none transcribed). Code `envvars.py`,
`fixedeffects.py`, `fragility.py`. Arena: **496 eFEDS systems × 3,365
(system, radial-bin) DECADE shear points**, reused unmodified with Run AI's
declared even/odd-by-ID split. Ingest counts asserted (542x19, 542x40, 5411x13);
the M_gas,500 gate re-run at median 0.9994, 0.0469 dex, PASS. KiDS and the wide
binaries never loaded.

This answers the Stage 0 instruction to separate *"potential depth, the vector
external field, a directionless well sum, and the tidal tensor"* into four
variables tested on one sample with one set of folds.

## AN.1 Two structural results, worth more than any fit

**1. V2 is external-only by theorem, and the cancellation costs 2.41 dex.**
`sum_a G M_a d_a/|d_a|^3` over *all* mass is exactly `g_bar` — Newton's shell
theorem, not an approximation — so a V2 that includes the object's own mass is
identically the acceleration `f(g_bar, r)` already carries, and the test is
vacuous. External-only, the real catalogue's external field is **5.5e-5 of a0**
and 8.5e-3 of local g_bar. The directionless sum V3 has no shell theorem: its
self term survives at **1.42 x g_bar** with a different radial shape. The
distinction the brief asked for is forced, not chosen.

**2. On this sample, "environmental" and "radially resolved" are mutually
exclusive.** The variables with radial structure inside an object — V1 1.24 dex,
V3 1.97, V4a 2.43 — are **98.4 / 96.8 / 97.2% reconstructible from a quadratic in
(log g_bar, log r)** at the shear radii, independently reproducing Run AI's
0.9863 / 0.087 dex for V1 (this lane gets 0.9845 / 0.094). The two genuinely
orthogonal to (g_bar, r) — V2 at R^2 = 0.12 and external tidal V4d at 0.08 — vary
by only **0.003 and 0.021 dex** inside an object.

That is a theorem too: an external field is constant across a small object to
leading order, and **its first radial derivative IS the external tidal tensor**,
measured here at 2.9e-4 of the internal one. **V2 can only ever be tested between
objects.**

## AN.2 The four-variable table — and the control decides it

beta in dex of log g per 1 sd of the variable, each against its own simulated
null:

| variable | within-object beta / z | within-class beta / z |
|---|---|---|
| V1 potential depth | +0.254 / +0.31 | +0.032 / **+4.93** |
| V2 vector g_ext | -0.173 / -0.15 | +0.037 / +0.35 |
| V3 directionless W | +0.129 / +0.30 | +0.149 / **+4.49** |
| V4a tidal magnitude | +0.175 / +0.55 | +0.241 / **+3.70** |
| V4b tidal shape | +0.079 / +0.24 | +0.145 / +2.17 |
| V4d external tidal | -2.0 (grid edge) | +0.038 / +0.38 |
| — radius tilt (control) | -0.156 / -1.14 | -0.328 / -1.44 |
| — **acceleration tilt (control)** | +0.088 / +0.42 | +0.117 / **+5.97** |

**The largest within-class z belongs to a bare acceleration tilt with no
environmental content whatsoever.** Every density-fit-derived regressor sits above
a strongly negative null, and the data sit above that null — but so does a
control that says nothing about environment, and by more. **The ordering does not
favour environment.** Within-object, every variable is inside 0.6 sigma.

Frozen held-out transfer: of 32 evaluations, exactly one reaches a negative dBIC
(V3 within-class, **-0.87** — below Jeffreys' "not worth a bare mention"), with
the environment-free acceleration tilt at +0.45 beside it.

## AN.3 The shared-quantity null fires harder than in Run AI

40 realisations redrawing every published density parameter plus
sigma_z = 0.005(1+z):

    E[beta | H0], V1 within-class    -0.306 +- 0.011    = -28 sigma_MC from zero

from X-ray fit noise alone. And the Fisher errors are badly optimistic wherever
the variable depends on the fit: **V2 within-object has Fisher sigma = 0.033
against a null sd of 1.000 — a factor of 30.** A Fisher-error analysis would have
announced a 5-sigma within-object detection of the external field that is
entirely propagated X-ray fit noise.

**Seventh instance of the shared-quantity family.** The fragility diagnostic
(new in this lane) explains the mechanism: for V2 and V4d within-object, **2
objects out of 248 carry 69% of the Fisher information** and the top 5% carry
over 99%; dropping 10% of objects moves V2's beta from -0.173 to **+1.797**. For
V1/V3/V4a/V4b the top 1% carry only 5-10%.

## AN.4 Coarse-graining: two pass, one fails as a catalogue quantity

Uniform x4 refinement gives scatter ~6e-17 dex for p = 0.5, 1 and 2 alike —
**toothless, exactly as the brief warned**, and now measured rather than
asserted.

    V2, V4d   PASS.  Resolving each neighbour into 8-512 pieces over 1.5 Mpc
                     moves |g_ext| by at most 0.034 dex, settling at 0.003,
                     against a 0.61 dex between-object spread.

    V3        FAILS as a catalogue quantity.  One row per object — how the
                     catalogue actually represents mass — puts W wrong by
                     0.09-1.06 dex against a 0.66 dex between-object spread.
                     Drift falls as N^-0.42...N^-0.46 with NO PLATEAU:
                     convergent-quadrature, not coherence-limited, so no
                     physical scale emerges and the answer is set by the
                     cataloguer.  ~2e2-2e4 rows per object for 0.01 dex.

V3 survives in this lane only because eFEDS publishes a resolved density fit for
the central object. **W is a quantity a mass map can deliver, not a catalogue.**

## AN.5 Power, stated without softening

Responsiveness `d(beta_hat)/d(beta_inj)` is **well below 1** — +0.07 to +0.45
within-class, unresolved within-object at +-0.5 to 1.1 — because an injected
effect lives on the true density profile while the analyst measures it through
the published one. For V1, V3 and V4a within-object the slope is consistent with
zero at n = 18. **Where the slope is consistent with zero this lane has NOT set an
upper limit; it has only failed to find the effect.**

Also stated: the external field is 4-5 dex below a0, so V2 and V4d are tested as
*orderings*, not amplitudes; Bahar+2022 publishes no redshift error, and the
assumed sigma_z implies a 29 Mpc line-of-sight error against a 33 Mpc median
neighbour separation; the well network is an X-ray flux-limited list, not a mass
census.

## AN.6 A correction to the brief's premise, and a re-description of the tidal gate

V4 on a spherical baryonic source collapses analytically to

    |T~| = sqrt(6) (g/r) |1 - rho/<rho>|,     lambda_r/lambda_t = -2 identically

So the tidal **shape** carries exactly one bit, the principal eigenvector is
radial by construction, and the eigenvector information is degenerate until an
external tide breaks it.

**Consequence for Run AH: the 151x ordering that separates a cluster member from
a cluster shell is `g/r` evaluated at two very different radii, not an
environmental quantity.** The tidal gate's advantage over the potential gate is
an advantage of `g/r` over `Phi` — it should be described as a **local-kinematic
gate**, not an environmental one.

This does not overturn Run AH's member screen and does not contradict Run AM,
which independently confirmed the gate is clean on the invariance gates. It
changes what the gate is called and therefore what it would be evidence *for*: a
law that reads the local ratio of acceleration to radius, which is a much less
exotic object than a law that reads its surroundings.

---

# Run AO — axis provenance and the 2-D shear phase, and a detector bug the blindness theorem caught

Full record in `work/wellnet-2026-09/axis-2d/REPORT.md`. Code `axis_power.py`,
`shear2d.py`, `selection.py`, `amplitudes.py`, `crosscheck.py`. The failing
pre-repair run is kept in `prerepair/` for audit rather than deleted.

This answers the Stage 0 instruction to *"separate the three axis provenances and
give each its own power surface"* and to *"score the 2-D shear phase, not the
azimuthal average."*

## AO.1 The detector was measuring extra scalar flexibility and calling it a tensor

The first version of the statistic **failed the spherical-limit check, and it was
a real bug, not a numerical artefact.** Source-axis power sat at 0.42/0.62/0.24
at axis ratio 0.970 and was flat in axis ratio — it should have collapsed.

Cause: in spherical symmetry `ghat` is an eigenvector of K, so

    K grad Phi_N = (ghat^T K ghat) grad Phi_N

The tensor atom acts as a **scalar rescaling with a radial profile the 70-atom
scalar bank cannot reproduce**. The blindness theorem says the transverse
*eigenvalue* is unobservable; it does not say the tensor *atom* is, and the
detector was collecting the difference.

Repair (`flux_orthogonal`): every non-identity atom becomes
`K_perp = B - (ghat^T B ghat) I`, which can only turn the flux, never rescale it.
It also kills the QUMOND `ghat ghat^T` degeneracy with no special case — the same
degeneracy Run AA proved by hand now falls out of the construction.

## AO.2 The theorem, measured — and a correction to Run AC

Flux-turning fraction, the observable part of the response:

| provenance | q=0.50 | 0.65 | 0.80 | 0.90 | 0.97 |
|---|---|---|---|---|---|
| source | 0.1137 | 0.0839 | 0.0538 | 0.0307 | **0.0215** |
| external | 0.7813 | 0.7786 | 0.7764 | 0.7753 | **0.7747** |
| network | 0.4461 | 0.3709 | 0.3386 | 0.3449 | 0.3472 |

**Source-axis collapses by 5.3x as the source rounds; external does not collapse
at all** — it has no reason to, since its axis is not the source's.

**Run AC's near-spherical control injected the `dd` (external) basis, so its
power was never supposed to collapse.** That control was measuring the wrong
hypothesis, not failing one. Run AC's surface stands for what it measured; its
near-spherical row should not be read as a validation of source-axis blindness.

Power over 15 geometry x noise rows, 400 calibration + **400 untouched audit** +
50 injections per cell: external-axis power rises 0.76 -> 1.00 with axis ratio at
top amplitude, while **source-axis and network power sit at the test size
everywhere** — not a search failure but a shortage of *observable* amplitude.
Audit FPR over 81 cells: median 0.055, mean 0.056, range 0.020-0.113 against a
binomial SE of 0.011. The Stage 0 null (five scalar families, deprojection,
miscentring, M/L gradients, shear m- and c-error) is carried throughout.

**Design lesson worth propagating: a misspecified axis is a null detector.** With
a single fixed direction in the bank, an external injection tilted 45 deg gave
power 0.03 at every amplitude while the aligned case reached 1.00. Hence five
spanning directions, with the look-elsewhere cost pushed through the calibration.

Coarse-graining is a partial failure: 1 -> 10 wells changes the network response
by 1.65x its own RMS — **the cataloguer sets the physics at the coarse end** —
but 60 -> 300 changes the drift by only 0.138, so the continuum limit is good.

## AO.3 The 2-D shear phase test: 27 DEV clusters, 181,949 background sources

| channel | alpha | randomisation sigma | p |
|---|---|---|---|
| **a2s, predicted monopole (THE TEST)** | +0.0136 | 0.6182 | 0.98 |
| a2s, per-cluster fixed effects | +0.3129 | 0.7608 | 0.68 |
| a2c, source-aligned (positive control) | +0.0749 | (0.1076 WLS) | — |
| x2s, cross / B-mode | +0.7427 | 0.6827 | 0.30 |
| CTRL (136 near-round) a2s | +0.2006 | 0.2582 | 0.43 |

Gates all pass: tangential monopole +3.2 sigma (DEV) / +8.4 sigma (CTRL); cross
monopole -1.0 / +1.1 sigma; the estimator agrees with the eFEDS lane's own
profiles at chi2/system 0.23 and 0.38; M_dyn monotone 27/27 and 136/136. The
neighbour uniform-shear column is fitted, not waved away.

**Two methodological catches, both of which change a reading:**

1. The measured-monopole variant looked **4x tighter** (-0.179 +- 0.153).
   End-to-end injection into the real data gives its response slope as **0.072** —
   regression dilution by a factor of 14, so it is really **3.7x worse**. The
   predicted-monopole design has slope **1.0000**. An error bar is meaningless
   without its response slope.

2. **The positive control does not fire.** a2c is +0.075 +- 0.108, consistent
   with zero and with the ~0.13 the baryons alone should produce. **The null
   therefore cannot be read as evidence of absence** — the instrument has not
   demonstrated it can see a signal it is known to contain.

## AO.4 Selection frozen by geometry that never touches a shape column

DEV 27 / CTRL 136 from 542 eFEDS systems. Axes come from member-galaxy
**positions** — the selection pass never requests a shape column — and from a
large-scale tidal axis built from neighbouring X-ray systems. The member slice
and the scored background slice are **disjoint by >= 0.04 in z**. Median |Delta|
= 47.7 deg; deconvolved RMS member ellipticity 0.584 (DEV) against 0.000 (CTRL)
and 0.276 (parent). Mask systematic correlation -0.176. Frozen at SHA-256
`aafc7def...`, which `shear2d.py` verifies before scoring.

**No X-ray position angle exists for eFEDS** — Bahar's fits are spherical. Stated,
not substituted. One control cut was changed (raw to noise-debiased amplitude)
before any shear was computed, and the change is recorded in the source.

## AO.5 The decisive structural result: the grammar does not contain this hypothesis

**Not one of the 3,123 tournament candidates carries an external tidal axis.**

    network   1560      source   780      isotropic   783      EXTERNAL   0

And of the 18 survivors, the two `tensor_d` ones satisfy
`K grad Phi_N = exp(2AW/3) grad Phi_N` **exactly** — scalar rescalings wearing a
tensor's name, predicting zero quadrupole of any phase. The rest predict
a2c/a0 of 0.020-0.257 and **a2s/a0 = 0**, which is to say they land entirely in
the channel that is degenerate with baryonic ellipticity.

**Could this test have seen it anyway?** End-to-end injection gives a 95%
exclusion of |alpha| < 1.158, which at the median |sin 2Delta| = 0.801 and the
measured C = 0.440 means `e_kappa < 2.11` — **above the geometric maximum of 1.**
**No external-axis tensor of any amplitude is excluded by this test.** Reaching
`e_kappa = 0.2` needs 111x the effective source count; DECADE gives 4.4 usable
sources per square arcmin here.

**The phase channel is the right instrument, correctly built and calibrated,
pointed at a hypothesis the searched grammar does not contain, with a sample too
shallow to have seen it if it did.** All three of those are worth fixing
separately, and the first is the cheapest: adding an external-axis provenance to
the bank costs nothing but a basis element.

## AO.6 A flagged discrepancy in shared code, not yet adjudicated

The lane reports that `tw_core.mond_invert`'s `rar` branch appears to give
deep-MOND `g ~ k^(-1/4)` where its own docstring and the `aqual` branch give
`k^(-3/4)` — the argument `F/(k^1.5 a0)` looking like it should be
`F/(k^0.5 a0)`. If that is right it puts the fitted amplitudes of the rar-based
and aqual-based halves of the tournament on different footings. **Recorded here
as flagged, not as established**; it is chased in the next run before any
amplitude that depends on it is quoted again.

---

# Run AP — the formation and stability gate: the tidal gate's linear response is identically zero

Full record in `work/wellnet-2026-09/formation/REPORT.md`. Code
`linear_response.py`, `growth.py`, `test_solver.py`. This answers the Stage 0
instruction that *"a law that cannot form structure is not a candidate however
well it fits a rotation curve."*

## AP.1 A correction to my own framing of the tidal candidate

I described the tidal gate as divergent at |T| = 0. **It is not.** The form the
tournament froze is `inv`, which `tw_core.W_of` implements as `W = 1/(1 + I^m)`
— **bounded by 1**. At |T| = 0 the gate *saturates at its maximum*: W = 1 and
a0 -> 17 a0 = 1.704e-9 m/s^2. The homogeneous background exists and the gate is
fully ON in it.

That correction is what makes the next result possible to state.

## AP.2 The mechanism that won the tournament does nothing at linear order

|T| is a norm and vanishes in the homogeneous background, so `I ~ |delta|`. With
the frozen **even** exponent m = 2, the gate's departure from saturation goes as
delta^2:

    measured log-log slope of (1 - W) against delta   1.9999998
    analytic                                          2

**At linear order the tidal candidate is exactly AQUAL with a0 -> 17 a0.** The
tidal gate — the thing that separated a member galaxy from a cluster shell by two
orders of magnitude, the thing Run AM cleared on Gates 1-3 — contributes
identically nothing to linear growth.

It is not inert at all times, because the gate's expansion parameter is I, not
delta, and I is not small: at recombination delta = 1e-5 already gives I = 2.89.
So a0_eff **runs from 3.28 a0 at z = 1000 to 17.0 a0 today**. That is a real,
global, parameter-free time dependence, and it is the only cosmological signature
the candidate has.

## AP.3 Both depth-gated tensors have no homogeneous background at all — three ways

1. **|Phi_N| is pure gauge.** The Jeans-swindle rule gives W = 7e-15 to 0.055;
   any external-reference rule gives W = 1.000000. **In cosmology the boundary
   ambiguity is the ENTIRE gate range, not Run AH's 0.87 dex.** Run AM's Gate 2
   flagged this on galaxies; in cosmology it is total.
2. **S is 0/0 in the continuum.** Measured with unmodified `wellnet.S_tensor`, N
   wells give shot noise with fitted slope **-0.489** and a factor **269** between
   N = 1 and N = 1e5.
3. **The frozen kernel is not normalisable.** Both settings are `plaw q=1, s=2`,
   so w ~ r^-2 and the integral of w r^2 dr grows as rmax^1.083 while the response
   falls as rmax^-1.098. **On an unbounded background S = 0 identically.**

And a caveat that applies to everything except Newton: the homogeneous state is a
solution but **not a unique one**. The deep-MOND source goes as delta^(1/2),
non-Lipschitz at zero, so delta = 0 and delta = (C^2/144) t^4 share initial data
(residual 5.4e-9). **"Linearise and read an eigenvalue" is not an available
operation in this class of theory** — which is why this lane integrates instead.

## AP.4 Growth: every MOND-family candidate is fast enough, Newton is not

delta(z=0), and the scale factor at which delta = 1, pancake geometry:

| scale | tidal scalar | S p=0 | S literal p=1 | AQUAL | QUMOND | Newton |
|---|---|---|---|---|---|---|
| 1 Mpc | 270.8 / 0.073 | 21.8 / 0.176 | 22.6 / 0.173 | 22.6 / 0.173 | 23.2 / 0.171 | 1.18e-4 / — |
| 10 Mpc | 34.4 / 0.149 | 2.26 / 0.568 | 2.34 / 0.556 | 2.34 / 0.556 | 2.40 / 0.548 | 1.18e-4 / — |
| 100 Mpc | 3.69 / 0.433 | 0.243 / — | 0.252 / — | 0.252 / — | 0.258 / — | 1.18e-4 / — |
| 1000 Mpc | 0.394 / — | 0.0283 / — | 0.0293 / — | 0.0293 / — | 0.0299 / — | 1.18e-4 / — |

**Newton on baryons alone fails by 3-4 orders** — amplification 11.8 against the
~1e5 required. **The depth-gated tensors reproduce AQUAL to every printed digit**
(p1_literal is identical; p0 differs only through its own fitted a0), which is the
cosmological restatement of Run AO's finding that they are scalar rescalings
wearing a tensor's name. The tidal gate is worth x9.0-14.9 — **entirely through
a0_eff**, consistent with AP.2.

Integrator validated against the closed-form LCDM growth factor to **2.0e-8**.

**The deep-MOND attractor erases the primordial spectrum.** Analytic
y* = (3f/20)^2 = 5.4686e-5 against measured 5.4591e-5 (ratio 0.9983); exponent
2.016 against 2; a 1e4 range in initial amplitude compresses to **1.57**; final
mode amplitude proportional to k (0.933 against 1). Whatever the initial
conditions were, this class of law forgets them.

## AP.5 The tidal candidate is ANTI-Zel'dovich — a new and awkward prediction

The concern going in was that a modified law would overproduce filaments and
pancakes. **The opposite is measured:**

    sphere / pancake     1.6537    (analytic 5/3)
    filament / pancake   1.4903    (analytic 3/2)
    Newton               1.0000    exactly

and **the tidal gate pushes sphere/pancake to 4.90 at z = 999**, because an
isotropic configuration has |T| = 0 and therefore receives the *full* 17x boost
while a pancake receives only 3.29x. **The gate systematically favours collapse
to spheres over collapse to sheets** — backwards from the observed cosmic web,
and a specific, falsifiable, parameter-free consequence of the same saturation
that makes AP.2 true.

Local anisotropy is present and mild (fastest-growing mode has k perpendicular to
g; ratio 1.9603 AQUAL, 1.9610 QUMOND, 1.9773 tidal, cross-checked against full
nonlinear solves to 0.33%; AQUAL and QUMOND differ by 8.2% at intermediate
angles). Global statistical isotropy holds, ensemble quadrupole falling as
N^-0.557. **But the tensors manufacture a spurious global axis out of catalogue
shot noise: growth anisotropy 12.4 at 100 wells, 1.10 at 1e5 — set by particle
count, zero in the continuum.** Seventh instance of a catalogue artefact
masquerading as physics. The tidal gate's own averaging bracket is only 0.0128
dex, unlike the cluster channel.

## AP.6 No UV cutoff, and a momentum violation with a decisive signature

No finite-time blow-up (delta ~ t^4) and the growth rate is bounded (f = 1.73 to
2.01 over four decades in k). But **Q ~ k^0.499: there is no UV cutoff in the
gravity law itself.** Both limits: IR finite and *Newtonian* for every candidate;
UV divergent for every candidate but Newton. The well-network form factor is the
one band-limited response (IR slope 2.000, UV -2.041, peak at kL = 0.23),
confirmed against unmodified `wellnet.S_tensor` at 0.5/1.6/0.9/1.3 sigma.

Momentum, analytically: `int rho grad psi = (1/4piG) int (dPsi/da0) grad a0`,
**zero if and only if a0 is constant.** A running a0 therefore *must* violate it.
Measured against the base law with the response off, the tidal candidate gives
**1.52e-4 +- 2.47e-4 against an AQUAL null of 1.65e-6 — a ratio of 92.**

The discriminator that settles it is not the amplitude but the convergence:
**the null falls as h^3.25 while the violation is flat at h^0.02.** A
discretisation artefact vanishes with the grid; this does not. It peaks at
z ~ 32, exactly where the gate is switching. The physical consequence is small
but real: a spurious bulk velocity of 0.33 +- 0.54 km/s per Hubble time in a
40 Mpc box.

## AP.7 Verification, and four test bugs worth propagating

Six solver gates pass: convergence order 1.99, operator symmetry 2.7e-15, flux
conservation 3.5e-19, 1-D AQUAL 1.4e-3. All seven headline statistics pass the
responsiveness gate with spreads printed. Imported modules' SHA-256s match the
tournament's recorded hashes.

Four bugs caught in the lane's own tests:

  * **Picard iteration from psi = 0 converges to the wrong fixed point**, because
    mu(0) = 0 — the iteration is not a contraction at the origin.
  * **Face-averaged cross terms are not self-adjoint** (1.9e-2), which breaks the
    variational structure the solver assumes.
  * **A sign error that the k-perpendicular test still passed at 0.14%** while
    k-parallel was wrong by 6.6x. A single-geometry test would have shipped it.
  * **A 0.53 convergence order that is not a discretisation error at all** — it is
    the square-root cusp of the deep-MOND field at every source node. Correct
    diagnosis rather than a papered-over tolerance.

## AP.8 What the linear gate cannot decide

Background expansion (the 10 Mpc amplitude moves by x130 across the three
carried), initial conditions, the power spectrum (single-mode scaling only), the
CMB, lensing, and whether the momentum violation is observationally relevant.
Nothing about the tensors' cosmology can be decided until a boundary rule and a
universal coherence scale are *declared*. Nothing past delta ~ 1 (the tidal
candidate leaves the regime at 100 Mpc by a = 0.43).

**And the neglected second-order tensor term is not small**: it reaches order
unity at delta = 0.21 for a 6 Mpc catalogue radius and delta = 111 for a 3000 Mpc
one — **a factor of 500 set by the cataloguer**, which is the same disease Gate 3
names.

## AP.9 Verdict

Nothing promoted, nothing eliminated. But the standing of the programme's best
candidate changes character: **the strongest tournament survivor reduces at linear
order to AQUAL with a running a0 (3.28 -> 17.0), the well-network mechanism is
invisible to a linear cosmological gate twice over, the only spontaneous global
axis in the whole set is an artefact of catalogue resolution, and the tidal gate
makes structure formation anti-Zel'dovich.** That last is the first cosmological
prediction this programme has that could be checked against a survey.

---

# Run AQ — two corrections: a real bug in half the tournament, and a wrong claim of mine about eps_T

A corrections run. Nothing new was searched; two things already recorded were
checked, and one of them was wrong in the code while the other was wrong in my
write-up.

## AQ.1 `tw_core.mond_invert` scored the `rar` half of the tournament on the wrong k-scaling

Flagged by the axis-provenance lane (AO.6) and confirmed here. The `rar` branch
read

    nu_rar(F / (k ** 1.5 * a0)) * F / k

Within the family `nu(F/(k^p a0)) F/k`, the exponent p is **fixed, not chosen**:
k = 1 reproduces plain RAR for every p, and the Newtonian limit gives F/k for
every p, but deep MOND gives `g ~ sqrt(F a0) k^(p/2 - 1)`, so matching the AQUAL
branch's `k^-3/4` requires **p = 1/2 uniquely**. Measured:

    d ln g / d ln k, deep MOND (F = 1e-4 a0)
      newton                     -1.0000
      aqual                      -0.7513
      rar  AS CODED              -0.2543     <-- its own comment claimed -0.75
      rar  with k**0.5           -0.7513     <-- agrees with aqual to 2.2e-06

**The error is exactly zero at k = 1** (relative deviation 0.00e+00 over seven
decades in F) and exactly zero in both Newtonian limits. That is why nothing
caught it: every `scalar_a0` candidate has `k_radial_pointwise = 1` identically,
so the entire tidal-gated scalar family — the programme's best candidate — was
never touched by it.

Fixed to `k ** 0.5`. `tournament/test_mond_invert.py` added: it asserts the
deep-MOND slope for all three bases, asserts **|rar − aqual| ≤ 1e-2 whatever the
shared value turns out to be**, asserts the k = 1 identity to 1e-14 over seven
decades, and asserts monotonicity in k. It **fails on the old code** (slope
−0.2543, gap 0.4969) and passes on the new.

## AQ.2 The blast radius, and a control that came out perfect

    exposed  (base='rar' AND struct != scalar_a0)   1,365 / 3,123  =  43.7%
    inert    (aqual, newton, or scalar_a0)          1,758

The whole tournament was re-scored from scratch (`--fresh`; the stale checkpoint
was moved aside, not reused) in 1,009 s, and the old run preserved as
`tournament_prefix_k15.json`.

    CONTROL -- inert candidates whose J moved:  0  of 1,758

Bit-identical, as the k = 1 identity requires. That is the check that makes the
rest of the comparison trustworthy.

    EXPOSED -- J change over the 1,239 scorable
      median -0.6509   mean -19.7563   sd 83.2320   |max| 1013.1223
      improved 745     worsened 494     unchanged 0

## AQ.3 What actually changed: an exact degeneracy that the bug was hiding

Survivor count 18 -> 26. But the survivor *count* is the wrong unit, because
every candidate appears twice — once per base. Counting distinct laws:

    OLD:  13 stems survive as aqual,  5 as rar,  4 under BOTH,  10 asymmetric
    NEW:  13 stems survive as aqual, 13 as rar, 13 under BOTH,   0 asymmetric

    aqual -> rar J gap over paired survivors
      OLD   median +0.0008   max +0.2004   (plus one at 10.16 vs 2.35, one nan)
      NEW   median +0.0017   max +0.0069

**AQUAL and QUMOND+RAR now agree to within 0.007 in J on every surviving law**,
which is what should happen — they are two formulations of nearly the same
physics, and their agreement is a strong internal consistency check that the
programme did not previously have. The bug was suppressing it, and had cost the
`rar` half 9 of its 13 laws.

**The tournament's conclusion does not move.** The same 13 distinct laws survive,
with the same structures (5 tensor_S, 3 scalar_a0, 2 tensor_d, 2 iso_K,
1 tensor_T per base), and the verdict remains promote nothing. All ten tensor_S
survivors still carry the `plaw_p0q1s2_L300` kernel, so **Run AM's Gate 3 finding
— p = 0 is inadmissible under selective refinement — applies unchanged, now to
ten rather than seven.** The six tidal-gated scalar survivors were provably
unaffected.

What was wrong was not the answer but half the scoreboard, for two days, in a
quantity that was reported.

## AQ.4 The eps_T claim I carried forward was wrong

I recorded, from Run AG, that `families.tidal_hat` uses `eps_T = 3.9e-31`, *"190x
the actual |T0| = 5.3e-33, suppressing That by 97x in galaxy outskirts."*
Measured on the screen's own galaxy:

| r (kpc) | \|T0\| (s^-2) | \|T0\|/eps_T | \|That\| |
|---|---|---|---|
| 0.5 | 3.343e-29 | 85.97 | 0.9999 |
| 2 | 1.636e-29 | 42.06 | 0.9997 |
| 5 | 2.383e-30 | 6.128 | 0.9869 |
| 10 | 4.535e-31 | 1.166 | 0.7591 |
| 20 | 6.667e-32 | 0.1714 | 0.1690 |
| 40 | 8.646e-33 | 0.0222 | 0.0222 |
| 80 | 1.081e-33 | 0.0028 | 0.0028 |

**Median |T0| over the 2-40 kpc probe range is 4.53e-31 against a frozen eps_T of
3.89e-31 — a ratio of 0.9. The frozen value is well scaled, not 190x too large.**
The 5.3e-33 I quoted is the value at roughly 45 kpc, i.e. the far outskirt, not
the typical one. The suppression is real but confined to r > 20 kpc, and it is
what a norm with a floor *does*.

## AQ.5 The re-run anyway, and why the verdict cannot move

Full Stage-1 screen, both couplings, seven eps_T values spanning frozen down to
**exactly zero**:

| eps_T | E1 gain | E1 S10 | E2 gain | E2 S10 |
|---|---|---|---|---|
| frozen (3.89e-31) | 1.079 | 0.197 | 1.257 | 0.399 |
| matched (4.53e-31) | 1.068 | 0.189 | 1.218 | 0.389 |
| x1e-1 | 1.435 | 0.287 | 2.957 | 0.587 |
| x1e-2 | 1.503 | 0.297 | 3.398 | 0.628 |
| x1e-3 | 1.504 | 0.300 | 3.403 | 0.631 |
| x1e-4 | 1.504 | 0.300 | 3.403 | 0.631 |
| **zero** | **1.504** | **0.300** | **3.403** | **0.631** |

**All fourteen runs FAIL, with the identical failure set every time**: S6
Newtonian limit, S10 reciprocity, S11 and S11b coarse-graining, S13 coherence.

Two things are visible. **Lowering eps_T buys gain and pays for it in the third
law, in lockstep** — E1 goes 1.079 -> 1.504 in gain while its reciprocity
violation goes 0.197 -> 0.300; E2 goes 1.257 -> 3.403 while its violation goes
0.399 -> 0.631. And **everything saturates by eps_T = 1e-3 x frozen**, so there
is no unexplored corner at smaller values.

The reason is structural, and independent of eps_T entirely: `|That|_F <= 1` by
construction, so `K = exp(fT That)` has maximum eigenvalue at most `exp(fT)` —
1.649 for E1, 4.482 for E2. **Family E's gain is bounded at every eps_T,
including zero, so the boundedness theorem disposes of it whatever the
normalisation.** eps_T controls *where* the response switches on, never *how
strong it can ever be*. The re-run was owed and is now done; it changes nothing,
and it was worth knowing that it changes nothing for a reason rather than by
coincidence.

---

# Run AR — the same-object orthogonal-orbit test: blind by theorem, and a fourth blindness result

Full record in `work/wellnet-2026-09/orthogonal/REPORT.md`. Code `orbit_model.py`,
`adyn_same_object.py`, `warp_pool.py`. This answers the Stage 0 instruction to
*"find one object whose in-plane and out-of-plane response are both measured, and
score the ratio rather than either leg."*

## AR.1 A_dyn is blind to three of the four candidates BY THEOREM

Any law of the form `g = F(scalar invariants) grad Phi_N` gives

    A_dyn == 1  exactly

so **Newton, the algebraic RAR, and the tidal-gated scalar are all identically
invisible to this test** — |T| is a scalar, so the programme's best candidate is
in the blind set. Only the curl field of a *solved nonlinear field equation*
moves A_dyn, which is why the lane solved QUMOND (linear, one solve) and AQUAL
(damped Picard, 13-14 iterations) rather than assuming a completion.

**Fourth blindness theorem in this programme**, after spherical blindness, the
QUMOND tensor degeneracy, and boundedness. The pattern is now hard to miss: the
observables this programme keeps reaching for are insensitive to the structures
it keeps proposing, and the insensitivity is provable in advance each time.

| law | A_dyn algebraic | predicted (field completion) | at 100 kpc | measured |
|---|---|---|---|---|
| Newton (control) | 1 exactly | 1.0011 | 1.0015 | 0.994 +- 0.125 |
| RAR | **1 exactly** | 1.0062 (QUMOND) | 1.0019 | 1.333 +- 0.108 |
| AQUAL | **1 exactly** | 1.0014 | 1.0028 | 1.243 +- 0.108 |
| tidal scalar A=+16 | **1 exactly** | 1.0026 | **1.0437** | 1.247 +- 0.114 |
| well-network tensor_S | not scalar | 1.0110 | 1.0032 | 1.347 +- 0.116 |

Reference point R = 11.28 kpc, |z| = 3.59 kpc, set by tracer geometry; every
prediction written before any stream met any model.

## AR.2 The power statement, which is the actual result

End-to-end simulation through the whole chain: null mean 1.0004, sd 0.0408,
calibration and audit sets consistent at KS p = 0.26. Injections give
z = -1.9 / +2.9 / +6.3 at Lambda_true = 0.90 / 1.15 / 1.40, so

    3 sigma needs |A_dyn - 1| ~ 0.16      5 sigma needs ~ 0.27
    the candidates' predictions span       0.010

**The test is 16x too insensitive to separate them.** That is a clean negative
with a number attached, and it was established rather than assumed.

The measured column above sits 2-3 sigma above every prediction, Newton excepted.
**It is not interpretable, and the lane says so**: the forward model fails its own
goodness-of-fit gate at chi2/n_obs = 22-29, each stream pins Lambda to a formal
+-0.0002 while the 38 streams disagree by +-0.42 — **an internal inconsistency
factor of 1400-3400**. The declared summed-chi2 statistic is therefore void, its
argmin pinning to the grid edge; a robust median-of-per-stream statistic is
reported in its place and **flagged as post-hoc and uncalibrated**. A coherent
2-3 sigma offset across four laws whose predictions differ by 0.01 is a
misspecified baryon model, not four separate detections.

## AR.3 Independent confirmation of Gate 4, from the opposite direction

Measured on the analytic field, `max |curl g| x 10 kpc / |g|`:

    Newton                  3.8e-05    (the estimator's noise floor)
    RAR                     0.048
    well-network tensor     0.137 max, 0.054 median
    tidal scalar            1.08

**Neither surviving family's pointwise form is derivable from a single static
scalar potential.** Run AM reached the same verdict from pure structure — Gate 4 rejects any response
reading a functional of rho that is not `grad Phi_N` — and this lane reaches it
by measuring the curl of the field the law actually produces. **Two lanes, no
shared code, no shared data, same conclusion.** The tidal scalar at 1.08 is the
worst offender in the set by a factor of eight.

> **CORRECTED IN RUN AS (§AS.6).** The sentence originally read "can come from an
> action", which is FALSE in general — the Lorentz force has curl and comes from
> an action. The identity `curl g_alg = (grad nu) x g_N` makes the RAR's 0.048
> EXPECTED in non-spherical systems, not anomalous, and AQUAL exists precisely to
> supply the action-based completion. What survives is the magnitude comparison:
> the tidal scalar's 1.08 is 22x the RAR's 0.048.

## AR.4 The error floor is the in-plane leg, and more streams make it worse

    null sd, rotation curve re-drawn      0.0408
    null sd, rotation curve frozen        0.0178

**81% of the null variance is the shared baryon fit**, and the A_dyn *ratio does
not cancel it* because the response is nonlinear in |g_N|. Confirmed the hard
way: **doubling the stream sample raised the error to 0.0683 rather than lowering
it.** More streams will not help this test. A better `v_c(R)` covariance will.

That is the same disease as Run AL's slip measurement, where the dominant
uncertainty is an unpublished X-ray parameter covariance rather than the shear.
The programme's error budget is now repeatedly dominated by the *baryon model
someone else published*, not by the measurement it is testing.

## AR.5 Three secondary systems, all with zero power and a diagnosable reason

  * **NGC 4651**, 45 gated Keck tracers (30 disc, 15 halo): the five laws span the
    entire grid, Lambda = 0.60 to 1.70, **on identical data**. The missing
    observable is the line-of-sight distance to the shells, not sample size.
  * **M31**, 98 HI rings against 115 Chapman stream A-D stars: same picture,
    Lambda = 0.73 to 1.49.
  * **Pooled warps**, 15 Verheijen & Sancisi galaxies, 80 warped rings, exact
    tilted-ring wire integration: Delta chi2 over Lambda in [0.5, 2.0] is only
    0.31-0.67. No power.

**But the warp pool contains a real residual the test was not looking for**: a
3.8 sigma term tracking sin^2(psi), c = +0.84 +- 0.22 at chi2/dof = 1.13, with
per-galaxy intercepts and a common ln R term removed. **A_dyn cannot reproduce
it.** Most likely tilted-ring inclination bias through `V_rot = V_los/sin i`;
recorded, not claimed. And Run W's orientation-radius degeneracy is confirmed and
worse than recorded: within-galaxy Spearman(R, psi) has median **+0.939** across
all 15 galaxies.

## AR.6 Data-integrity findings beyond the declared checklist

All four declared `galstreams` defects carried, plus two the brief did not list:

  * **InfoFlags digits take values 0/1/2, not just 0/1**, so a `== "1"` test
    silently drops `M68-Fjorm.palau2019` and three others. The decode must be
    `> 0`.
  * **A fifth defect family**: `M68-Fjorm.ibata2021` carries distances down to
    1.8e-4 kpc and `M5.ibata2021` a radial-velocity track spanning -641 to
    +838 km/s — **both advertising InfoFlags 1111**, i.e. the catalogue's own
    quality flag says these are clean.

The lane's gate yields 68 usable-3D and 29 usable-6D against the summary table's
69 and 33; the entire difference is those physics failures, and **the gate only
ever demotes**.

Two silent bugs found and fixed, both of the worst kind — wrong answers with no
error raised:

  * **Point-sampling the density loses 27% of the disc mass** on the solver grid,
    driving the solved rotation curve to 0.73 of analytic *with no visible
    symptom*. Fixed with 12x12 Gauss-Legendre cell averaging.
  * **The M31 VizieR table uses space-separated sexagesimal**, so a colon-based
    parser returned NaN for all 115 stream stars and raised nothing.

And one design bug of the lane's own, caught and reported: putting a `sin^2 psi`
column into the warp Lambda-scan makes chi2(Lambda) flat **by construction** —
Delta chi2 range 0.07 with it in, 0.67 with it out.

## AR.7 Verdict

Nothing promoted, nothing eliminated, and unusually the *reason* is worth more
than the result: **the test was blind to three of its four targets before it
began, and the fourth's prediction is 16x below its detection threshold.** The
lane's lasting contributions are the curl measurement — an independent route to
Gate 4 that agrees with it — and the finding that the in-plane leg, not the
tracer sample, is the error floor. Both are reusable; the A_dyn number is not.

---

# Run AS — six corrections from external review, one of them a provenance failure

A corrections run, prompted by an external critique of the Runs J–AR register.
All six criticisms are accepted. Four are overstatements of mine, one is a false
general claim I published, and one is a provenance contradiction that had been
sitting in this document unnoticed. Nothing new was measured here; three lanes
were opened to do the work the critique implies (AS.8).

## AS.1 The provenance failure: KiDS and the wide binaries are VALIDATION, not sealed confirmation

**This is the most serious item and it is a contradiction internal to this
document.** The Runs J–AR register says the holdouts *"have never been loaded."*
But this same document records their outcomes, repeatedly:

    line 2478   KiDS [BLIND], 35-2600 kpc   nu/nu_RAR = 1.31, canonical a0
    line 2479   wide binaries [BLIND]       nu/nu_RAR = 0.90
    line 2023   "Both blind holdouts land on the RAR. KiDS tracks it to 0.134 dex"
    line 3410   wide binaries | 6 | 0.022 | 0.050 | Newton
    line 3707   KiDS [blind] 0.1340 | wide binaries [blind] 0.0497 | unchanged

Those cannot all be true under the ordinary meaning of "never loaded."

**What actually happened, stated precisely.** The ORIGINAL rule (line 62) was
*"Never FIT to a holdout"*, and that rule has been kept without exception — neither
dataset has ever entered a fit, screen, calibration or model-selection step. But
both were **scored**, as one-shot blind checks, in the earlier rounds. Their
numerical outcomes are therefore known. The stricter rule — *"Do not load them.
Do not look at them."* — was adopted at the start of the wellnet-2026-09 round
and **has been honoured within that round**; every lane from Run J onward states
it and none has touched them.

So the correct three-way classification, which this programme should have been
using all along:

    training      used to estimate parameters
    validation    outcomes examined during development
    confirmation  never examined until a single final frozen test

**KiDS and the wide binaries are VALIDATION.** Any dataset whose result is
already known cannot serve as untouched confirmation, no matter how carefully it
was quarantined afterwards. The "permanent sealed holdout" language is withdrawn
and replaced by "never fitted; scored once in round 1; sealed for rounds 2-3."

**Consequence.** This programme currently has **no confirmation set at all**.
Acquiring one is now a prerequisite for any headline claim: select it, hash it,
freeze the model and the statistic, and evaluate through a one-shot script. Until
then every result in this document is at best validation-grade.

## AS.2 "No slip, to within 2%" conflates a central value with a precision

AL.5 reported the RAR landing on raw cluster shear as *"no slip, to within 2%."*
**Withdrawn.** What was measured is

    Sigma_s(RAR)   = 0.981      so   |Sigma_s - 1| = 0.019 = 1.9%

which is a **central value 1.9% from unity**, not slip measured to 1.9%
precision. And AL.7, two sections later in the same run, says the opposite in
plain terms: the X-ray-fit-noise null drags a fitted Sigma_s down by up to a
factor 2.2 at -17 sigma, *"its factor-two width is the lane's dominant
uncertainty"*, and *"every Sigma_s must be read against this null rather than
against 1."* The null-corrected bracket in AL.9 is **1.06-2.17**, which does not
even contain the raw central value. **The run contradicted itself and I quoted
the wrong half.**

The accurate statement:

> Under the adopted baryonic profiles, spherical mapping, and no-slip closure,
> the parameter-free RAR prediction lands within 2% of the best-fitting
> weak-lensing normalisation. Its uncertainty is presently dominated by
> unreported baryonic-profile covariance and is much larger than 2%.

That is still the cleanest result the cluster work has produced. A prediction can
land strikingly close before the uncertainty is small enough to claim precision.

**Also withdraw "parameter-free" as a description of the pipeline.** The correct
phrase is **zero new fitted GRAVITY parameters**. The pipeline still contains
estimated X-ray profile parameters, gas conversions, stellar masses, centring
assumptions, source-redshift distributions, shear calibration and cosmological
distance factors. None of that diminishes the frozen transfer; it does mean a
parameter-free *law* is not a parameter-free *measurement*.

And AL.7's projection that publishing the Vikhlinin covariance *"would take the
slip measurement from a factor of two to about 10%"* is downgraded to a
conjecture. It has not been propagated. The right ask of the authors is the full
covariance **plus** parameter ordering and units, whether it is per-object or
population-level, the density-normalisation-to-shape covariances, and preferably
posterior samples — because the Vikhlinin parameters are strongly correlated and
need not be Gaussian. The quantity actually wanted is

    p(Sigma_s | D_gamma, D_X) = INT p(D_gamma | Sigma_s, theta_X) p(theta_X | D_X) d theta_X

## AS.3 RAR and AQUAL are one result here, not two

AL.5 listed them as separate rows and I read the agreement as corroboration. In
spherical symmetry an AQUAL equation reduces to an algebraic
`mu(g/a0) g = g_N`, so with an interpolation function chosen to approximate the
RAR the two are near-identical by construction. The measured
chi2 = 3588.4 against 3588.2, and Sigma_s = 0.981 against 0.992, are **one
physical result reported twice.**

Further, **the empirical RAR does not by itself specify photon propagation.**
Predicting shear requires a closure — here `Phi = Psi`, so light responds to
`Phi + Psi = 2 Psi`. The honest object is therefore the conjunction

    RAR/AQUAL-like dynamics + spherical projection + no-slip lensing closure

which matches the central eFEDS shear normalisation. That is a compelling target
for a complete theory; it is not itself a complete theory.

## AS.4 The weak-versus-strong discrepancy is not yet "regime-dependent photon coupling"

AL.9 concluded that *"MOND requires a regime-dependent photon coupling."*
**Too strong.** What was demonstrated is an **effective closure mismatch**
between lower-mass or larger-radius weak-lensing systems and one massive
strong-lensing core. That mismatch is consistent with a failure of the MOND-like
potential in cluster cores, missing or misestimated BCG and member-galaxy
baryons, line-of-sight structure, triaxial deprojection, mass-sheet or
source-position degeneracies, the assumed closure, **or** a genuinely changing
photon-to-matter response.

And the caveat that undercuts it was already in the same run: AL.8 records that
**one closure parameter brings EVERY law, including unmodified Newton, onto
Refsdal's 376.02 d**, and that *"a single time delay can never test a gravity
law."* The delay is measured from the light curve to 1.48%, but predicting it
needs the Fermat potential and therefore a lens model — so it is not a model-free
measurement of slip. AL.9 then leaned on it as though it were.

Replacement wording:

> Weak-lensing outskirts and a strong-lensing core require inconsistent effective
> normalisations under the current common-potential implementation. More systems
> are required to determine whether the transition is driven by radius, host
> mass, baryonic modelling, or lensing closure.

## AS.5 The vertical slope does NOT reject the RAR or AQUAL

Run AF is titled *"the vertical channel is real, and it rejects Newton and the
RAR."* The second clause is not supported by the slope, and the title invites
exactly the wrong reading. On the slope itself:

    observed  -0.346      RAR  -0.291      AQUAL  -0.264      Newton  0.000

    against the quoted +-0.173:   RAR 0.32 sigma,  AQUAL 0.47 sigma
    against the robust +-0.067:   RAR 0.82 sigma,  AQUAL 1.22 sigma

**Under either error bar the observed slope AGREES with the MOND-like
predictions.** The rejection in AF.3 comes from a *different* statistic — the
joint amplitude-and-shape comparison including the within-galaxy radial scale
length, where AF.5 records observed 2.086 against Newton 2.499 and RAR 2.896.
That is a real model comparison, but it is not "the slope rejects the RAR."

Correct statement:

> The between-galaxy slope resembles the RAR/AQUAL prediction; the within-galaxy
> radial profile does not. Their JOINT vertical prediction fails, and only under
> the adopted constant-Upsilon_K model.

That tension is itself informative: it separates a local response
`B_z(R) = f[g_b(R)]` from a global one `B_z(R) = f(Sigma_0, Phi_global)`, and
from a stellar-population systematic.

**And a numerical correction.** AF.1 reports `P = 0` from 2,000 null trials. A
finite simulation cannot establish zero. The Monte-Carlo p-value with no
exceedances is

    p <= 1/(N+1) = 1/2001 = 5.0e-04

which is quite strong enough. (The `95% bound 0.0075` printed beside it is the
rule-of-three bound on the per-scenario rate, 3/400, and is correct as such.)

**The degeneracy remains the decisive issue, and it is large.** The alternative
to gravity is `Upsilon_K ~ Sigma_0^-0.395`, which across the sample's factor of
35 in central surface density demands

    Upsilon_K(high) / Upsilon_K(low) = 35^-0.395 = 0.246,  a factor of 4.07

That is a big, highly testable stellar-population claim — and it is exactly
degenerate with the gravity reading, to 5e-15 dex. Until Upsilon_K is
independently constrained the vertical result is an **astrophysical anomaly, not
a gravity anomaly.** The one piece of evidence against the systematic reading
stands: the required trend is opposite in sign to the observed B-K colour trend.

## AS.6 "A field with curl cannot come from an action" is FALSE, and I published it

In Run AR and in the register I wrote that the measured curl shows neither
surviving family's form *"can come from an action."* **That is wrong in
general.** The Lorentz force has nonzero curl and follows from

    L = (1/2) m v^2 + q A.v - q phi

because it is velocity-dependent and involves a vector potential; gravitomagnetism
is the gravitational analogue.

What the measurement actually shows is narrower and follows from an identity. For
an algebraic vector prescription `g_alg = nu(|g_N|) g_N` with `curl g_N = 0`,

    curl g_alg = (grad nu) x g_N

which is generally nonzero in **non-spherical** systems. So the measured curl
demonstrates that the *algebraic vector prescription* is not derivable from a
single static scalar potential — a known property, and precisely the reason AQUAL
was constructed to supply an action-based completion. The RAR's measured 0.048 is
therefore expected, not anomalous.

**The gate is renamed** from a reciprocity-and-action gate to
*"scalar-potential integrability under the declared static, velocity-independent
model class"*, and its scope stated: it says nothing about velocity-dependent
forces, vector-potential or gravitomagnetic sectors, or theories with extra
propagating fields.

What survives of AR.3: the tidal scalar's 1.08 is **22x the RAR's 0.048**, so the
*magnitude* of its non-integrability is in a different class from the known and
repairable case. That comparison stands; the general claim does not.

## AS.7 The compiler's 97.2% mixes four different verdicts

The headline *"3,036 of 3,123 rejected before any data"* conflates:

    mathematically inconsistent     a legitimate kill
    physically incomplete AS WRITTEN   might admit a variational completion if a
                                    gating field were promoted to a dynamical one
    representation/convention dependent   kills the FORMULA, not every
                                    potential-dependent theory: a physical boundary
                                    condition or covariant field can define a
                                    meaningful potential DIFFERENCE
    non-identifiable on this bench  an EXPERIMENT cannot see it; the theory is not
                                    thereby inconsistent

Only the first is an unconditional rejection. Run AM already said "read Gate 4 as
a to-do list, not an extermination" and then reported a single undifferentiated
rate anyway.

Also accepted: **35/35 against this programme's own recorded verdicts is
regression testing**, and risks validating the compiler against the conclusions
that shaped it. External positive controls are required — Newtonian Poisson,
AQUAL, QUMOND, a Yukawa scalar from an action, a symmetric nonlocal action, a
scalar-tensor weak-field limit, and critically a **vector-potential force with
nonzero curl and a valid action**, which must be *labelled outside the
scalar-potential class* rather than rejected. That last case is the one my AS.6
error would have mishandled.

## AS.8 What was opened in response

Three lanes, running as this is written:

  * **`r500-audit/`** — the shared-variable risk in *"a cluster-only excess
    organised by r/R500"*. If R500 is derived from the same mass that sets the
    excess, both axes share a noisy quantity. Synthetic clusters with NO true
    scaled-radius dependence, run through the real inference, will say how much
    organisation is manufactured automatically; then the real relation under four
    radial definitions — physical kpc, r/R500_WL, r/R500_X, and a
    **baryon-only** r/R_b that cannot be tautological. Assume this is the eighth
    shared-quantity artefact until shown otherwise. (The record already tested one
    law-free replacement, `t = r/r_a0`, and found it fails to unify CLASH with
    SPARC; that is a different question from whether the original is circular.)
  * **`transition/`** — one common observable space
    `S(M, r) = observed / RAR-no-slip-predicted` across eFEDS, LoCuSS and
    strong-lensing cores, with a prespecified hierarchy H0/H_M/H_R/H_MR, one
    survey-level offset per dataset, and **one entire survey held out**. This is
    the fastest route to deciding whether the cluster residual is mass, radius,
    acceleration regime, or pipeline — four different physical stories the
    programme currently cannot separate.
  * **`compiler/` v2** — the AS.7 taxonomy, the AS.6 curl identity verified
    numerically and the gate renamed, the external control suite, and the
    **external-axis basis element** `K = exp[f0 I + f_E e_ext e_ext^T]` added to
    the grammar. Run AO showed not one of 3,123 candidates carried an external
    axis, so the built and calibrated 2-D shear channel is aimed at a hypothesis
    the grammar cannot express. That is a grammar-completeness fix and **no
    observational null for it may be interpreted**: AO's 95% exclusion sits at an
    ellipticity of 2.11, above the geometric maximum of 1.

Also scoped: generating candidates **from admissible actions** rather than
generating arbitrary force laws and rejecting 97% afterwards, starting from

    L = -(1/8 pi G)(grad Phi)^T K(q,I)(grad Phi) - (Z(q)/2)|grad q|^2 - V(q) - rho Phi

with automatic variation producing the field equations, so symmetry, reciprocity
and scalar-potential integrability hold by construction.

## AS.9 The corrected headline

> No new force law constructed by the programme survives its mathematical and
> observational controls. A galaxy-calibrated RAR/AQUAL-like potential, combined
> with a no-slip spherical lensing closure, lands within 2% of the best-fitting
> normalisation of eFEDS tangential shear without fitting new gravity parameters
> — although missing baryonic-profile covariance prevents any 2% precision claim.
> DiskMass contains a robust surface-density trend in vertical support, but it is
> exactly degenerate with a factor-4 variation in stellar mass-to-light ratio and
> is therefore not yet evidence for modified gravity. The admissibility compiler
> is a powerful screening tool, but its verdicts must distinguish inconsistency
> from convention dependence and observational non-identifiability. The remaining
> empirical target is the transition from RAR-compatible group outskirts to
> excess gravity in massive cluster interiors.

## AS.10 One item the critique raised that the record can already answer

The redshift branch's CMB cap of 0.28-0.44% (AK.2) applies to the **line-of-sight
environmental modulation around a homogeneous background**, not to a total
cosmological redshift mechanism — it is derived from `dT/T = -c2 dI_q` on the
CMB, i.e. entirely from LOCAL FOREGROUND structure, and `c2` is by construction
the coefficient of the void-path term in `ln(1+z) = c1 D + c2 I_q + ...`.

Stated as a consequence, which AK did not do: **a sub-percent differential
mechanism cannot produce redshifts of order unity, so this branch is not a
replacement for cosmological expansion.** What survives is a possible secondary
effect,

    Delta z / (1+z)  <~  4e-3   correlated with intervening void structure,

which is legitimate and testable, and is what the Planck cross-correlation would
target. If a homogeneous geometric clock evolution were to supply the main
redshift with only the differential component CMB-limited, that decomposition
would have to be stated explicitly and it has not been.

---

# Run AT — the R500 audit: the eighth artefact is real, the phrase dies, the trend survives

Full record in `work/wellnet-2026-09/r500-audit/REPORT.md`; 11 code files,
`results.json`, 13/13 self-tests. KiDS and the wide binaries never loaded.
Opened in response to the external critique (§AS.8).

The reviewer's suspicion was correct and the instruction to *"assume this is the
eighth shared-quantity artefact until shown otherwise"* was the right posture. It
**is** the eighth, and it lands almost exactly where the seventh did. But it is
not what produced the number, and the audit ends up killing a *phrase* rather
than a result.

## AT.1 The naive significance is destroyed; the trend is not

Reproduced exactly: `Spearman(r/R500, RAR residual) = -0.7884` on 588 X-COP
points, 12 clusters.

    observed                                          -0.7884
    R500-scrambling permutation null      -0.7724 +- 0.0102, obs at pct 5.8
    collapse statistic, same null                     obs at pct 4.2
    FPR of that test under a FLAT truth, nominal 5%    0.53 (S1), 0.70 (S2)
    **correctly calibrated p**                        **0.580 and 0.653**

    forward synthetic null, NO true radial dependence  -0.1417 +- 0.0403
    observed against it                                **z = -16.1**, pct 0.00
    responsiveness d(corr_meas)/d(corr_inj)            0.867 (0.757 near null)

So the naive p of 0.04-0.06 becomes **p = 0.58 / 0.65** — statistically
indistinguishable from the retracted LoCuSS partial correlation, which sat at
p = 0.563. A permutation test whose own false-positive rate is **53-70% against a
nominal 5%** is not a test.

**But a forward null that contains no radial dependence at all cannot manufacture
-0.788.** The trend survives at 16 sigma, with responsiveness 0.87 — so this is a
measurement, not a failure to exclude.

## AT.2 Why the tautology cannot do it — two structural facts

**The cancellation lemma.** X-COP tabulates temperature as `T/T500` against
`R/R500`, so reconstructing `T_X x T500(R500)` at `RW_X x R500` returns the
observed physical temperature for *any* R500. Verified numerically: scaling R500
by 0.55x and 2.30x moves g_obs by **1.6e-13**. **R500 enters the x-axis only.**

**Monotone invariance.** A per-cluster normaliser cannot change a *within-cluster*
rank statistic — bit-identical over a 10x range in R500 — and **90.3% of the
residual variance is within-cluster**. R500 explains r^2 = 0.25 of the between
part, i.e. at most **2.43% of the total**. Fifth instance of the
monotone-invariant-statistic pattern, and the first time it has *protected* a
result rather than voided one.

The shared-quantity channel is nonetheless real and measurable:
`M500_hdr / [(4/3) pi 500 rho_c R500^3] = 1.0003` for all twelve, and
`ERR_M500/M500 = 3 x ERR_R500/R500` **exactly** — **M500 and R500 are one
number**, not two. Directly measured,
`corr(per-cluster mean residual, ln R500) = +0.500`, with the sign the tautology
requires. The channel exists; it is simply too weak to reach a within-cluster
statistic.

## AT.3 The phrase "organised by r/R500" is dead — by an exact identity

    r, physical, no normalisation      -0.7790
    r / R500                           -0.7884

The normalisation is worth **0.0095**, and it cannot be worth more, because

    log(r / R500_i) = log r - log R500_i

and `log R500_i` is constant within cluster i, hence **already in the span of the
cluster indicators**. Any fit that gives each cluster its own level therefore
sees the two regressors as identical. Verified in the lane (rank 13 = 13,
residual 5.2e-14, identical fixed-effects RSS and slope) and independently here:
stacking BOTH columns beside twelve indicators still gives rank 13, not 14;
fitted values agree to 4.7e-16 and the radial slopes are equal to ten decimals.

Leverage confirms it: `sd(ln R500) = 0.109` against `sd(ln r) = 0.723` within a
cluster, a ratio of 0.15, with R500 spanning only a factor 1.36 across the
sample. The one well-posed comparison — two global-parameter hypotheses, no
per-object parameter — separates by **0.68 sigma**.

**Ninth instance of the collapsing-variable-list pattern.** `r/R500` and `r` are
the same variable. The claim this programme should have been making is *"the
excess is organised by radius"*; the R500 normalisation carried no information
and implied a self-similarity that was never tested.

## AT.4 Four radial definitions, and the reviewer's decisive test

Slopes under each normalisation:

    r physical      -0.4996
    r / R500,X      -0.4974
    r / R_b,gas     -0.4875      <- baryon-only, no total mass
    r / R_b,ne      -0.4841      <- baryon-only, no total mass

**Identical.** The trend survives under radii that contain no total mass at all,
which is exactly the test the critique asked for.

> **CORRECTED IN RUN AY.** This sentence claims too much and I wrote it. Both
> baryon-only normalisers measure **|corr| = 1.0000** against the baryon
> amplitude — worse than the -0.99 case AX.5 warned about. But the reason the
> slope is unchanged is not that they are independent: it is that `R_b` is a
> **per-cluster constant**, and by this run's own rank identity a per-cluster
> constant cannot move a within-cluster slope. Measured: the slope under
> `r/R_b,gas` is **bit-identical** to physical `r` at every amplitude.
> **The baryon-only control was VACUOUS, not passed** — it could not have been
> informative about this statistic. It is not evidence for the trend, and it was
> reported as though it were: *a transition visible only
under the mass-derived radius would be suspect.* It is not visible only there.

## AT.5 A real bug in shared bench code, and claims quoted past the data

`invariant_bench._cluster_profile` interpolates the coarse temperature profile
onto the fine density grid with a bare `np.interp` — no `left`/`right`, so it
**clamps** past the last measured temperature bin and forces `dlnT/dlnr -> 0`
there. Confirmed by reading the source: the hydrostatic `g_obs` carries
`(dln n_e/dln r + dln kT/dln r)`, so the temperature-gradient term is silently
deleted in the outskirts, precisely where the radial trend is read.

    affected                       93 / 588 points = 15.8%, all at the outer end
    measured T grid ends at        median r/R500 = 0.91
    the relation is quoted to      r/R500 = 1.52
    slope with clamped points cut  -0.4803 -> -0.4092   (15%)

**So the record's "factor 1.4 at R200" and "crossing at r/R500 = 1.9-2.5" are,
for X-COP, beyond any measured temperature.** Those are extrapolations of a
clamped profile, not measurements, and they should not be quoted again without
the correction. This is shared code (`work/gravity-wells-2026-09/`), so other
lanes that call it inherit the same defect.

**And the pipeline is biased even with no noise at all:** a perfectly flat truth
run through it returns S1 = -0.207 and a slope of -0.139 dex/dex, which is **29%
of the observed slope**. The residual 71% is either modified gravity or an
outward-rising non-thermal pressure, and X-ray data alone cannot separate them.

## AT.6 A methodological finding worth propagating: correlations saturate

Injecting a slope of only **-0.25** already drives the correlation coefficient to
**-0.92**. A correlation near -0.8 therefore says almost nothing about the size
of the effect, and comparing two such numbers says less. **Future claims in this
programme must be quoted as slopes, not correlations.** This retroactively
explains why -0.788 felt impressive: it would have looked much the same for a
third of the effect.

## AT.7 The law-free replacement, revisited (Job 3)

`t = r/r_a0` carries the *same* rank degeneracy — it adds zero rank directions
beside the cluster indicators — but its shared-quantity channel is
**+1.148 dex/dex, positive**, so a baryon error cannot manufacture a negative
correlation. **`t` is conservative exactly where `R500` is not**, which is a
better argument for it than the record's original one. It scores -0.7045, worse
than raw radius.

And the extrapolation the record merely flagged **is boundable**: the a0 crossing
is directly measured in only **2 of 12** clusters, `g_bar` turns over inward in
**5 of 12** (so `r_a0` does not exist under a continuation of the measured
slope), and across the defensible baryon family — bare gas plus a BCG of
0.5/1/2 x 10^12 Msun — `log10 r_a0` spreads **0.87 dex, a factor of 7**, set
*entirely* by the unmeasured BCG stellar mass. That is a precise price tag on the
sub-30-kpc acquisition the record proposed.

## AT.8 What could NOT be established

  * **Whether the excess is organised by r/R500 rather than by physical r** — not
    answerable on this sample at 0.68 sigma, and algebraically the same model once
    each cluster has its own level. It needs 10^13 Msun groups measured alongside
    10^15 Msun clusters in the same way.
  * **Whether the within-cluster trend is physics or hydrostatic bias.** 29% is
    pipeline; the rest is modified gravity or non-thermal pressure, and X-rays
    alone cannot separate them.
  * **Only 4 of 12 X-COP clusters are in Herbonnet+2020 by name**, so the
    record's "n = 7" must draw three from LC2/Sereno, which is not in the repo.
    That provenance needs closing.
  * **CLASH's binned table was NOT re-audited, and it is the reviewer's tautology
    in pure form** — Umetsu+2016 M500c on the x-axis against a numerator built
    from the same lensing profiles, with no cancellation lemma to protect it and
    the Run AL.3 provenance failure inherited on top. It needs the Umetsu+2016
    per-cluster masses, which are not in the repo. **This is the open half of the
    audit** and is the next acquisition.
  * The forward null's error covariance is the lane's own construction, not
    X-COP's — the true L1-deprojection covariance is unpublished. The sensitivity
    table is stable across everything tried, but that is a sensitivity check, not
    the real covariance. Same disease as Run AL's missing Vikhlinin covariance.

## AT.9 Net effect on the register

The surviving statement changes from

> a cluster-only excess organised by r/R500

to

> **a cluster-only excess organised by radius, measured within clusters, at
> 16 sigma against a forward null and unchanged under baryon-only radial
> normalisations — of which 29% is a known pipeline bias, whose outer 16% of
> points sit beyond any measured temperature, and whose separation from an
> outward-rising non-thermal pressure X-ray data alone cannot make.**

Less tidy, and considerably better supported.

---

# Run AU — the compiler rebuilt: 12.5% inconsistent, not 97.2% rejected

Full record in `work/wellnet-2026-09/compiler/REPORT_v2.md` (`REPORT.md` is
untouched — it is committed and cited). 48/48 tests in 75 s; retrospective over
3,123 candidates in ~50 s. `test_no_observational_data_is_opened` extended to
cover every new path: **0 files opened, 0 outside the lane.** Opened in response
to the external critique (§AS.7).

## AU.1 The taxonomy, and why the first cut of it was wrong

Run AM's headline — *"3,036 of 3,123 rejected before any data"* — mixed verdicts
that are not scientifically equivalent. Partitioned:

| bin | n | % |
|---|---|---|
| **mathematically inconsistent** | **390** | **12.5%** |
| representation / convention dependent | 1,482 | 47.5% |
| physically incomplete as written | 1,008 | 32.3% |
| not decidable on this bench | 100 | 3.2% |
| non-identifiable on this bench | 52 | 1.7% |
| admissible | 91 | 2.9% |

**Only 12.5% are unconditionally dead**, and those 390 are the well-network
settings with **no continuum limit at all** — the only candidates this bench can
call inconsistent rather than merely unusable as written. The critique was right
that the undifferentiated rate was doing too much work.

**The first taxonomy was wrong, and finding that justified the whole exercise.**
It put 2,075 candidates (66%) in `mathematically inconsistent` on the strength of
Gate 4's `cond(K) > 1e8` health check — which is a **float64 CG-solvability
limit, not an ill-posed PDE** — and Gate 4 *returns on it first*, so a solver
limitation was masking whatever structural defect lay underneath. Of those 2,075,
**1,975 carried a structural defect as well; only 100 were conditioning alone.**
The taxonomy now consults structural findings before conditioning, and the
inconsistency bin fell from 66% to 12.5%.

Two bins had to be invented that Run AM did not have:
`outside_declared_model_class` (a label that is neither admit nor reject — see
AU.3) and `not_decidable_on_this_bench`.

**Verdict invariance is asserted mechanically**, not claimed: the committed
pre-v2 compiler was checked out and run against the same `tournament.json`,
giving 3032/91/26 and gate kills 150/0/1560/2980 — **bit-identical** to the new
code, and `retrospective.py` re-asserts it every run. (The 3,036 → 3,032 drift
against Run AM's REPORT is not this lane's: `tournament.json` was itself re-run
in Run AQ after the `mond_invert` fix, which took the survivor count from 18
to 26.)

## AU.2 The curl identity, verified exactly — and 0.048 was a prediction all along

`curl g_alg = (grad nu) x g_N`, verified with **complex-step** derivatives (no
subtractive cancellation) on a declared closed-form disc, reproducing all four of
Run AR's rows from an independent implementation:

| law | identity residual | curl g_N control | exact max q | at Run AR's h | AR recorded | rel |
|---|---|---|---|---|---|---|
| newton | both sides round-off | 7.7e-16 | 7.67e-16 | 3.81836e-05 | 3.81836e-05 | **0** |
| rar | **2.52e-14** | 6.7e-16 | 0.0484537 | 0.0482477 | 0.0482477 | **0** |
| aqual | **2.02e-14** | 4.8e-16 | 0.048958 | 0.0487499 | 0.0487499 | 5.4e-13 |
| tidal scalar | **1.51e-14** | 6.7e-16 | 1.07847 | 1.08253 | 1.08253 | 2.1e-14 |

**The RAR's 0.048 is a prediction of the identity, not an anomaly.** Its continuum
value is 0.0484537; Run AR's 0.0482477 is that same number seen through a central
difference at h = 0.05 kpc, and the FD residual converges at order 2.00 in h.

**The AQUAL row settles the general claim by itself.** AQUAL was constructed
precisely to give MOND an action, and its algebraic form still carries curl
0.049. If non-zero curl meant "no action", that row alone would refute it. My
Run AR wording is therefore not merely imprecise but decisively wrong, and it is
corrected in the source, the report and here.

**Fifth blindness result, and this one indicts the programme's own instruments.**
On a spherical probe the antisymmetry is **2.0e-10** — so *every* spherical
channel in this programme, **including this compiler's own radial Jacobian**, is
blind to the obstruction the curl measures. Measured, not assumed, and it forced
a new non-spherical channel: `u_space_integrability` tests the gate's actual
criterion — is `K(u)u` a gradient in `u`? — directly on a 3-D cloud of `u`
vectors, with the floor measured on laws that are gradients exactly (2.2e-10
AQUAL/QUMOND, 0 Newton, 3.0e-10 gn-gated `tensor_d`) against a declared 1e-7.

**The gate is renamed** `gate4_reciprocity_action` -> **`gate4_scalar_potential_integrability`**,
titled *"scalar-potential integrability under the declared static,
velocity-independent model class"*, with the old key kept as a deprecated alias
outside `GATES` so it cannot double-count. Declared **out of scope and never
rejected**: `velocity_dependent` (the Lorentz force, from
`L = 1/2 m v^2 + q A.v - q phi`), `vector_potential_gravitomagnetic`,
`extra_propagating_field`, `relativistic_completion`. The correction is carried
as a comment block in the source so it cannot be lost.

## AU.3 External positive controls: 12 of 12, and the sharp case passes

| control | required | got |
|---|---|---|
| Newtonian Poisson | ADMIT | ADMIT |
| AQUAL | ADMIT | ADMIT |
| QUMOND | ADMIT | ADMIT |
| Yukawa from an action | ADMIT | ADMIT |
| symmetric nonlocal action (Gaussian kernel) | ADMIT | ADMIT |
| scalar-tensor weak field (Brans-Dicke omega = -1) | ADMIT | ADMIT |
| **vector potential, non-zero curl, valid action** | **OUTSIDE-CLASS** | **OUTSIDE-CLASS** |
| non-reciprocal catalogue force | REJECT | REJECT |
| coarse-graining well-count law (p = 0) | REJECT | REJECT |
| indefinite kinetic energy | REJECT | REJECT |
| *contrast:* sub-threshold Yukawa | REJECT | non-identifiable |
| *contrast:* f(R) scalar-tensor, alpha = 1/3 | REJECT | non-identifiable |

**The vector-potential case is labelled, not rejected** — `_verdict =
"OUTSIDE-CLASS"`, `_failed = []` (Gates 1-3 still apply and it passes them),
label *"action-based but outside the scalar-potential class"* — with a standalone
regression test so a regression names itself. That is exactly the case my AS.6
error would have mishandled.

Making positive-definiteness of the kinetic operator a real check was necessary
for the indefinite-energy control, because the exponential grammar's own sign
theorem hides that failure mode; a non-exponential structure was added to
exercise it.

**Two findings the lane did not tidy away.** Gate 1 is an *identifiability*
statement, so it must depend on amplitude and range, and `gate1_identifiability_scan`
measures the threshold rather than letting it be tuned: a Yukawa escapes only
when its range is comparable to the probes' 10-30 kpc span **and** its coupling
is O(1). Consequently **f(R) gravity, which fixes alpha = 1/3, cannot be
identified on this bench's three probes at any range** — a two-parameter stretch
absorbs it to 0.019 dex. That is a probe-geometry limitation; it lands in
`non_identifiable_on_this_bench` and never in an inconsistency bin, which is
precisely the distinction the taxonomy exists to preserve. (The scalar-tensor
ADMIT row therefore uses Brans-Dicke omega = -1, the string dilaton, where
alpha = 1 exactly.)

## AU.4 The external axis lands in two places, by derivation

Run AO found that **not one** of the 3,123 candidates carried an external tidal
axis. Added to the grammar:

| element | verdict | bin |
|---|---|---|
| `F1_ext_axis_const` (f0, f_E constant) | **ADMIT** | admissible |
| `F2_ext_axis_gn_gated` | REJECT | incomplete as written |
| `F3_ext_axis_tidal_gated` | REJECT | incomplete as written |

**Constant couplings are admissible, exactly.** K is then a constant symmetric
positive-definite tensor, so `div[K grad Psi] = 4 pi G rho` is precisely the
Euler-Lagrange equation of
`L = -(1/8 pi G)(grad Psi)^T K (grad Psi) - rho Psi`; u-space antisymmetry
**0**. And it escapes Gate 1 on all three escapes **including (b), the
independently measured axis** — misaligned 60 degrees from the probes' radial
direction against a declared 10-degree threshold. **This is the only axis
provenance for which escape (b) is available at all**, which is the pre-data
counterpart of Run AO's measurement that external-axis power does not collapse as
the source rounds. Two lanes, structural and observational, converging again.

**The gated versions fail by derivation, not by fit.** With
`(Ku)_i = a(|u|) u_i + b(|u|)(e.u) e_i`, the antisymmetric part of `dM_i/du_j` is
`(e.u) b'(|u|) [u^_j e_i - u^_i e_j]`, non-zero off the axis; measured
antisymmetry **0.0494** against a 1e-7 floor. **The spherical radial Jacobian
could not have found this** — it is the curl identity seen in u-space, i.e. AU.2's
blindness result biting a real candidate.

**No observational claim is attached and none can be**: Run AO's 95% exclusion
sits at ellipticity 2.11, above the geometric maximum of 1. This is grammar
completeness, not evidence.

## AU.5 Action-first generation: worth doing, not as a replacement

Scored against the compiler's own defect census, **86.1% of 4,796 defect
instances would be prevented by construction** if candidates were generated from
admissible actions rather than generated freely and screened afterwards.

  * **Gate 1 survives intact and becomes the binding gate.** A constant K is
    degenerate with `x -> K^(-1/2) x` whatever action produced it, and generating
    only admissible actions raises the fraction of the search that reaches it.
  * **Gate 2 survives intact.** Nothing stops an author writing `|Phi|` into
    `V(q)` or into the coupling — 418 defects here.
  * **Gate 3 becomes vacuous for this grammar** (no row-list atom) but must be
    kept dormant: the cost of keeping it is zero and the cost of losing it is
    family C.
  * **Gate 4's verdict becomes vacuous; its measurements do not.** The u-space
    antisymmetry, the curl identity and the reciprocity measurement are the only
    things that would catch **a bug in the variation code itself** — an AD layer
    emitting the wrong field equation produces an asymmetric Jacobian, and
    nothing else on this bench would notice. Demote to a self-test of the
    generator; do not delete.
  * **Cost.** Symbolic or automatic variation through a matrix exponential is the
    visible item. The larger structural cost is that the emitted system is **two
    coupled equations**, so either every tournament channel changes or each
    candidate needs a coupled solve — which is the PDE-per-candidate cost the
    pre-data compiler exists to avoid.
  * Reusable unchanged: probe geometry, gauge-rule population, all Gate 1 and
    Gate 3 routines, the curl module, the u-space test, the taxonomy, the control
    suite.

**Recommendation: build it, but alongside.** It prevents the row-list half of the
convention bin and all of the incomplete bin; it does not touch the gauge half
(418), non-identifiability (150) or conditioning (100). **A generator cannot tell
you whether what it generated is measurable.**

## AU.6 A provenance error of mine, and how it was repaired

A `git add -A work/wellnet-2026-09` of mine swept **3,613 lines of this lane's
mid-edit files** into commit `8970c943`, which was labelled "Run AT: the R500
tautology audit" and had nothing to do with the compiler. It also caught the
transition lane. The lane flagged it rather than working around it, which was the
right call in a repository with receipt discipline.

Repaired by splitting: `8970c943` was unpushed and at HEAD, so the branch pointer
was moved back with `reset --soft` (working tree untouched throughout), Run AT
was re-committed as `de388cd1` containing **only** the r500-audit lane, and this
lane follows in its own commit. **The lesson is procedural and worth keeping:
never `git add -A` a shared work directory while background lanes are writing to
it.** Stage the lane you mean.

---

# Run AV — the temperature clamp: three recorded verdicts change, and the relation reaches nowhere near where it was quoted

Full record in `work/wellnet-2026-09/tempclamp/REPORT.md`; `python run_all.py`
reproduces the lane end to end. KiDS and the wide binaries never loaded. Opened
in response to Run AT's discovery of the bug.

## AV.1 The inventory is worse than one function

`invariant_bench._cluster_profile` has **0 direct callers** outside the bench and
`_xcop` is its only in-repo caller — so one patch reaches every lane through it.
**15 lanes import `Bench`, and all 15 consume X-COP.** The lane's first inventory
pass wrongly cleared three of them because they never write `d["xcop"]`; they
call `b.confound(...)`, `b.score(...)` or iterate `b.d`, each of which pulls
X-COP in. Derived with `ast`, not asserted.

But the patch is not sufficient, and that is the important half:

    9 files re-implement the identical unguarded interpolation, same direction
    8 more (the ben-executor family) clamp onto the T grid by the same mechanism
    NOT ONE np.interp on a temperature anywhere in the repo passes left= or right=

**The defect is a habit, not a line.** Fixing the bench fixes none of the nine.

## AV.2 The fix, and a default that cannot move a recorded number

`_cluster_profile(d, temp_extrapolation=...)` with `clamp` (default), `drop`,
`loglinear`, `forbid`. It returns a `ClusterProfile` — a **4-tuple subclass**, so
every existing `r, gb, go, R500 = ...` unpack still works — carrying
`.extrapolated`, `.stencil` (the mask dilated by `np.gradient`'s stencil),
`.frac_extrapolated`, `.r_tmin/.r_tmax`, `.kT`, `.mode`. A
`TemperatureExtrapolationWarning` now fires on every affected load carrying the
fraction.

**The default is bit-identical**: 12 of 12 recorded quantities reproduce exactly,
and `test_default_is_bit_identical` requires `np.array_equal` on `r`, `gb`, `go`
for all twelve clusters against the pre-patch body inlined verbatim. 93 tests
pass; `prove_test_fails_prepatch.py` shows **7 of 7** of the suite's demands fail
against the untouched pre-patch source.

**Five bugs the lane's own tests found**, one of which matters beyond this lane:
its first synthetic truth produced **negative temperatures** — a mid-bin pressure
anchor made `P(r_out) < 0` for 5 of 12 clusters, `log(kT)` went `nan`, and the
reconstruction silently dropped those points, **which flipped the sign of the
reported bias.** Also: a `deep`-mask test that admitted a one-sided-stencil
endpoint; `mode="drop"` initially silent while deleting 93 points; a bare
`except Exception` in `Bench._load` swallowing the `forbid` error into a silently
missing probe; and a pre-patch proof that counted warnings instead of reading
them, scoring a false pass because astropy emits two FITS warnings of its own.

## AV.3 Three of nine recorded verdicts change

93 of 588 points (**15.82%**), all outermost, zero inner extrapolations.

| quantity | clamp | drop | loglinear | verdict moves |
|---|---:|---:|---:|---|
| within-cluster radial slope | -0.4817 | -0.4499 | -0.4590 | no |
| Spearman(r/R500, residual) | -0.7871 | -0.7310 | -0.7862 | no |
| **rho_T (kT vs excess, n=12)** | **0.6154** | **0.3287** | 0.6503 | **YES** |
| **p_T** | **0.0374** | **0.2970** | 0.0260 | **YES** |
| c70 dln(exc)/dln(kT) | +0.5165 | +0.2923 | +0.5491 | no (strengthens) |
| c71 median-ratio (power C) | 0.0683 (p .064) | 0.0504 (p .072) | 0.0692 (**p .0487**) | decided by policy alone |
| **kappa** | **1.563e5** | **1.892e5** | 1.556e5 | **YES** — drop lies outside the recorded 68% interval |

The `p_T` permutation test was **sized first**: realised FPR 0.040 +- 0.010
against a nominal 0.05, correctly calibrated — but at n = 12, one cluster's
median moving 4% takes rho_T from 0.615 to 0.650. A placebo confirms it is
*which* points, not how many: dropping 93 **random** points moves the slope by
-0.0030 +- 0.0115 against the real +0.0317, i.e. p = 0.005.

## AV.4 The honest reach of X-COP, and two claims that die

    all 12 clusters have measured temperature only to   r/R500 = 0.78
    half reach                                          0.92
    ONE of twelve reaches                               1.00
    R200 (1.52 R500)                                    0 of 12
    the recorded "crossing at 1.9-2.5"                  0 of 12

**The outermost X-COP point of any kind is 1.519**, so the crossing is beyond
every X-COP datum — not merely beyond every measured temperature. `g_bar = a0` is
bracketed in **0 of 12** within the bench's own cut.

Fitting only measured-T points gives a within-cluster slope of
-0.3755 +- 0.1535 and an excess of **1.493** at the last measured T radius.
Extrapolating to R200 — 0.22 dex beyond the data — gives **1.22, 16-84th
[0.93, 1.54]**, with 3 of 12 clusters below 1.0. So **X-COP neither contradicts
nor supports "factor 1.4 at R200"; that number came from lensing and X-COP cannot
corroborate it, because it measures no temperature there.** The excess reaches
1.0 at r/R500 = **2.49, [1.36, 6.89]**.

> **WITHDRAWN IN RUN BA.** Widening the interval was the wrong repair. A crossing
> radius obtained by extrapolating a CLAMPED temperature profile past the last
> measured bin is not a measurement with a large error bar — it is not a
> measurement. The 1.9-2.5 crossing, the 2.49 [1.36, 6.89] that replaced it, and
> "factor 1.4 at R200" are all withdrawn outright. **X-COP is limited to
> r <~ 0.915 R500**; beyond that the honest entry is an absence.

## AV.5 Clamping is the worst available policy, and it is 92% of the bias

Synthetic truth, temperature falling outward in 12 of 12, no noise:

| policy | bias on the slope | response | % of observed |
|---|---:|---:|---:|
| `clamp` (the default) | **-0.0765** | 0.849 | 15.9% |
| `loglinear` | -0.0613 | 0.850 | 13.4% |
| `drop` | **-0.0339** | 0.929 | 7.5% |
| `full_coverage` | -0.0064 | 0.978 | 1.3% |
| `perfect` | -0.0042 | 0.993 | 0.9% |

**`drop` is least biased and tracks a real signal best; clamping is the worst of
the three and should not be the scientific default.** It remains the code default
only so that no recorded number moves without someone choosing it. Decomposing
the clamp's -0.0765: `np.gradient` -0.0042 (5.4%), coarse grid -0.0022 (2.9%),
**the clamp itself -0.0702 (91.7%)**.

**And it accounts for most of Run AT's pipeline bias.** Reproducing AT's noiseless
forward null bit-identically (S1 -0.2067, S3 -0.1359) and flipping one switch
inside AT's own machinery — the simulated cluster observed on a T grid extended
to the last density bin — drops the S3 bias to **-0.0487**. So **64.2% of AT's
29% is this bug** (73.6% on S1). The clamp alone is 18.2% of the observed slope
and everything else is 10.1%.

**A second, unrecorded consequence:** the clamp was biasing the *inferred R500*
low by **8%** (recovery ratio 0.9186 -> 0.9887). That touches Run AT's own R500
inference, and it is the kind of coupling that makes a shared-quantity audit
harder than it looks.

## AV.6 Two provenance findings, one of them a standing hazard

**Stale receipts.** `repro/inputs_c70.json` and `inputs_c71.json` pin a
20,139-byte `invariant_bench.py` at `fe817b22...` that lived in a scratchpad and
no longer exists; the surviving repo copy was 19,707 bytes at `00cfbf28...`.
Those receipts already pinned an uncheckable file before this patch moved the
hash again. They need **a legitimate reseal, not a hash edit.**

**A standing seal hazard, pre-existing and serious.** `Bench.__init__` calls
`_widebin()`, which returns hard-coded El-Badry boosts **from the source file
itself** — so **any bare `Bench()` loads a sealed probe, and every one of the 15
lanes in the inventory does exactly that.** The values are a published summary
compiled into the source rather than a data file, which is why it was never
noticed. It is nonetheless the mechanism by which a "sealed" holdout stopped
being sealed, and it corroborates §AS.1 from the code side: the wide binaries
have been in the room the whole time. Any future seal must be enforced by the
loader, not by intention.

---

# Run AW — Stage 10: the confirmation budget is 23/25 spent, and nobody was counting

Code and record in `work/wellnet-2026-09/confirmation/` — `ledger.py`,
`reserve.py`, `seal.py`, `ledger.json`, `seal.json`. **No observational data is
opened by this lane**; sealing a dataset means recording its identity, never its
content. First work directed by the charter
(`C:/Users/henry/dev/invariant-gravity-discovery-charter.md`), whose Stage 10
states the governing rule:

> *"Once any numerical result has been examined, that sample is no longer
> pristine confirmation data."*

## AW.1 The ledger

Every dataset the programme has touched, classified by searching the research
record for evidence of **scoring**, not merely of mention:

    PRISTINE      2 of 25
    VALIDATION   23 of 25   -- spent; cannot confirm anything, however
                               carefully quarantined afterwards

SPARC (49 scoring contexts), X-COP (28), CLASH (21), MaNGA (21), SAMI (14),
eFEDS (12), DiskMass (9), Pantheon+ (8), DECADE (8), VoidFinder (8), Planck (7),
REVOLVER (6), LoCuSS (4), DESIVAST (2), Frontier Fields (2), ACCEPT (1),
SN Refsdal (1) — all spent. KiDS and the wide binaries are spent by §AS.1.

**The programme burned 92% of its corpus without anyone tracking it.** That is
the finding, and it is a process failure rather than a scientific one.

## AW.2 The classifier's own false-positive, caught by a second channel

The first version called **galstreams PRISTINE**. It is not — Run AR scored it
(68 usable-3D, 29 usable-6D, A_dyn measured). The classifier missed it because
Run AR's *scoring* text says "streams" while naming the catalogue only in its
data-integrity section.

**A false PRISTINE is the worst error this lane can make**: it would burn an
already-spent dataset as confirmation and produce a fraudulent-looking success.
So the classifier is now a **screen, not an oracle**, and a dataset is called
pristine only when three independent channels agree: no scoring context in the
record *with aliases*, no reference anywhere in the lane source tree, and no
manual override.

**The second channel immediately caught three more false PRISTINEs** that the
record alone had cleared — lanes had acquired data they never wrote up. Which is
the general lesson: *the research record is not a complete account of what the
programme has touched.*

## AW.3 The scoping rule: by data product, not by survey name

Chasing those three produced the rule that matters:

> **A survey can be simultaneously spent and pristine.**

  * **MUSE** appears in ~98 lane files as **spectroscopic redshifts** setting
    cluster membership — spent. Its **internal velocity dispersions** (Granata
    2026, 213 of them) were inventoried but never entered a gravity statistic —
    a row count is not a result.
  * **Gaia** appears in ~178 lane files as an **astrometric frame** ("RA/Dec
    aligned to Gaia DR2") — a coordinate calibration, not a gravity measurement.
    Its **dynamical products** are untouched.

Naming the survey would have either burned the pristine half or falsely cleared
the spent half. **Scope the reserve by product.**

## AW.4 The reserve, sealed

| product | tier | why it survives |
|---|---|---|
| **SPT clusters** | A | 0 record mentions, 0 lane-tree references |
| **X-GAP** | A | record L5425 *proposes* it — "the obvious next lane". Never opened |
| **CLoGS** | A | a lane attempted acquisition and recorded a confirmed **absence** |
| **Gaia dynamical products** | B-scoped | frame use only; wide-binary products excluded |
| **MUSE/Granata dispersions** | B-scoped | redshifts excluded; dispersions inventoried only |

**CLoGS is pristine for an instructive reason.** `potential-depth/code/probe.py`
line 18 labels `J/A+A/601/A95` as O'Sullivan's CLoGS, but that identifier is
Calabro+2017, "Star-forming dwarfs at intermediate-z in VUDS". The probe
therefore **never fetched CLoGS** — which is precisely why the data stay clean.
The same VizieR mislabelling family is already on record; the ID must be fixed
before any future acquisition, and the failed attempt cost nothing because a
failed acquisition examines no result.

## AW.5 The statistic, frozen before any law exists

The charter requires freezing the model *and* the statistic, then evaluating
once. No law is promoted, so the model slot is **empty by design** — but the
statistic is declared now, so it cannot later be chosen to fit:

    primary     median |log10(g_pred / g_meas)| over the object's MEASURED
                radial range, on RAW observables, after both the candidate law
                and the instrument forward model are applied
    comparator  the same quantity for the RAR with a0 frozen at its
                galaxy-calibrated value and no free parameter
    decision    the candidate must beat the comparator with every parameter
                frozen; a tie or a loss is a null result and is reported as one

    forbidden   refitting anything on the reserve, global or per-object;
                choosing the radial range after seeing residuals;
                dropping objects after seeing residuals;
                reporting a subset without the full-sample number beside it;
                a second evaluation of the same reserve for the same law family

    precondition  responsiveness d(estimate)/d(injected) must be measured on
                  synthetic data and reported BEFORE the reserve is opened

    evaluations permitted: 1        used: 0

Note the "measured radial range" clause: it is there because of Run AV, which
found the X-COP relation quoted to r/R500 = 1.52 while temperature is measured to
a median of 0.915 and **0 of 12** clusters reach R200.

## AW.6 The tripwire, and an honest limitation

`reserve.py` scans the lane tree and reports contact with the reserve, with 18
hand-verified benign contacts allowlisted *with reasons* so a genuinely new
reference stands out. It currently returns **clean**.

**But it cannot police Gaia, and pretending otherwise would be worse than not
having it.** "Gaia" is cited by essentially every modern astronomy paper, so over
a tree containing downloaded eprints, `.bbl` files and SESAME name-resolver
output, the scan cannot separate a citation from an ingest. After allowlisting
bibliographies, paper sources and acquisition scratch, what remained was still
prose. **Gaia's reserve is therefore policed by an explicit acquisition
manifest** — a lane may use only a Gaia dynamical product it has declared first —
and the scan defers on it by name rather than reporting a false clean. The scan
is reliable for distinctive names (SPT, X-GAP, CLoGS, Granata) and unreliable for
ubiquitous ones; that asymmetry is now in the code.

## AW.7 The seal must be enforced by the loader, not by intention

Run AV found the mechanism by which the last seal failed: `Bench.__init__` calls
`_widebin()`, which returns hard-coded values **from the source file itself**, so
**any bare `Bench()` loads a sealed probe — and all 15 lanes that import `Bench`
do exactly that.** No lane intended it and none would have caught it, because
there was no loader-level check.

That is the design requirement this lane inherits: a sealed dataset must be
unreachable, not merely undiscussed. The reserve's tripwire is a start; the next
step is a loader that refuses.

## AW.8 What this costs the programme

**With 5 reserved products and one evaluation each, the programme has at most
five confirmation shots left** — and three of them are group- or cluster-scale
X-ray samples that test the same regime. There is no reserved galaxy rotation
sample, no reserved lensing sample, and no reserved local-gravity probe. A
candidate law that needed to demonstrate cross-channel transfer on untouched data
**cannot currently do so in more than one channel.**

The charter's Stage 10 assumed a confirmation set exists. It does not, in any
meaningful sense, and acquiring more is now a precondition for the charter's
final deliverable rather than a tidy-up. **The cheapest repair is to stop
spending**: every lane from here should declare, before it runs, which datasets
it will touch — which is what the manifest above is for.

---

# Run AX — CLASH is inadmissible, null-consistent, and flat on its own full sample

Full record in `work/wellnet-2026-09/clash-audit/REPORT.md` (every number rendered
programmatically); 13 code files, 8 result JSONs, `ACQUISITION.json`, 18/18
self-tests. KiDS and the wide binaries never loaded. This completes the half of
the R500 audit that Run AT could not reach.

Run AT protected X-COP with two structural facts. **CLASH has neither**, and the
audit finds against it three separate ways.

## AX.1 The numerator and the x-axis are two functionals of ONE two-parameter fit

| quantity | derived from | root |
|---|---|---|
| numerator g_obs(r) | `G M_NFW(<r \| M200_i, c200_i) / r^2` | an NFW fit to a GR convergence map |
| denominator g_bar(r) | `G [M_gas + M_star] / r^2` | Chandra X-ray + stellar |
| x-axis radius r | fixed grid, 14-600 kpc | none |
| x-axis R500_i | `M_NFW(<R500 \| M200_i, c200_i)` | **the same NFW fit** |

This is not "shares inputs". **It is one two-parameter fit read twice**, and the
lane proved it rather than asserting it: regenerating Tian's published
`log g_tot` from Umetsu's (M200c, c200c) alone reproduces **all 84 rows to
sd 0.0084 dex**, ten times smaller than the quoted error. **84 rows carry 40 free
numbers, and with per-cluster levels the entire radial shape is one number per
cluster — c200_i.**

**No cancellation lemma exists.** X-COP's protection required the numerator
tabulated in R500-scaled units; Tian tabulates absolute m/s^2 against absolute
kpc, so R500 is not an input and there is nothing to cancel. Moving the mass that
*generates* R500 moves the numerator by **2.02 dex per dex of R500** — against
X-COP's 1.6e-13. The induced slope is **-2.02** (y) and **-4.04** (a0),
sign-definite negative. 64% of the R500 spread is quoted measurement error shared
with the numerator.

**And the within/between protection is inverted.** X-COP was 90.3% within-cluster,
where a per-cluster normaliser provably cannot reach. CLASH is **66.3% between**.
Worse, its radial grid is common across clusters, so at fixed r,
`log(r/R500)` varies *only* through R500 — **the tautology in pure form**. It
shows: `corr(excess, log R500)` climbs 0.07 -> 0.15 -> 0.53 -> **0.71** from 100
to 600 kpc, while the *independent* Chandra R500 moves the other way.

## AX.2 Null-calibrated, the trend does not survive

Forward null — flat truth, Abel projection, NFW fit over R <= 2.9 Mpc, publish —
with noise calibrated to both of Umetsu's error budgets:

    pooled slope (y)    observed -0.174   null -0.182 +- 0.035   z = +0.23, pct 58
    pooled slope (a0)   observed -0.459   null -0.466 +- 0.068   z = +0.10, pct 53
    the +0.71 contrast  observed +0.71    null +0.654 +- 0.141   z = +0.42

    83-85% of the observed slope is manufactured by the NFW template
    with NO NOISE AT ALL          (Run AT's X-COP figure was 29%)

    responsiveness 0.199 -- the pipeline attenuates a true trend 5x
    implied true slope +0.040 +- 0.175, consistent with zero
    no limit tighter than |s| < 0.38

    FPR, R500-label permutation   0.855      (nominal 0.05)
    FPR, naive t-test             0.970

**A test with a false-positive rate of 0.97 is not a test.** Both are worse than
Run AT's 0.53-0.70, and the pattern is now consistent: every obvious permutation
scheme in this problem is anti-conservative because the permuted label is itself
built from the measured quantity.

The obvious escape — a broken-power-law truth with a break radius the data cannot
see — is **not** undecidable. Umetsu measured Sigma out to 2.9 Mpc even though
Tian tabulates only to 600 kpc, and truths imposed past ~1 Mpc are excluded at
chi2/dof = 5-17. **The admissible nulls are exactly the ones containing the
observation.** The single residual — a within-cluster slope at 2.8 sigma —
reduces to the published NFW concentrations sitting **8% (0.029 dex) above** a
flat-excess fit, which is **0.22 of one cluster's own quoted c200 uncertainty.**

## AX.3 Inadmissible under the no-dark-matter-presupposition rule

Every published CLASH lensing product was checked: NFW parameters, overdensity
masses, Zitrin parametric strong-lensing masses, S/N values, and a non-parametric
kappa that exists only inside figures. **All are fitted mass models.** Umetsu+2016
has no VizieR entry at all — verified properly, with the identifier echoed back in
the error and no `CatalogsExamined=` fallback, plus a `METAcat title=*CLASH*`
search returning 14 catalogues without it and a positive control present.

**The numerator cannot be rebuilt from raw shear, because it IS a two-parameter
NFW fit to a GR-derived convergence map.** That is a stronger version of the
Run AL.3 failure: there an amplitude was *selected* against published lensing
masses; here the observable *is* one. Under the standing constraint — a
parametric lens model whose mass is tied to light by construction is not a raw
observation — **CLASH is inadmissible as evidence for or against a gravity law.**

The masses were nonetheless acquired properly, from the arXiv e-print source of
1507.04385v4 (Tables 1-3), together with Donahue+2014 CLASH-X (1405.7876v3),
which supplies the Chandra hydrostatic r500 — **the independent radius Run AT
could not build for X-COP.** That acquisition is reusable even though this
verdict is negative.

## AX.4 Corrections owed to the record

  * **"CLASH has no object identity" is a BENCH DEFECT, not a data limitation.**
    The bench's CLASH input is Tian+2020 `fig2.dat`, 84 rows over **20 named
    clusters**; `invariant_bench._clash()` reads `q[2], q[3], q[4]` and throws
    away `q[1] = AName`. Every per-object analysis this programme declined to run
    on CLASH was declined for a reason that was not true.
  * **Lane 12 used ONE pooled R500 ~ 1372 kpc for all of CLASH.**
  * **Every CLASH point sits at r/R500 <= 0.59**, so R500 is 2.8x beyond the
    outermost measurement — and it is a property of the NFW fit, not of the data.
    This is the same disease Run AV found in X-COP (relation quoted to 1.52,
    temperature measured to 0.915), in a more extreme form.
  * The record's `-0.347 +- 0.057` **reproduces exactly** (-0.3495 +- 0.0498,
    10 of 11 negative). **The number is right; the inference is not.**
  * **On the full 84 points the RAR-residual slope is +0.020 — flat.** The trend
    exists only after dropping the BCG points or switching to the a0
    parametrisation. Neither choice was declared in advance.

## AX.5 A shared-denominator instance found in a CONTROL, which touches Run AT

Six bugs surfaced in the lane's own tests: Abel truncation losing **21% of Sigma**
at the outer fit radius; an inner-sphere mass written rather than added
(M[0] > M[1] by 180x); the null's R500 population 1.2-1.7x too large; **a c200
uncertainty 4x too small**, because a coherent amplitude term reproduces
`e_M500` but not `e_c200` — which had inflated one statistic's z by 4x; and a
shadowed accumulator that silently truncated the report to 6 bytes.

The sixth is the one that generalises. The lane's contamination diagnostic was
**wrong twice**, and the corrected answer is that the worst normaliser is
**R_b,M at -0.99** against the baryon amplitude — *not* R_b,gas.

**This is the ninth shared-quantity instance and the first found in a CONTROL
rather than a measurement.** The lesson is sharp: **a "baryon-only" radius is not
automatically clean, because `g_bar` already sits in the excess's denominator.**
Run AT's reassurance rested on `r/R_b,gas` and `r/R_b,ne` holding the slope at
-0.4875 and -0.4841; those particular normalisers are not the -0.99 case, but
**AT's baryon-only control must be re-audited against this diagnostic before it
is quoted again**, because it was chosen on the assumption that baryon-only
implies independent, and that assumption is now measured to be false in general.

## AX.6 What the cluster claim now rests on

CLASH's FITTED MASS PRODUCTS are out — inadmissible on provenance,
null-consistent on significance, and flat on its own full sample. **Corrected in
BA.2: this retires the derived masses, NOT the survey.** CLASH's calibrated
images, background ellipticities, image-family positions, arc shapes, source
redshifts, time delays and member photometry remain primitive observations and
stay potentially admissible once a new-theory forward model exists. X-COP survives Run AT's tautology audit but Run AV
showed it measures no temperature past r/R500 = 0.915 and reaches R200 in 0 of 12
clusters. **The surviving cluster-only excess therefore rests on LoCuSS weak
lensing and on X-COP inside its measured range**, and every claim about R200, the
crossing radius, or self-similar organisation beyond ~0.9 R500 is currently
unsupported by either.

---

# Run AY — Run AT's baryon-only control was vacuous, not passed

Code `work/wellnet-2026-09/r500-audit/reaudit_baryon_radii.py`, results
`reaudit_baryon_radii.json`. Demanded by AX.5, which found — in its own control,
which is why it generalises — that a "baryon-only" radius is not automatically
independent, because `g_bar` already sits in the excess's denominator.

## AY.1 Both of AT's normalisers are at |corr| = 1.0000, worse than AX.5's case

Perturbing the gas amplitude and watching the normaliser and the excess co-move:

    normaliser   d log R_b / d amp   d log E / d amp   corr(R_b, E)
    R_b,gas          +0.5801            -0.8970        median -0.9993, worst -1.0000
    R_b,ne           +0.3623            -0.8914        median -0.9993, worst -1.0000

AX.5's threshold for concern was |corr| near 0.99. **Both sit at 1.0000.** The
mechanism is exactly as AX.5 described: `R_b,gas` is defined as the radius where
the mean enclosed *gas* density reaches a threshold, so it is an
amplitude-dependent radius whose amplitude is `M_gas` — precisely what `g_bar` is
built from.

## AY.2 But it does not overturn AT's slope, and the reason matters

`R_b` is a **per-cluster constant**, and by the rank identity Run AT itself proved,
a per-cluster constant cannot move a within-cluster slope. Measured directly, at
five gas amplitudes spanning +-0.10 dex:

    slope under r / R_b,gas   [-0.1902 -0.1995 -0.2085 -0.2173 -0.2257]
    slope under physical r    [-0.1902 -0.1995 -0.2085 -0.2173 -0.2257]

**Bit-identical at every amplitude.**

## AY.3 The correction, which is to my own reporting of Run AT

I wrote, of the four radial definitions: *"the trend survives under radii that
contain no total mass at all, which is exactly the test the critique asked for."*
**That is wrong.** The two baryon-only normalisers are not independent — they are
maximally contaminated — and the reason the slope does not move is that they are
*algebraically incapable* of moving it.

**The control was VACUOUS, not passed.** It could never have been informative
about the within-cluster statistic, so it is not evidence for the trend, and I
reported it as though it were. The reviewer's Job-2 request — four radial
definitions including a baryon-only one — cannot be satisfied by *any*
per-cluster normaliser for a within-cluster statistic. Answering it requires a
radius that varies WITHIN a cluster, or a between-cluster statistic with its own
null.

What survives of AT untouched: the forward-null result (z = -16.1), the
cancellation lemma, the within/between decomposition, and the rank identity
itself. Those never depended on the baryon-only radii.

## AY.4 One real systematic the exercise did surface

A +-0.10 dex gas-amplitude error moves the within-cluster slope by **0.0355** —
about 7% of the observed -0.4996 — but it does so through `g_bar` inside the
excess, **not** through the normaliser. That is a genuine, separately reportable
systematic on the slope, and it had not been quantified.

---

# Run AZ — Stage 0: the search has explored one corner of a 15-axis space

Code and results in `work/wellnet-2026-09/grammar/` — `axioms.py`,
`axioms.json`. No data opened. First construction of the charter's Stage 0
grammar, which asks for a specification of allowed source types, field ranks,
locality, directions, memory and path operations, matter-light relationships and
conservation properties, *"so that the final search is over sparse combinations
of admissible physical building blocks, not arbitrary columns."*

## AZ.1 The size of the space

The charter's fifteen axiom axes, transcribed verbatim, carry 78 axis-values
between them and a product of

    32,006,016,000 distinct universes

That number is not a search target — most combinations are incoherent, and the
compiler exists to say which. It is a denominator.

## AZ.2 What the programme actually visited

**Eleven families** have been constructed and scored across the whole programme:
Newton, AQUAL, QUMOND/RAR, families B through E, the tidal-gated scalar, the
external-axis tensor, the nonlocal path kernel, and the path-redshift branch.
Each commits only **2 to 8 of the 15 axes**; the rest are left implicit.

| axis | options | touched | coverage |
|---|---:|---:|---:|
| locality | 5 | 4 | 80% |
| directionality | 7 | 4 | 57% |
| superposition | 6 | 3 | 50% |
| axis_origin | 4 | 2 | 50% |
| field_type | 7 | 3 | 43% |
| conservation | 3 | 1 | 33% |
| matter_light | 5 | 1 | 20% |
| geometry | 6 | 1 | 17% |
| cosmology | 6 | 1 | 17% |
| source | 7 | 1 | 14% |
| **propagation** | 5 | **0** | **0%** |
| **eff_dimension** | 3 | **0** | **0%** |
| **equivalence** | 4 | **0** | **0%** |
| **vacuum** | 6 | **0** | **0%** |
| **initial_conditions** | 4 | **0** | **0%** |
| **TOTAL** | **78** | **21** | **27%** |

**Five of fifteen axes have never been varied at all.** Not "explored and
rejected" — *never varied*: every family silently takes the same implicit default
(instantaneous propagation, three-dimensional spreading, an exact equivalence
principle, a passive vacuum, standard initial conditions). **An axis that is never
varied is not an axis; it is an unexamined assumption.**

And the most-explored axes are the ones that cost nothing to vary — locality at
80%, directionality at 57% — while **source sits at 14%: every family this
programme has ever built takes rest mass as its only source.** The charter lists
seven options there, including the full stress-energy tensor, and the failed
pressure hypothesis was the single attempt to leave that corner.

## AZ.3 The 3,123-candidate tournament, in perspective

The tournament that felt exhaustive was 3,123 **parameter settings inside one
corner** — source = rest mass, geometry = force in fixed space, propagation =
instantaneous, vacuum = passive, equivalence = exact — with the variation
confined to field type, directionality, locality and the gate's functional form.
That is precisely the charter's diagnosis of the earlier exhaustive run:
*"generating more powers and exponents from the same columns cannot create new
physical information."*

## AZ.4 63% of what can even be assessed is out of reach

For the eight axes the record can speak to, each value was classified as
reachable, partial or unreachable **by the current bench** — meaning it can be
both expressed and scored:

    of 41 axis-values assessed:  13 reachable,  2 partial,  26 UNREACHABLE (63%)

    field_type    3 reachable, 4 unreachable  -- no vector-potential sector at
                                                 all; Run AU had to declare it
                                                 outside the compiler's class
    geometry      1 reachable, 5 unreachable  -- no relativistic solver; the
                                                 lensing closure is IMPOSED,
                                                 not derived (Run AL)
    propagation   1 reachable, 4 unreachable  -- static solver only
    vacuum        1 reachable, 5 unreachable  -- no latent-field dynamics
    equivalence   2 reachable, 2 unreachable  -- no composition channel
    matter_light  2 reachable, 1 partial, 2 unreachable -- no spectral or
                                                 polarization channel
    cosmology     2 reachable, 1 partial, 3 unreachable
    conservation  1 reachable, 1 unreachable  -- no candidate has ever declared
                                                 a momentum carrier

**The unreachable set is not a list of bad ideas.** It is the list of physics the
bench cannot express or cannot score, and making it visible is what Stage 0 is
for. Three entries are already load-bearing elsewhere in this record: Run AU had
to put the vector-potential sector out of scope; Run AL's slip is a fitted
closure standing in for the absent relativistic geometry; and Run AP noted that
no candidate carries a momentum carrier, which is why Gate 4 reads as a to-do
list rather than an extermination.

## AZ.5 What follows, and a correction to the action-first plan

The charter's proposed inversion — generate candidates **from admissible
actions** rather than screening arbitrary force laws — was scoped in Run AU at
86.1% of defect instances prevented by construction. AZ makes the complementary
point, and it changes the priority:

**The binding constraint is not the generator but the SCORER.** Adding axes to
the grammar is cheap; **63% of the values already in it cannot be scored**, so a
richer generator would mostly emit candidates the bench must decline to evaluate.

The next capability investment that actually enlarges the search is therefore a
solver and a scoring channel for one currently unreachable axis. On this record
the cheapest is **propagation** — a time-dependent channel — because Run AP
already integrates growth and Run AK already carries a path-geometry branch, so
two of the three pieces exist.

---

# Run BA — the standing rule, and five corrections from the second external review

A corrections run. The review's central instruction is adopted as a standing rule
and takes precedence over the charter's stage ordering wherever they conflict:

> **No expanded law tournament until one complete scene can be generated, solved,
> and scored against root observations.**

The next milestone is therefore **not** more axis coverage. It is *one resolved
development cluster, represented from primitive observations, run through several
genuinely different universe models, with the same law predicting internal
member-galaxy motion and photon propagation through the same cluster.*

The timing of the criticism is fair: Run AZ spent its effort computing axis
coverage, which is precisely the "catalogue possibilities instead of finishing an
end-to-end test" failure mode. AZ's diagnosis stands; its *priority* does not.

## BA.1 WITHDRAWN: every X-COP statement beyond the measured support

Run AV reported the crossing radius as `r/R500 = 2.49, [1.36, 6.89]`, calling the
interval "3.8x wider than recorded". **That was the wrong repair.** An interval
computed by extrapolating a clamped temperature profile past the last measured
bin is not a measurement with a large error bar; it is not a measurement.

**Withdrawn outright, not widened:**

  * the crossing at `r/R500 = 1.9-2.5`;
  * the value 2.49 [1.36, 6.89] that replaced it;
  * "factor 1.4 at R200";
  * any statement about the outer transition or behaviour at R200 from X-COP.

**The X-COP result is now limited explicitly to `r <~ 0.915 R500`**, the median
radius at which temperature is actually measured. Outside that, X-COP says
nothing, and the honest register entry is an absence rather than a wide interval.

**And the default changes.** The tempclamp lane built four policies; the review's
rule is right:

    fail closed outside the declared data support.

`forbid` becomes the default for any headline statistic. `clamp` remains only as
a bit-identical reproduction mode for re-deriving historical numbers, and any run
using it must print its extrapolated fraction.

## BA.2 CORRECTED: "CLASH is dead" is too broad, and I wrote it

Run AX's finding was about **CLASH-derived fitted mass products**, and I
generalised it to the survey. The correct scope:

    RETIRED, inadmissible as root evidence
      GR-derived convergence masses
      NFW masses reconstructed from those maps
      accelerations derived from the same fitted profile
      any trend against a radius derived from that same fit

    NOT retired -- primitive observations, still potentially admissible
      calibrated cluster images
      background-galaxy ellipticities
      image-family positions and arc shapes
      source redshifts
      time-delay measurements
      member-galaxy photometry and spectra

**The register entry becomes: "CLASH fitted mass products retired. CLASH root
imaging and lens constraints remain potentially admissible once a new-theory
forward model exists."** That distinction is the charter's whole point — score
detector-facing quantities after the candidate law and the instrument model are
both applied — and collapsing it would have thrown away exactly the data the
charter asks for.

## BA.3 CORRECTED: the propagation priority was reasoned from cost, not from what binds

Run AZ concluded that a time-dependent channel for the propagation axis was "the
cheapest genuinely enlarging investment". **Cheapness was the wrong criterion.**
The binding constraint remains the inability to build and score a resolved static
scene from root matter and photon data, and adding time dependence introduces
more unmeasured state variables before the static source-to-observable chain has
been validated at all.

**Corrected priority, adopted:**

    1. resolved scene and raw-observation scorer
    2. vacuum / latent-field dynamics
    3. matter-light metric closure
    4. external-axis and well-network tensor response
    5. finite propagation or memory
    6. linear structure formation
    7. cosmological initial-condition alternatives

Time dependence comes **after** a static tensor or nonlocal law can be propagated
through both matter and light in one scene.

## BA.4 CORRECTED: 32,006,016,000 is not a count of anything physical

AZ called it "a denominator, not a search target", which was not enough of a
caveat. The Cartesian product over axis labels counts combinations that are
logically incompatible, mathematically equivalent, redundant under field
redefinition, incomplete without another axis, or unscoreable. Concretely: a
path-dependent redshift law *requires* a propagation and clock model; nonmetric
geometry has no meaning against a fixed-space force scorer; initial-condition
alternatives require time evolution; a vector field requires a coupling and a
momentum-exchange mechanism; matter-light non-universality requires two sectors;
and effective dimensionality may be an emergent property of a solution rather
than an axiom at all.

**Replaced by a typed compatibility graph**

    sources -> fields -> dynamics -> matter coupling -> photon coupling
            -> observables

with each node declaring prerequisites, compatible choices, required solver,
required root data, known equivalences, and its decisive observable. **The
quantity to count is observationally distinct, internally complete model
classes** — not label combinations. The axis table survives as a coverage map,
which is all it was ever good for.

**And coverage is not a target.** Varying an axis to raise a percentage is not
progress. The axes that matter are the ones this programme's own hypothesis needs
— a non-passive vacuum or spacetime state, a directional or relational response,
nonlocality or propagation, a consistent matter-light relation, and eventually
time evolution — and they matter because they express the proposed physics, not
because they fill blank cells.

## BA.5 SHARPENED: the compiler must not encode the sought physics as forbidden

Run AU already labels rather than rejects one out-of-class case. The review
generalises it correctly, and the generalisation bites: **if reciprocity,
scalar-potential integrability, metric compatibility or exact equivalence are
compiled in as unquestionable, the machine deletes the very physics the charter
is looking for.**

The requirement is not "ordinary momentum conservation". It is:

> **Identify where energy and momentum go.**

A non-reciprocal matter force is admissible if another dynamical field carries
the compensating momentum. A force with spatial curl can come from an action when
velocity or vector-potential terms are present — which is AS.6 restated as a
design rule rather than an erratum. Such models sit outside the current static
scalar scorer; that is a statement about the scorer.

The verdict vocabulary gains one term, and `unsupported_by_current_scorer` must
never be reported alongside the rejection bins:

    mathematically inconsistent | incomplete as specified |
    equivalent to another class | convention dependent |
    non-identifiable with current data | UNSUPPORTED BY CURRENT SCORER |
    admissible and testable

## BA.6 The confirmation register needs a finer ladder, and product scoping is not independence

The AW ledger used a binary spent/pristine split. Adopted instead:

    Mentioned      named in prose or metadata            -- NOT spent
    Acquired       bytes downloaded, values not inspected -- NOT spent
    Transformed    processed or joined                    -- NOT spent
    Scored         entered a model comparison             -- SPENT
    Inspected      numerical outcome seen                 -- SPENT
    Decision-used  steered model or test development      -- SPENT

Appearing in a bibliography does not spend a dataset; that resolves the Gaia
tripwire noise properly instead of allowlisting it.

**And product-level sealing is not statistical independence.** MUSE redshifts and
MUSE dispersions share objects, selection, spectra, calibration and reduction;
Gaia frame and Gaia dynamics share upstream systematics. The register must record
four separate things:

    untouched outcome | untouched objects | untouched survey |
    untouched reduction pipeline

The AW reserve currently guarantees only the first, and in two of five cases not
even the second.

## BA.7 A2029 can develop the pipeline; it cannot validate anything

The environment inventory found A2029 is the only system with the several-Mpc
spectroscopic environment the well-network and void-direction hypotheses require.
**It is also already heavily spent.** So the gold sample for every environmental
claim has an effective size of **one, and that one is validation data.**

Consequence, stated plainly: **no population-level claim about gravity aligning
with external wells, void-directed response, well-network tensors, path geometry
through the cosmic web, or environment-dependent cluster lensing is available to
this programme at present**, and none will be until additional resolved scenes
with multi-Mpc environmental spectroscopy are acquired and sealed. Developing and
debugging on A2029 is legitimate; validating on it is not.

## BA.8 What is being built in response

  * **A run registry with invalidation** — the acquisition rules changed mid-flight
    in Run AZ and could not be pushed to running lanes, so a known-invalid
    assumption kept producing plausible output. Every run records its code commit,
    data-manifest version, schema version and active warnings; when a loader,
    interpolation or catalogue-validation rule changes, affected runs are marked
    `INVALIDATED_PENDING_RERUN` and their outputs are quarantined from the
    register automatically.
  * **Loader-level seal enforcement** — a regex tripwire is not enough, and
    `Bench.__init__` calling `_widebin()` proved that intentions and manifests are
    not either. Sealed products move outside the ordinary data tree; access
    requires an explicit one-shot confirmation token naming dataset, product,
    statistic, commit and output; constructors get no data-loading side effects;
    every open appends to a ledger; CI fails if an ordinary test touches a sealed
    product.
  * **Stage 4 as a sensitivity certificate**, ahead of any new Stage 6 search. No
    candidate/statistic pair opens real data until it demonstrates that the
    statistic moves when the claimed effect moves, that it is not restating a
    fitted normalisation, that null and signal pipelines are exchangeable where a
    permutation is used, that it has power at the *predicted* effect size, that
    support overlaps, that an injected law from **outside** the inference grammar
    is recovered, and that common nuisances do not manufacture the same signature.
    That single gate would have caught the kappa-invariant rank statistic, the
    vacuous per-cluster radius normalisation, the same-NFW-fit-on-both-axes
    result, several best-case injection controls, and the out-of-support
    temperature use.
  * **Catalogue contracts instead of accumulating heuristics** — the three-detector
    rule is still name-matching, and both prior detectors have now separately
    accepted wrong content. Each acquisition declares identifier, DOI, authors,
    year, table names, mandatory columns, units, row-count range, object
    identifiers, sentinel records, sky footprint and a post-canonicalisation
    checksum. For VOTables, **HTTP 200 is not success**: the parser must read the
    query-status metadata and fail on an error result, and a zero-row result is
    never an absence until success is verified.
  * **Object identity preserved structurally.** The CLASH input had names and the
    loader discarded them. In the scene graph that must be impossible, not merely
    discouraged.

The alternate-universe suite is additionally constrained against the **inverse
crime**: generators must not share basis, discretisation, solver or nuisance
assumptions with the recovery machinery, at least one must lie outside the
inference grammar, and calibration and audit simulation sets must be disjoint —
a 95th percentile set on one must be checked on an untouched other, never
declared correctly sized by construction.

## BA.9 The progress metric

Replaced. Not the number of universe combinations, the percentage of axiom values
covered, the number of compiler rules, or search throughput, but:

> **How many fundamentally different universes can the system distinguish from
> root observations in one complete physical scene?**

---

# Run BB — the run registry and the Stage 4 certificate, built and applied

Two of the review's operational demands, implemented and applied to the
programme's own work rather than demonstrated on toy input. Code in
`work/wellnet-2026-09/registry/` and `work/wellnet-2026-09/stage4/`. Neither
opens observational data.

## BB.1 The run registry, and six runs quarantined by it

The motivating failure was concrete: in Run AZ the catalogue-validation rule was
found to be wrong while three lanes were already running against it, and there
was no way to reach them. A known-invalid assumption kept producing
plausible-looking output with nothing in the pipeline to stop it entering the
register.

Five shared rules are now versioned, each carrying the history of *why* it
changed:

    catalogue_validation  v3   the three-detector rule (v2 was BROKEN: neither
                               detector is sufficient alone)
    temperature_support   v2   fail closed outside measured support (v1 clamped
                               silently)
    holdout_seal          v2   loader-level enforcement (v1 was intention, and
                               Bench.__init__ loaded a sealed probe)
    confirmation_status   v2   the six-level ladder (v1 was binary spent/pristine)
    identifiability_gate  v1   the Stage 4 certificate below

Applying it to the runs actually launched under the older versions:

    AT-r500-audit          INVALIDATED  catalogue_validation, temperature_support,
                                        holdout_seal
    AV-tempclamp           INVALIDATED  temperature_support, holdout_seal
    AX-clash-audit         INVALIDATED  catalogue_validation, holdout_seal
    AY-baryon-reaudit      INVALIDATED  temperature_support, holdout_seal
    IN-FLIGHT-transition   INVALIDATED  all three -- and still running
    IN-FLIGHT-scene        INVALIDATED  catalogue_validation, holdout_seal
    IN-FLIGHT-universes    VALID        synthetic; opens no catalogues

**Six of seven runs are quarantined pending re-run, two of them mid-flight.**
That is the honest state, and it is uncomfortable in the right way: Runs AT, AV,
AX and AY are all recorded above and all four now carry a machine-readable mark
saying their outputs may not enter the register until re-run under the current
rules. Their *conclusions* may well survive; the point is that the pipeline no
longer decides that by assumption.

**A bug in the first version, caught by using it:** a run stopped accumulating
reasons after the first rule matched, so `AT-r500-audit` showed only
`catalogue_validation`. A re-run would have fixed one cause and silently
inherited two others. Reasons now accumulate, and every run carries its full set.

## BB.2 Stage 4: the certificate refuses 5 of 5 failures this programme committed

The charter listed Stage 4 as "remove redundant information" and this programme
had it as ad hoc. The review makes it a precondition: no candidate/statistic pair
opens real data until it demonstrates seven things.

    C1 responsive         dS/d(effect) non-zero over the physical range
    C2 not a restatement  S is not reproducible by moving a fitted normalisation
    C3 exchangeable       null and signal pipelines exchangeable under permutation
    C4 powered            power AT THE PREDICTED effect size, not a convenient one
    C5 support            the statistic reads only where data exist
    C6 out-of-grammar     an injection from OUTSIDE the grammar is recovered
    C7 nuisance-distinct  no common nuisance reproduces the signature

Its test suite is this programme's own history. **All five are refused:**

| case | fails on | measured |
|---|---|---|
| the monotone-invariant rank statistic | C1 | moves **0.000e+00** over the whole effect range |
| the per-cluster baryon radius control (AY) | C1, C2 | moves 0.000e+00; a pure normalisation reproduces **inf x** its range |
| CLASH r/R500 vs an NFW-derived excess (AX) | C2, C3, C4 | a normalisation shift reproduces **8.08x** the statistic's range; null mean −0.465 where zero was assumed; the predicted effect is **0.59 sigma** through a responsiveness of 0.199 |
| the degenerate in-grammar injection | C6 | recovers **0%** of an out-of-grammar law |
| X-COP quoted past measured temperature (AV) | C5 | **42.6% outside support** |

**And it issues a certificate to a well-posed control**, so it is a gate rather
than a rejector. That two-sidedness is the part worth checking, and it is checked
on every run.

**A second bug caught by using it**, and it is the programme's recurring one:
the harness selected its expected-failure set by testing whether the case name
contained the substring "control" — which silently excused
*"per-cluster baryon radius **control**"* and scored 4/4 instead of 5/5.
Replaced with an explicit flag. **Name-matching keeps costing this programme
correctness**, in VizieR identifiers, in the galstreams alias, and now in its own
test harness.

## BB.3 What this changes about how work proceeds

The standing rule from BA now has machinery behind it. Any future lane must:

  1. register its rule dependencies before it runs;
  2. obtain a Stage 4 certificate for its candidate/statistic pair before it
     opens real data;
  3. accept quarantine automatically if a rule it depends on changes.

None of that produces a gravity result, and it is not meant to. It is the
apparatus that makes the next gravity result mean something — and on the evidence
of five refusals out of five, its absence is why several previous results did
not.

---

# Run BC — a radius hypothesis, REFUSED by the gate; not an empirical result

> **STATUS, set in BE.1.** The heading of this run previously read *"the
> transition map answers RADIUS"*. That was wrong to state as a result and it is
> withdrawn. **A radius-dependent transition is an exploratory hypothesis
> generated by partly out-of-support and partly theory-contaminated data. It is
> not presently an empirical finding.** The model rankings below are recorded for
> software diagnosis only and must not steer candidate selection, data
> acquisition, boundary-rule choices, or any later claim of confirmation.

Full record in `work/wellnet-2026-09/transition/REPORT.md` (749 lines, rendered
from JSON), 16 modules, 19/19 tests. Certification in
`work/wellnet-2026-09/stage4/certify_transition.py`. **Registry status:
`INVALIDATED_PENDING_RERUN`** on catalogue_validation, temperature_support and
holdout_seal — recorded here with that mark, not despite it.

## BC.1 What the lane computed (DIAGNOSTIC ONLY -- see the status note above)

All three record numbers reproduce in **one forward framework with one frozen
law** — the RAR, a0 from SPARC, never refitted: eFEDS **0.982**, LoCuSS
**1.689**, strong-lens cores **4.062**, against the record's 0.981, 1.62 and
4.11.

| model | k | BIC | dBIC | fitted |
|---|---|---|---|---|
| **H_R (radius)** | 2 | 3549.03 | **0.00** | beta = -0.350 |
| H_T (transition) | 3 | 3550.78 | +1.75 | not bought |
| H_MR | 3 | 3555.04 | +6.00 | alpha = -0.063 |
| H_P (pipeline) | 3 | 3567.64 | +18.61 | |
| H_G (acceleration) | 2 | 3571.14 | +22.11 | |
| H0 | 0 | 3574.32 | +25.29 | |
| **H_M (mass)** | 2 | 3577.41 | **+28.38** | alpha = +0.020 |

**Mass is the worst model in the set — worse than no model at all.** On the
declared held-out prediction (train eFEDS + strong lensing, predict LoCuSS),
worst-case leave-one-out gives H_R **1.67 sigma** — the only model never
rejected — against H_M 3.18 and H0 3.87.

And the sharpest internal number: what eFEDS measures *by itself* versus what
each story needs — **radius 0.9 sigma, acceleration 3.3 sigma, mass 19.8
sigma.**

## BC.2 The gate refuses it, on two of four applicable checks

This is the first result produced after the Stage 4 certificate existed, so it is
the first to be gated rather than reported.

    PASS  C1 responsive   beta moves 0.650 across the declared windows
    FAIL  C2 restatement  choosing the radial window reproduces 1.62x the
                          statistic's entire effect range
    PASS  C4 powered      beta to +-0.05 against -0.35 is 7.00 sigma
    FAIL  C5 support      reads ln x = -3.5; eFEDS measured from -1.5;
                          28.6% outside support
    -> REFUSED

**C5 is the same failure family as the withdrawn X-COP R200 claim.** The
headline extrapolates eFEDS's beta *7x inward in radius* to the strong-lens
cores — and the lane's own occupancy table shows eFEDS contributes **zero
points** below ln x = -1.5 (r/R500 = 0.223), while the cores sit at ln x = -3.5
to -1.5 (r/R500 0.030-0.223). The two surveys barely overlap in radius at all.

**C2 is the lane's own dominant systematic, promoted from caveat to refusal:**

    all radii        beta = -0.400  [-0.450, -0.350]   n = 3365
    r/R500 < 2.0     beta = -0.250  [-0.400, +0.000]   n = 1036
    r/R500 > 2.0     beta = -0.800  [-1.000, -0.600]   n = 2329

**The inner and outer 68% intervals do not overlap.** The lane said so —
*"whether the Bahar Vikhlinin fit may be extrapolated past 2 R500 ... is worth
0.55, the entire disagreement"* — and the gate turns that into a verdict.

**The result is not wrong. It is not yet believable**, and it names precisely
what would make it so: the Bahar covariance published, resolved weak-lensing
profiles inside 0.3 R500 for ~13 massive clusters (20-30 with systematics), and
raw Subaru shear for LoCuSS. Statistics are not the limit; support is.

## BC.3 Four bugs, one of them in shared code again

  * **`pipeline.sigma_from_g`'s `Sigma_bar` is wrong at small radius** — 8.1% at
    27 kpc, 4.1% at 54, 2.1% at 108, with a **flat error curve against grid
    density**, so refining the grid never revealed it. Affects any lane using it
    inside ~50x its inner grid radius. Replaced by an exact form, SIS to 8e-6.
    Second shared-code defect found this round, after the temperature clamp.
  * **The declared radius axis was circular** — `ln(r/R500_dyn) = ln S/(3-m)`
    exactly for a single-aperture dataset, corr +0.885 on LoCuSS. Caught and the
    declaration amended **pre-data**. That is the R500 tautology a third time,
    and the first time a lane caught it in itself before scoring.
  * **The strong-lens internal radial slope is an artefact** (+0.205 +- 0.025,
    wrong sign; `d ln x/d ln theta = 1` exactly) **and was outvoting 3,365 shear
    points** — it moved beta by 64.7 in -2lnL where all of eFEDS moved it by
    12.3. Fixed by cluster-level aggregation.
  * **The transfer significances were inflated ~3.5x** by treating 49 image
    systems in 4 clusters as independent and omitting the held-out survey's
    prior.

## BC.4 What is not separated, and an admissibility flag

Mass and pipeline are **not** distinguished: both cluster samples sit a
consistent ~1.3x above the eFEDS radial law, described equally well by
`alpha ~ +0.09` or by a survey constant. LoCuSS carries no radial information
under either aperture.

And a flag that connects to Run AX: **LoCuSS `M_WL` is an NFW-fitted mass**, and
**no public per-source shear exists for any LoCuSS cluster.** The lane labels it.
By the AX standard that makes the LoCuSS leg theory-contaminated in the same way
the CLASH masses were — a fitted mass model standing where a raw observation is
required. It is not as severe (LoCuSS is not on both axes), but it is the same
class, and the surviving cluster excess now rests on **one admissible leg**:
eFEDS raw shear, inside its measured radial support.

No time delay is used anywhere in this lane. KiDS and the wide binaries never
loaded.

---

# Run BD — Stage 1 exists: the gravitational scene graph, and no cluster satisfies Corpus E

Full record in `work/wellnet-2026-09/scene/` — `SCHEMA.md`, `REPORT.md`,
`scene_results.json`, eleven modules, **38/38 tests**. Reproduces from
`python run_scene.py && python test_scene.py && python write_report.py`. All work
on synthetic scenes per the standing constraint. KiDS and the wide binaries not
loaded, queried or referenced — enforced by a test, as is a test that no file
outside the lane is read.

This is the charter's fundamental data object, which did not exist.

## BD.1 The contract is enforced at construction, not audited afterwards

15 node types, 10 edge types, 8 field types — the charter's own lists, checked by
test. **67 ontology quantities, each carrying all 17 contract items**, audit
reports `all_complete = True`.

The design decision that matters: **the contract cannot be violated, rather than
being checked later.** A potential declared `SHIFTS_BY_CONSTANT` cannot be
constructed without naming a boundary rule. A log of a dimensionful quantity
raises. An unregistered quantity cannot enter a scene, and **neither can a bare
number** — it must be `Fixed` or `Uncertain`, so "known" and "sampled" can never
be confused. Units are exact `Fraction` exponent vectors over (M, L, T, Θ, Q), so
dimensional consistency is a real test rather than a naming convention.

`bridge.py` verdicts a candidate from its list of consumed quantities alone,
**without opening a file**, and caught three real defects that way: a nonlinear
function applied to a dimensionful temperature, a turbulent velocity defined only
in one frame, and a well count that changes under deblending.

**Two new taxonomy branches earn their place**, and one of them is Run AX made
structural: `convention_dependent` (carrying the 0.87 dex gauge-rule spread
against a 0.9 dex margin) and **`theory_contaminated`** — a candidate scored
against a convergence map or an NFW-defined R500 is being tested against a
product of the theory it is meant to replace. That is now a compile-time verdict
rather than something an auditor has to notice.

## BD.2 Depth is absent, not noisy — and collapsing the scene costs 28%

The posterior is `n_3d(r) x N(v; sigma_los(r)) x p(morph|r) x S(R,z)`, with **no
term that is a mass model** — the profiles are Abel deprojections of observed
counts and observed velocities, asserted by test. Depths are drawn jointly
through a substructure bulk offset so the correlated lumpy geometry survives, and
credible intervals are calibrated against known synthetic truth at all four
levels.

Two numbers to carry:

  * **The velocity information is worth 4.6%.** The depth posterior is 0.954 of
    the density-prior-only width. *A scene ensemble does not recover depth; it is
    a way of being honest that depth is not there.*
  * **Collapsing to the mean scene understates every mean 3-D radius by 28%**,
    because it puts every member back in the plane of the sky.

## BD.3 The commutation gate refuses 7 of 8 substitutions — and sharpens the charter

`erased = 1 - dev(A.S)/dev(S)`, where `dev` is the candidate's deviation from a
linear control on the same scene with the same probe configuration. A linear law
has `dev = 0` by construction, so the gate cannot manufacture an erasure. Null
control against the closed-form shell-averaged Plummer potential: floor
**2.65e-4**.

**Replacing ~300 members with a spherical source, measured against the gate:**

    200 kpc   2.09%   REFUSE at 1% precision
    300 kpc   1.08%   REFUSE
    2 Mpc     0.11%   ALLOW

> **CORRECTED IN BE.2. This is NOT a reversal of the charter's 0.4%, and calling
> it one conflated two different observables.** The record already contained
> both, at Run V:
>
>     3-D shell-averaged |g|, lumpy/smoothed     1.0040   (max 0.77%)
>     projected deflection at 150 kpc            0.9731 +- 0.0183, ~3 sigma,
>                                                sign does NOT flip
>     projected deflection beyond ~300 kpc       mean on 1, sign flips between
>                                                realisations -- shot noise
>
> So the 2.09% at 200 kpc is **consistent with the previously measured ~2.7%
> inner projected-deflection systematic**, not a contradiction of the 0.77%
> shell-averaged monopole. The scientific lesson survives and is worth stating
> precisely: **smoothing member galaxies is harmless for large-radius monopole
> calculations and inadmissible for percent-level inner or directional lensing
> tests.** Future statements must name the observable — shell-averaged |g|,
> projected deflection, tangential shear, shear quadrupole, or local field near a
> member — because they differ by more than the effect being measured.

**And the sharpest result is a directional pair**: two laws of identical form and
amplitude, differing only in where the axis comes from.

> **RESTATED IN BE.3 as a transfer coefficient**, because "erases 120%" is not
> interpretable. With `R = dev(A.S)/dev(S)` — the fraction of the resolved
> deviation that survives azimuthal averaging, in the shell-radial observable:
>
>     R_source   = -0.200      (from source_axis_erased = 1.2002)
>     R_external = +1.298      (from external_axis_erased = -0.2983)
>
> **R_source = -0.200** means the averaged observable retains a fifth of the
> resolved signal *with the sign reversed* — at that magnitude, against the
> lane's 0.24% quadrature floor, it should be read as consistent with zero and
> the sign as not meaningful. **R_external = +1.298** means averaging leaves the
> external-axis law 30% STRONGER, by removing the competing source quadrupole.
> The complete erasures are cleaner: path, memory and network laws all give
> `R = 0.000` to machine precision.
>
> **The result is that averaging is not merely information loss — it biases one
> tensor mechanism relative to another by a factor of ~6.5 in this
> configuration.** That is Gate 1's identifiability question, measured rather than
argued — and it is the mechanism behind Run AO's finding that external-axis power
does not collapse as the source rounds.

## BD.4 No cluster satisfies Corpus E, and the binding constraints are named

Best is **MACS J1149 at 9 of 10**; A2029 at 5 of 10. Four layers newly acquired
(member IFU, SZ, time delays, environment) and joined to the existing seven.

  * **A public raw shear catalogue exists for one of seven targets** — Abell 370,
    18,556 sources to 6.2 Mpc — **and A370 is one of the two without resolved
    member Sersic fits. No target has both.**
  * **Exactly one target has a measured time delay** (SN Refsdal in MACS J1149).
    The whole-sky census of cluster-scale delays is four systems plus three
    cluster-lensed quasars.
  * **The IFU layer is not what the charter asked for.** Every HFF sigma is a
    single 1.5-arcsec aperture value, not a resolved map. The only resolved
    member kinematic maps anywhere are SAMI's, which cover no target cluster and
    one X-COP cluster.
  * **SZ and environment are anti-correlated** across the sample: the three
    southern primaries have zero SDSS and zero DESI, while the two clusters
    outside the ACT footprint have the best DESI. **Only A2029 supports an
    external-tidal-axis reconstruction** (15.8 Mpc, four independent layers) —
    confirming BA.7 from a second direction.
  * Strongest single asset: the X-COP `Y-PROF-COVMAT`, a *measured* Compton-y
    profile with full bin-bin covariance, geometrically derived only.
  * **Two layers are invisible to a catalogue search** — the 213 member
    dispersions and the Refsdal delays are machine-readable only *inside arXiv
    LaTeX*. Granata et al. deposit Appendix B at CDS; the dispersions are
    Appendix C. **New trap: a published data-availability statement can be
    narrower than the paper.**

BUFFALO carries an expiry date: if it lands with per-source shapes, weak lensing
goes from one cluster to six and time delays become the sole binding constraint.

## BD.5 Eight bugs from its own tests, and two the tests could not see

Documented at their sites: prior volume not equal to declared scene volume
(over-dispersed depths); an unchunked pair array; a probe-lattice quadrature
floor at 0.24% that **does not fall monotonically with point count**; a path law
self-normalised so the observable could not see it; a discrete shell
representation that made radial averaging appear to *amplify* a path law
twelvefold; S6 flagging **Newtonian gravity itself** as non-identifiable; and
importance reweighting collapsing effective sample size to 17/64.

The most important is conceptual: **the observable was erasing the direction
before the source averaging could** — so the gate was measuring the probe's
blindness, not the law's.

**Two were caught by reading the output rather than by any test**: the gauge flag
could never fire, and **reading a convergence map as data scored admissible.**
That second one is the `theory_contaminated` branch's reason for existing, and it
is a reminder that a test suite only checks what someone thought to ask.

## BD.6 New acquisition traps, including one that breaks a mirror assumption

  * **The VizieR fuzzy-fallback trap is MIRROR-DEPENDENT.** CfA returns a clean
    error where CDS serves 5.9 MB of an unrelated catalogue. The memory's advice
    to prefer the CfA mirror was right for a reason nobody had established.
  * **A silently aliased catalogue that `CatalogsExamined` cannot see** — a third
    instance of that detector failing.
  * **NOIRLab TAP returned errors as HTTP 200 VOTABLE**, zeroing every count
    *including the positive control* — so the positive control did not protect
    against it. A positive control only works if it can fail differently from the
    thing it guards.

## BD.7 Limits

The gate's floor is quadrature, not machine precision: a commutator below ~0.1%
is unresolvable. Everything ran on synthetic scenes, so the erasure fractions are
properties of a representative cluster, not measurements of A2744. The path and
memory laws are illustrative — **the gate is a measurement device, not a
theorem, and must be re-run per candidate.** Filament catalogues were only partly
surveyed. And whether an aperture dispersion can stand in for a resolved map is
untested; by this lane's own rule it must pass the commutation gate first.

---

# Run BE — corrections from the third review, and the infrastructure exit condition

## BE.1 The transition headline is withdrawn

Run BC was titled *"the transition map answers RADIUS"*, with the refusal
underneath. **That ordering was wrong.** A dramatic result followed by a caveat
reads as a result. The heading now says what is true: *a radius hypothesis,
REFUSED by the gate; not an empirical finding*, with a status note at the top of
the run marking its rankings diagnostic-only.

**A radius-dependent transition is an exploratory hypothesis generated by partly
out-of-support and partly theory-contaminated data.** It is not a result.

## BE.2 A quarantined result can still steer the researcher — and mine did

The registry marked the transition outputs `INVALIDATED_PENDING_RERUN`, and I
then reported their model rankings, compared hypotheses, and described what they
appeared to answer. **That is contamination even when the numbers never enter the
register**, because seeing that an invalid run favours radius shapes which
parameterisations get generated next, which data get acquired, which boundary
rules get emphasised, and which later result gets called confirmatory.

Invalidation now has **two** effects, and the second was missing:

    DIAGNOSTIC_ONLY            locate implementation defects ONLY. May not alter
                               hypotheses, priors, candidate selection,
                               acquisition priorities, or register conclusions.
                               A hypothesis inspired by such a run is EXPLORATORY
                               and needs a fresh, independently specified test.
    SCIENTIFICALLY_ADMISSIBLE  may alter the register and future experiment design

Headline estimates and model rankings are sealed keys — `hierarchy`, `ranking`,
`bic`, `dbic`, `beta`, `alpha`, `best_model`, `significance`, `sigma`,
`transfer`, `held` — and stay sealed until the corrected re-run. Logs and failure
diagnostics stay open, because that is what diagnostic-only is for.

**Six of seven registered runs are now `DIAGNOSTIC_ONLY`.**

## BE.3 The 0.4% "flip" was a conflation of two observables

BD.3 said the charter's ~0.4% number "flips". **It does not.** The record already
contained both quantities, at Run V:

    3-D shell-averaged |g|, lumpy/smoothed      1.0040  (max 0.77%)
    projected deflection at 150 kpc             0.9731 +- 0.0183, ~3 sigma,
                                                sign does NOT flip
    projected deflection beyond ~300 kpc        mean on 1, sign flips between
                                                realisations -- shot noise

So the scene lane's **2.09% at 200 kpc is consistent with the previously measured
~2.7% inner projected-deflection systematic**, not a contradiction of the 0.77%
shell-averaged monopole. Corrected in place.

The surviving lesson is worth stating exactly: **smoothing member galaxies is
harmless for large-radius monopole calculations and inadmissible for
percent-level inner or directional lensing tests.** Every future statement must
name the observable — shell-averaged |g|, projected deflection, tangential shear,
shear quadrupole, or local field near a member — because they differ by more than
the effect being measured.

## BE.4 "Erases 120%" restated as a transfer coefficient

With `R = dev(A.S)/dev(S)`, the fraction of the resolved deviation surviving
azimuthal averaging in the shell-radial observable:

    R_source   = -0.200        (from source_axis_erased   = 1.2002)
    R_external = +1.298        (from external_axis_erased = -0.2983)
    R_path = R_memory = R_network = 0.000   (complete, to machine precision)

`R_source = -0.200` means a fifth of the signal survives with the sign reversed;
against the lane's 0.24% quadrature floor that should be read as **consistent
with zero, sign not meaningful**. `R_external = +1.298` means averaging leaves
the external-axis law **30% stronger**, by removing the competing source
quadrupole.

**Averaging is not merely information loss: it biases one tensor mechanism
relative to another by a factor of ~6.5 in this configuration.**

## BE.5 The certificate now has typed identifiers and a prospective suite

Stable IDs replace human-readable names everywhere logic depends on them —
`CERT.SHARED_DENOMINATOR.001`, `CERT.NULL.NONEXCHANGEABLE.001` — because the
development harness had selected its expected-failure set by substring-matching
"control" in a case *title*. And C2 no longer prints `inf x` when the target
responsiveness is zero; a ratio against zero is not a number, so the verdict is
now `target responsiveness = 0, control > 0, statistic INSENSITIVE`.

Five mechanisms the gate was **not** designed against:

| id | mechanism | outcome |
|---|---|---|
| `CERT.SHARED_DENOMINATOR.001` | statistic and effect share a denominator | CAUGHT |
| `CERT.SELECTION_MIMIC.001` | the selection function has the signal's shape | CAUGHT |
| `CERT.NONMONOTONE_RESPONSE.001` | responsive where tested, flat where predicted | CAUGHT |
| `CERT.UNDERPOWERED_AT_PREDICTION.001` | powered at a convenient amplitude, not the predicted one | CAUGHT |
| `CERT.NULL.NONEXCHANGEABLE.001` | the permuted label is a function of the measurement | CAUGHT |

    5/5 caught, 0 coverage gaps, 0 false alarms on two valid controls

**Stated limitation: this is not the hidden suite the review asked for.** I wrote
both the mechanisms and the gate, so the exercise shows the seven checks are
*sufficient for failure modes I can imagine*, which is weaker than independence.
A genuinely hidden suite requires someone else to author it.

## BE.6 Theory-dependence becomes graded, not binary

`theory_contaminated` was a single flag. Adopted instead:

    T0  primitive detector observable        admissible for final scoring
    T1  calibration-derived                  admissible with a propagated
                                             calibration model
    T2  geometrically transformed            admissible if the transformation is
                                             common across candidate universes
    T3  theory-dependent reconstruction      diagnostic or screening only
    T4  fitted under the competing law       CANNOT be primary evidence for
                                             selecting among laws

By this grading the CLASH NFW masses are **T4**, LoCuSS `M_WL` is **T4**, the
eFEDS raw shear is **T0**, and the X-COP `Y-PROF-COVMAT` is **T0/T2**. T3 and T4
products keep legitimate uses — pipeline debugging, screening, reproducing
published results, and prioritising where root-data acquisition actually matters
— which a binary flag denied them.

## BE.7 The development suite is federated, and the gold cluster does not exist

**No public cluster satisfies the charter's complete experiment.** Development
therefore proceeds on complementary already-spent systems, each validating a
different part of the pipeline and **none of them establishing that one field
predicts all channels in one physical object**:

    A2029             environment and well-network construction (15.8 Mpc)
    MACS J1149        strong-lensing images and the only measured time delay
    eFEDS / raw shear weak-lensing ingestion and scoring (T0)
    SAMI or similar   resolved member internal dynamics

Results from this suite are **cross-system, not same-scene**, and must be
labelled so. The decisive test still requires one object carrying resolved
baryons, member internal dynamics, cluster dynamics, raw weak and strong lensing,
and environment together.

**One blocker is softer than reported:** the absence of published resolved
Sersic fits is not hard, because calibrated multi-band imaging exists and the
programme can fit member light profiles itself — which is more consistent with
the root-data rule than depending on someone else's catalogue. **The genuinely
missing layer is resolved internal stellar kinematics for member galaxies in the
same lensing cluster.** Every HFF dispersion is a single 1.5-arcsec aperture
value, and an aperture number must not be treated as equivalent to a resolved
map — by this programme's own commutation gate it would have to pass that gate
first.

A Gold Cluster Acquisition Specification is to be written before a target is
chosen, so candidates rank by how much new telescope time they need rather than
by how much code already exists for them.

## BE.8 The infrastructure phase gets a firm exit condition

Recorded as binding, because the standing risk is now an endlessly improving
audit platform:

    Stage 1 scene round-trip passes
    Stage 4 certificate passes PROSPECTIVE validation (independently authored)
    Stage 5 distinguishes injected universe classes
        -> the programme MUST then test the two actual new-gravity families:
           action-derived void/tensor gravity, and reciprocal path-dependent
           gravity.

**Adding further registries, axes, certificates or benchmark universes after
those three conditions are met, without solving and testing those two families,
is the definition of going off track.** Two of the three conditions are now
close; the third is running.

And the progress metric stands as set in BA.9: **how many fundamentally different
universes the system can distinguish from root observations in one complete
physical scene** — not combinations counted, axes covered, rules written, or laws
per second.

---

# Run BF — Stage 5: seven of the ten universes are ONE observational class, and the tensor detectors fire on dark matter

Full record in `work/wellnet-2026-09/universes/REPORT.md` (690 lines, every number
rendered from JSON); 18 modules, 13 result JSONs. **Registry status: VALID** —
this is the only lane of the round not quarantined, because it opens no
catalogues. Provenance enforced by patching `open` and `numpy.load`: **0 foreign
reads, 0 sealed-token matches, no real observational file opened.** KiDS was
excluded even as a noise-model source.

## BF.1 The design that makes the answer trustworthy

One corpus = 30 disk galaxies + 12 clusters + 200 supernovae, drawn from a scene
library **shared by all ten universes**, so a separation can never come from the
scene prior. Each universe supplies only a matter potential, a light potential
and a redshift map; **the instrument is identical for all ten.**

**Nothing in a corpus is a mass.** Gas temperature is *predicted* by each
universe's own hydrostatic equilibrium; the stellar dispersion by a Jeans
solution in its own potential. The emitted products are detector-facing:
PSF-convolved aperture-integrated IFU velocity fields with per-spaxel errors,
per-source WL ellipticities with weights and photo-z outliers, a multiplicative
shear bias and a spatially coherent additive PSF residual, X-ray annulus **photon
counts** with a radially rising non-thermal pressure fraction, SZ, multiple-image
positions and time delays where supercritical, and SN magnitudes **and
light-curve durations**.

U3 (MOND scalar) is the base and U4-U9 are one-knob deformations returning
exactly U3 at zero. U5 is solved exactly to first order in A via the l=2 Green's
function; U6's kernel is manifestly reciprocal.

## BF.2 The equivalence-class map — the result

**At the a-priori fiducial amplitudes all 45 pairs separate.** That is not a
success, it is a finding about the fiducials: they sit **2.9x to 26x above what
this corpus can already see.** Recomputed at amplitudes taken from the scans
rather than chosen by hand, at both the threshold and half-threshold sets:

    { U3 MOND, U4 environment-scalar, U5 tensor, U6 well-network,
      U7 memory, U8 EP-slip, U9 path-redshift }        ONE CLASS

    { U2 dark matter }        separate at z = 8.5 against everything
    { U10 systematics-only }  separate at z = 8.5 against everything

**Every modified-gravity universe this programme has proposed is observationally
indistinguishable from every other one**, on a realistic corpus, at the
amplitudes that corpus can detect. Only dark matter and pure systematics stand
apart.

**Nine pairs are separated by none of three simulated improvements** — 16x source
density, 4x better systematics, 1.5x larger survey. The hardest is U3-vs-U4,
plain scalar against a potential-depth-gated scalar, which fails at both
amplitude sets.

This is exactly the report the charter asked for: *"These theories belong to the
same observational equivalence class on this corpus; the following missing
observation would separate them."* The answer is negative and it is more
informative than a positive would have been.

## BF.3 The dark-matter control: the tensor detectors fire on CDM two-thirds of the time

| detector | FP on the calibration audit | **FP on U2 (dark matter)** |
|---|---|---|
| tensor, external-minus-baryon axis | 0.051 | **0.479 [0.435, 0.524]** |
| tensor, external axis | 0.054 | **0.294** |
| well-network | 0.048 | **0.233** |
| galaxy m=3 / environment | 0.055 / 0.054 | 0.065 / 0.065 |
| memory / EP-slip / path | 0.044 / 0.044 / 0.061 | 0.004 / 0.000 / 0.002 |

    family-wise, any of 8 detectors:  0.226 on the audit
                                      0.648 [0.604, 0.689] on the DARK MATTER
                                      universe   (0.785 with 3x systematics)

**A triaxial collisionless halo misaligned from the baryons IS the tensor
signature.** The detectors this programme built to find directional gravity
cannot distinguish it from ordinary dark matter, and they fire on a CDM universe
**two thirds of the time**.

> **DECOMPOSED IN RUN BK.** The 0.648 is two detector defects and a library
> accident, not a wall: a two-sided test of an asymmetric quantity (splitting the
> tail takes it from 0.456 to 0.000), variance inflation from a randomly-phased
> halo quadrupole (a detector provably blind to the tensor still fires at 0.524),
> and a -2.2 sigma chance alignment baked into the shared scene library. A
> signed joint procedure reaches FP 0.002 on CDM at power 0.989 -- **but only
> while halo-filament alignment is set to zero, which BF's generator does and
> real haloes do not.** Above f_lss = 0.38 it is no better than this. The
> separation is a statement about the alignment prior, not about gravity.

**Consequence, and it is binding: no tensor or network detector may be run on
real data without a dark-matter-universe null beside it.** Every anisotropy
result this programme has produced was calibrated against scalar nulls only,
which is why their sizes looked acceptable.

## BF.4 The seven Stage 5 questions, answered

1. **Recover an injected scalar law** — yes. `d(log a0_hat)/d(log a0) = 0.925 +-
   0.049`, bias −0.004 dex, scatter 0.043 dex. **But a CDM universe yields an
   equally well-defined a0** (bias 0.067 dex). *Recovering a0 is not evidence for
   modified gravity.*
2. **Scalar misspecification vs genuine anisotropy** — yes, in **one channel
   only**. The galaxy m=3 harmonic has power **1.000** at FP 0.069 against seven
   qualitatively different scalar families, three of which are not functions of
   `g_N/a0` at all. The **cluster shear quadrupole has power 0.000**: a 0.5
   tensor puts only 0.083 of its l=2 into the lensing potential.
3. **External axis** — yes, per object: median error **11.9 deg** in U5 against
   46.1 deg in U3, responsiveness 0.415 +- 0.019. **The 45-degree misspecified
   control gives 0.005 +- 0.003, consistent with zero** — a misspecified axis
   sets no limit. Run AO's finding, reproduced from a generator that shares no
   code with it.
4. **Network vs source ellipticity** — **the network detector is BLIND.** Across
   the whole B range it moves **0.0003**, which is 0.023 of its own critical
   value; power **0.000**. U6 is detected at z = 8.5 only through its
   **monopole**, never through the member-locked azimuthal signature that is its
   whole physical premise.
5. **Path effect after survey systematics** — the raw slope detector has power
   **0.025** at fiducial epsilon, while the calibrated SN channel separates at
   z = 8.5. **Systematics alone fake a path effect at 0.131.** Durations stretch
   as (1+z) exactly, so the geometric mechanism is not excluded by time dilation.
6. **False new gravity in a dark-matter universe** — BF.3 above.
7. **Amplitude thresholds** — kappa 0.062, A 0.019, B 0.0079, M 0.035, zeta
   0.034, epsilon 0.0082; all six scans responsive at
   `dz/dlog10(A) = 4.1-11.2` on 4-6 unsaturated points.

Gates: coarse-graining 1.4e-3 PASS, reciprocity **0.0 exactly** PASS, gauge rules
differing by 0.290 dex at rank correlation 0.9959.

## BF.5 Sizing done first, and two bugs it caught

Realised false-positive rate on an **untouched** null half: 0.050 by replicate,
0.062 transferring the critical value to arms it was never calibrated on. **At
nominal 0.01 it realises 0.033 — the tail is not calibrated**, so every verdict
is taken at the measured 0.05 family-wise value rather than a nominal one.

The sizing caught a real bug: `argsort(argsort())` produces **sequential rather
than tied ranks**, so a degenerate channel returned AUC = 1.0 deterministically —
an apparent **z = 4.8 between two universes identical in that channel by
construction**. Fixed with mid-ranks. A second: the hydrostatic estimator treated
volume-integrated counts as a surface density, leaving +1.5 in `dln n/dln r` and
clipping `M_HE`. Fixed, and it now returns `M_HE/M_bar` of 1.0 (Newton), 2.8
(MOND), 4.8 (CDM).

## BF.6 What this does to the exit condition

BE.8 set the third condition as *"Stage 5 distinguishes injected universe
classes."* **That phrasing assumed a positive answer.** Stage 5 is complete and
its answer is negative: on a realistic corpus, the programme's seven candidate
families are one observational class.

The condition is nonetheless **met**, because the charter's actual requirement was
to produce the equivalence map and name the missing observation — not to succeed
at separation. What changes is what the two theory families must now be tested
*for*:

  * **Not "which of our candidates fits best."** They are indistinguishable, so
    a fit contest between them is meaningless on this corpus at these amplitudes.
  * **But whether either is distinguishable from DARK MATTER**, which is the one
    boundary the corpus can resolve — and where the current detectors have a
    family-wise false-positive rate of **0.648**.

So the next step is unchanged in identity and changed in target: solve and test
action-derived void/tensor gravity and reciprocal path-dependent gravity, **each
against a dark-matter null rather than against each other**, and report the
amplitude at which each becomes separable from CDM rather than from its siblings.

## BF.7 Limits, stated by the lane

The tensor is first-order in A and asymptotes to a constant K — **degenerate with
a coordinate stretch**, which is Gate 1 reappearing inside the generator. U6's
150 kpc coherence length makes it cluster-only *by construction*, so its blindness
in BF.4 is partly a property of the choice. U7's age proxy precision is
optimistic. And `Delta Sigma_bar` uses an SIS factor, so the absolute values of
`clu_wl` and `ep_ld` are uncalibrated — differences between universes are not.

---

# Run BG — the charter, read back and scored: NOT SATISFIED, 27% met

Code and results in `work/wellnet-2026-09/charter-eval/` — `evaluate.py`,
`charter_eval.json`. The charter
(`C:/Users/henry/dev/invariant-gravity-discovery-charter.md`, 1,242 lines) had
been written and worked *from* but never read back and scored *against*. This is
that evaluation, over its own enumerated requirements, with the run and number
supporting each verdict.

## BG.1 The score

    41 enumerated charter requirements

    MET       11    27%
    PARTIAL   21    51%
    NOT_MET    7    17%
    BLOCKED    2     5%

| set | result |
|---|---|
| the 12 questions a final law must answer | 0 MET, 9 PARTIAL, 3 NOT_MET |
| the 12 promotion criteria for new gravity | 5 MET, 3 PARTIAL, 3 NOT_MET, 1 BLOCKED |
| Stages 0-10 | 5 MET, 5 PARTIAL, 1 NOT_MET |
| Corpora A-F | 1 MET, 4 PARTIAL, 1 BLOCKED |

**Not one of the twelve questions a final law must answer is fully met.** Three
are outright unanswered: *what creates gravity* (every family ever built takes
rest mass as its only source, 1 of 7 charter options), *why galaxies and clusters
differ* (Run BC's radius hypothesis was refused by the gate; Run AX removed the
CLASH leg), and *what would falsify it* (Run BF: the seven candidate families are
one observational class, so no candidate has a falsifier that separates it from
its siblings).

## BG.2 The two blockers, neither solvable by computation

**Criterion 10 — a distinctive SEALED prediction — is BLOCKED.** Run AW: 23 of 25
datasets are spent, the reserve holds at most five one-shot evaluations, three of
them in the same regime, and there is no reserved galaxy-rotation, lensing or
local-gravity probe at all.

**Corpus E — gold resolved clusters — is BLOCKED.** Run BD: no cluster satisfies
it. Raw shear exists for 1 of 7 targets and that one lacks resolved member Sersic
fits, so no target has both; exactly one target has a measured time delay; and
every HFF dispersion is a single 1.5-arcsec aperture value rather than the
resolved map the charter specifies.

**Criterion 9 — survives alternate-universe controls — is NOT_MET, and it is the
sharpest.** Run BF measures the family-wise false-positive rate on a dark-matter
universe at **0.648 [0.604, 0.689]**. A candidate cannot survive a control the
detector fails two-thirds of the time.

## BG.3 What the charter's own final deliverable requires

The charter asks for one of two things:

> *"either a new generative field law with distinctive successful predictions, or
> a precise statement of which broad classes of new gravity have been ruled out
> and which observation would distinguish the remaining equivalence classes."*

**(a) a new generative field law — NOT_MET.** No candidate survives its own
controls.

**(b) an equivalence-class map plus the missing observation — PARTIAL.** Run BF
produced exactly this object, with amplitude thresholds and the specific
improvements that fail to separate nine pairs. **But on a synthetic corpus.** The
charter asks for it over real observations, and Stage 10 says no untouched real
data remains to close it with.

## BG.4 The honest verdict

**The charter is not satisfied.** Deliverable (b) is reachable and half-built;
deliverable (a) is not supported by any surviving candidate.

What the programme has built is real and is mostly the apparatus the charter
demanded: Stages 3, 4, 5, 6 and 9 are MET, and the invariance compiler, the
sensitivity certificate, the alternate-universe suite and the commutation gate
did not exist before. What it has *not* built is a law, and the two things
standing between it and the charter's fallback deliverable are **data**, not
computation — a cluster with the required layers, and a confirmation set that has
not been spent.

That is the state, scored against the document rather than asserted.

---

# Run BH — the charter's fallback deliverable, assembled: it was a synthesis gap, not a data gap

Code and results `work/wellnet-2026-09/charter-eval/deliverable_b.py`,
`deliverable_b.json`. Run BG scored the charter's deliverable (b) as PARTIAL on
the reasoning that Stage 5's equivalence map is synthetic. **That was wrong.**

The charter asks for:

> *"a precise statement of which broad classes of new gravity have been ruled out
> and which observation would distinguish the remaining equivalence classes."*

It does not require that statement to come from synthetic or from real data. It
requires the statement. The programme's eliminations on **real** observations are
real, its mathematical eliminations need no data at all, and the missing-observation
list has been accumulating since Run K. **The pieces existed across some sixty
runs and had never been assembled into the object the charter names.** That is a
synthesis gap, and it is now closed.

## BH.1 Ruled out, with the programme's own admissibility standards applied

Each elimination is graded: an elimination resting on T3/T4 data is not
admissible evidence about the world (BE.6), one refused by the Stage 4
certificate does not count, and one made on synthetic data alone is a statement
about the *detector* rather than the world.

    13 classes ruled out admissibly
        5  on MATHEMATICS ALONE, no data
        6  on REAL observations
        2  detector statements (about the instrument, not the world)
     1 elimination WITHDRAWN as inadmissible

**On mathematics alone:** well-network laws with no continuum limit (390 of
3,123 settings); bounded-response laws as an explanation of flat curves; pair and
graph laws with p = 0 mass weighting; tensor laws whose only signature is the
transverse eigenvalue (spherical blindness); and the field-direction projector as
a mechanism, which is the identity exactly.

**On real observations:** energy-drain and tired-light redshift mechanisms, dead
at **90 sigma** on 1,504 DES supernovae; Newtonian gravity on baryons alone as a
structure-formation law, short by 3-4 orders; the tidal-gated scalar as an
*environmental* law, since `|T~| = sqrt(6)(g/r)|1-rho/<rho>|` makes it a local
kinematic ratio; the tidal gate as a cluster-lensing improvement, fitting worse
than predicting no lensing at all on 3,365 raw shear points; pressure and
amplified stress as the cluster source, over-predicting at −13.0 sigma; and the
nonlocal path kernel as a rotation-curve law.

**Withdrawn as inadmissible:** the cluster excess organised by `r/R500` — CLASH's
numerator and x-axis are two functionals of one NFW fit (T4), and `r/R500` is the
same regressor as `r` given per-cluster levels.

## BH.2 The six remaining classes, each with its separating observation

| class | separating observation | cost |
|---|---|---|
| **scalar MOND-like** | the Vikhlinin parameter covariance from Bahar+2022 | **an email** |
| **environment-gated scalar** | \|Phi_b\| varying >= 1 dex at fixed g_bar within one class — eFEDS paired with X-GAP or CLoGS | **one pairing, data exists** |
| **tensor / directional** | the 2-D shear phase against an independently measured external axis at 111x the source count, scored against a DARK-MATTER null | a deeper survey |
| **well-network / graph** | member-locked azimuth needs resolved member positions AND raw shear in the same cluster; no public target has both | new observation, or self-fit member light |
| **geometric path redshift** | cross-correlate the void path-length map with Planck | **a cross-correlation, both public** |
| **memory / hysteresis** | no time-dependent scoring channel exists — UNREACHABLE, not untested | a solver, not an observation |

**Two of the six are separable with data that already exists**, and a third needs
only an author request. That is a materially different picture from "blocked on
acquisition".

## BH.3 The corrected verdict on the charter

**The charter's final-output requirement is SATISFIED, via its fallback branch
(b).** Its primary goal (a) — a new generative field law with distinctive
successful predictions — is **NOT met**, and no surviving candidate supports it.

The two blockers on (a) stand and remain acquisition problems: Corpus E, where no
cluster carries the required layers, and Stage 10, where no confirmation set
remains. Neither is solvable by more computation.

**What BG got wrong is worth keeping as a lesson.** I scored a deliverable
unavailable because one *input* to it was synthetic, without checking what the
deliverable actually required. The charter's own words were the test, and I had
not applied them — which is the same error class as reading a claim's headline
instead of its support, committed against the governing document itself.

---

# Run BI — void path length x Planck: the geometric redshift class is measured, and excluded at AK's own amplitude

Full record in `work/wellnet-2026-09/voidcmb/` (`REPORT.md` rendered entirely
from JSON). Registered as `BI-voidcmb` under the v3 rules **before** any work.
Inputs graded: Planck TT T1, VoidFinder T1-T2, the LCDM ISW template T3 — which
is why the headline marginalises it rather than trusting it. KiDS, the wide
binaries and the entire confirmation reserve untouched, confirmed by code audit.

This is the first lane to open real data **after** the Stage 4 certificate
existed, and the first to attempt a distinctive prediction on real, public,
unreserved data since the charter was adopted.

## BI.1 The certificate was issued with a blind guard armed

All seven checks passed, and the way C3 was sized matters: the certificate read
real Planck temperatures **only at rotated sky placements overlapping the true
footprint by <= 5%**, so the test was sized against the real sky — foregrounds,
noise, mask — while the measurement stayed invisible. 7,085 guard checks, 0
refusals, 0 identity reads.

    C1  d(estimate)/d(injected)  1.0000 in-grammar x 0.9814 pixelisation
    C2  100 uK monopole moves c2/c1 by 1.2e-18; 0.1% gain by 3.5e-7;
        predicted signal 4.0e-3
    C3  realised FPR 0.050 (leave-one-out), 0.052 (Gaussian skies from the
        published Planck TT spectrum) vs nominal 0.05; sd(sim)/sd(null) = 0.98
    C4  13.6 sigma at 0.28%, 21.3 sigma at 0.44% -- AK's OWN bound
    C5  reads chi in [0, 332.4] Mpc/h, z <= 0.1125 -- the catalogue's limits
    C6  out-of-grammar recovery 0.31-0.68 across four foreign functionals
    C7  worst nuisance is the ISW template at |r| = 0.763; dust -0.099

## BI.2 The measurement

    c2/c1 = -0.0266% +- 0.0206%        -1.28 sigma null-calibrated, p = 0.206
    |c2/c1| < 0.0678% at 95%
    responsiveness 0.9814; injections of +-0.10/0.20/0.40% return their input

Nothing moves across **eleven declared arms** — two component-separation
pipelines from two Planck releases, three footprint erosions, two resolutions, a
tomographic near/far split, edge-void exclusion — nor across eleven systematic
splits. Every arm sits between -1.9 and +0.03 sigma.

**Against AK's derived bound of 0.28-0.44%, this is 4.1-6.5x tighter, with 14-21
sigma of power AT that amplitude, and it found nothing.** The 3-sigma floor is
0.063% against AK's own supernova floor of 3.9-5.9% — 62-94x better.

## BI.3 The ISW is separated, and a template was caught wearing a physics label

The LCDM integrated Sachs-Wolfe signal is a real contaminant with the same sign
structure. Separated three ways; the decisive number is that the *theory-
normalised* ISW (Omega_m = 0.315, |delta| = 0.7, f = 0.530), rather than a fitted
one, biases an ISW-free fit by **+0.0008% = 0.057 sigma** — 17x below the noise.
Separation works because a top-hat void contributes ~2R to the path length but
~R^3 to the potential integral.

An earlier ISW template with an untruncated far field correlated **-0.83 with
distance-to-footprint-edge and only -0.14 with the path length** — a
survey-geometry template wearing a physics label. Caught pre-data, truncated at
3 R_eff, and **removed rather than fitted.**

## BI.4 The analytic-error trap, reproduced on a third dataset

    OLS analytic sigma            0.0044%
    rotation-null sigma           0.0206%
    ratio                         4.72

**The analytic error bar turns this null into a 6.0-sigma detection** — 8.1 sigma
in the non-edge arm, 9.5 sigma at nside 128. Run AK found 6.1 sigma analytic
against 1.8 simulated on a different dataset; this is the third. The cleanest
demonstration: nside 128 uses four times the pixels, its analytic error falls to
0.0027%, and **its null width does not change at all.** The correlated sky is the
noise; the pixels are not independent samples of it.

## BI.5 What it settles, and what it does not

**Settled.** The geometric half of the path-redshift class — the half supernova
time dilation cannot reach, because it predicts b = 1 identically — is now
*measured*, not bounded by an anisotropy budget. **Run BH's "geometric path
redshift" row moves from the remaining classes to the ruled-out list**, at the
amplitude AK's own derivation identified, on real observations, T1-T2 inputs.

**Not settled.** A coefficient that vanishes below z = 0.11 and revives above it
— the map reaches z = 0.1125 and no further, stated as support rather than assumed
away. A law keyed to a functional orthogonal to void path length; the
out-of-grammar recoveries of 0.31-0.68 measure exactly how much is covered. The
tidal terms c3/c6, where AK.5's obstruction stands at n = 46. And anything in the
gravity lanes — this branch is logically independent.

**Reach.** Cosmic-variance-limited by the CMB projected onto this template over
5,810 deg^2. More Planck does not help; DESIVAST VoidFinder over the DESI BGS
footprint buys roughly a factor of two. **This observable is within about a factor
of two of its ultimate reach with existing data.**

## BI.6 What this does to the charter's deliverables

Deliverable (b) sharpens: **five remaining classes, not six**, and the ruled-out
list gains its seventh real-observation entry.

Deliverable (a) does not move. This was the one remaining class separable with
real, public, unreserved data, it was tested at full power, and the answer was
null. That is a fact about the data, not about the effort: the class was reachable
and it is not there.

---

# Run BJ — the fourth review: three accomplishments, only two of them made

The review separates three things this programme has been at risk of conflating:
discovering a new physical principle; discovering which theories current
observations cannot distinguish; and building machinery that prevents false
discoveries. **The programme has done the second and third. It has not done the
first.** Recorded as two statuses that must never be merged:

    FALLBACK EQUIVALENCE MAP:          COMPLETE   (Runs BH, BI)
    PRIMARY NEW-PRINCIPLE DISCOVERY:   NOT YET ACHIEVED

## BJ.1 "False positive" was the wrong word, and it hid the real failure

BF.3 called the detectors' 0.648 rate on the dark-matter universe a false-positive
rate. **Physically that is not right.** A triaxial collisionless halo misaligned
from the visible matter really does produce anisotropy, so a generic anisotropy
detector that fires on it is *working*. The error was never

    the detector falsely found anisotropy

but

    the programme interpreted generic anisotropy as evidence for TENSOR GRAVITY.

That is a **construct-validity failure**, not a sizing problem, and it
generalises into a rule that now binds every future claim:

> **Anisotropy by itself is not evidence for anisotropic gravity.** Nor is a
> recovered a0 evidence for modified gravity (BF: a CDM universe yields one at
> bias 0.067 dex); nor a shear quadrupole evidence for a tensor vacuum; nor
> galaxy-aligned structure evidence for a well network; nor a MOND-like monopole
> evidence for MOND. **A new principle must predict something more specific than
> the phenomenon that motivated it.**

## BJ.2 The equivalence is conditional, and I wrote it as if it were intrinsic

BF said seven universes are "one class" and BF.6 said testing the two theory
families against each other is "meaningless". Both overstate. The statement is

    U_i  ~_{D, A, G, N}  U_j

— indistinguishable **given** the present detector-level corpus D, the assumed
amplitudes A, the tested geometries G, and the adopted noise-and-nuisance
distribution N. It is not a proof that tensor gravity, memory, path transport,
network gravity and environment-gated scalars are physically the same. It says
the present corpus is dominated by observables — static monopole strength, radial
profiles, azimuthally averaged lensing, equilibrium responses — to which their
differences are invisible. The corrected sentence: **testing them against each
other with the present observable set is uninformative.** That is a
specification for the next experiment, not a reason to abandon the families.

| hidden property | observation that would expose it |
|---|---|
| tensor direction | 2-D phase information, or orthogonal orbits |
| well-network dependence | actual member geometry vs mass-preserving scrambles |
| memory | same present configuration, different histories |
| matter–photon difference | dynamics and lensing/timing in the same scene |
| path dependence | equal-distance sources with different intervening paths |
| finite propagation | time-dependent sources and response lags |

## BJ.3 "Fourteen classes ruled out" was one number where four were needed

Recategorised in `deliverable_b.py`, and the recount changes the meaning:

    THEORY FAMILIES ruled out        4    energy-drain redshift; Newton on baryons
                                          as a growth law; bounded response as a
                                          flat-curve explanation (within its
                                          functional class); the identity projector
    SPECIFIC IMPLEMENTATIONS         7    the tested gates, couplings, weightings
                                          and amplitude ranges -- related theories
                                          remain possible
    STATISTICS ruled out             3    the tensor/network detectors (construct
                                          validity), the blind network detector,
                                          the withdrawn r/R500 organisation
    CURRENT DATA NON-IDENTIFYING     1    spherical blindness

**Spherical blindness was miscategorised.** I had it as a mathematical
elimination. It is a theorem that a spherical source carries no directional
leverage — an *experimental limitation*, not a falsification of tensor gravity.
And **a blind detector is not a ruled-out theory**: the network premise is
untested, not refuted. Keeping these apart is what stops the compiler from
quietly shrinking "new physics" to whatever the current software can express.

## BJ.4 A caveat BI inherits from AK and did not restate

The void path lengths in Run BI are built from redshifts that already assume the
conventional distance relation, and the ray truncation point reuses the source's
own redshift (AK.7 measured that endpoint reuse at 18% on the leverage
variable). **BI therefore tests environmental modulation of redshift *within* the
conventional coordinate frame.** It does not test a complete no-expansion
universe, which would require re-running the void finders under the candidate's
own distance law. The exclusion stands for what it measured; the category entry
now says so.

## BJ.5 "No statistic can separate them" cannot come from a finite bank

The CDM-separation lane's brief offered, as one outcome, a proof that no statistic
in the observable set separates the classes. **A finite bank of statistics
cannot prove that.** The defensible form is: approximate the full likelihood
ratio with a high-capacity discriminator on detector-level data, test it on
untouched scenes and an independently implemented simulator, and report a
**bound on distinguishability under the stated corpus and simulators.** Chance
performance from a flexible, calibrated discriminator would then read: *under
these source distributions, noise levels, nuisance models and channels, no
measurable separation has been demonstrated* — strong, and not a theorem about
every future observation.

## BJ.6 The reframing that matters: the corpus already separates CDM, and nobody knows why

BF's two findings together say something the programme has not acted on:

    the corpus contains information separating CDM from the modified-gravity
    class (U2 separates at z = 8.5)
                                    -- but --
    the named mechanism detectors are NOT what carries it (they fire on CDM
    at 0.648)

**Some other feature is doing the separating.** Finding it, and expressing it as
a physical principle, is the shortest known path to the primary objective. The
review's leading candidate, to be *tested* rather than assumed: **baryonic
closure** — given a complete baryonic scene and environment, gravity has very
low residual freedom under a universal law, whereas a CDM universe carries an
independently evolving halo whose shape, orientation and substructure are not
fixed by the visible baryons. The distinguishing quantity would then be the
*conditional predictability* of gravity from baryons — the scatter, covariance
and phase of P(G | B) — not anisotropy itself.

## BJ.7 What is opened in response

  * **A Principle Synthesis Lane, now.** Theory construction spends no
    confirmation data, so the independent-certificate gate does not block it.
    For action-derived void/tensor gravity and reciprocal path-dependent gravity:
    a machine-readable principle card (physical statement, source, state,
    propagation, matter coupling, photon coupling, known limits, counterfactual
    signature, unique falsifier, **CDM distinction**), then the simplest action
    enforcing the principle, run through the Stage 3 compiler. The programme had
    become better at explaining why existing ideas cannot be distinguished
    without constructing the law that is its objective.
  * **A Principle Extraction Lane, when the CDM-separation lane lands.** Its
    result is the input. Protocol: pair every universe on the same baryonic scene
    and noise; match away the scalar monopole so strength cannot carry the
    answer; a flexible invariant discriminator to *locate* the information;
    channel, radius, harmonic and object ablations; then **causal
    counterfactuals** — move baryons holding the halo fixed, move the halo
    holding baryons fixed, rotate the external axis, scramble members preserving
    every radial profile, change history preserving present matter, change the
    photon path preserving endpoints. The object of interest becomes the response
    `dO/dB`, not the value of O. Distil to a sparse invariant; validate on an
    independently implemented generator; only then translate into a principle.

## BJ.8 Operating rules adopted

Stop calling CDM responses false anisotropy. Stop treating present-data
equivalence as theoretical equivalence. Stop broadening the statistic bank
without causal interventions. Stop counting a blind detector as a ruled-out
theory. **And do not let completion of the fallback deliverable satisfy the
primary discovery objective** — it is recorded as complete, beside the primary
objective recorded as not achieved.

---

# Run BK — the 0.648 decomposed: two detector defects and a library accident, and a separation that is a statement about the alignment prior

Full record in `work/wellnet-2026-09/cdm-separation/` (`REPORT.md`, 577 lines,
rendered from JSON; 15 modules; 16/16 tests). Registry `BH-cdm-separation`
VALID; 0 foreign reads; sealed and reserve tokens guarded in every worker, with
the guard exercised as a test rather than asserted. Estimator shares nothing with
BF's: monopole, m=2 and m=4 fitted jointly per radial bin in both tangential and
cross ellipticity, with a covariance, so power is noise-debiased and phase carries
an error.

## BK.1 What actually differs between a halo quadrupole and a tensor one

| property | CDM halo | external-axis tensor |
|---|---|---|
| phase | 20.1 deg from the **baryon** major axis (null 44.7) | 12.3 deg from the **external** axis (null 45.4) |
| radial profile | 97.8% of power inside 0.55 R500; studentised power **falls** outward (+3.42, +1.30, +0.23) | **rises** outward (-0.05, +0.27, +0.63) |
| slope on baryon ellipticity | **+8.84 +- 0.35** (systematics-only +7.57) | **-0.035 +- 0.065**, consistent with zero |
| survives azimuthal averaging | 0.040 | **0.946** |
| survives spherical averaging | 0.004 | 0.957, matching A(2/3)<P2^2> analytically |

The axes differ, the radial gradients have opposite signs, and the halo's
quadrupole is sourced by the baryons' shape while the tensor's is not. Member-
dispersion m=2 is zero in every arm (+-0.005) — BF's generator writes both
quadrupoles into the lensing map only, so the matter/light axis sets no limit and
could not have separated two metric quadrupoles anyway.

## BK.2 The 0.648 is two detector defects and an accident, not a wall

**(1) Variance inflation.** A large, randomly-phased halo quadrupole inflates the
variance of *any* external-axis projection. `S_45` — a detector **provably blind
to the tensor**, d/dA = -0.011 +- 0.047 — still fires on CDM at **0.524**
two-sided. That is BJ.1's construct-validity failure measured: the detector
responds to anisotropy of any origin.

**(2) A two-sided test of an asymmetric quantity.** `aniso_ext_minus_bar` is
driven strongly *negative* by a halo and strongly *positive* by a tensor; taking
|S| puts them on the same side. The independent reimplementation reproduces BF's
number — **0.456 [0.425, 0.487]** against BF's 0.479 — and it drops to
**0.000 [0.000, 0.004]** once the tail is split.

**(3) A library accident.** BF's shared 18-cluster scene library has mean
`cos 2(pa_bar - axis_ext) = -0.369`, a -2.2 sigma chance alignment that enters
every corpus and never averages out. With axes redrawn per cluster, the halo mean
on `S_ext` is consistent with zero and only the variance is inflated.

BF's number was reproduced independently, then explained. The record's 0.648
stands as what BF's detectors do; **BK is why.**

## BK.3 A procedure that beats it by ~300x

Signed, studentised projection on the external axis **fires** and the baryon-axis
projection **does not**:

    FP on the dark-matter universe        0.002 [0.001, 0.007]
    with 3x systematics                   0.000 [0.000, 0.004]
    on U10 systematics-only               0.000 [0.000, 0.004]
    on the scalar null                    0.049
    power on the tensor at A = 0.5        0.989 [0.980, 0.994]
    BF family-wise, for comparison        0.648

Sizing is clean on an untouched audit half: 0.052/0.050 at nominal 0.05 and
0.009/0.011 at nominal 0.01 (BF's 0.01 realised 0.033).

## BK.4 Where the answer changes — and the second threshold is not an amplitude

**Amplitude.** Galaxy m=3 reaches 3 sigma at **A = 0.115**; the cluster
quadrupole at **A = 0.78**. At BF's own detectable amplitude, A = 0.020, the joint
procedure has power **0.059 — its own size.** A factor of ~6 short. The tensor at
A = 0.020 needs **~2,000 galaxies** with resolved velocity fields *and*
independently measured external axes (sqrt-N law measured to 0.8%). CDM is
detectable in **~1.4 clusters**.

**Alignment.** The joint FP on CDM crosses 0.05 at **f_lss = 0.38** — once ~38%
of the halo's projected alignment is inherited from surrounding large-scale
structure rather than from the baryons, the procedure is no better than BF's
(0.226 at f_lss = 0.5, 0.808 at 1.0). **BF's generator sets halo-LSS alignment to
exactly zero. Real haloes align with their filament.**

> **The separation is a statement about the alignment prior, not about gravity.**

That sentence is the lane's most important one, and it binds the Extraction Lane:
any CDM null that omits a realistic halo-filament alignment will rediscover this
same separation and mistake it for physics. BF's galaxy channel was likewise never
tested against a triaxial galaxy halo; supplied here, a disc-aligned one leaves
the m=3 detector at nominal (0.056-0.110 up to q = 0.2) while a **tidally aligned
one fires it at 0.44-0.76.**

## BK.5 Two families produce no directional signature at all

    d(S_ext)/dB      = +0.29 +- 0.41     (well network)
    d(S_ext)/deps    = -1.21 +- 1.10     (path redshift)

Both consistent with zero: **no upper limit on B or epsilon is set.** For the
reciprocal well-network and path families the question is not answered
negatively — **it is not posed.** A directional statistic cannot see a
non-directional mechanism, which is exactly why the Extraction Lane must carry a
non-directional candidate (conditional scatter, BJ.6's baryonic closure) beside
the directional ones.

## BK.6 Stage 4, and the bug that would have been published

15 certificates evaluated, **3 issued, 12 refused with a named check.** The
cluster `S_ext` fails C2 at every amplitude below ~1.0 because a dark-matter halo
moves it **1.9-48x as much as the tensor does.** `S_morph` fails C6 at 11%
recovery. The certificate did its job on fresh statistics, not only on history.

Test T6 caught **a sign error in this lane's own `forward.f_halo`** — minor axis
where the major belongs — which had reported `S_bar = -2.4` where BF's generator
gives +10.6. A single implementation would have published a confidently
wrong-signed alignment scan; the independent forward model is what caught it.
Also flagged: the Stage 4 C7 check is a correlation and saturates on
amplitude-sequence signatures (0.98 between mechanisms differing 50x in
amplitude), so every signature here is reported as a response pattern across
conditions, not a correlation.

## BK.7 What this changes

BF.3 read as a wall between the programme and any tensor claim. **It is not a
wall; it is two fixable detector defects, a library accident, and an
astrophysical prior.** The fixable parts are fixed. The prior — how much of a
real halo's alignment comes from its filament — is now the quantity the CDM
distinction turns on, and it is measurable in principle, which is a better place
to be than 0.648 was. The primary objective is unchanged: not yet achieved.

---

# Run BL — the two theory families, constructed: actions, field equations, cards, and where each one bites itself

Full record in `work/wellnet-2026-09/synthesis/` — `REPORT.md`, `cards.json`,
`card_tensor.md`, `card_path.md`, the derivations and compiler verdicts.
Registered before any work; every script asserts 0 foreign reads, 0 sealed-token
matches, 0 confirmation-reserve matches. **This is the first run since the
charter was adopted that constructs a law rather than screening one.** The lane
labelled itself BK; that letter was already taken and it is recorded here as BL.

## BL.1 The tensor action, and three things it forced rather than allowed

    L_T = -(1/8 pi G) [ a0^2 F(|u|^2/a0^2) + f_E h(|u|/a0) u^T That_env u ] - rho Phi
    u = grad Phi,   F' = mu = x/(1+x),   h = mu(1-mu) = x/(1+x)^2

with `That_env` the unit-norm traceless tidal tensor of the *environment* — the
Hessian of Phi_N smoothed on L_env, principal-axis form sqrt(3/2)(e e^T - I/3).
Field equation `div M(u) = 4 pi G rho`, radial reduction
`g[mu + f_E lambda eta] = g_N`. **Two new universal constants**, f_E and L_env.

Three properties are **derived, not chosen**: `h` must vanish in deep MOND or the
operator loses ellipticity (a constant h is indefinite on 59% of a low-x cloud);
`h` must vanish at high g for the Solar System; and the admissible interval
f_E in (-0.95, 1.85) follows from the Hessian. **The same gate written as K(u)u
in QUMOND form is not a gradient — compiler bin F2 — while in the Lagrangian it
is one identically.** That is the whole argument for action-first generation,
measured on the programme's own candidate: the grammar rejected what the action
admits.

Compiler: ADMIT / `admissible` at f_E = 0.1, 0.3, 1.0, 1.8; u-space antisymmetry
1.4-2.2e-10; Gate 1 escaped via the independent axis. With a *dynamical* axis
field: `outside_declared_model_class` — a scorer statement, gates 1-3 pass.

## BL.2 The path action, with its carrier derived in closed form

    S = -1/2 INT INT rho(x) W(x,y) rho(y),   W = -(G/|x-y|)[1 + eps v]
    v = fraction of the straight segment with rho < rho_*

No local PDE. `Phi = dE/drho = Phi_dir + Phi_3`, and the momentum carrier is
**derived**, not declared: `Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z)`, with
`P = INT dOmega C(z,n) C(z,-n)`, the product of opposite half-columns. Closed
form for Plummers, checked to 2e-5. Verified mechanically on five bodies: total
forces sum to **3e-9** of the mean; endpoint forces alone to 0.030; the carrier
closes the gap. **Two new constants**, eps and rho_*; rho_* = rho-bar is the
no-new-scale variant. Compiler: ADMIT at eps = 0.3 and 0.03, reciprocity exactly
0; `non_identifiable_on_this_bench` at eps = 0.003 — **a scorer statement: the
bench has no two-body probe.**

## BL.3 The cards' CDM-distinction fields — neither is anisotropy

Both pass BJ.1's rule. Neither falsifier is a generic quadrupole.

**T:** a **delta-function phase lock to the present tidal axis**, one universal
sign, **zero radial twist** (a halo twists with radius), a universal A2(g/a0)
profile that dies in deep MOND, and **matter-light covariance exactly +1**
(same Phi, same chi, no slip). To mimic it a halo would need a delta-function
shape distribution slaved to the present tidal field with an ellipticity profile
keyed to baryonic acceleration and dying outward. Baryonic closure: yes — the
scatter is a pure m=2 harmonic of universal amplitude, random-phased only when
the axis is unobserved.

**P:** a **compensated bridge** between two clusters — Sigma_eff -0.49 kg/m^2 on
the axis, +0.16 in the wings, **zero net mass** — scaling as M_A M_B (measured
2.001, 4.000). **rho_DM >= 0 cannot make a net-zero-mass bridge: a positivity
obstruction, not fine-tuning.** No halo arrangement scales with the product of
endpoint masses and saturates in gas density. Falsifiers: a bridge with net
positive mass; scaling with the filament's own mass; survival as one endpoint
goes to zero. Solar System, lab, pulsars and wide binaries are *exactly*
Newtonian because those media are denser than rho_* (v = 0, Phi_3 ~ 1e-8).

## BL.4 The counterfactual signatures, signed — the Extraction Lane's inputs

    T   rotate e_hat            m=2 phase of dynamics AND lensing co-rotates,
                                d phi/d psi = +1, zero lag, all radii
        move baryons, hold e    amplitude re-evaluated on the new g0(r) instantly
        move halo / scramble members / change history     0
        photon path             same chi: matter-light covariance +1
        radial twist            d phi / d ln r = 0    (CDM: != 0)

    P   rotate axis / move halo / change history          0
        scramble members at fixed radial profiles         pair forces move 0.1%
        photon path through a bridge                      core -, wings +
        insert a filament at rho_f = rho_*                dF/F = -0.070, log-slope
                                                          0.87 in far mass,
                                                          saturating in rho_f
                                                          (Newton: 0 and 1)

**The 0.1% line explains BK.5.** Scrambling members at fixed radial profiles
barely moves the path family's pair forces — *the member-locked signal is not
where this physics is.* The well-network detector was built for a signature the
reciprocal path law does not predict, which is why it measured nothing.

## BL.5 Where each family bites itself

**P confines cluster members by order unity** at any bench-identifiable
amplitude: a member's internal gravity x7.6 at 20 kpc at the fiducial, net force
outward beyond 38 kpc. That is the tensor lane's "switches on hardest inside
cluster galaxies" trap from a different construction. Its surviving distinctive
signal at member-safe amplitudes is the bridge, at the (100 km/s)^2 level — and
**the bench has no two-concentration scene to probe it.** Whether the fiducial's
member confinement is already excluded could not be checked without opening
member dynamics, which are in the confirmation reserve.

**T's endpoint quadrupole has the wrong sign for the cluster excess** at f_E > 0
(0.92-0.95 inside 1 Mpc); its signal is directional, not a boost.

## BL.6 Two findings the brief did not ask for

A **transition-band-gated tensor and a constant tensor of the same axis and sign
have OPPOSITE force quadrupoles** — the potential is deeper along the axis in
both, but chi approaches a constant from above in the gated case. Validated
against the exact constant-K solution (4.7e-4) and both asymptotes. And **the
compiler's declared radial caricature has the wrong sign for constant K and a
magnitude wrong by up to 40x for the gated case.** No verdict depends on it —
Gate 1 uses the full stretch — but **no sign may be read off it**, and it is
flagged in the compiler source.

The compiler extension is additive: +153/-6, 29 pre-existing verdicts unchanged
field by field, external controls 12/12, suite 48/48 (re-verified in BL.8).

## BL.7 Fields filled with an assumption the lane had to invent

Stated by the lane, kept here so they are not mistaken for derivations: that the
relativistic completion leaves tensor waves at c (a static action cannot decide
it); L_env is declared, not derived, and the axis is a background field held
fixed under variation; rho_* = 1e-24 kg/m^3 was chosen so the probes straddle it;
the signs of f_E and eps are free, and the falsifier is that one sign holds
universally. The Solar-System comparison quotes published bounds as an order of
magnitude only.

## BL.8 What this changes about the charter

Criterion 2 (generative) moves from PARTIAL toward MET for these two families:
given a scene and boundary conditions, each produces a gravitational state rather
than a fitted ratio. Criterion 1 (a dependence absent from the RAR) now has
candidates with **specific, non-anisotropy falsifiers** — a zero-twist phase lock
and a compensated bridge — where before it had none. **Criterion 4 (multiple
probes) is where both stand or fall**: T predicts matter-light covariance +1 with
no slip; P predicts a photon feature with zero net mass. Neither has been tested
on data, and P's testable signature needs a scene the bench cannot build. The
primary objective remains not achieved; **it now has two concrete things to
fail.**

---

# Run BM — the push to GitHub main: a 38-commit history rewrite, 6.7 GB of LFS, and an edit I clobbered

A provenance entry, not a science one. Everything the programme has produced
since Run AL is now on `origin/main` at `d5c1f2b8` (52 commits ahead of the
previous `58caee92`, 0 behind, verified). Two things happened on the way that
must be on the record.

## BM.1 GitHub rejected the first push, and the fix rewrote 38 commits

The pre-receive hook refused one blob over the 100 MB hard limit —
`env-data/raw/groups/tempel2017_table1_galaxies.tsv` at 105.9 MB — with three
more raw catalogues at 61-98 MB (warnings). All four are public catalogues that
already carry `.manifest.json` hash pins, so the bytes never needed to be in git;
committing them was an oversight against a partial `raw/` ignore rule.

Removed from the 38 unpushed commits that carried them (`git filter-repo`,
scoped to `origin/main..main` only), with the user's explicit choice of that
over an LFS migration — the repo already holds LFS objects, but the added
volume risked a quota failure the user could not rule out. Verified before the
push: 0 blobs over 100 MB in the range, exactly 4 deletions and **zero other
changes** against the pre-rewrite tree, 51 commits preserved, `origin` intact.
The pre-rewrite `main` is kept as `backup/pre-rewrite-main` (`9a0ac7bc`) for
one-command reversal. The four files remain on disk and are now `.gitignore`d
so a future `add` cannot re-commit them; their manifests stay tracked.

**The push then uploaded 1,511 LFS objects, 6.7 GB.** The repo's existing
`.gitattributes` LFS rules cover far more of what the lanes committed (caches,
arrays, maps) than the four files under inspection. It succeeded, so quota did
not block, but that volume now sits in the user's GitHub LFS storage and may
carry a billing consequence. Recorded so it is not a surprise.

## BM.2 The rewrite clobbered a running lane's uncommitted edit — my error

`filter-repo`'s final checkout reset the working tree to HEAD. Three tracked
files carried uncommitted edits from live lanes at that moment: the bridge
lane's `+147/-6` two-body-probe extension to `compiler/compiler.py`, its
regenerated `compiler_results.json`, and its `registry.json` row. **All three
were reverted.** I had snapshotted their hashes precisely to detect this and it
detected it — but detection is not protection. The AU.6 lesson was about
`git add -A` sweeping a lane's files *into* a commit; this is its mirror: a
checkout sweeping a lane's edits *out of* the working tree. **Stash or copy
every modified tracked file before any operation that checks out.**

**Scope, established rather than assumed:**

    extraction lane   UNHARMED. Every output (discriminator 17:18:10 ...
                      certificates and REPORT.md 17:19:19) predates the revert
                      at 17:19:59; it does not import the compiler; its registry
                      row survived because registry.json was committed at BL.
    bridge lane       COMPUTE COMPLETE AND SAFE. Its five substantive results --
                      compile_bridge 16:51, cdm_attack 16:57, bridge.json 17:03,
                      detect.json 17:14 -- all predate the revert, and its
                      post-patch compiler test log (16:36) proves the suite
                      passed with its edit. Zero python processes remain.
                      PENDING AND BROKEN: its final certify -> tests -> report
                      steps read `two_body_probe_consulted` and escape (d) from
                      the compiler, which the reverted file lacks. They will
                      fail or mis-report when the agent resumes.
    registry          the bridge lane's row is the one lost edit.

**Not reconstructed by hand.** The lost ~150 lines are gate logic the bridge
lane designed; guessing at them from `compile_bridge.py`'s field names would
manufacture a compiler extension and call it recovered. The lane's own files and
results are the specification; on its completion report a targeted repair
re-applies from them.

## BM.3 Bookkeeping: registry letters drifted from record letters

Lanes self-registered before record letters were assigned: the registry carries
`BK-synthesis` and `BL-extraction`, where the record has BK = the CDM
decomposition, BL = the synthesis, and BM = this entry. The registry's `run_id`
is a lane label, not a record pointer; the record is authoritative. Noted so
neither is "corrected" to match the other by a later reader.

## BM.4 Working-tree state after the push

Branch and `main` both at `d5c1f2b8`, matching origin. Two untracked lane
directories (`bridge/` 49 files, `extraction/` 49 files) remain uncommitted by
design until their lanes report. Four ignored raw catalogues on disk. The `dev/`
copy of this record is byte-identical to the tracked one.

---

# Run BN — the charter, read and compared: 11 of 41 requirements met, deliverable (b) delivered, the goal not achieved

The charter (`invariant-gravity-discovery-charter.md`, 1,242 lines) was opened in
this session and its requirement-bearing text compared clause by clause against
the record. This is that comparison, not the programme's self-assessment.
Machine-readable form in `work/wellnet-2026-09/charter-eval/charter_eval.json`
(refreshed for Runs BI-BM; 11 rows changed from the 21:04Z table).

## BN.1 The goal statement (L11-15), clause by clause

> *"Discover the simplest universal law of gravity or spacetime that starts only
> from observable matter, energy, motion, light, and their complete spatial and
> temporal arrangement, and predicts the behavior of massive objects and photons
> across all accessible scales."*

**NOT ACHIEVED.** Two candidate laws exist (Run BL), constructed as actions with
two universal constants each; neither has been tested on data, and each carries a
disqualifying property at bench-identifiable amplitude (P confines cluster
members x7.6; T's endpoint quadrupole has the wrong sign for the cluster excess).

> *"...must determine whether gravity depends only on local matter or also on
> the emptiness, geometry, direction, connectivity, motion, and history of the
> space between concentrations of matter."*

**NOT DETERMINED.** Run BF: seven such dependences form ONE observational class
on a realistic corpus (conditional on D, A, G, N — BJ.2). The corpus cannot yet
tell local from non-local, and Run AZ found five of the charter's fifteen axes
have never been varied.

> *"...must treat individual galaxies, gas structures, stars or stellar
> populations, and light paths before any averaging is imposed."*

**BUILT, NOT EXERCISED ON DATA.** Stage 1 exists (Run BD, 38/38, contract
enforced at construction) and the commutation gate refuses 7 of 8 averaging
substitutions — but every scene it has processed is synthetic.

> *"...one global set of physical constants, predict both internal motion within
> galaxies and motion through groups and clusters, reproduce known
> high-precision local gravity, and explain or sharply characterize the
> remaining galaxy-cluster discrepancy."*

**Global constants: MET** (never violated). **Internal + through-cluster motion
by one law: NOT MET** (never same-scene; Run BC's cross-scene transfer was
refused by the gate). **Local gravity: PARTIAL** (derived for both BL actions,
with an invented relativistic assumption). **Galaxy-cluster discrepancy: sharply
characterised, not explained** — Run AL: a galaxy-calibrated RAR lands within
1.9% of the eFEDS shear normalisation with zero new gravity parameters, and the
surviving excess rests on one admissible leg (eFEDS T0 shear inside its measured
support) after CLASH (AX) and LoCuSS (BC) masses were graded T4.

> *"Invariant must eliminate candidates that depend on arbitrary coordinate
> choices, an arbitrary zero of potential, catalog deblending, hidden
> object-specific parameters, non-identifiable quantities, or inconsistent
> conservation rules."*

**MET.** The compiler (AM/AU) does exactly this, in the charter's own five bins,
with 12/12 external controls — and BJ.3/BA.5 keep "non-identifiable" and
"unsupported by this scorer" from being read as "false".

> *"It must test surviving laws against synthetic alternate universes and
> untouched real observations."*

**Synthetic: MET** (BF, BK, the bridge lane). **Untouched real observations:
BLOCKED** — Run AW: the programme has no confirmation set; 23 of 25 datasets are
spent, and the reserve is five product-scoped shots, three in one regime.

> *"Its final output must be either a new generative field law with distinctive
> successful predictions, or a precise statement of which broad classes of new
> gravity have been ruled out and which observation would distinguish the
> remaining equivalence classes."*

**Branch (a): NOT MET. Branch (b): MET** — assembled in `deliverable_b.py`
(BH/BI/BJ): 4 theory families ruled out, 7 specific implementations, 3
statistics, 1 experimentally non-identifying; 5 equivalence classes remaining,
each with its named separating observation; 1 separable with data that already
exists. **The output clause is satisfied on its fallback branch and not on its
primary one.**

## BN.2 The twelve questions a final law must answer (L25-38)

    0 MET   10 PARTIAL   2 NOT_MET

Not one is answered *"from the same underlying description"* — the charter's own
standard. The two NOT_MET: *what creates gravity* (every family ever built takes
rest mass as its only source) and *why galaxies and clusters differ* (BC's
radius hypothesis refused; no admissible answer stands). The best-served: *what
creates redshift* — both tested mechanisms answered negatively (AK at 90 sigma;
BI at 13.6-21.3 sigma of power) — and *what would falsify it*, now PARTIAL
because Run BL supplies two falsifiers specific by construction, for candidates
not for an established law.

## BN.3 The twelve promotion criteria (L1194-1216)

    6 MET   4 PARTIAL   1 NOT_MET   1 BLOCKED

MET: generative (2), one global parameter set (3), preserves known limits (5),
matches galaxy regularities (6), survives representation changes (8), sparse
(11) — all by the two BL actions and the gates. PARTIAL: a dependence absent from
the RAR (1: constructed, not demonstrated), multiple probes (4: predicted, never
same-scene), alternate-universe controls (9: BK's FP 0.002 holds only at zero
halo-filament alignment), can evolve (12: linear only). **NOT_MET: improves the
cluster problem using ROOT observations (7).** **BLOCKED: a distinctive SEALED
prediction (10) — there is nothing sealed to predict against.** The charter says
a candidate is promoted only if it satisfies *all twelve*. None does.

## BN.4 Stages 0-10 and Corpora A-F

    Stages   4 MET   6 PARTIAL   1 NOT_MET      (Stage 10, confirmation: NOT_MET)
    Corpora  1 MET   4 PARTIAL   1 BLOCKED      (Corpus E, gold cluster: BLOCKED)

Stage 4 is downgraded from the earlier table's MET to PARTIAL: the certificate
refuses 5/5 historical and 5/5 new failures and refused a fresh result (BC), but
BE.8 requires *independently authored* prospective validation and BE.5 records
that it was self-authored. Corpus E is blocked by the world, not the programme:
Run BD found no public cluster carrying resolved baryons, member internal
dynamics, raw lensing and environment together.

## BN.5 The answer to "has the charter been met"

    total   41 requirements:  11 MET   24 PARTIAL   4 NOT_MET   2 BLOCKED

**No.** The charter is met on its fallback output branch and on its process
requirements for elimination, synthetic testing and representation-invariance.
It is not met on its goal, on any of the twelve questions, on the promotion
criteria as a set, on cluster evidence from root observations, on a sealed
prediction, or on the gold-cluster corpus. The bridge lane still running cannot
change that: at best it certifies one falsifier of one candidate on synthetic
data. What would change it is named in the record — a sealed confirmation scene
(AW), a gold cluster (BD), and one of the two BL actions surviving a matter-and-
light test in the same real scene (BE.8's exit condition).

This entry is the evaluation the goal check asked for. Its verdict is that the
condition is not satisfied, stated so rather than declared met to release it.
