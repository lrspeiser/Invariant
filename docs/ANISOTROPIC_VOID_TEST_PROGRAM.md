# An explicit test program for the anisotropic-void gravity hypothesis

The uploaded analysis is highly compatible with the earlier “void-polarized spacetime” proposal. It supplies the strongest missing piece: a conservative weak-field equation that preserves a scalar potential and gravitational-flux conservation,

$$
\boxed{
\nabla\cdot\!\left(\mathbf K\nabla\Psi\right)=4\pi G\rho_b,
}
$$

with

$$
\mathbf g=-\nabla\Psi,
\qquad
\mathbf D_g=\mathbf K\mathbf g.
$$

That is a better computational foundation than directly multiplying Newtonian gravity by an angular function. The uploaded analysis correctly notes that a field like

$$
\mathbf g=-\frac{GM}{r^2}f(\theta,\phi)\hat{\mathbf r}
$$

will generally have nonzero curl and therefore cannot globally be derived from a potential. Its tensor equation avoids that problem and follows from an action when \(\mathbf K\) is symmetric and positive definite.

The earlier analysis should therefore be reorganized as a set of modules around this core:

$$
\boxed{
\text{baryons}
\rightarrow
\text{void state }q
\rightarrow
\text{direction tensor }\mathbf K
\rightarrow
\Psi
\rightarrow
\begin{cases}
\text{matter dynamics}\\
\text{lensing}\\
\text{clock/redshift transport}\\
\text{structure formation}
\end{cases}
}
$$

The goal is not to make one enormously flexible equation fit everything. It is to identify which modules are actually required and whether one universal parameter set predicts multiple kinds of observations.

---

## 1. Comparison of the two analyses

### What the uploaded analysis establishes especially well

First, it distinguishes ordinary anisotropy caused by a flattened source from anisotropy in the law or medium itself. A galaxy’s ordinary quadrupole field declines much faster than its monopole field, so the visible disk’s shape alone cannot maintain a strong outer-galaxy effect.

Second, it gives a mathematically controlled constitutive equation:

$$
\mathbf D_g=\mathbf K\mathbf g,
\qquad
\nabla\cdot\mathbf D_g=-4\pi G\rho_b,
$$

and therefore

$$
\boxed{
\nabla\cdot(\mathbf K\nabla\Psi)=4\pi G\rho_b.
}
$$

This lets the conserved flux point differently from the acceleration while keeping

$$
\mathbf g=-\nabla\Psi.
$$

Third, it provides a clean axisymmetric tensor:

$$
\boxed{
\mathbf K=
\kappa_\parallel\hat{\mathbf e}\hat{\mathbf e}^{T}
+
\kappa_\perp
\left(\mathbf I-\hat{\mathbf e}\hat{\mathbf e}^{T}\right).
}
$$

A smaller eigenvalue means a larger field is needed to transmit a given flux component. The use of \(\hat{\mathbf e}\hat{\mathbf e}^{T}\), rather than an oriented vector alone, makes this an axis rather than a one-way arrow.

Fourth, the uploaded analysis identifies the central geometric mechanism we want to test:

$$
1/R^2\ \text{spreading over a sphere}
\quad\longrightarrow\quad
1/R\ \text{spreading through a layer}.
$$

If the effective area available to gravitational flux changes from

$$
4\pi R^2
$$

to approximately

$$
4\pi R h,
$$

then

$$
g_R\sim\frac{GM}{Rh}
$$

and the orbital speed becomes approximately constant. But it also correctly points out that the baryonic Tully–Fisher relation requires

$$
h_{\rm eff}\propto\sqrt{\frac{GM_b}{a_0}},
$$

which does **not** follow automatically from generic anisotropy. The field equation must produce that scaling rather than having it inserted separately for every galaxy.

### What the earlier void-polarization analysis adds

The earlier analysis contributes the pieces needed to turn \(\mathbf K\) into a predictive physical model:

1. A bounded field \(q(\mathbf x,t)\) describing how “void-like” the local spacetime state is.
2. A nonlocal equation determining \(q\) from density and acceleration.
3. A tidal tensor determining the locally preferred direction.
4. A nonlinear low-acceleration function \(\mu\).
5. A relativistic metric determining how photons respond.
6. A memory equation for colliding clusters.
7. Candidate clock-transport laws capable of producing cumulative redshift.
8. A near-signature-flip metric inspired by black-hole interiors.

The uploaded analysis is therefore the cleaner weak-field core. The previous analysis is the broader physical completion.

### Three important corrections exposed by comparing them

**A constant \(\mathbf K\) does not produce flat rotation curves.** Its point-source solution has ellipsoidal equipotentials, but still behaves as an anisotropic inverse-square field. The uploaded analysis gives that analytic solution explicitly. To obtain a transition toward \(1/R\), we need at least one of:

$$
\nabla\mathbf K\neq0,
\qquad
\mu\neq1,
\qquad
q\ \text{nonlocal},
\qquad
\text{a finite flux-confinement layer},
$$

or a combination of them.

**Spatial anisotropy in a metric is not enough by itself.** In a weak-field metric

$$
ds^2=-e^{2A}c^2dt^2+g_{ij}dx^idx^j,
$$

the leading slow-body acceleration is approximately

$$
\mathbf g=-c^2\nabla A.
$$

Changing only the spatial factors affects light and higher-order motion but does not automatically produce the required extra radial acceleration. The lapse \(A\) must be connected to \(q\), \(\mathbf K\), or both.

**A static gravitational potential cannot create cumulative path redshift.** A photon falling into and climbing out of a stationary potential does not simply retain an accumulated energy loss. The redshift sector must involve evolving geometry, non-metric transport, mode conversion with energy exchange, or a clock-calibration field.

---

# 2. The combined master model

We should define all candidate equations using the same intermediate fields.

## 2.1 Baryonic source

Given the three-dimensional baryonic density,

$$
\rho_b(\mathbf x)
=
\rho_\star(\mathbf x)+
\rho_{\rm gas}(\mathbf x),
$$

calculate the ordinary baryonic potential:

$$
\boxed{
\nabla^2\Phi_N=4\pi G\rho_b,
\qquad
\mathbf g_N=-\nabla\Phi_N.
}
$$

No dark halo is included in the candidate models.

## 2.2 Smoothed density

Define density on an environmental scale \(L_\rho\):

$$
\boxed{
\rho_L(\mathbf x)
=
\int W_{L_\rho}(\mathbf x-\mathbf x')
\rho_b(\mathbf x')\,d^3x'.
}
$$

A Gaussian kernel is a reasonable first choice:

$$
W_L(\mathbf r)
=
\frac{1}{(2\pi L^2)^{3/2}}
e^{-r^2/(2L^2)}.
$$

The value of \(L_\rho\) is a global model parameter, not a separately fitted galaxy size.

## 2.3 Candidate local void fields

We should test density, acceleration, and combinations independently:

$$
\boxed{
q_\rho=
\left[
1+\left(\frac{\rho_L}{\rho_c}\right)^m
\right]^{-1},
}
$$

$$
\boxed{
q_g=
\left[
1+\left(\frac{|\mathbf g_N|}{a_0}\right)^n
\right]^{-1},
}
$$

and

$$
\boxed{
q_{\rho g}=
\left[
1+
\left(\frac{\rho_L}{\rho_c}\right)^m+
\left(\frac{|\mathbf g_N|}{a_0}\right)^n
\right]^{-1}.
}
$$

All satisfy

$$
0<q\le1.
$$

Dense or high-acceleration regions have \(q\rightarrow0\), while low-density, low-acceleration regions have \(q\rightarrow1\).

## 2.4 Nonlocal void field

The simplest nonlocal version is a screened Poisson or Helmholtz equation:

$$
\boxed{
\left(1-L_q^2\nabla^2\right)q
=
q_{\rho g}.
}
$$

This means the vacuum state at one location depends on an extended surrounding region.

Boundary conditions matter. For an isolated-galaxy calculation, use an environmental value \(q_{\rm env}\) at the outer boundary:

$$
q|_{\partial\Omega}=q_{\rm env}.
$$

That value should eventually come from a reconstructed large-scale density field rather than being fitted independently for every galaxy.

## 2.5 Tidal direction

Calculate the external, smoothed potential and its traceless tidal tensor:

$$
\nabla^2\Phi_{\rm ext}=4\pi G\rho_{\rm ext},
$$

$$
\boxed{
T_{ij}
=
\partial_i\partial_j\Phi_{\rm ext}
-
\frac13\delta_{ij}\nabla^2\Phi_{\rm ext}.
}
$$

Its eigenvectors are

$$
\mathbf e_1,\mathbf e_2,\mathbf e_3.
$$

Each candidate must specify in advance whether its preferred axis is:

* the most compressive tidal eigenvector;
* the most extensive tidal eigenvector;
* the galaxy disk normal;
* the galaxy angular-momentum axis; or
* the full tidal tensor without choosing a single eigenvector.

The uploaded analysis explicitly proposes external acceleration or a principal tidal axis as the environmental direction.

---

# 3. A 30-case initial equation tournament

Rather than write 30 unrelated equations, use a controlled factorial design. This reveals whether performance comes from the void definition, the directional rule, or the low-acceleration nonlinearity.

## 3.1 Four void definitions

$$
Q_1=q_\rho,
\qquad
Q_2=q_g,
\qquad
Q_3=q_{\rho g},
\qquad
Q_4=(1-L_q^2\nabla^2)^{-1}q_{\rho g}.
$$

## 3.2 Four tensor geometries

### \(K_1\): scalar void response

$$
\boxed{
\mathbf K=e^{-\alpha q}\mathbf I.
}
$$

This tests stronger gravity in voids without directionality.

### \(K_2\): disk-axis tensor

Let \(\hat{\mathbf d}\) be the disk normal:

$$
P_d=\hat{\mathbf d}\hat{\mathbf d}^{T}.
$$

Then

$$
\boxed{
\mathbf K=
e^{-\alpha_dq}P_d+
e^{-\alpha_pq}(\mathbf I-P_d).
}
$$

This directly tests whether disk geometry alters gravitational transmission.

### \(K_3\): tidal-axis tensor

For a selected tidal eigenvector \(\hat{\mathbf e}\),

$$
\boxed{
\mathbf K=
e^{-\alpha_\parallel q}
\hat{\mathbf e}\hat{\mathbf e}^{T}
+
e^{-\alpha_\perp q}
\left(\mathbf I-\hat{\mathbf e}\hat{\mathbf e}^{T}\right).
}
$$

### \(K_4\): full tidal tensor

Normalize the tidal field:

$$
\widehat{\mathbf T}
=
\frac{\mathbf T}
{\sqrt{T_{ij}T^{ij}+\epsilon_T^2}}.
$$

Define

$$
\boxed{
\mathbf K=
e^{-\alpha_0q}
\exp\!\left[-\alpha_Tq\widehat{\mathbf T}\right].
}
$$

The matrix exponential of a real symmetric matrix is symmetric and positive definite, so this is numerically safer than independently fitting unconstrained tensor components.

## 3.3 Three nonlinear response choices

$$
M_0:\quad \mu(X)=1,
$$

$$
M_1:\quad
\mu(X)=\frac{X}{1+X},
$$

$$
M_2:\quad
\mu(X)=\frac{X}{\sqrt{1+X^2}},
$$

where

$$
\boxed{
X=
\frac{
\sqrt{\nabla\Psi^{T}\mathbf K\nabla\Psi}
}{a_0}.
}
$$

The unified field equation is

$$
\boxed{
\nabla_i
\left[
\mu(X)K^{ij}\nabla_j\Psi
\right]
=
4\pi G\rho_b.
}
$$

### The reduced 24-model set

Run:

$$
Q_{1\ldots4}\times K_{1\ldots4}\times M_0
$$

for 16 models, followed by

$$
Q_{3,4}\times K_{1\ldots4}\times M_1
$$

for eight additional models.

That gives 24 controlled candidate equations. \(M_2\) is held in reserve as a robustness test rather than multiplying the first search unnecessarily.

## 3.4 Six diagnostic models

These are not all complete physical theories. They tell us what mathematical behavior is required.

### D1: ordinary baryons

$$
\nabla^2\Psi=4\pi G\rho_b.
$$

### D2: scalar AQUAL benchmark

$$
\nabla\cdot
\left[
\mu\left(\frac{|\nabla\Psi|}{a_0}\right)
\nabla\Psi
\right]
=
4\pi G\rho_b.
$$

### D3: cylindrical-confinement target

Let

$$
R_t=\sqrt{\frac{GM_b}{a_0}}.
$$

Use the piecewise target law

$$
g_R=
\begin{cases}
GM_b(<R)/R^2,&R<R_t,\\[4pt]
\sqrt{GM_ba_0}/R,&R\ge R_t.
\end{cases}
$$

This is not yet a field theory. It tests whether the desired \(1/R\) asymptote and Tully–Fisher normalization are sufficient to fit the data.

### D4: flattened logarithmic potential

$$
\boxed{
\Psi_{\log}(R,z)
=
\frac{v_f^2}{2}
\ln\left[
r_c^2+R^2+\frac{z^2}{q_g^2}
\right],
}
$$

with

$$
v_f^4=GM_ba_0.
$$

This tests the combined flat-rotation and flattened-potential behavior.

### D5: linear near-flip metric

$$
ds^2
=
-e^{2A(q)}c^2dt^2+
e^{2B_\perp(q)}d\ell_\perp^2+
e^{2B_\parallel(q)}d\ell_\parallel^2,
$$

where

$$
A=\frac{\Phi_N}{c^2}+\alpha_Aq,
\qquad
B_a=\alpha^B_aq.
$$

The slow-body acceleration is

$$
\boxed{
\mathbf g
=
-\nabla\Phi_N
-\alpha_Ac^2\nabla q.
}
$$

### D6: saturating near-flip metric

Use

$$
\boxed{
A=
\frac{\Phi_N}{c^2}
-
\alpha_A\ln(1-sq),
\qquad
0<s<1.
}
$$

Then

$$
\mathbf g
=
-\nabla\Phi_N
-
\frac{\alpha_A s c^2}{1-sq}\nabla q.
$$

As \(sq\) approaches one, the lapse response becomes strong, but the exponential metric coefficient remains positive. This explores black-hole-like behavior without actually changing metric signature.

---

# 4. Lensing modules

The gravity equation predicts slow-particle motion through \(\Psi\). It does not yet uniquely predict photon motion.

Use the weak-field metric

$$
ds^2
=
-\left(1+\frac{2\Psi}{c^2}\right)c^2dt^2
+
\left(\delta_{ij}-\frac{2\Phi_{ij}}{c^2}\right)
dx^idx^j.
$$

## L0: GR-like no-slip closure

$$
\boxed{
\Phi_{ij}=\Psi\delta_{ij}.
}
$$

The deflection potential is \(\Psi+\Phi=2\Psi\).

## L1: scalar void-dependent slip

$$
\boxed{
\Phi=\gamma(q)\Psi,
\qquad
\gamma(q)=1+\gamma_q q.
}
$$

Then

$$
\hat{\boldsymbol\alpha}
=
\frac1{c^2}
\int
\nabla_\perp[(1+\gamma)\Psi]\,d\ell.
$$

## L2: tensor spatial curvature

$$
\boxed{
\Phi_{ij}
=
\gamma_\parallel(q)\Psi P_{\parallel,ij}
+
\gamma_\perp(q)\Psi P_{\perp,ij}.
}
$$

For a light ray with direction \(\hat{\mathbf k}\), the first working lens potential is

$$
U_{\rm lens}
=
\Psi+
\Phi_{ij}k^ik^j.
$$

The direct calculation should ultimately integrate null geodesics through the complete metric rather than rely only on this reduced expression.

After the gravity tournament, take the three best gravity models and test all three lensing closures:

$$
3\times3=9
$$

joint dynamics-and-lensing models.

---

# 5. Redshift and photon modules

These should be tested only after the gravity and lensing parameters have been frozen.

## Z0: endpoint gravitational redshift

$$
\boxed{
\ln(1+z)
\simeq
\frac{\Psi_o-\Psi_e}{c^2}.
}
$$

This is the stationary-metric control. It should not produce large cumulative cosmological redshift.

## Z1: void-path transport

Define

$$
I_q=\int_0^Dq(\ell)\,d\ell.
$$

Then

$$
\boxed{
\ln(1+z)
=
\frac{H_*}{c}I_q
+
\zeta(q_o-q_e).
}
$$

## Z2: gravitational-field path integral

$$
I_g
=
\frac1{c^2}
\int_0^D|\nabla\Psi|\,d\ell,
$$

$$
\boxed{
\ln(1+z)=\xi I_g.
}
$$

## Z3: directional void transport

$$
I_T
=
\int_0^D
q(\ell)
P_2[\hat{\mathbf k}\cdot\hat{\mathbf e}(\ell)]
\,d\ell,
$$

$$
\boxed{
\ln(1+z)
=
\frac{H_*}{c}
\left(I_q+\eta I_T\right).
}
$$

## Z4: conformal clock field

$$
\tilde g_{\mu\nu}=e^{2\omega(q,t)}g_{\mu\nu},
$$

$$
\boxed{
1+z=e^{\omega_o-\omega_e}.
}
$$

Use

$$
\omega(q,t)=\bar\omega(t)+\omega_q q.
$$

A purely homogeneous \(\bar\omega(t)\) may be observationally equivalent to an ordinary scale factor. The distinctive part is the environmental contribution \(\omega_q q\).

## Z5: hybrid expansion plus void effect

Before attempting to eliminate expansion entirely, fit

$$
\boxed{
\ln(1+z_{\rm obs})
=
\ln(1+z_{\rm FLRW})
+
\lambda_z I_q.
}
$$

This determines whether the data permit or prefer any path-dependent component at all.

## P1: photon–graviton conversion

$$
P_{\gamma\rightarrow g}
=
\sin^2(2\vartheta)
\sin^2\left(\frac{\Delta kL}{2}\right).
$$

This changes photon survival,

$$
P_\gamma=1-P_{\gamma\rightarrow g},
$$

but by itself it produces dimming, not necessarily redshift. It belongs in the opacity and spectral-distortion test rather than being presumed to explain \(z\).

---

# 6. Numerical tests before using astronomical data

These are mandatory. A model that fails them should never reach the fitting stage.

## 6.1 Dimensional consistency

Every argument to an exponential, logarithm, or interpolation function must be dimensionless. Automatically inspect the symbolic expressions or unit-test them with a units package.

## 6.2 Positive-definite tensor

At every grid cell compute the eigenvalues

$$
\lambda_1,\lambda_2,\lambda_3
$$

of \(\mathbf K\). Require

$$
\boxed{
\lambda_i>0.
}
$$

A proposed engineering threshold is

$$
\lambda_i>10^{-6},
$$

to prevent the elliptic equation from becoming numerically singular. This threshold must be varied in sensitivity tests.

## 6.3 Flux conservation

For every closed numerical surface \(S\),

$$
\oint_S
\mu\mathbf K\nabla\Psi\cdot d\mathbf A
=
4\pi GM_b(<S).
$$

Define

$$
\epsilon_{\rm flux}
=
\frac{
\left|
\oint_S\mu\mathbf K\nabla\Psi\cdot d\mathbf A
-
4\pi GM_b
\right|
}{
4\pi GM_b
}.
$$

Require initially

$$
\boxed{
\epsilon_{\rm flux}<10^{-5}.
}
$$

## 6.4 Constant-\(\mathbf K\) analytic test

For a point mass and constant tensor, compare the numerical solution to

$$
\Psi(\mathbf r)
=
-\frac{GM}
{\sqrt{\det\mathbf K}}
\frac1{
\sqrt{\mathbf r^T\mathbf K^{-1}\mathbf r}
}.
$$

This exact solution is supplied by the uploaded analysis.

Run at three grid resolutions and require second-order or better convergence outside the softened central cell.

## 6.5 Curl test

Calculate

$$
\nabla\times(-\nabla\Psi).
$$

It should converge toward zero with increasing resolution. This specifically guards against accidentally implementing the prohibited angular-force shortcut.

## 6.6 Newtonian recovery

For

$$
q\rightarrow0,\qquad X\gg1,
$$

require

$$
\mathbf K\rightarrow\mathbf I,
\qquad
\mu\rightarrow1,
\qquad
\Psi\rightarrow\Phi_N.
$$

Test a Solar-like source over accelerations much greater than \(a_0\).

## 6.7 Domain and boundary convergence

Repeat every representative galaxy calculation with:

$$
R_{\max},\quad 2R_{\max},\quad4R_{\max}.
$$

Require predicted velocities within the observed region to change by less than approximately \(0.5\%\).

This is especially important for a \(1/R\) force. A \(1/R\) field cannot continue unchanged to infinity without producing a logarithmically divergent potential. The model must predict an outer coherence scale, environmental transition, or overlap with neighboring wells.

## 6.8 Angular quadrupole diagnostic

For

$$
p(\theta)=\frac1{4\pi}
[1+\beta P_2(\cos\theta)],
$$

positivity requires

$$
\boxed{-1\le\beta\le2.}
$$

Use this only to compare the angular distribution produced by the tensor solver. Do not use it as the force equation itself.

---

# 7. Test 1: galaxy rotation curves

## Data required

The first dataset should be SPARC. It contains 175 late-type galaxies with 3.6-micron photometry, gas data, rotation curves, distances, inclinations, luminosities, characteristic radii, surface brightnesses, gas masses, flat velocities, and quality flags. Its public mass-model tables include radius, observed velocity and uncertainty, and the separate Newtonian velocity contributions of gas, stellar disk, and bulge. ([Astroweb][1])

For each point we need:

$$
R_0,\quad
V_{\rm obs,0},\quad
\sigma_V,\quad
V_{\rm gas},\quad
V_{\rm disk},\quad
V_{\rm bulge}.
$$

For each galaxy we also need:

$$
D_0,\ \sigma_D,\ i_0,\ \sigma_i,\ 
L_{3.6},\ M_{\rm HI},\ R_d,\ 
\Sigma_{\rm eff},\ V_f,\ Q.
$$

## Data transformations

Sample distance \(D\), inclination \(i\), disk mass-to-light ratio \(\Upsilon_d\), and bulge mass-to-light ratio \(\Upsilon_b\).

Correct the radius:

$$
\boxed{
R=R_0\frac{D}{D_0}.
}
$$

Correct the deprojected observed velocity:

$$
\boxed{
V_{\rm obs}
=
V_{\rm obs,0}
\frac{\sin i_0}{\sin i}.
}
$$

Calculate the baryonic Newtonian contribution:

$$
\boxed{
V_b^2
=
\frac{D}{D_0}
\left[
V_{\rm gas}|V_{\rm gas}|
+
\Upsilon_dV_{\rm disk}^2
+
\Upsilon_bV_{\rm bulge}^2
\right].
}
$$

The signed gas term is retained because tabulated gas contributions can be negative in regions where the gas distribution produces an outward radial contribution.

Then

$$
g_{\rm bar}(R)=\frac{V_b^2}{R},
\qquad
g_{\rm obs}(R)=\frac{V_{\rm obs}^2}{R}.
$$

## First-pass algebraic predictions

Models D1–D4 and scalar low-acceleration models can be screened without a full three-dimensional PDE.

For example:

$$
g_{\rm D1}=g_{\rm bar},
$$

$$
g_{\rm D3}
=
\begin{cases}
g_{\rm bar},&g_{\rm bar}\ge a_0,\\
\sqrt{a_0g_{\rm bar}},&g_{\rm bar}<a_0,
\end{cases}
$$

$$
V_{\rm model}(R)=\sqrt{Rg_{\rm model}(R)}.
$$

This will quickly determine whether the needed asymptotic scaling is present.

## Full PDE predictions

For the tensor models, build an axisymmetric density model:

$$
\rho_\star(R,z)
=
\Sigma_\star(R)f_\star(z),
$$

with, for example,

$$
f_\star(z)
=
\frac1{2h_z}
\operatorname{sech}^2\left(\frac z{h_z}\right).
$$

Create a similar gas layer with a smaller vertical scale height. Then solve

$$
\frac1R\frac{\partial}{\partial R}
\left[
R
\left(
A_{RR}\frac{\partial\Psi}{\partial R}
+
A_{Rz}\frac{\partial\Psi}{\partial z}
\right)
\right]
+
\frac{\partial}{\partial z}
\left(
A_{zR}\frac{\partial\Psi}{\partial R}
+
A_{zz}\frac{\partial\Psi}{\partial z}
\right)
=
4\pi G\rho_b,
$$

where

$$
\mathbf A=\mu\mathbf K.
$$

The predicted midplane circular velocity is

$$
\boxed{
V_{\rm model}^2(R)
=
R
\left.
\frac{\partial\Psi}{\partial R}
\right|_{z=0}.
}
$$

## Why SPARC is only the first layer

SPARC’s one-dimensional mass-model products are excellent for screening equations, but a genuinely directional PDE needs resolved baryonic maps. Use a smaller high-quality subset drawn from THINGS, LITTLE THINGS, and WALLABY, which provide resolved H I maps, velocity fields, data cubes, and, for some releases, spatially resolved kinematic models. ([ADS Abstracts][2])

For those galaxies:

1. Fit the full data cube or velocity field, not merely the published rotation curve.
2. Infer inclination, position angle, warps, noncircular motions, and gas surface density jointly.
3. Construct \(\rho_b(x,y,z)\).
4. Solve the three-dimensional tensor PDE.
5. Project the model velocity field back into observed line-of-sight velocities.
6. Convolve with the telescope beam and spectral response.
7. Compare directly with the observed cube or moment maps.

That avoids treating a rotation curve already extracted under axisymmetric assumptions as the raw observation.

## Likelihood

For a simple rotation-curve fit:

$$
\ln\mathcal L_i
=
-\frac12
(\mathbf V_{\rm obs}-\mathbf V_{\rm model})^T
C_i^{-1}
(\mathbf V_{\rm obs}-\mathbf V_{\rm model})
-\frac12\ln|C_i|.
$$

The global gravity parameters are shared by every galaxy:

$$
\theta_G=
\{a_0,\rho_c,m,n,L_\rho,L_q,
\alpha_\parallel,\alpha_\perp,\ldots\}.
$$

The galaxy-specific nuisance parameters are:

$$
\eta_i=
\{D_i,i_i,\Upsilon_{d,i},
\Upsilon_{b,i},h_{z,i}\}.
$$

No galaxy receives its own \(a_0\), \(\rho_c\), or anisotropy coefficient in the primary test.

## Train, validation, and blind split

Split by whole galaxies, never by individual radial points:

$$
60\%\ \text{training},
\qquad
20\%\ \text{validation},
\qquad
20\%\ \text{blind test}.
$$

Stratify by:

$$
M_b,\quad
\Sigma_b,\quad
f_{\rm gas},\quad
V_f,\quad
\text{rotation-curve quality}.
$$

Freeze the blind list before fitting.

## Required outputs

For every candidate produce:

$$
\chi^2_{\rm train},
\quad
\chi^2_{\rm blind},
\quad
\text{held-out log likelihood},
$$

$$
\operatorname{RMS}
\left[
\log_{10}g_{\rm obs}
-
\log_{10}g_{\rm pred}
\right],
$$

and residual correlations with:

$$
R/R_d,\quad
\Sigma_b,\quad
M_b,\quad
f_{\rm gas},\quad
i,\quad
D,\quad
\text{environment}.
$$

A candidate that fits only by pushing distances, inclinations, or stellar mass-to-light ratios to the edges of their priors is not successful.

---

# 8. Test 2: baryonic Tully–Fisher and the confinement scale

For each galaxy calculate

$$
M_b=
\Upsilon_dL_d+
\Upsilon_bL_b+
M_{\rm gas}.
$$

Use the model’s predicted outer velocity \(V_{f,\rm pred}\), not the observed velocity, and fit

$$
\boxed{
\log_{10}M_b
=
b+s
\log_{10}\left(
\frac{V_{f,\rm pred}}
{100\ {\rm km\,s^{-1}}}
\right)
}
$$

with intrinsic scatter \(\sigma_{\rm int}\).

The important test is not merely whether a regression can return \(s\approx4\). It is whether the field equation itself produces it with universal parameters.

For every tensor solution define an empirical effective confinement thickness:

$$
\boxed{
h_{\rm eff}(R)
=
\frac{GM_b(<R)}
{R\,g_R(R)}.
}
$$

In the approximately flat portion of the curve, calculate a weighted mean

$$
\bar h_{\rm eff}.
$$

Then fit

$$
\log\bar h_{\rm eff}
=
c+
s_h\log M_b.
$$

The cylindrical argument requires

$$
\boxed{
s_h=\frac12.
}
$$

This is a particularly clean test of the uploaded analysis. If each galaxy can be fit only by assigning an arbitrary \(h_{\rm eff}\), the model has merely renamed the halo scale.

Also inspect whether the transition radius satisfies

$$
\boxed{
R_t\approx\sqrt{\frac{GM_b}{a_0}}.
}
$$

The scatter in this relation must be predicted by measured structure or environment, not absorbed into arbitrary per-galaxy parameters.

---

# 9. Test 3: vertical gravity and galaxy thickness

A model can fit radial speeds by overconcentrating gravity in the disk plane while predicting completely wrong vertical forces.

## Data required

Use stellar velocity-dispersion and disk-thickness measurements. The DiskMass Survey provides stellar and gas kinematics for a sample of nearly face-on spiral galaxies designed to measure disk dynamics, while MaNGA’s final release contains spatially resolved spectroscopy for more than ten thousand galaxies. ([arXiv][3])

For each test galaxy we need:

$$
I(R,z),\quad
\sigma_{\rm los}(R),\quad
h_z(R),\quad
i,\quad
\text{PSF},\quad
\text{aperture geometry}.
$$

## Calculation

From the solved potential calculate

$$
K_z(R,z)
=
\frac{\partial\Psi}{\partial z}.
$$

For tracer density \(\nu(R,z)\), the vertical Jeans equation is

$$
\boxed{
\frac{\partial[\nu\sigma_z^2]}
{\partial z}
=
-\nu K_z.
}
$$

Assuming \(\sigma_z\rightarrow0\) as \(z\rightarrow\infty\),

$$
\boxed{
\sigma_z^2(R,z)
=
\frac1{\nu(R,z)}
\int_z^\infty
\nu(R,z')K_z(R,z')\,dz'.
}
$$

Project the velocity ellipsoid along the line of sight, integrate through the galaxy, and convolve with the PSF and fiber or slit aperture.

## Key diagnostic

Define

$$
\mathcal A_{\rm dyn}(R,z)
=
\frac{
g_R/g_{R,N}
}{
K_z/K_{z,N}
}.
$$

A model that enhances radial gravity while leaving vertical gravity relatively unchanged predicts

$$
\mathcal A_{\rm dyn}>1.
$$

A model that simply multiplies gravity isotropically predicts approximately

$$
\mathcal A_{\rm dyn}\approx1.
$$

This may be the cleanest way to distinguish true planar flux channeling from scalar modified gravity.

## Rejection condition

Freeze the gravity parameters obtained from SPARC. Permit only observational nuisance parameters. Reject a model if fitting the vertical data requires a different \(a_0\), \(\alpha_\parallel\), \(\alpha_\perp\), or \(L_q\) than the rotation-curve fit.

---

# 10. Test 4: environmental and tidal dependence

This is the direct test of the claim that gravity becomes stronger **toward other wells** and through voids.

## Data required

Use a reconstructed nearby density field and public void/cosmic-web catalogs. DESIVAST provides multiple DESI Year-1 bright-galaxy void catalogs extending to low redshift, while 2M++ and Cosmicflows-type reconstructions provide complementary density and peculiar-velocity fields. ([DESI Data][4])

For every galaxy we need:

$$
\mathbf x_{\rm gal},\quad
\hat{\mathbf d},\quad
\rho_{\rm env}(\mathbf x),\quad
\mathbf T(\mathbf x),\quad
d_{\rm void},\quad
R_{\rm void}.
$$

## Calculation of the tidal tensor

Place the baryonic tracer density on a three-dimensional grid. Correct for survey selection and mask. Smooth at several predeclared scales:

$$
L=1,\ 3,\ 5,\ 10\ {\rm Mpc}.
$$

In Fourier space,

$$
\Phi_{\rm ext}(\mathbf k)
=
-\frac{4\pi G\rho_b(\mathbf k)}{k^2}.
$$

Then

$$
\boxed{
T_{ij}(\mathbf k)
=
4\pi G
\left(
\frac{k_ik_j}{k^2}
-
\frac13\delta_{ij}
\right)
\rho_b(\mathbf k).
}
$$

Transform back to real space and diagonalize at every galaxy position.

## Direct residual test

Fit the gravity model without environmental information, then calculate

$$
\Delta\log g
=
\log g_{\rm obs}
-
\log g_{\rm base}.
$$

Test whether

$$
\Delta\log g
=
c_0+
c_q q_{\rm env}+
c_T|T|+
c_\theta
P_2(\hat{\mathbf d}\cdot\hat{\mathbf e})
$$

predicts held-out galaxies.

The tensor model should do better than this post-hoc regression because it predicts the entire rotation curve from the environmental field. The regression is a diagnostic of which variable contains the signal.

## Scrambled-axis control

Randomly rotate the tidal eigenvectors among galaxies while preserving each galaxy’s density, mass, and tidal eigenvalues. Re-run the model.

A genuine directional effect should lose predictive power when the axes are scrambled. A model that performs identically is using only scalar environment, not anisotropy.

## Geographic cross-validation

Hold out complete sky regions or survey volumes rather than random galaxies. This prevents the density reconstruction and nearby correlated systems from leaking environmental information into both training and test samples.

## Major circularity warning

Existing void catalogs generally convert redshifts into distances using a fiducial cosmology. That is acceptable for an exploratory low-redshift gravity test. A serious test of **no cosmological expansion** must rerun the density reconstruction and void finder using each candidate redshift–distance law.

---

# 11. Test 5: galaxy lensing and stellar dynamics

## Weak galaxy–galaxy lensing

DES Year 6 combines a very large source-shape sample, lens-galaxy clustering, and galaxy–galaxy lensing over roughly 5,000 square degrees. ([arXiv][5])

For every lens–source pair, use:

$$
z_l,\quad z_s,\quad
\text{source ellipticity},\quad
\text{shape weight},\quad
\text{photo-}z\ \text{distribution},
$$

plus lens:

$$
M_\star,\quad
R_e,\quad
\text{morphology},\quad
\text{position angle},\quad
\text{environment}.
$$

Do not begin from a published “lensing mass,” because that conversion already assumes a lensing law. Predict the reduced shear directly:

$$
g_{\rm red}
=
\frac{\gamma}
{1-\kappa}.
$$

For every lens:

1. Construct a baryonic stellar and gas mass model.
2. Solve the gravity equation for \(\Psi\).
3. Select lensing closure L0, L1, or L2.
4. Ray trace through the metric.
5. Predict \(\kappa,\gamma_1,\gamma_2\) at every source.
6. Average using the survey weights and redshift distributions.
7. Compare with the raw shape data or predeclared stacked data vector.

Measure both the monopole tangential shear,

$$
\gamma_{t,0}(R),
$$

and the quadrupole relative to the disk and tidal axes,

$$
\gamma_{t,2}(R)
=
\left\langle
\gamma_t(R,\phi)\cos2\phi
\right\rangle.
$$

A directional-gravity model makes a much more specific prediction for \(\gamma_{t,2}\) than an ordinary spherical halo fit.

## Strong lensing plus stellar kinematics

The SLACS sample supplies strong-lensing systems with lens and source redshifts, galaxy properties, stellar velocity dispersions, and image constraints. ([arXiv][6])

For each lens:

1. Build the stellar density from its light profile and \(\Upsilon_\star\).
2. Solve for \(\Psi\).
3. Solve the lens equation,

$$
\boxed{
\boldsymbol\beta
=
\boldsymbol\theta
-
\frac{D_{ls}}{D_s}
\hat{\boldsymbol\alpha}(\boldsymbol\theta).
}
$$

4. Predict image positions, Einstein radius, and magnification ratios.
5. Solve the spherical or axisymmetric Jeans equation,

$$
\frac{d(\nu\sigma_r^2)}{dr}
+
\frac{2\beta_{\rm ani}}r
\nu\sigma_r^2
=
-\nu\frac{d\Psi}{dr}.
$$

6. Project and aperture-average \(\sigma_{\rm los}\).
7. Fit image and dispersion data jointly.

This directly tests whether the potential required by stars is the same potential required by photons.

## Rejection condition

A candidate fails if:

$$
\theta_G^{\rm dynamics}
$$

and

$$
\theta_G^{\rm lensing}
$$

have incompatible posterior distributions, or if it requires a different lensing slip for each galaxy.

---

# 12. Test 6: ordinary galaxy clusters

Clusters test whether the equation scales from tens of kiloparsecs to megaparsecs.

## Hydrostatic data

X-COP provides public XMM-Newton density and temperature information, Planck pressure information, and derived thermodynamic and mass profiles for a sample of massive clusters extending toward the virial region. ([Dominique Eckert][7])

Required profiles are:

$$
n_e(r),\quad
T(r),\quad
P_{\rm SZ}(r),\quad
\rho_\star(r),\quad
C_{\rm profile}.
$$

## Calculation

Construct

$$
\rho_b(r)=\rho_{\rm gas}(r)+\rho_\star(r).
$$

Solve the candidate equation for \(\Psi(r)\) in the spherical first pass. The model acceleration is

$$
g_{\rm model}(r)=\frac{d\Psi}{dr}.
$$

The acceleration required by gas equilibrium is

$$
\boxed{
g_{\rm HSE}(r)
=
-\frac1{\rho_{\rm gas}}
\frac{dP_{\rm tot}}{dr}.
}
$$

Include nonthermal pressure as a nuisance model:

$$
P_{\rm tot}
=
\frac{P_{\rm th}}
{1-f_{\rm nt}(r)}.
$$

Use external priors for \(f_{\rm nt}\), rather than letting it absorb all discrepancies.

The profile likelihood is

$$
\ln\mathcal L_{\rm X}
=
-\frac12
(\mathbf P_{\rm obs}-\mathbf P_{\rm model})^T
C_P^{-1}
(\mathbf P_{\rm obs}-\mathbf P_{\rm model}).
$$

## Lensing cluster test

The Hubble Frontier Fields publish multiple cluster-lensing products for six massive clusters, including convergence, shear, deflection, and magnification reconstructions. ([MAST][8])

Use those reconstructed maps only for rapid diagnostics. They were derived using conventional lensing equations.

The final test should use:

$$
\text{multiple-image positions},
\quad
\text{source redshifts},
\quad
\text{weak-shear catalogs},
\quad
\text{cluster-member light},
\quad
\text{X-ray gas maps}.
$$

Given only the stars and gas, solve the candidate gravity equation and predict image positions and shear directly.

## Strong pass criterion

The global parameters inferred from galaxies must predict:

$$
g_{\rm HSE}(r)
$$

and the lensing data without introducing a separate cluster-scale \(a_0\), \(\rho_c\), or anisotropy strength.

A cluster-specific gas-equilibrium nuisance is legitimate. A cluster-specific gravity law is not.

---

# 13. Test 7: merging clusters and spacetime memory

The Bullet Cluster and similar systems are the most direct challenge to a local baryon-following modification. In the Bullet Cluster, the dominant X-ray gas and the lensing distribution are spatially offset. ([arXiv][9])

A static equation

$$
q=q(\rho_b)
$$

will usually make its strongest response follow the gas. The earlier analysis therefore proposed a dynamic memory field.

## Dynamic equation

Use

$$
\boxed{
\tau_q
\left(
\frac{\partial q}{\partial t}
+
\mathbf v_q\cdot\nabla q
\right)
=
D_q\nabla^2q
-
\frac{\lambda_q}{2}
q(1-q)(1-2q)
+
\alpha_TT_{ij}T^{ij}
-
\beta_\rho\rho_bq.
}
$$

The double-well term comes from

$$
V(q)=\frac{\lambda_q}{4}q^2(1-q)^2.
$$

## Data required

For each merger:

$$
\Sigma_{\rm gas}(x,y),\quad
T_{\rm gas}(x,y),\quad
\Sigma_\star(x,y),\quad
v_{\rm shock},\quad
\text{merger axis},\quad
\text{time since passage},
$$

plus raw weak-lensing shapes and strong-lensing image positions.

## Calculation

1. Construct pre-collision baryonic models consistent with the observed galaxies and gas.
2. Run baryonic gas hydrodynamics and collisionless stellar dynamics.
3. Evolve \(q\) and \(\Psi\) simultaneously.
4. Project the present gas and stellar distributions.
5. Ray trace through the resulting metric.
6. Compare gas morphology, shock position, image positions, shear field, and lensing-peak locations.

## Summary statistics

Measure:

$$
\Delta_{\rm gas-lens}
=
|\mathbf x_{\rm lens}-\mathbf x_{\rm gas}|,
$$

$$
\Delta_{\star-\rm lens}
=
|\mathbf x_{\rm lens}-\mathbf x_\star|,
$$

image-position RMS, and full shear-field likelihood.

## Critical design rule

Fit

$$
\tau_q,\ D_q,\ \lambda_q,\ \alpha_T,\ \beta_\rho
$$

jointly across a sample of mergers at different collision stages.

A model that chooses a different memory time for every merger is not predictive. The strongest result would be a single memory timescale that predicts how lensing offsets decay with merger age.

---

# 14. Test 8: direct void-path redshift

This is the most direct test of the user’s central redshift claim.

## Data required

We need objects with:

1. measured spectroscopic redshift;
2. a distance estimate not obtained from the candidate redshift law;
3. a reconstructed three-dimensional foreground density field;
4. a quantified peculiar-velocity covariance.

The preferred primary distances are geometric masers, Cepheid and TRGB calibrators, surface-brightness fluctuations, carefully reprocessed Type Ia supernovae, and eventually standard sirens. Tully–Fisher distances should not be primary here because the same project is changing the relation between rotation and mass.

Pantheon+ provides supernova data, covariance products, calibrator information, and cosmological chains suitable for an initial distance-redshift analysis. ([GitHub][10])

## Sightline integrals

For each source at trial geometric distance \(D\), sample the reconstructed field in small steps and calculate:

$$
\boxed{
I_q(D)
=
\int_0^Dq[\mathbf x(\ell)]\,d\ell,
}
$$

$$
\boxed{
I_g(D)
=
\frac1{c^2}
\int_0^D
|\nabla\Psi[\mathbf x(\ell)]|\,d\ell,
}
$$

$$
\boxed{
I_T(D)
=
\int_0^D
q[\mathbf x(\ell)]
P_2[
\hat{\mathbf k}\cdot\hat{\mathbf e}(\mathbf x)
]\,d\ell.
}
$$

Numerically,

$$
I_q
\approx
\sum_j
\frac{q_j+q_{j+1}}2
\Delta\ell_j.
$$

## Model predictions

For Z1:

$$
z_{\rm pred}
=
\exp\left[
\frac{H_*}{c}I_q+
\zeta(q_o-q_e)
\right]-1.
$$

For Z2:

$$
z_{\rm pred}
=
e^{\xi I_g}-1.
$$

For Z3:

$$
z_{\rm pred}
=
\exp\left[
\frac{H_*}{c}
(I_q+\eta I_T)
\right]-1.
$$

Include peculiar velocities multiplicatively:

$$
\boxed{
1+z_{\rm obs}
=
(1+z_{\rm pred})
(1+z_{\rm pec}).
}
$$

At low velocity,

$$
z_{\rm pec}\simeq
\frac{\mathbf v_{\rm pec}\cdot\hat{\mathbf k}}c.
$$

## Direct regression test

At fixed independent distance, test whether redshift residuals correlate with \(I_q\):

$$
\Delta\ln(1+z)
=
\ln(1+z_{\rm obs})
-
\ln(1+z_{\rm distance-only}).
$$

Fit

$$
\Delta\ln(1+z)
=
\lambda_q I_q+
\lambda_g I_g+
\lambda_T I_T+
\epsilon.
$$

Then hold out complete sightlines and sky regions.

The sharp prediction is:

> Two sources at the same geometric distance should have systematically different redshifts if one sightline crosses more void-polarized spacetime.

That is distinct from a homogeneous distance-only law.

---

# 15. Test 9: time dilation, brightness, and distance duality

A redshift mechanism must predict more than photon frequency.

Define:

$$
R_\nu=1+z_\nu
$$

as the photon-energy redshift,

$$
R_t
$$

as the observed stretching of time intervals, and

$$
P_\gamma
$$

as photon survival.

Let \(D_G\) be the geometric radius over which the emitted radiation is distributed. Then

$$
\boxed{
F
=
\frac{L\,P_\gamma}
{4\pi D_G^2R_\nu R_t}.
}
$$

Therefore,

$$
\boxed{
D_L
=
D_G
\sqrt{
\frac{R_\nu R_t}{P_\gamma}
}.
}
$$

## Energy-loss-only prediction

For a simple tired-light-like law,

$$
R_\nu=1+z,\qquad
R_t=1,\qquad
P_\gamma=1.
$$

Then

$$
D_L=D_G\sqrt{1+z}.
$$

It also predicts no \(1+z\) time dilation.

## Geometric clock-transport prediction

For a geometric redshift,

$$
R_\nu=R_t=1+z,
$$

so

$$
D_L=D_G(1+z)
$$

if photons are conserved.

But in a static Euclidean geometry,

$$
D_A=D_G,
$$

which would give

$$
\frac{D_L}
{(1+z)^2D_A}
=
\frac1{1+z}.
$$

Thus reproducing time dilation alone is not enough. The same geometry must alter ray-bundle angular distances so that the observed luminosity and angular-distance relations agree.

## Supernova time-dilation test

Fit raw light curves with

$$
\boxed{
\Delta t_{\rm obs}
=
\Delta t_{\rm em}(1+z)^b.
}
$$

An energy-loss-only theory predicts

$$
b=0.
$$

An expansion-like or geometric clock-rescaling mechanism predicts

$$
b=1.
$$

A large DES Type Ia sample found time stretching consistent with \(b=1\), with a reported best fit very close to unity, so simple frequency-loss mechanisms already face a stringent test. ([arXiv][11])

For a clean alternate-theory analysis, re-fit the observed photometry and spectra with \(b\) free rather than using a light-curve pipeline that has already divided observed times by \(1+z\).

## Distance-duality test

Calculate

$$
\boxed{
\eta_{\rm DD}(z)
=
\frac{D_L}
{(1+z)^2D_A}.
}
$$

Metric propagation with photon conservation predicts

$$
\eta_{\rm DD}=1.
$$

Photon–graviton conversion changes \(P_\gamma\) and therefore changes \(\eta_{\rm DD}\). A Weyl-like geometric transport law may also change the relation, but it must calculate the change explicitly.

---

# 16. Test 10: BAO

DESI’s current BAO releases provide radial and transverse distance constraints over a large redshift range, with the DR2 cosmology products based on the first three years of DESI observations. ([DESI Data][12])

The two key observables are approximately

$$
\frac{D_M(z)}{r_d}
$$

and

$$
\frac{D_H(z)}{r_d},
$$

where

$$
D_H(z)=\frac{dD_\parallel}{dz}.
$$

For Z1,

$$
\ln(1+z)
=
\frac{H_*}{c}I_q(D).
$$

Differentiate:

$$
\frac{1}{1+z}\frac{dz}{dD}
=
\frac{H_*}{c}
\frac{dI_q}{dD}.
$$

Since

$$
\frac{dI_q}{dD}=q_{\rm end}
$$

for a specified line of sight,

$$
\boxed{
\frac{dD}{dz}
=
\frac{c}
{H_*(1+z)q_{\rm end}}.
}
$$

The transverse distance must come from ray tracing through the same geometry. It cannot simply be set equal to the radial mapping unless the theory predicts that.

## Two-level BAO test

First, treat \(r_d\) as a nuisance scale. This tests only the late-time redshift and geometry relation.

Second, derive \(r_d\) from the candidate early-universe theory. That is required before claiming the model eliminates dark matter or conventional expansion.

For a radical non-FLRW geometry, compressed BAO likelihoods are only an initial test. The final analysis should remeasure the clustering feature from angles and redshifts under each candidate coordinate mapping.

---

# 17. Test 11: CMB spectrum and anisotropies

## Blackbody spectral test

COBE/FIRAS found the cosmic microwave background spectrum to be extraordinarily close to a blackbody, with residuals constrained at approximately the tens-of-parts-per-million level. ([arXiv][13])

For each photon path define a frequency scale factor

$$
s=1+z.
$$

Under ideal metric transport, the phase-space result is

$$
I_{\nu,o}(\nu)
=
\frac1{s^3}
I_{\nu,e}(s\nu).
$$

A single value of \(s\) maps a blackbody at \(T_e\) to a blackbody at

$$
T_o=\frac{T_e}{s}.
$$

But if different paths or frequencies acquire different \(s\), the observed spectrum becomes a mixture:

$$
\boxed{
I_{\nu,o}(\nu)
=
\int p(s)
\frac{P_\gamma(s,\nu)}{s^3}
I_{\nu,e}(s\nu)\,ds.
}
$$

Calculate this spectrum, subtract the best-fit blackbody, and compare the residual vector with the FIRAS covariance.

This strongly constrains:

* stochastic path-dependent redshift;
* frequency-dependent photon–graviton conversion;
* unequal redshift histories across the last-scattering surface.

## Linearized gravity response

For every candidate gravity equation, determine its cosmological linear response by applying a small density perturbation:

$$
\delta\rho
=
\bar\rho\delta_0
\cos(\mathbf k\cdot\mathbf x).
$$

Solve the equation at several sufficiently small \(\delta_0\) and extract the Fourier amplitudes \(\Psi_k,\Phi_k\).

Define

$$
\boxed{
\mu_{\rm eff}(k,a,\hat{\mathbf k})
=
-\frac{k^2\Psi_k}
{4\pi Ga^2\bar\rho_b\delta_k},
}
$$

and

$$
\boxed{
\eta_{\rm eff}(k,a,\hat{\mathbf k})
=
\frac{\Phi_k}{\Psi_k}.
}
$$

These functions can be placed into a modified cosmological perturbation solver to calculate:

$$
C_\ell^{TT},
\quad
C_\ell^{TE},
\quad
C_\ell^{EE},
\quad
C_\ell^{\phi\phi},
\quad
P(k),
\quad
f\sigma_8(z).
$$

Planck and ACT provide public temperature, polarization, and lensing likelihoods and data products for this comparison. ([arXiv][14])

## Three distinct cosmological ablations

Do not initially test “no dark matter and no expansion” as one inseparable proposition.

### C1: replace dark matter only

Set

$$
\Omega_c=0
$$

but retain an FLRW background and conventional dark energy. Ask whether the \(q,\mathbf K\) sector supplies the missing gravitational growth.

### C2: modify redshift or late-time geometry only

Retain conventional early-universe matter content, but replace or supplement the late-time redshift relation.

### C3: replace both dark matter and standard expansion

Set

$$
\Omega_c=0
$$

and derive the entire background clock and distance evolution from the new action.

Only C3 represents the full hypothesis. C1 and C2 identify which part fails if C3 does not work.

## Important theoretical requirement

The phenomenological galaxy equation alone is insufficient for CMB calculations. The \(q\) field carries energy, momentum, pressure, and perturbations. Those must be derived from a covariant action before a self-consistent early-universe prediction exists.

---

# 18. Test 12: does the force actually self-organize galaxies?

Fitting present-day gravity does not demonstrate that the same law forms galaxies.

## Controlled isolated simulations

Create identical rotating baryonic gas clouds under each gravity model.

Vary only predeclared initial conditions:

$$
M_b,\quad
R_0,\quad
\lambda_{\rm spin},\quad
T,\quad
\text{turbulence},\quad
\text{initial perturbation spectrum}.
$$

Run:

1. Newtonian baryons only;
2. scalar modified gravity;
3. local tensor void gravity;
4. nonlocal tensor void gravity;
5. dynamic \(q\)-field gravity.

Use the same hydrodynamics, cooling, star-formation, and feedback prescriptions.

Measure:

$$
t_{\rm collapse},
\quad
R_d,
\quad
h_z,
\quad
h_z/R_d,
\quad
B/T,
\quad
j_\star,
\quad
V_c(R),
\quad
Q_{\rm Toomre},
$$

plus bar strength, fragmentation rate, and angular-momentum loss.

The distinctive prediction should be more than “a disk forms.” Ordinary dissipative gas with angular momentum already forms disks. The new theory must predict a measurable difference such as:

* thinner disks at fixed angular momentum;
* a specific connection between disk thickness and outer rotation speed;
* preferred alignment with the tidal tensor;
* more rapid filament or sheet formation;
* a universal transition acceleration.

## Cosmological simulation

The final test requires baryons, radiation, neutrinos, and the \(q\)-gravity sector, but no cold dark matter in the strongest version.

The simulation must reproduce jointly:

$$
\text{galaxy abundance},
\quad
\text{stellar-mass function},
\quad
\text{disk/spheroid fraction},
\quad
\text{two-point clustering},
\quad
\text{void distribution},
\quad
\text{filament structure},
\quad
\text{cluster abundance}.
$$

Galaxy Zoo morphology and MaNGA kinematics can be used for large-sample morphology and internal-dynamics comparisons. ([Galaxy Zoo Data][15])

---

# 19. Test 13: Solar System, high-latitude motion, and strong fields

The model should be screened in high-acceleration environments by

$$
q\rightarrow0.
$$

That has to be demonstrated, not assumed.

## Planetary and spacecraft test

Integrate the Sun, planets, major moons, and selected spacecraft using the candidate equation.

The preliminary comparison can use JPL DE440 and SPICE trajectories, which are based on extensive ground- and spacecraft-tracking information. A rigorous test would eventually refit range and Doppler observations directly, since a conventional ephemeris may absorb some deviations into fitted initial conditions. ([JPL Solar System Dynamics][16])

For observation \(a\),

$$
r_a=
y_{a,\rm observed}
-
y_{a,\rm model}(\theta_G,\eta_{\rm orbit}).
$$

Fit initial states and standard nuisance parameters for each gravity model, then compare range and Doppler residuals.

Include spacecraft with strongly out-of-ecliptic trajectories because they test directional terms more directly than the nearly coplanar planets.

## Light propagation

Calculate Shapiro delay from the same metric used for galactic lensing. Cassini’s radio tracking provides a stringent Solar-System test of the relation between temporal and spatial gravitational potentials. ([PubMed][17])

## Wide binaries

Gaia supplies precise astrometry for a very large stellar sample. Wide binaries reach much lower mutual accelerations than planets and can test whether the transition depends on internal acceleration, external Galactic field, or orientation relative to the Galactic tidal tensor. ([Cosmos][18])

For each binary, predict the full vector relative acceleration:

$$
\ddot{\mathbf r}
=
-\nabla\Psi_1+
\nabla\Psi_2
$$

including the external \(q\) and tidal field. Compare projected separation and velocity distributions after forward-modeling orbital orientation and selection effects.

## Gravitational-wave propagation

Any photon–graviton unified model must reproduce the near equality of electromagnetic and gravitational-wave propagation speeds inferred from GW170817 and its electromagnetic counterpart. ([arXiv][19])

The strong-field completion must also be compared with binary-pulsar timing, including orbital decay and post-Keplerian parameters. ([ADS Abstracts][20])

---

# 20. Statistical architecture

## Global and nuisance parameters

Use one global parameter vector per candidate:

$$
\theta_{\rm global}
=
\{
a_0,\rho_c,m,n,L_\rho,L_q,
\alpha_0,\alpha_T,
\alpha_\parallel,\alpha_\perp,
\gamma_q,\ldots
\}.
$$

Object-specific parameters may describe measured uncertainty:

$$
\eta_i=
\{
D_i,i_i,\Upsilon_i,
\text{velocity anisotropy},
\text{gas pressure nuisance},
\text{source position}
\}.
$$

They may not redefine the gravity law.

## Broad initial priors

A usable exploratory range is:

$$
-13<
\log_{10}
\left(
\frac{a_0}{\mathrm{m\,s^{-2}}}
\right)
<-8,
$$

$$
-1<
\log_{10}
\left(
\frac{L_q}{\mathrm{kpc}}
\right)
<3,
$$

$$
0.25<m,n<8,
$$

$$
-5<\alpha_a<5.
$$

These are broad engineering priors, not physical measurements. Repeat the inference with wider bounds to check whether the posterior is prior-limited.

## Dataset likelihood

$$
\boxed{
\ln\mathcal L_{\rm total}
=
\sum_j
\ln\mathcal L_j.
}
$$

But do not immediately fit every dataset together. Freeze parameters as the model progresses:

$$
\text{galaxies}
\rightarrow
\text{vertical dynamics}
\rightarrow
\text{lensing}
\rightarrow
\text{clusters}
\rightarrow
\text{redshift}
\rightarrow
\text{cosmology}.
$$

A powerful theory should predict later datasets using parameters inferred from earlier ones.

## Benchmarks

Every candidate should be compared with:

$$
\text{Newtonian/GR baryons only},
$$

$$
\text{a scalar AQUAL-like model},
$$

and

$$
\text{GR plus a standard halo model}.
$$

The halo benchmark may have more object-specific parameters, so compare both fit quality and complexity.

Report:

$$
\text{held-out log score},
\quad
\text{posterior predictive residuals},
\quad
\text{WAIC or LOO},
\quad
\text{Bayesian evidence where feasible},
\quad
\text{BIC as a secondary summary}.
$$

Raw training \(\chi^2\) is not sufficient.

---

# 21. Sources of circularity to avoid

| Test                  | Potential circularity                                        | Correct treatment                                                |
| --------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| Galaxy rotation       | Rotation curve extracted assuming a disk model               | Refit resolved velocity fields or cubes for the final test       |
| Stellar mass          | Mass-to-light ratio inferred using a preferred gravity model | Use stellar-population priors and marginalize                    |
| Void maps             | Distances inferred from redshift using FLRW                  | Reconstruct iteratively under each candidate distance law        |
| Lensing maps          | Published convergence assumes conventional lens equation     | Fit raw shapes and multiple-image positions                      |
| Cluster mass          | Published hydrostatic mass assumes GR                        | Use density, temperature, and pressure profiles directly         |
| Supernova distance    | Light-curve pipeline prescales time by \(1+z\)               | Refit raw photometry with time-dilation exponent free            |
| BAO                   | Compressed likelihood assumes a fiducial mapping             | Use compressed data for screening, raw clustering for final test |
| CMB                   | Insert \(\mu(k,a)\) without background stress-energy         | Derive the \(q\)-field stress tensor from an action              |
| Tully–Fisher distance | Uses the very relation being tested                          | Exclude from the primary redshift-distance sample                |

---

# 22. Explicit first executable run

The first complete run should be narrow enough to finish and strong enough to eliminate most equations.

## Run A: one-dimensional galaxy screening

1. Ingest SPARC galaxy and mass-model tables.
2. Apply quality cuts that are defined before examining model residuals.
3. Freeze a stratified 60/20/20 galaxy split.
4. Sample \(D,i,\Upsilon_d,\Upsilon_b\) as nuisances.
5. Run D1–D4 and scalar versions of \(Q_1\)–\(Q_4\).
6. Calculate rotation-curve likelihood, RAR residuals, BTFR slope/scatter, and inferred \(h_{\rm eff}\).
7. Reject candidates that fail blind galaxies or require nonuniversal transition scales.

## Run B: axisymmetric PDE tournament

Run the 24 \(Q\times K\times\mu\) candidates on a representative subset spanning:

$$
\text{dwarf},
\quad
\text{gas dominated},
\quad
\text{low surface brightness},
\quad
\text{massive spiral},
\quad
\text{high surface brightness}.
$$

Use a finite-volume solver and verify exact flux conservation.

Promote only models that:

1. converge numerically;
2. recover Newtonian gravity at high acceleration;
3. produce an outer effect without object-specific parameters;
4. predict held-out curves;
5. generate the required \(h_{\rm eff}\)–mass scaling.

## Run C: resolved galaxies

Use resolved H I maps and velocity fields for approximately ten high-quality systems. Construct full baryonic maps and compare predicted and observed two-dimensional velocities.

This tests noncircular and directional signatures that a one-dimensional rotation curve erases.

## Run D: radial-plus-vertical prediction

Freeze the top three models. Use DiskMass or comparable data to predict \(\sigma_z(R)\) and disk thickness.

Promote models only if one parameter set predicts both radial and vertical dynamics.

## Run E: dynamics plus lensing

Test:

$$
3\ \text{gravity models}
\times
3\ \text{lensing closures}
=
9\ \text{combinations}.
$$

Use SLACS first for compact joint dynamics/lensing constraints, followed by galaxy–galaxy weak lensing.

## Run F: clusters

Apply the surviving models without changing gravity parameters to X-COP gas profiles and cluster lensing.

## Run G: merger memory

Activate the dynamic \(q\) equation only if the static model fails the merger offsets. Fit one set of memory parameters across several mergers.

## Run H: redshift

Freeze all gravity and lensing parameters. Compare Z0–Z5 using:

$$
I_q,\quad I_g,\quad I_T,
$$

independent distances, supernova brightness, time dilation, BAO, and distance duality.

## Run I: CMB and formation

Only models surviving the preceding stages are promoted to a covariant action, perturbation solver, and structure-formation simulation.

---

# 23. Solver design

A finite-volume method is preferable because the theory is written as a conserved flux equation.

For cell \(c\),

$$
\boxed{
\sum_{f\in c}
A_f
\left[
\mu\mathbf K\nabla\Psi
\right]_f
\cdot\hat{\mathbf n}_f
=
4\pi G\rho_{b,c}V_c.
}
$$

A nonlinear iteration is:

```text
1. Solve ordinary Poisson gravity for ΦN.
2. Compute ρL, gN, qlocal, and the tidal tensor.
3. Solve for nonlocal q when required.
4. Build the positive-definite tensor K.
5. Initialize Ψ = ΦN.
6. Compute X and μ(X).
7. Solve div[μ K grad Ψ] = 4πGρb.
8. Update μ and repeat until converged.
9. Check flux conservation, tensor eigenvalues, curl, and domain convergence.
10. Project Ψ into the requested observable.
```

A practical repository layout is:

```text
gravitylab/
├── configs/
│   ├── models/
│   ├── datasets/
│   └── splits/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── models/
│   ├── baryons.py
│   ├── qfield.py
│   ├── tensor_rules.py
│   ├── mu_functions.py
│   ├── near_flip.py
│   ├── metric.py
│   ├── redshift.py
│   └── memory.py
├── solvers/
│   ├── poisson.py
│   ├── axisymmetric_fv.py
│   ├── tensor_fv_3d.py
│   ├── nonlinear.py
│   ├── geodesics.py
│   └── raytrace.py
├── observables/
│   ├── rotation.py
│   ├── vertical_jeans.py
│   ├── weak_lensing.py
│   ├── strong_lensing.py
│   ├── hydrostatic.py
│   ├── supernova.py
│   ├── bao.py
│   └── cmb_response.py
├── inference/
│   ├── priors.py
│   ├── likelihoods.py
│   ├── sampling.py
│   └── cross_validation.py
├── tests/
│   ├── test_flux.py
│   ├── test_constant_tensor.py
│   ├── test_newtonian_limit.py
│   ├── test_grid_convergence.py
│   └── test_raytrace.py
└── outputs/
    ├── posteriors/
    ├── predictions/
    ├── residuals/
    └── model_cards/
```

Each model card should record:

```text
model_id
q_source
tensor_rule
mu_rule
lensing_closure
redshift_closure
global_parameters
nuisance_parameters
training_datasets
blind_datasets
numerical_test_results
held_out_scores
known failures
status
```

---

# 24. Clear survival and rejection rules

## Immediate mathematical rejection

Reject a model if it:

* loses positive definiteness of \(\mathbf K\);
* has no well-defined boundary condition;
* violates flux conservation after convergence;
* fails to recover Newtonian gravity in screened regions;
* produces a metric with unintended signature change;
* has unstable or nonunique static solutions;
* requires a path-dependent force not derivable from an action.

## Galaxy-level rejection

Reject it if:

* it fits training galaxies but fails held-out galaxies;
* it needs galaxy-specific gravity parameters;
* it cannot reproduce the BTFR scaling without explicitly assigning \(h_{\rm eff}\propto\sqrt M\);
* its residuals strongly correlate with surface brightness, gas fraction, or radius;
* it fits radial motion but predicts incorrect vertical forces.

## Relativistic rejection

Reject it if:

* the lensing potential needed by the data is incompatible with the dynamical potential;
* the required lensing slip fails Solar-System light-propagation constraints;
* it fits galaxies but fails clusters with the same parameters;
* cluster-merger offsets require arbitrary system-specific memory.

## Redshift rejection

Reject it if:

* it shifts photon frequency but does not reproduce time dilation;
* it fits luminosity distance but not angular distance;
* it produces frequency-dependent spectral distortions beyond FIRAS limits;
* it fits radial BAO but not transverse BAO;
* its line-of-sight prediction disappears when tested on independent distances;
* it is merely a change of notation for an FLRW scale factor while claiming a distinct observable theory.

## Full-theory rejection

Reject it if it cannot, with one covariant action and one parameter set, produce acceptable:

$$
\text{galaxy dynamics}
+
\text{lensing}
+
\text{clusters}
+
\text{redshift}
+
\text{CMB}
+
\text{structure growth}.
$$

---

# The most important first scientific question

The initial test should not be “Can this explain all of cosmology?”

It should be:

$$
\boxed{
\begin{aligned}
&\text{Given baryonic mass maps alone, does a universal}\\
&\text{void-dependent tensor equation predict both the}\\
&\text{radial and vertical gravitational fields of galaxies?}
\end{aligned}
}
$$

Specifically, start from

$$
\boxed{
\nabla_i
\left[
\mu(X)
\left(
e^{-\alpha_\parallel q}
P_\parallel^{ij}
+
e^{-\alpha_\perp q}
P_\perp^{ij}
\right)
\nabla_j\Psi
\right]
=
4\pi G\rho_b,
}
$$

with

$$
\boxed{
(1-L_q^2\nabla^2)q
=
\left[
1+
\left(\frac{\rho_L}{\rho_c}\right)^m+
\left(\frac{|\mathbf g_N|}{a_0}\right)^n
\right]^{-1}.
}
$$

Then ask whether the solution produces, rather than assumes:

$$
g_R\rightarrow\frac{\sqrt{GM_ba_0}}R,
$$

$$
v_f^4\rightarrow GM_ba_0,
$$

$$
h_{\rm eff}\rightarrow
\sqrt{\frac{GM_b}{a_0}},
$$

and a vertical field consistent with observed disk thicknesses.

If it does, the next decisive question is whether the same \(\Psi\), with a tightly constrained metric closure, predicts galaxy and cluster lensing. Only after those two results should the cumulative-redshift sector be attached. That sequencing gives the idea the best chance to reveal a real mathematical regularity without allowing redshift, lensing, and memory parameters to compensate for a gravity equation that does not actually work.

[1]: https://astroweb.case.edu/SPARC/ "https://astroweb.case.edu/SPARC/"
[2]: https://ui.adsabs.harvard.edu/abs/2008AJ....136.2563W/abstract "https://ui.adsabs.harvard.edu/abs/2008AJ....136.2563W/abstract"
[3]: https://arxiv.org/abs/1307.8130?utm_source=chatgpt.com "The DiskMass Survey. VI. Gas and stellar kinematics in ..."
[4]: https://data.desi.lbl.gov/doc/releases/dr1/vac/desivast/ "https://data.desi.lbl.gov/doc/releases/dr1/vac/desivast/"
[5]: https://arxiv.org/abs/2601.14559?utm_source=chatgpt.com "[2601.14559] Dark Energy Survey Year 6 Results"
[6]: https://arxiv.org/html/0805.1931v1 "https://arxiv.org/html/0805.1931v1"
[7]: https://dominiqueeckert.wixsite.com/xcop/data "https://dominiqueeckert.wixsite.com/xcop/data"
[8]: https://archive.stsci.edu/prepds/frontier/lensmodels/?utm_source=chatgpt.com "Frontier Fields Lens Models"
[9]: https://arxiv.org/abs/astro-ph/0608407?utm_source=chatgpt.com "A direct empirical proof of the existence of dark matter"
[10]: https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/README.md "https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/README.md"
[11]: https://arxiv.org/abs/2406.05050 "https://arxiv.org/abs/2406.05050"
[12]: https://data.desi.lbl.gov/doc/papers/dr2/ "https://data.desi.lbl.gov/doc/papers/dr2/"
[13]: https://arxiv.org/abs/astro-ph/9605054 "https://arxiv.org/abs/astro-ph/9605054"
[14]: https://arxiv.org/abs/1807.06209 "https://arxiv.org/abs/1807.06209"
[15]: https://data.galaxyzoo.org/ "https://data.galaxyzoo.org/"
[16]: https://ssd.jpl.nasa.gov/doc/de440_de441.html?utm_source=chatgpt.com "The JPL Planetary and Lunar Ephemerides DE440 and DE441"
[17]: https://pubmed.ncbi.nlm.nih.gov/14508481/?utm_source=chatgpt.com "A test of general relativity using radio links with the Cassini ..."
[18]: https://www.cosmos.esa.int/web/gaia/dr3?utm_source=chatgpt.com "Gaia Data Release 3 (Gaia DR3) - ESA Cosmos"
[19]: https://arxiv.org/abs/1711.04137 "https://arxiv.org/abs/1711.04137"
[20]: https://ui.adsabs.harvard.edu/abs/2021PhRvX..11d1050K/abstract "https://ui.adsabs.harvard.edu/abs/2021PhRvX..11d1050K/abstract"
