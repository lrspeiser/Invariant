# Complete tested-formula scorecard

This inventory contains **128 scientifically distinct formula/test rows**. It consolidates repeated controls but retains force-law, screen, exponent, path, tensor, metric, and reconstruction variations when they change the tested hypothesis.

## How to read the percentages

The percentages are normalized proximity scores created for this audit; the original scientific scores remain beside them. They are **not** probabilities that a theory is true, confidence levels, or interchangeable likelihoods.

- Galaxy velocity: `100 x max(0, 1 - RMSE / RMS(observed speed))`.
- Observed or derived log acceleration: `100 x 10^(-RMSE_dex)`. A 0.301-dex error is therefore 50% proximity (a typical factor-of-two miss).
- Raw lensing: `100 x max(0, 1 - image-position RMS / RMS(observed image radius))`.
- Raw image positions and GR/NFW-derived acceleration products are kept separate. NFW's 100% derived score is circular by construction, not a prediction.
- A blank means that observable was not tested. Incomplete image roots are not assigned a percentage.

## Highest descriptive proximity scores

Galaxy tests are not all the same split, and raw-lens rows can use different clusters; these lists are navigation aids, not a leaderboard.

### Galaxy

| Formula | Proximity | Original error | Test |
|---|---:|---:|---|
| Fixed RAR | 93.79% | 10.348 km/s | 131 SPARC galaxies, 968 untouched outer points |
| Fixed RAR with scalar metric slip s=5 | 93.79% | 10.348 km/s | 131 SPARC galaxies, 968 untouched outer points |
| Member tidal-contrast metric | 93.79% | 10.348 km/s | inherits locked fixed-RAR matter law |
| Full member tidal metric (new test) | 93.79% | 10.348 km/s | inherits locked fixed-RAR matter law |
| Simple MOND | 93.76% | 10.385 km/s | 131 SPARC galaxies, 968 untouched outer points |
| RAR + squared coherence-gated RG (current empirical bridge) | 93.64% | 10.586 km/s | 131 SPARC galaxies, 968 untouched outer points |
| Causal catch-up Sigma completion | 93.64% | 10.586 km/s | static limit; 131 SPARC galaxies, 968 outer points |
| Curvature power p=2 | 91.35% | 14.403 km/s | 131 SPARC galaxies, 968 untouched outer points |

### Raw lensing

| Formula | Proximity | Original error | Test |
|---|---:|---:|---|
| RAR + squared coherence-gated RG (current empirical bridge) | 92.13% | 1.064 arcsec | RXJ2129: 7 held-out images |
| curvature log | 92.05% | 1.075 arcsec | RXJ2129: 7 held-out images |
| curvature loglog | 91.65% | 1.128 arcsec | RXJ2129: 7 held-out images |
| tensor competition | 91.55% | 1.142 arcsec | RXJ2129: 7 held-out images |
| tensor dominance | 91.45% | 1.155 arcsec | RXJ2129: 7 held-out images |
| tensor alignment | 90.99% | 1.218 arcsec | RXJ2129: 7 held-out images |
| isotropic completion | 90.47% | 1.288 arcsec | RXJ2129: 7 held-out images |
| coherence completion | 90.22% | 1.321 arcsec | RXJ2129: 7 held-out images |

## Full formula table

| Family | Formula | Schematic equation | Galaxy proximity (error) | Derived-lens proximity (error) | Raw-lens proximity (error) | Verdict |
|---|---|---|---:|---:|---:|---|
| Controls and bridge | Newtonian | `g = g_bar` | 65.00% (60.721 km/s) | 13.07% (0.884 dex) | 7.28% (25.199 arcsec) | baseline fails galaxies and clusters |
| Controls and bridge | Fixed RAR | `g/g_bar = [1-exp(-sqrt(g_bar/a0))]^-1` | 93.79% (10.348 km/s) | 30.00% (0.523 dex) | 5.53% (25.673 arcsec) | excellent galaxy control; fails raw cluster lensing |
| Controls and bridge | Simple MOND | `g/g_bar = (1+sqrt(1+4 a0/g_bar))/2` | 93.76% (10.385 km/s) | 30.31% (0.518 dex) | 5.67% (25.636 arcsec) | excellent galaxies; fails cluster lensing |
| Controls and bridge | Cluster-retuned RAR | `RAR law with a0 retuned to cluster data` | 73.77% (0.132 dex) | 77.94% (0.108 dex) | 79.16% (2.816 arcsec) | cluster-only retuning; not universal |
| Controls and bridge | Density-only refracted gravity | `g = g_bar / epsilon(rho_b)` | 80.64% (0.093 dex) | 69.88% (0.156 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | RG with acceleration-moving density threshold | `g = g_bar / epsilon[rho_b; rho_c(g_bar)]` | 82.24% (0.085 dex) | 76.57% (0.116 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | RG with potential-moving density threshold | `g = g_bar / epsilon[rho_b; rho_c(Phi_bar)]` | 81.86% (0.087 dex) | 77.42% (0.111 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | RG with acceleration-dependent floor | `g = g_bar / epsilon[rho_b; epsilon0(g_bar)]` | 81.83% (0.087 dex) | 78.77% (0.104 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | RG + Sigma additive | `E = E_RG + B E_Sigma` | 81.53% (0.089 dex) | 79.03% (0.102 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | RG + Sigma quadrature | `E = sqrt(E_RG^2 + (B E_Sigma)^2)` | 80.87% (0.092 dex) | 78.86% (0.103 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | RG x Sigma product | `E = E_RG (1 + B E_Sigma)` | 79.91% (0.097 dex) | 78.26% (0.106 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | Density-gated Sigma/RG | `E = E_RG + B S(rho_b) E_Sigma` | 80.58% (0.094 dex) | 79.46% (0.100 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | Linear g-rho surface | `log g = b0 + b1 log g_bar + b2 log rho_b` | 81.89% (0.087 dex) | 79.19% (0.101 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | Quadratic g-rho surface | `log g = quadratic(log g_bar, log rho_b)` | 82.14% (0.085 dex) | 79.32% (0.101 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | Quadratic g-rho-potential surface | `log g = quadratic(log g_bar, log rho_b, Phi_bar/c^2)` | 81.21% (0.090 dex) | 78.36% (0.106 dex) | — | diagnostic only; no independent SPARC transfer |
| Controls and bridge | RAR + squared coherence-gated RG (current empirical bridge) | `E = E_RAR + [1-(3C^2-2C^3)]^2 (epsilon(rho_b)^-1 - 1)` | 93.64% (10.586 km/s) | 72.66% (0.139 dex) | 92.13% (1.064 arcsec) | best empirical bridge; one-cluster raw success, not yet multi-cluster validated |
| Dark-matter controls | Per-galaxy NFW halo | `rho(r)=rho_s/[x(1+x)^2], x=r/r_s` | 89.31% (17.804 km/s) | — | — | dark-matter galaxy control; flexible per galaxy and weak outer extrapolation here |
| Dark-matter controls | Compact cluster halo (RXJ2129) | `alpha = alpha_baryons + alpha_compact_halo` | — | — | 81.24% (2.536 arcsec) | limited one-halo dark-matter comparator |
| Dark-matter controls | Compact cluster halo (four-cluster transfer) | `alpha = alpha_baryons + alpha_compact_halo` | — | — | 64.28% (9.048 arcsec) | best four-cluster raw comparator, though still inadequate in two systems |
| Dark-matter controls | CLASH NFW construction | `g_obs is deprojected from the same fitted NFW lensing profile` | — | 100.00% (0.000 dex) | — | 100% by construction; not an independent prediction |
| One-parameter cluster laws | Baryon-normalized isothermal tail (lambda=9) | `g=g_bar+9 g_bar(200 kpc)(200 kpc/r)` | — | — | 65.33% (9.423 arcsec) | one shared parameter improves both replay-holdout clusters and narrowly beats the compact-halo equal-cluster aggregate, but fails the 2-arcsec gate and the RXJ2129 stress comparison |
| One-parameter cluster laws | Solar-screened baryon-normalized isothermal tail | `g=g_bar+lambda g_bar(200 kpc)(200 kpc/r) a0/(a0+g_bar); lambda=10.5` | 88.83% (18.602 km/s) | — | 80.64% (5.261 arcsec) | one shared parameter passes the published Mercury-margin diagnostic and beats the limited compact-halo replay aggregate, but fails the frozen galaxy transfer (18.60 versus 10.35 km/s for RAR), the earlier 2-arcsec target, and the RXJ2129 halo comparison |
| Unified acceleration laws | joint a0 | `RAR with one a0 fit jointly to galaxies and clusters` | 86.62% (23.213 km/s) | 31.66% (0.499 dex) | — | failed joint universal gates |
| Unified acceleration laws | U0 emond like | `a_eff=a0 exp[ln(F) S(Phi_bar/c^2)]; insert a_eff in RAR` | 86.56% (23.318 km/s) | 67.78% (0.169 dex) | — | failed joint universal gates |
| Unified acceleration laws | U1 coherence length | `U0 multiplied by a coherence/length gate` | 84.78% (26.397 km/s) | 57.97% (0.237 dex) | — | failed joint universal gates |
| Unified acceleration laws | domain oracle | `fixed galaxy RAR for galaxies; cluster RAR for clusters` | 86.69% (23.085 km/s) | 77.38% (0.111 dex) | — | not universal |
| Void and environment | Free-p low-acceleration law | `Delta g/g_bar = A S(g_bar) (g_bar/a_t)^(-p)` | 85.23% (25.622 km/s) | — | — | beats Newtonian but loses to RAR |
| Void and environment | Fixed p=1/2 low-acceleration law | `Delta g/g_bar = A S(g_bar) sqrt(a_t/g_bar)` | 86.69% (23.085 km/s) | — | — | radial holdout strong; strict galaxy CV no better than RAR |
| Void and environment | CF4 grouped-64 environment law | `Delta g -> Delta g (E_CF4/E0)^beta` | 85.12% (25.822 km/s) | — | — | environment worsens held-out prediction |
| Void and environment | CF4 ungrouped-64 environment law | `Delta g -> Delta g (E_CF4/E0)^beta` | 84.76% (26.437 km/s) | — | — | environment worsens held-out prediction |
| Void and environment | CF4 ungrouped-128 environment law | `Delta g -> Delta g (E_CF4/E0)^beta` | 85.43% (25.284 km/s) | — | — | sign is unstable; no robust detection |
| Void and environment | direct harmonic blind | `Delta v^2 = kappa r^2 (no measured environment)` | 73.84% (45.383 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | direct harmonic primary | `Delta v^2 = kappa(E_CF4) r^2` | 72.93% (46.966 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened radial blind | `Delta v^2 = V0^2 r^2/[r^2+(c_R R_d)^2]` | 75.38% (42.707 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened primary | `Delta v^2 = V0^2 E_CF4^m r^2/[r^2+(c_R R_d)^2]` | 75.32% (42.819 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened primary shuffled | `same screened law with shuffled E_CF4 control` | 74.83% (43.673 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened grouped 64 power p3 | `screened law; E from inverse-cube exterior force` | 75.38% (42.707 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened grouped 64 yukawa l31p25 | `screened law; E from Yukawa lambda=31.25 Mpc/h` | 74.29% (44.612 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened grouped 64 yukawa l62p5 | `screened law; E from Yukawa lambda=62.5 Mpc/h` | 75.09% (43.220 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened grouped 64 yukawa l7p8125 | `screened law; E from Yukawa lambda=7.8125 Mpc/h` | 75.38% (42.707 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened ungrouped 128 yukawa l15p625 | `screened law; E from ungrouped-128 Yukawa map` | 75.38% (42.707 km/s) | — | — | failed frozen void-cage gates |
| Void and environment | screened ungrouped 64 yukawa l15p625 | `screened law; E from ungrouped-64 Yukawa map` | 75.26% (42.919 km/s) | — | — | failed frozen void-cage gates |
| Galaxy-scaled void laws | catalog mass concentration internal | `Delta v^2=V0^2 M^eta r^2/[r^2+(c_R R_d C^gamma)^2]` | 86.53% (23.363 km/s) | — | — | mass scaling is competitive but void exponent is zero |
| Galaxy-scaled void laws | catalog mass concentration void | `mass/concentration law x E_CF4^m` | 86.53% (23.363 km/s) | — | — | mass scaling is competitive but void exponent is zero |
| Galaxy-scaled void laws | catalog mass surface internal | `Delta v^2=V0^2 M^eta r^2/[r^2+(c_R R_d Sigma^beta)^2]` | 86.63% (23.189 km/s) | — | — | mass scaling is competitive but void exponent is zero |
| Galaxy-scaled void laws | catalog mass surface void | `mass/surface law x E_CF4^m` | 86.63% (23.189 km/s) | — | — | mass scaling is competitive but void exponent is zero |
| Galaxy-scaled void laws | catalog mass surface void shuffled | `mass/surface law with shuffled E_CF4` | 86.62% (23.210 km/s) | — | — | mass scaling is competitive but void exponent is zero |
| Galaxy-scaled void laws | legacy size only | `Delta v^2=V0^2 r^2/[r^2+(c_R R_d)^2]` | 75.38% (42.707 km/s) | — | — | failed |
| Galaxy-scaled void laws | local acceleration internal | `screened law with S(g_bar)=[1+(g_bar/g*)^n]^-1` | 72.00% (48.571 km/s) | — | — | failed |
| Galaxy-scaled void laws | local acceleration void | `local-acceleration law x E_CF4^m` | 72.00% (48.571 km/s) | — | — | failed |
| Galaxy-scaled void laws | local acceleration void shuffled | `local-acceleration law with shuffled E_CF4` | 71.94% (48.678 km/s) | — | — | failed |
| Galaxy-scaled void laws | local acceleration void ungrouped 128 | `local-acceleration law x ungrouped-128 E_CF4^m` | 72.00% (48.571 km/s) | — | — | failed |
| Galaxy-scaled void laws | local acceleration void ungrouped 64 | `local-acceleration law x ungrouped-64 E_CF4^m` | 71.93% (48.703 km/s) | — | — | failed |
| Galaxy-scaled void laws | mass amplitude only | `Delta v^2=V0^2 M^eta r^2/[r^2+(c_R R_d)^2]` | 86.67% (23.126 km/s) | — | — | retained empirical mass-amplitude control |
| Galaxy-scaled void laws | mass transition | `same, with r_t=c_R R_d M^alpha` | 86.63% (23.187 km/s) | — | — | transition dependence not supported |
| Galaxy-scaled void laws | surface transition | `same, with r_t=c_R R_d Sigma^beta` | 86.63% (23.189 km/s) | — | — | transition dependence not supported |
| Galaxy-scaled void laws | concentration transition | `same, with r_t=c_R R_d C^gamma` | 86.53% (23.363 km/s) | — | — | transition dependence not supported |
| Unbounded running | curvature log | `E=1+alpha[ln(1+T*/T)]^p` | 89.17% (18.036 km/s) | 70.82% (0.150 dex) | 92.05% (1.075 arcsec) | failed universal gates |
| Unbounded running | curvature loglog | `E=1+alpha{ln[1+ln(1+T*/T)]}^p` | 87.21% (21.292 km/s) | 71.32% (0.147 dex) | 91.65% (1.128 arcsec) | failed universal gates |
| Unbounded running | curvature rootlog | `E=1+alpha sqrt[ln(1+T*/T)]^p` | 88.98% (18.352 km/s) | 70.81% (0.150 dex) | — | failed universal gates |
| Unbounded running | curvature power | `E=[1+(T*/T)^p]^epsilon` | 91.24% (14.579 km/s) | 67.92% (0.168 dex) | 87.86% (1.641 arcsec) | failed universal gates |
| Unbounded running | path log running | `E=1+alpha ln[1+(ell/L)^p]` | 84.77% (25.356 km/s) | 70.49% (0.152 dex) | — | failed universal gates |
| Unbounded running | path power running | `E=[1+(ell/L)^p]^epsilon` | 87.82% (20.276 km/s) | 67.91% (0.168 dex) | — | failed universal gates |
| Unbounded running | tensor alignment log | `log running x tidal-eigenvector alignment` | 83.47% (27.527 km/s) | 70.42% (0.152 dex) | — | failed universal gates |
| Unbounded running | tensor dominance log | `log running x tidal-eigenvalue dominance` | 83.68% (27.176 km/s) | 70.23% (0.154 dex) | — | failed universal gates |
| Unbounded running | tensor alignment power | `power running x tidal-eigenvector alignment` | 85.98% (23.342 km/s) | 66.89% (0.175 dex) | — | failed universal gates |
| Unbounded running | tensor dominance power | `power running x tidal-eigenvalue dominance` | 86.11% (23.126 km/s) | 66.54% (0.177 dex) | — | failed universal gates |
| Variable exponent | curvature variable mass power | `E=[1+(T*/T)^p(M_eq)]^epsilon` | 63.04% (61.544 km/s) | 73.26% (0.135 dex) | 29.66% (9.507 arcsec) | failed universal gates |
| Variable exponent | curvature variable density power | `E=[1+(T*/T)^p(rho_b)]^epsilon` | 76.66% (38.865 km/s) | 70.97% (0.149 dex) | 35.11% (8.770 arcsec) | failed universal gates |
| Variable exponent | curvature variable shape power | `E=[1+(T*/T)^p(rho_b/rho_mean)]^epsilon` | 90.84% (15.260 km/s) | 67.65% (0.170 dex) | — | failed universal gates; raw roots incomplete |
| Bounded tensor completion | tensor isotropic | `C_ij=C_solar delta_ij+(1-C_solar) A_T delta_ij` | 82.63% (28.918 km/s) | 71.64% (0.145 dex) | 88.64% (1.536 arcsec) | failed universal gates |
| Bounded tensor completion | tensor alignment | `C_ij=C_solar delta_ij+(1-C_solar) A_T P_align,ij` | 81.52% (30.779 km/s) | 70.71% (0.151 dex) | 90.99% (1.218 arcsec) | failed universal gates |
| Bounded tensor completion | tensor competition | `C_ij=C_solar delta_ij+(1-C_solar) A_T P_compete,ij` | 81.48% (30.844 km/s) | 70.94% (0.149 dex) | 91.55% (1.142 arcsec) | failed universal gates |
| Bounded tensor completion | tensor dominance | `C_ij=C_solar delta_ij+(1-C_solar) A_T P_dom,ij` | 81.77% (30.353 km/s) | 70.47% (0.152 dex) | 91.45% (1.155 arcsec) | failed universal gates |
| Bounded vector completion | isotropic completion | `C=C_solar+(1-C_solar)A_T` | 84.25% (26.228 km/s) | 71.39% (0.146 dex) | 90.47% (1.288 arcsec) | failed universal gates |
| Bounded vector completion | coherence completion | `C=C_solar+(1-C_solar)A_T(1-Coh)^q` | 80.83% (31.915 km/s) | 71.37% (0.147 dex) | 90.22% (1.321 arcsec) | failed universal gates |
| Bounded path completion | distance path | `logit C=logit C_solar+integral dr/ell` | 76.94% (38.398 km/s) | 63.17% (0.199 dex) | 87.09% (1.745 arcsec) | failed universal gates |
| Bounded path completion | tidal path | `logit C=logit C_solar+integral A_T dr/ell` | 74.72% (42.092 km/s) | 67.88% (0.168 dex) | 86.86% (1.776 arcsec) | failed universal gates |
| Bounded path completion | matter path | `logit C=logit C_solar+integral A_rho dr/ell` | 79.06% (34.864 km/s) | 69.80% (0.156 dex) | 84.51% (2.094 arcsec) | failed universal gates |
| Bounded path completion | hybrid path | `logit C=logit C_solar+integral A_T A_rho dr/ell` | 79.23% (34.589 km/s) | 66.84% (0.175 dex) | 86.08% (1.881 arcsec) | failed universal gates |
| Mass-conditioned path completion | mass weighted path | `d tau/dr=[1+(M*/M_history)^q]^-1/ell` | 81.53% (30.763 km/s) | 63.26% (0.199 dex) | 84.58% (2.084 arcsec) | failed universal gates |
| Mass-conditioned path completion | mass amplified path | `d tau/dr=(M_history/M*)^q/ell` | 80.62% (32.268 km/s) | 60.69% (0.217 dex) | 87.01% (1.755 arcsec) | failed universal gates |
| Mass-conditioned path completion | mass ceiling path | `distance recovery capped by [1+(M*/M_history)^q]^-1` | 85.18% (24.671 km/s) | 63.26% (0.199 dex) | 83.98% (2.165 arcsec) | failed universal gates |
| Refined scalar and spatial lens | Curvature power p=2 | `E=[1+(T*/T)^2]^epsilon` | 91.35% (14.403 km/s) | 68.00% (0.168 dex) | 26.46% (18.630 arcsec) | balanced phenomenology but fails four-cluster raw lensing |
| Refined scalar and spatial lens | Curvature additive alpha=10 | `E=1+10 ln[1+(T*/T)^p]` | 89.93% (16.765 km/s) | 70.37% (0.153 dex) | 28.29% (18.165 arcsec) | balanced phenomenology but fails four-cluster raw lensing |
| Refined scalar and spatial lens | Curvature p=2 + member vector (GR-linear) | `alpha=alpha_spherical+f Delta alpha_members` | 91.35% (14.403 km/s) | — | 26.23% (18.688 arcsec) | member-light directions do not improve transfer |
| Refined scalar and spatial lens | Curvature p=2 + member vector (running dressed) | `alpha=alpha_spherical+f E(r) Delta alpha_members` | 91.35% (14.403 km/s) | — | 26.09% (18.724 arcsec) | member-light directions do not improve transfer |
| Refined scalar and spatial lens | Additive alpha=10 + member vector (GR-linear) | `alpha=alpha_spherical+f Delta alpha_members` | 89.93% (16.765 km/s) | — | 27.59% (18.342 arcsec) | member-light directions do not improve transfer |
| Refined scalar and spatial lens | Additive alpha=10 + member vector (running dressed) | `alpha=alpha_spherical+f E(r) Delta alpha_members` | 89.93% (16.765 km/s) | — | 28.09% (18.216 arcsec) | member-light directions do not improve transfer |
| Metric lens closures | Fixed RAR with scalar metric slip s=5 | `g_lens=g_bar+(1+s/2)(g_dyn-g_bar)` | 93.79% (10.348 km/s) | — | 32.18% (18.432 arcsec) | scalar slip selected but fails halo-competitiveness gate |
| Metric lens closures | Member tidal-contrast metric | `partial_i[(delta_ij+t Qcontrast_ij) partial_j Phi]=source` | 93.79% (10.348 km/s) | — | 32.18% (18.432 arcsec) | selected t=0; retired |
| Metric lens closures | Full member tidal metric (new test) | `partial_i[(delta_ij+t Qfull_ij) partial_j Phi]=source` | 93.79% (10.348 km/s) | — | 32.17% (18.433 arcsec) | selected t=0; retired; negative t lost exact roots |
| Spherical spacetime/cavity | closed global cluster safe | `g/g_bar=[(r/L)/sin(r/L)]^2` | 56.53% (72.387 km/s) | 13.42% (0.872 dex) | 7.45% (25.153 arcsec) | failed galaxy and/or cluster-domain gates |
| Spherical spacetime/cavity | closed global galaxy only diagnostic | `same closed-space law, galaxy-only L` | 46.71% (88.741 km/s) | — | — | failed galaxy and/or cluster-domain gates |
| Spherical spacetime/cavity | local amplified screened | `closed-space amplification x local acceleration screen` | 0.00% (177.972 km/s) | — | — | failed galaxy and/or cluster-domain gates |
| Spherical spacetime/cavity | Hard spherical cavity flow analogy | `v_flow/v_inf = 1 + O[(R_body/r)^3] with potential-flow angular factors` | 56.77% (71.990 km/s) | — | — | geometric effect far too small; analytic net force is zero |
| Action-level Sigma | sigma refracted AQUAL | `div[mu(Sigma,\|grad Phi\|/a0) grad Phi]=4 pi G rho_b` | — | 59.83% (0.223 dex) | 71.49% (3.853 arcsec) | exploratory action; synthetic galaxy test only, no covariant completion |
| Action-level Sigma | sigma gated AQUAL | `AQUAL mu with Sigma activation gate` | — | 51.96% (0.284 dex) | — | exploratory action; synthetic galaxy test only, no covariant completion |
| Action-level Sigma | conformal symmetron | `Box Sigma=dV_eff/dSigma; matter follows A(Sigma)^2 g_mn` | — | 16.70% (0.777 dex) | — | exploratory action; synthetic galaxy test only, no covariant completion |
| Action-level Sigma | Sigma refracted AQUAL (cluster-tuned diagnostic) | `same AQUAL law with cluster-selected Sigma parameters` | — | 85.14% (0.070 dex) | 83.14% (2.279 arcsec) | not universal; included only to expose galaxy/cluster tension |
| Action-level Sigma | Causal catch-up Sigma completion | `div(mu grad Phi) - Q(Sigma,y)c^-2 d_t^2 Phi = 4 pi G rho_b` | 93.64% (10.586 km/s) | 72.66% (0.139 dex) | 83.37% (2.248 arcsec) | causal and stable, but time term is exactly invisible to static tests |
| Potential and boundary screens | P0 baryonic-potential screen | `activation S=S(\|Phi_bar\|/c^2); Delta g/g_bar=A S chi^(-p)` | 86.51% (23.409 km/s) | — | — | competitive with but does not beat fixed RAR |
| Potential and boundary screens | P1 CF4-shifted potential threshold | `log chi_t=log chi_t0+zeta V_CF4` | 85.88% (24.495 km/s) | — | — | environmental threshold shift worsens prediction |
| Potential and boundary screens | B1 potential boundary layer | `g_B=kappa a_star dS_Phi/d ln R` | 85.91% (24.447 km/s) | — | — | boundary coefficient sign is unstable; rejected |
| Potential and boundary screens | W1 measured void-wall threshold | `log chi_t=log chi_t0+zeta_w d_to_measured_void_wall` | 85.19% (25.693 km/s) | — | — | measured wall depth worsens prediction |
| Potential and boundary screens | T0 ordinary CF4 gravity tide | `div g_delta=-(3/2) Omega_m H0^2 delta; internal effect from tidal Hessian` | — | — | — | median inward tide is 1.33e-5 of the required acceleration; rejected |
| Covariant/Aether attempts | H7a simple-mu potential-dependent AQUAL | `div[mu(g/a_X(Phi_bar)) grad Psi]=4 pi G rho; mu=x/(1+x)` | 86.64% (23.175 km/s) | 68.43% (0.165 dex) | — | phenomenologically competitive, but it recreates MOND/AQUAL and the environment is noncovariant |
| Covariant/Aether attempts | H7s standard-mu potential-dependent AQUAL | `div[mu_s(g/a_X(Phi_bar)) grad Psi]=4 pi G rho; mu_s=x/sqrt(1+x^2)` | 86.69% (23.091 km/s) | 69.82% (0.156 dex) | — | phenomenologically competitive, but it recreates MOND/AQUAL and the environment is noncovariant |
| Covariant/Aether attempts | EA-Q0 reciprocal environmental Aether | `S~F(Q)R + Aether[K, a_Q(Q)] + beta[(grad Q)^2+Q^2/L_Q^2]` | — | — | — | retired before fit: reciprocal Aether source changes Q by orders of magnitude |
| Covariant/Aether attempts | EMOG-Q0 chameleon scalar + Proca vector | `S~F(s)R-(grad s)^2-U(s)-B^2/4-mu^2 phi^2/2-phi_a J^a` | — | — | — | retired before fit: wrong density ordering, Yukawa shape, and Solar-System conflict |
| Measured density/coherence | CPR0 measured coherence-partitioned RG | `epsilon_mix=w(C)+(1-w)epsilon_RG; nu_src=1+w B0 h(g_bar)` | 80.68% (0.093 dex) | 70.05% (0.155 dex) | — | 0.00070-dex gain over RG; fails frozen improvement gate |
| Measured density/coherence | NBP0 nonlocal scalar permittivity morphology | `div[epsilon(X) grad Phi]=4 pi G rho_b; (1-L_X^2 Laplacian)X=rho_b` | 74.73% (0.127 dex) | — | — | morphology worsens RMSE by 7.03%; structural scalar failure |
| Finite mechanism closure | NBM0 A0 canonical conformal scalar | `g~=exp(2 alpha X) g; canonical massive X` | — | — | — | Weyl-potential contribution cancels |
| Finite mechanism closure | NBM0 A1 disformal scalar with prescribed U | `g~=exp(2 alpha X)(g+2 beta X U U)` | — | — | — | no reciprocal equation for preferred direction |
| Finite mechanism closure | NBM0 A2 canonical scalar + dynamical Aether | `E(r)=1+A(1+r/L)e^(-r/L)` | — | — | — | positive Yukawa response is never flatter than Keplerian |
| Finite mechanism closure | NBM0 A3 massless canonical scalar | `E(r)=1+A` | — | — | — | constant Newtonian rescaling; no flat curve or screening |
| Finite mechanism closure | NBM0 A4 positive Yukawa spectrum | `E(r)=1+sum A_i(1+r/L_i)e^(-r/L_i), A_i>=0` | — | — | — | nonnegative spectrum cannot turn gravity on at large radius |
| Finite mechanism closure | NBM0 A5 fractional p=3/2 operator | `(-Laplacian)^(3/2) Phi proportional to rho` | — | — | — | flat radial shape but v_flat^4 proportional to M^2 |
| Finite mechanism closure | NBM0 A6 nonlinear p-Laplacian | `div(\|grad Phi\| grad Phi) proportional to rho` | — | — | — | unique flat+BTFR limit is already AQUAL/MOND |
| Finite mechanism closure | NBM0 A7 smooth external void basin | `X=X0+grad X.r+X2 r^2/2+...` | — | — | — | uniform terms cancel; leading internal force is harmonic |
| Finite mechanism closure | NBM0 A8 nonlinear nonlocal basin | `localized nonlinear memory kernel with auxiliary fields` | — | — | — | no healthy non-MOND action survived the closure audit |
| Finite mechanism closure | NBM0 A9 self-gravitating basin phase | `G_ab=8 pi G(T_b+T_basin)` | — | — | — | can lens only by adding an independent gravitating energy reservoir |
| Action-level Sigma | Sigma complete reciprocal action | `S~R/16piG - Z(Sigma)(grad Sigma)^2/2 - V(Sigma) + F(X,Sigma)` | — | — | — | reciprocal feedback/stress-energy completion did not resolve the galaxy-cluster tension |
| Action-level Sigma | Sigma covariant weak-field metric closure | `Box Sigma=V_eff,Sigma; div[mu(Sigma,X) grad Phi]=4 pi G rho_b` | — | — | — | mathematical weak-field closure only; no new independent observational gain |
| Conservative profile diffusion | No-flux fractional-excess diffusion | `dX/dtau=d2X/d(ln r)^2; X=F; zero boundary flux` | 56.88% (71.794 km/s) | — | 25.29% (18.927 arcsec) | measurable but worsens the local-control galaxy/lensing compromise |
| Conservative profile diffusion | No-flux added-acceleration diffusion | `dX/dtau=d2X/d(ln r)^2; X=F g_N; ell=0.7, mu=1` | 52.92% (78.398 km/s) | — | 26.39% (18.648 arcsec) | strongest diffusion raw improvement, paired with an 8.74-km/s galaxy penalty |
| Conservative profile diffusion | No-flux circular-speed-squared diffusion | `dX/dtau=d2X/d(ln r)^2; X=F g_N r; ell=0.15, mu=0.5` | 58.15% (69.678 km/s) | — | 25.56% (18.857 arcsec) | nearly galaxy-neutral but only a 0.089-arcsec development-sample raw gain |
| Conservative profile diffusion | One-sided memory plus no-flux diffusion | `F_memory(p=1.927,q=9,ell=0.35) then diffuse F at ell=0.35, mu=1` | 81.46% (30.868 km/s) | — | not scoreable (27.971 arcsec; incomplete roots) | improves the memory-control galaxy score but loses a held-out lens root |

## Bottom-line interpretation

1. Fixed RAR and simple MOND remain the strongest universal galaxy controls, at about 93.8% velocity proximity on the untouched SPARC outer points, but their two-cluster raw-lens proximity is only about 5–6%.
2. The RAR + squared coherence-gated RG bridge is the only tested project formula that is simultaneously close to RAR on observational SPARC data and very close on its one-cluster raw-lens test. That raw result has not yet transferred to multiple clusters, so it is promising evidence, not a universal solution.
3. On the four-cluster transfer, the best locked modified-gravity scalar law is roughly 29% raw-lens proximity, while the compact-halo comparator is roughly 65%. Adding member-light vectors does not close the gap.
4. The new full member-tidal tensor test selects zero coupling and gives essentially the same two-cluster error as the scalar-slip parent. Strong negative couplings improve a local fitting cost but lose exact image roots, so changing the number does not rescue it.
5. The Solar-screened isothermal tail now has a direct morphology-stratified galaxy test. Its locked cluster value scores 18.60 km/s overall and is especially poor for disk-dominated, dwarf, late-type, flat/rising systems; it is therefore not the missing universal bridge.
6. Conservative radial diffusion confirms that the transported physical quantity matters: added acceleration is lensing-favored, while short-scale circular-speed-squared transport is nearly galaxy-neutral. No diffusion carrier improves the complete-root cross-domain control, and diffusion after the best memory response loses lens roots.
7. No tested formula yet matches both trusted galaxy controls and multi-cluster dark-matter lens reconstructions with one universal setting. The most defensible next test is the existing RAR + coherence/RG candidate on several clusters using complete baryonic maps (gas, BCG, ICL, and member galaxies), with its constants frozen before image scoring.

Machine-readable versions: `results/formula_scorecard/formula_scorecard.csv` and `results/formula_scorecard/formula_scorecard.json`.
