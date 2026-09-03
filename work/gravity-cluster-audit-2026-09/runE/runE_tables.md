# Run E -- raw analysis output

```
==============================================================================
RUN E : SLACS joint strong-lensing + stellar-dynamics test of three gravity laws
==============================================================================
cosmology  : flat LCDM, H0=70.0 km/s/Mpc, Om=0.30
a0         : 1.2e-10 m/s^2   (FIXED, global; never fitted)
stars      : Hernquist sphere, a = Re/1.8153, Re from Bolton+2008 Table 4 as tabulated
             (Jaffe sphere, a = Re/0.7447, carried as a shape systematic)
aperture   : circular, radius 1.50 arcsec (SDSS 3-arcsec fibre diameter);
             no seeing convolution -- carried as a systematic by varying the radius
lensing    : mean convergence inside theta_E equals 1, Sigma_cr = c^2 D_S/(4 pi G D_L D_LS)
             RAR/AQUAL assume NO SLIP (photons deflect on the same effective
             potential the stars feel), as in TeVeS-like completions.

SELF-TESTS
  [PASS] D_A(z=0.20)                                    got=680.6027 want=680.6027 rel=3.51e-15
  [PASS] D_A(0.20,0.60)                                 got=868.52958 want=868.52958 rel=1.83e-15
  [PASS] D_A(z=0.44)                                    got=1172.7433 want=1172.7433 rel=2.71e-15
  [PASS] D_A(0.44,1.19)                                 got=937.40551 want=937.40551 rel=8.49e-16
  [PASS] D_A(z=0.05)                                    got=201.62273 want=201.62273 rel=4.93e-15
  [PASS] D_A(0.05,0.30)                                 got=755.9177 want=755.9177 rel=1.96e-15
  [PASS] M2D point mass                                 got=9.9999984e+10 want=1e+11 rel=1.57e-07
  [PASS] M2D SIS                                        got=4.5388384e+41 want=4.5388384e+41 rel=1.70e-16
  [PASS] M2D Hernquist vs Sigma-quad                    got=4.1256318e+10 want=4.1256318e+10 rel=0.00e+00
  [PASS] Sigma numeric vs analytic                      got=0.5856158 want=0.5856158 rel=1.90e-15
  [PASS] theta_E(SIS) end-to-end                        got=1.2279097 want=1.2279097 rel=4.70e-15
  [PASS] sigma_SIS_from_thetaE inverse                  got=260000 want=260000 rel=1.12e-16
  [PASS] SIS_REF mass_required_for_thetaE               got=8.4460123e+10 want=8.4460123e+10 rel=1.81e-16
  [PASS] SIS_REF sigma scales as sqrt(M)                got=2 want=2 rel=1.11e-16
  [PASS] RAR deep limit                                 got=1.20601e-12 want=1.2e-12 rel=5.01e-03
  [PASS] AQUAL deep limit                               got=1.206015e-12 want=1.2e-12 rel=5.01e-03
  [PASS] AQUAL Newtonian limit                          got=0.00012000012 want=0.00012 rel=1.00e-06
  [PASS] RAR Newtonian limit                            got=0.00012 want=0.00012 rel=0.00e+00
  [PASS] AQUAL root residual                            got=1 want=1 rel=0.00e+00
  [PASS] sigma_ap angular-quadrature convergence        got=117290.47 want=117290.47 rel=1.06e-09
  [PASS] sigma_ap radial-grid convergence (Newton)      got=117290.47 want=117290.38 rel=8.01e-07
  [PASS] sigma_ap radial-grid convergence (RAR,b=-0.2)  got=123919.24 want=123919.09 rel=1.16e-06
  [PASS] aperture denominator == projected stellar mass got=2.3298146e+10 want=2.3298146e+10 rel=5.39e-10
  [PASS] Jaffe M3d -> total mass                        got=1e+11 want=1e+11 rel=1.00e-09
  [PASS] Jaffe rho integrates to M3d                    got=8.3333333e+10 want=8.3333333e+10 rel=0.00e+00
  [PASS] Hernquist rho integrates to M3d                got=6.9444444e+10 want=6.9444444e+10 rel=0.00e+00
  self-tests PASS

CUT (declared before any residual was computed): exploration split AND Good=='Yes' AND finite sigma/e_sigma/bSIE AND finite Re,b/a AND all four Grillo masses present.
rows in exploration-responses.tsv : 45
rejected by cut : 3
    J0008-0004   Good!=Yes;missing response value
    J0903+4116   Good!=Yes;missing response value
    J1100+5329   Good!=Yes;missing response value
SAMPLE AFTER CUT : N = 42


TABLE 0 -- WHERE THIS SAMPLE LIVES (regime and probe geometry)
  g_N/a0 at observed theta_E   median   4.435   16-84% [  3.351,   6.560]   range [  2.353,  11.723]
  theta_E / Re                 median   0.529   16-84% [  0.429,   0.727]   range [  0.205,   0.917]
  R_aperture / Re              median   0.711   16-84% [  0.474,   1.027]   range [  0.312,   1.596]
  g/g_N at theta_E, NEWTON   median  1.0000   range [ 1.0000,  1.0000]
  g/g_N at theta_E, RAR      median  1.1386   range [ 1.0337,  1.2750]
  g/g_N at theta_E, AQUAL    median  1.1895   range [ 1.0791,  1.3215]
  ISOTHERMAL REFERENCE  log10(sigma_SIS/sigma_obs): median +0.0212  sd 0.0386
     (an SIS whose theta_E matches the data predicts the observed sigma to 5.0% -- the classic SLACS result, and an end-to-end check of this pipeline)

TABLE 1 -- THE DISCRIMINANT
  log10( M required by LENSING / M required by DYNAMICS ), per lens.
  This ratio is independent of the IMF and of the catalogue stellar mass:
  each side is an absolute mass demand, so the stellar-population model cancels.
  law     beta           N    median median 95% CI            sd      MAD
  NEWTON  beta=+0.0     42   -0.0252 [-0.0632,-0.0046]   0.0810   0.0902
  NEWTON  beta=+0.2     42   -0.0113 [-0.0486,+0.0156]   0.0814   0.0904
  NEWTON  beta=-0.2     42   -0.0413 [-0.0761,-0.0144]   0.0811   0.0872
  RAR     beta=+0.0     42   -0.0677 [-0.0944,-0.0394]   0.0832   0.0879
  RAR     beta=+0.2     42   -0.0498 [-0.0799,-0.0145]   0.0846   0.0858
  RAR     beta=-0.2     42   -0.0846 [-0.1119,-0.0521]   0.0825   0.0872
  AQUAL   beta=+0.0     42   -0.0693 [-0.0986,-0.0418]   0.0839   0.0894
  AQUAL   beta=+0.2     42   -0.0510 [-0.0820,-0.0159]   0.0853   0.0849
  AQUAL   beta=-0.2     42   -0.0871 [-0.1093,-0.0547]   0.0832   0.0869
  ---- below: NOT a gravity law, an isothermal mass-profile SHAPE control ----
  SIS_REF beta=+0.0     42   -0.0361 [-0.0636,-0.0100]   0.0796   0.0847
  SIS_REF beta=+0.2     42   -0.0186 [-0.0475,+0.0108]   0.0809   0.0877
  SIS_REF beta=-0.2     42   -0.0520 [-0.0795,-0.0242]   0.0787   0.0782
  within-lens differential (systematics largely cancel):
    RAR    minus NEWTON : median -0.0364  95% CI [-0.0395,-0.0329]  sd 0.0106
    AQUAL  minus NEWTON : median -0.0379  95% CI [-0.0427,-0.0350]  sd 0.0110

TABLE 2 -- ABSOLUTE mass demand vs catalogue stellar mass:  log10(M_req / M_*)
  law     IMF               lensing      sd   dynamics      sd
  NEWTON  Salpeter/BC03      +0.175   0.112     +0.196   0.130
  NEWTON  Salpeter/M05       +0.202   0.145     +0.235   0.164
  NEWTON  Chabrier/BC03      +0.406   0.110     +0.414   0.125
  NEWTON  Kroupa/M05         +0.391   0.139     +0.406   0.162
  RAR     Salpeter/BC03      +0.116   0.115     +0.174   0.135
  RAR     Salpeter/M05       +0.136   0.142     +0.207   0.166
  RAR     Chabrier/BC03      +0.347   0.112     +0.385   0.130
  RAR     Kroupa/M05         +0.322   0.136     +0.376   0.164
  AQUAL   Salpeter/BC03      +0.096   0.115     +0.156   0.136
  AQUAL   Salpeter/M05       +0.115   0.142     +0.188   0.166
  AQUAL   Chabrier/BC03      +0.327   0.112     +0.367   0.131
  AQUAL   Kroupa/M05         +0.302   0.136     +0.358   0.164

TABLE 3 -- forward residuals at the catalogue stellar mass, no rescaling
  law     IMF            dlog thetaE      sd dlog sigma      sd
  NEWTON  Salpeter/BC03      -0.162   0.129     -0.098   0.065
  NEWTON  Salpeter/M05       -0.171   0.185     -0.118   0.082
  NEWTON  Chabrier/BC03      -0.437   0.199     -0.207   0.062
  NEWTON  Kroupa/M05         -0.383   0.260     -0.203   0.081
  RAR     Salpeter/BC03      -0.108   0.120     -0.079   0.061
  RAR     Salpeter/M05       -0.112   0.162     -0.093   0.073
  RAR     Chabrier/BC03      -0.324   0.174     -0.172   0.057
  RAR     Kroupa/M05         -0.300   0.214     -0.169   0.069
  AQUAL   Salpeter/BC03      -0.089   0.117     -0.070   0.061
  AQUAL   Salpeter/M05       -0.095   0.157     -0.084   0.073
  AQUAL   Chabrier/BC03      -0.302   0.167     -0.163   0.057
  AQUAL   Kroupa/M05         -0.277   0.206     -0.161   0.070

TABLE 4 -- GLOBAL Upsilon: ONE mass-scale factor for the whole sample, per probe
  law     IMF              F_lens    F_dyn  log10 ratio
  NEWTON  Salpeter/BC03     1.495    1.570      -0.0213
  NEWTON  Salpeter/M05      1.591    1.719      -0.0335
  NEWTON  Chabrier/BC03     2.549    2.591      -0.0072
  NEWTON  Kroupa/M05        2.462    2.545      -0.0144
  RAR     Salpeter/BC03     1.307    1.494      -0.0582
  RAR     Salpeter/M05      1.366    1.609      -0.0710
  RAR     Chabrier/BC03     2.225    2.425      -0.0374
  RAR     Kroupa/M05        2.101    2.377      -0.0536
  AQUAL   Salpeter/BC03     1.247    1.432      -0.0600
  AQUAL   Salpeter/M05      1.304    1.541      -0.0724
  AQUAL   Chabrier/BC03     2.125    2.326      -0.0393
  AQUAL   Kroupa/M05        2.006    2.282      -0.0560

TABLE 4b -- how well the catalogue stellar masses are known (bounds TABLE 2/4 only)
  IMF                  +dex       -dex    round-off  worst round
  Salpeter/BC03       0.091      0.119        0.005        0.041
  Salpeter/M05        0.111      0.201        0.006        0.030
  Chabrier/BC03       0.076      0.122        0.009        0.067
  Kroupa/M05          0.116      0.225        0.009        0.051

TABLE 5 -- SYSTEMATICS on the discriminant (beta = 0), median log10(M_lens/M_dyn)
  config               NEWTON       RAR     AQUAL    RAR-NEWT  AQUAL-NEWT
  baseline            -0.0252   -0.0677   -0.0693     -0.0364     -0.0379
  Re_circularised     -0.0224   -0.0633   -0.0654     -0.0383     -0.0410
  Jaffe_profile       +0.0705   +0.0328   +0.0267     -0.0396     -0.0456
  aperture_1.0as      -0.0068   -0.0522   -0.0550     -0.0425     -0.0445
  aperture_2.0as      -0.0439   -0.0841   -0.0855     -0.0314     -0.0321

TABLE 6 -- is the discriminant structured?  Spearman rho of log10(M_lens/M_dyn)
            against structural variables, with a MEASUREMENT-NOISE NULL.
            'noise null' = rho produced by sigma and theta_E errors alone.
  -- stars-only Hernquist (NEWTON), beta=0
     variable                rho        p noise-null 95%     beyond null?
     thetaE_over_Rap      +0.459    0.002 [-0.271,+0.340]   YES
     thetaE_over_Re       +0.244    0.119 [-0.265,+0.318]   no
     Rap_over_Re          -0.046    0.772 [-0.314,+0.320]   no
     sigma_obs            -0.341    0.027 [-0.562,-0.014]   no
     z_lens               +0.031    0.845 [-0.299,+0.308]   no
     Re_kpc               +0.115    0.468 [-0.313,+0.300]   no
     bSIE                 +0.459    0.002 [-0.270,+0.331]   YES
     axis_ratio           -0.310    0.046 [-0.297,+0.319]   YES
  -- isothermal control (SIS_REF), beta=0
     variable                rho        p noise-null 95%     beyond null?
     thetaE_over_Rap      +0.456    0.002 [-0.273,+0.345]   YES
     thetaE_over_Re       -0.007    0.964 [-0.273,+0.333]   no
     Rap_over_Re          -0.245    0.119 [-0.301,+0.301]   no
     sigma_obs            -0.306    0.048 [-0.555,-0.003]   no
     z_lens               -0.013    0.936 [-0.307,+0.316]   no
     Re_kpc               +0.238    0.129 [-0.306,+0.308]   no
     bSIE                 +0.456    0.002 [-0.274,+0.342]   YES
     axis_ratio           -0.375    0.014 [-0.297,+0.296]   YES

ERROR MODEL CALIBRATION (diagnostic only)
  d logM / d log sigma   = 2.000 ; median fractional error on sigma = 0.0624 -> 0.0542 dex
  d logM / d log theta_E = 1.392 ; ASSUMED fractional error on bSIE = 0.030 -> 0.0181 dex
  expected scatter from measurement error alone : 0.0571 dex
  observed scatter (NEWTON, beta=0)             : 0.0810 dex
  implied chi2/dof = 2.01  ->  ERROR MODEL IS NOT CALIBRATED.
  No chi2, likelihood, AIC or BIC is quoted as evidence anywhere in this run.
  implied intrinsic (unmodelled) scatter        : 0.0575 dex

PER-LENS TABLE (NEWTON, beta=0; predictions use Chabrier/BC03)
  name            zl    zs   Re"  thE"  sig thEpred sigprd  sigSIS  lgMlen  lgMdyn    dlgR
  J0029-0055   0.227 0.931  2.16  0.96  229   0.345  122.4   217.1  11.636  11.775  -0.139
  J0044+0113   0.120 0.197  2.61  0.79  266   0.151  137.8   268.6  11.670  11.748  -0.077
  J0109+1500   0.294 0.525  1.38  0.69  251   0.435  165.8   243.1  11.628  11.775  -0.147
  J0216-0813   0.332 0.523  2.67  1.16  333   0.409  198.2   347.3  12.252  12.296  -0.043
  J0330-0020   0.351 1.071  1.20  1.10  212   0.697  174.6   251.5  11.723  11.631  +0.092
  J0405-0455   0.075 0.810  1.36  0.80  160   0.337   99.3   176.8  10.871  10.891  -0.020
  J0728+3835   0.206 0.688  1.78  1.25  214   0.369  125.0   256.2  11.704  11.613  +0.091
  J0737+3216   0.322 0.581  2.82  1.00  338   0.326  168.5   292.2  12.112  12.321  -0.208
  J0822+2652   0.241 0.594  1.82  1.17  259   0.661  179.0   270.6  11.803  11.840  -0.037
  J0912+0029   0.164 0.324  3.87  1.63  326   0.340  171.5   346.0  12.180  12.191  -0.011
  J0935-0003   0.347 0.467  4.24  0.87  396   0.046  170.9   360.5  12.514  12.639  -0.124
  J0946+1006   0.222 0.609  2.35  1.38  263   0.388  141.5   283.3  11.916  11.918  -0.002
  J0956+5100   0.240 0.470  2.19  1.33  334   0.586  189.4   317.8  12.015  12.126  -0.111
  J0959+0410   0.126 0.535  1.39  0.99  197   0.412  119.6   215.6  11.274  11.278  -0.005
  J0959+4416   0.237 0.531  1.98  0.96  244   0.479  160.3   253.4  11.751  11.812  -0.061
  J1016+3859   0.168 0.439  1.46  1.09  247   0.459  141.7   253.0  11.545  11.596  -0.052
  J1023+4230   0.191 0.696  1.77  1.41  242   0.670  161.7   266.9  11.730  11.693  +0.037
  J1134+6027   0.153 0.474  2.02  1.10  239   0.439  138.4   242.2  11.579  11.651  -0.072
  J1142+1001   0.222 0.504  1.91  0.98  221   0.340  137.4   254.1  11.720  11.691  +0.029
  J1143-0144   0.106 0.402  4.80  1.68  269   0.225  133.7   285.3  11.941  11.949  -0.009
  J1153+4612   0.180 0.875  1.16  1.05  226   0.376  111.1   219.9  11.377  11.462  -0.085
  J1204+0358   0.164 0.631  1.47  1.31  267   0.425  125.0   253.7  11.570  11.659  -0.089
  J1205+4910   0.215 0.481  2.59  1.22  281   0.363  151.1   285.0  11.936  12.001  -0.065
  J1213+6708   0.123 0.640  3.23  1.42  292   0.520  146.8   251.2  11.720  11.920  -0.200
  J1218+0830   0.135 0.717  3.18  1.45  219   0.479  142.0   253.8  11.759  11.699  +0.060
  J1250+0523   0.232 0.795  1.81  1.13  252   0.716  170.7   243.4  11.692  11.801  -0.109
  J1402+6321   0.205 0.481  2.70  1.35  267   0.525  170.9   293.5  11.967  11.956  +0.011
  J1403+0006   0.189 0.473  1.46  0.83  213   0.434  140.3   224.6  11.450  11.509  -0.058
  J1420+6019   0.063 0.535  2.06  1.04  205   0.608  138.8   203.9  11.091  11.184  -0.093
  J1430+4105   0.285 0.575  2.55  1.52  322   0.266  144.0   336.6  12.183  12.204  -0.021
  J1436-0000   0.285 0.805  2.24  1.12  224   0.474  152.1   255.9  11.874  11.841  +0.032
  J1443+0304   0.134 0.419  0.94  0.81  209   0.496  136.8   206.9  11.118  11.213  -0.095
  J1451-0239   0.125 0.520  2.48  1.04  223   0.579  151.1   221.8  11.502  11.593  -0.091
  J1525+3327   0.358 0.717  2.90  1.31  264   0.393  171.2   317.6  12.234  12.147  +0.087
  J1538+5817   0.143 0.531  1.58  1.00  189   0.489  135.0   222.1  11.387  11.334  +0.053
  J1630+4520   0.248 0.793  1.96  1.78  276   0.833  189.2   310.8  12.014  11.930  +0.084
  J1636+4707   0.228 0.674  1.68  1.09  231   0.395  131.5   247.0  11.672  11.693  -0.022
  J2238-0754   0.137 0.713  2.33  1.27  198   0.456  131.8   238.2  11.586  11.500  +0.087
  J2300+0022   0.229 0.464  1.83  1.24  279   0.415  155.4   300.6  11.885  11.888  -0.004
  J2303+1422   0.155 0.517  3.28  1.62  255   0.534  160.5   289.5  11.942  11.894  +0.049
  J2321-0939   0.082 0.532  4.11  1.60  249   0.726  168.5   259.0  11.691  11.720  -0.029
  J2341+0000   0.186 0.807  3.15  1.44  207   0.526  151.8   261.8  11.896  11.761  +0.136

  lenses with |log10(M_lens/M_dyn)| > 0.1 dex (26%) under NEWTON, beta=0: 8 / 42
```
