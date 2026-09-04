J/A+A/661/A11       eFEDS. Masses of galaxy clusters and groups    (Chui+, 2022)
================================================================================
The eROSITA Final Equatorial-Depth Survey (eFEDS). X-ray
observable-to-mass-and-redshift relations of galaxy clusters and groups with
weak-lensing mass calibration from the Hyper Suprime-Cam Subaru Strategic
Program survey.
    Chiu I.-N., Ghirardini V., Liu A., Grandis S., Bulbul E., Bahar Y.,
    Comparat J., Bocquet S., Clerc N., Klein M., Liu T., Li X., Miyatake H.,
    Mohr J., More S., Oguri M., Okabe N., Pacaud F., Ramos-ceja M.E.,
    Reiprich T.H., Schrabback T., Umetsu K.
   <Astron. Astrophys., 661, A11 (2022)>
   =2022A&A...661A..11C    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Clusters, galaxy ; X-ray sources
Keywords: galaxies: clusters: general -
          galaxies: clusters: intracluster medium -
          gravitational lensing: weak - large-scale structure of Universe -
          cosmology: observations - dark energy

Abstract:
    We present the first weak-lensing mass calibration and X-ray scaling
    relations of galaxy clusters and groups selected in the eROSITA Final
    Equatorial Depth Survey (eFEDS) observed by Spectrum Roentgen
    Gamma/eROSITA over a contiguous footprint with an area of ~=140
    deg^2^, using the three-year (S19A) weak-lensing data from the Hyper
    Suprime-Cam (HSC) Subaru Strategic Program survey. In this work, we
    study a sample of 434 optically confirmed galaxy clusters (and groups)
    at redshift 0.01 <= z <= 1.3 with a median of 0.35, of which 313
    systems are uniformly covered by the HSC survey to enable the
    extraction of the weak-lensing shear observable. In a Bayesian
    population modeling, we perform a blind analysis for the weak-lensing
    mass calibration by simultaneously modeling the observed count rate
    {eta} and the shear profile g_+_ of individual clusters through the
    count-rate-to-mass-and-redshift ({eta}-M_500_-z) relation and the
    weak-lensing-mass-to-mass-and-redshift (M_WL_-M_500_-z) relation,
    respectively, while accounting for the bias in these observables using
    simulation-based calibrations. As a result, the count-rate-inferred
    and lensing-calibrated cluster mass is obtained from the joint
    modeling of the scaling relations, as the ensemble mass spanning a
    range of 10^13^h^-1^M_{sun}_ <= M_500_ <= 10^15^h^-1^M_{sun}_ with a
    median of ~=10^14^h^-1^M_{sun}_ for the eFEDS sample. With the mass
    calibration, we further model the X-ray
    observable-to-mass-and-redshift relations, including the rest-frame
    soft-band and bolometric luminosity (L_X_ and L_b_), the
    emission-weighted temperature T_X_, the mass of intra-cluster medium
    M_g_, and the mass proxy Y_X_, which is the product of T_X_ and M_g_.
    Except for L_X_ with a steeper dependence on the cluster mass at a
    statistically significant level, we find that the other X-ray scaling
    relations all show a mass trend that is statistically consistent with
    the self-similar prediction at a level of <=1.7{sigma}. Meanwhile, all
    these scaling relations show no significant deviation from the
    self-similarity in their redshift scaling. Moreover, no significant
    redshift-dependent mass trend is present. This work demonstrates the
    synergy between the eROSITA and HSC surveys in preparation for the
    forthcoming first-year eROSITA cluster cosmology.

Description:
    We present the estimates of the cluster true mass for the eFEDS
    clusters in Table C1. In addition to the secure sample of 434 clusters
    with fcont<0.2, we also show the mass of clusters with 0.2<=fcont<0.3.
    This leads to a total number of 457 clusters in Table C1.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
tablec1.dat       70      457   The estimates of the cluster true mass M500
                                 of the eFEDS clusters
--------------------------------------------------------------------------------

See also:
    J/A+A/661/A1 : The eFEDS X-ray catalogs (V7.4) (Brunner+, 2022)
    J/A+A/661/A2 :  Galaxy clusters and groups in eFEDS (Liu+, 2022)

Byte-by-byte Description of file: tablec1.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label         Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---      Name          eFEDS name (JHHMMSS.s+DDMMSS)
      17  A1    ---    n_Name          [*] * for clusters with 20.1<=f_cont_<0.3
  19- 24  F6.3  [Msun]   logM500R0.2A  Mass M500 with the core in the
                                        weak-lensing mass calibration for
                                        R>0.2h^-1^Mpc (in h^1^M{sun} unit)
  26- 31  F6.3  [Msun]   logM500R0.2B  Mass M500 without the core in the
                                        weak-lensing mass calibration for
                                        R>0.2h^-1^Mpc (in h^1^M{sun} unit)
  33- 37  F5.3  [Msun] e_logM500R0.2B  Error on logM500R-0.2B
  39- 44  F6.3  [Msun]   logM500R0.5A  Mass M500 with the core in the
                                        weak-lensing mass calibration for
                                        R>0.5h^-1^Mpc (in h^1^M{sun} unit)
  46- 51  F6.3  [Msun]   logM500R0.5B  Mass M500 without the core in the
                                        weak-lensing mass calibration for
                                        R>0.5h^-1^Mpc (in h^1^M{sun} unit)
  53- 57  F5.3  [Msun] e_logM500R0.5B  Error on logM500R-0.5B
  59- 64  F6.3  [Msun]   logM500Rzsp   ? Mass M500 estimated with spectroscopic
                                        redshifts and without the core in the
                                        weak-lensing calibration
  66- 70  F5.3  [Msun] e_logM500Rzsp   ? Error on logM500R-zsp
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

================================================================================
(End)                                      Patricia Vannier [CDS]    24-Aug-2022
