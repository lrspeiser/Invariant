J/ApJS/276/59   The DiskMass Survey (DMS). XI. Full Ha sample   (Swaters+, 2025)
================================================================================
The DiskMass Survey.
XI. Disk geometries and star formation surface densities from ionized gas
kinematics and line intensities for the full Ha sample.
    Swaters R.A., Andersen D.R., Bershady M.A., Martinsson T.P.K., Scholz P.,
    Verheijen M.A.W., Westfall K.B.
   <Astrophys. J. Suppl. Ser., 276, 59 (2025)>
   =2025ApJS..276...59S
================================================================================
ADC_Keywords: Galaxy catalogs; Spectra, optical; Photometry, infrared;
              Magnitudes, absolute; Extinction; Velocity dispersion;
              Equivalent widths
Keywords: Galaxy dynamics ; Galaxy kinematics ; Unbarred spiral galaxies ;
          Galaxy structure

Abstract:
    We present H{alpha}-region integral-field spectroscopy for
    137 low-inclination, intermediate to late-type galaxies. Spectroscopic
    data, obtained with SparsePak and the Bench Spectrograph on the WIYN
    3.5m telescope, span 6475-6880 {AA} with an instrumental resolution of
    13km/s ({sigma}). The spectral range includes H{alpha} and
    [NII]{lambda}{lambda}6548,6584 for every source, and in most cases
    includes [SII]{lambda}{lambda}6717,6731. We present and publicly
    release 18288 calibrated spectra and visually inspected Gaussian line
    fits to the H{alpha} emission. Most measurements yield a
    signal-to-noise ratio above 5 in integrated H{alpha} line flux,
    adequate to derive reliable line centroids and widths. Second
    kinematic components are required to adequately describe the
    emission-line profile in 15% of reliable data. The H{alpha} velocity
    dispersion distribution peaks at 18km/s, modestly increasing with
    H{alpha} surface brightness, reaching 20km/s at
    {Sigma}H{alpha}=10^40^erg/s/kpc^2^. Lower-flux secondary components,
    when present, have widths of ~50km/s. These results agree well with
    previous echelle measurements of nearby galaxies. Velocity-field
    analysis yields kinematic inclinations, with a sample mean of 26{deg}.
    Large kinematic asymmetries systematically affect kinematic
    inclination estimates in a small fraction of our sample. When
    deviations from circular motion are below 10% of the projected
    velocity, kinematic inclinations are consistent, within errors, to
    estimates from inverting the Tully-Fisher relation. This confirms
    previous disk-submaximality estimates for galaxies with regular
    kinematics based on inclinations derived from inverting the
    Tully-Fisher relation.

Description:
    As described in the DiskMass Survey (DMS) Paper I
    (Bershady+ 2010, J/ApJ/716/198), we produced an initial sample of
    231 galaxies from the UGC.

    From the Phase-A sample, we obtained Ha observations of 146 galaxies.
    Nine of the galaxies that met these conditions were already observed
    using DensePak: UGC 1322, UGC 3091, UGC 4380, UGC 4978, UGC 5730,
    UGC6135, UGC 7072, UGC 7208, and UGC 12784. The observations and
    reductions for these galaxies are described in
    Andersen+ (2006, J/ApJS/166/505) and are not reported here.

    H{alpha} observations for the balance of 137 galaxies were carried out
    at the 3.5m WIYN telescope, between 2001 and 2005, using the SparsePak
    integral field unit. SparsePak feeds the WIYN Bench Spectrograph, a
    fiber-fed spectrograph designed to provide low to medium-resolution
    spectroscopy. We used the Bench Spectrograph Camera (BSC) and a
    316line/mm echelle grating in order 8 to cover 6475{AA}<{lambda}<6880{AA}.
    Of the 137 galaxies observed with SparsePak as part of the DMS,
    125 had sufficient signal to attempt velocity-field modeling.

    As discussed by DMS I, the DiskMass Survey obtained new broadband
    imaging data for nearly all galaxies in the Ha sample using the KPNO
    2.1m telescope. Optical data in UBVRI were obtained during four
    observing runs (2003-Sept-25, 7 nights; 2004-Feb-13, 7 nights;
    2006-Feb-24, 4 nights; 2007-Apr-16, 8 nights).

    See Section 2.

File Summary:
--------------------------------------------------------------------------------
 FileName  Lrecl Records  Explanations
--------------------------------------------------------------------------------
ReadMe        80       .  This file
table3.dat    88     125  H{alpha} velocity field parameters
table4.dat    59     125  Photometric data and inverse Tully-Fisher inclinations
table7.dat   123   18288  H{alpha} data table for individual fiber
                           measurements
--------------------------------------------------------------------------------

See also:
 II/183  : UBVRI Photometric Standards (Landolt 1992)
 VII/233 : 2MASS All-Sky Extended Source Catalog (XSC) (IPAC/UMass, 2003-2006)
 J/AJ/114/2402    : Rotation curves of early-type galaxies (Courteau+, 1997)
 J/ApJ/533/744    : Calibration of the Tully-Fischer relation (Tully+, 2000)
 J/A+A/370/765    : HI synthesis observations in UMa cluster (Verheijen+, 2001)
 J/A+A/390/863    : CCD R Photometry of WHISP Dwarf Galaxies (Swaters, 2002)
 J/ApJS/166/505   : H{alpha} photometry of face-on galaxies (Andersen+, 2006)
 J/MNRAS/390/466  : GHASP: H{alpha} data cubes for 97 galaxies (Epinat+, 2008)
 J/ApJ/716/198    : The DiskMass survey. I. (Bershady+, 2010)
 J/MNRAS/401/2113 : GHASP: H{alpha} data cubes for 153 galaxies (Epinat+, 2010)
 J/MNRAS/465/123  : SAMI Galaxy Survey asymmetries (Bloom+, 2017)
 J/MNRAS/495/2265 : Veloc. dispersions in star-forming galaxies (Varidel+, 2020)

Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label   Explanations
--------------------------------------------------------------------------------
   1-  5 I5     ---      UGC     [16/12808] UGC galaxy identifier
   7-  9 I3     ---      N       [25/393] Number
  11- 14 F4.1   km/s     sigMod  [0/16.1] Extra error term, {sigma}_mod_
                                  (see Equation 6)
  16- 19 F4.1   deg      inc     [14.5/47.5]? Inclination
  21- 24 F4.1   deg    e_inc     [0.8/15]? inc uncertainty
      26 A1     ---    f_inc     Flag on inc (1)
  28- 32 F5.1   deg      phi0    [8/374.1] Position angle of the major axis,
                                  {phi}_0_
  34- 36 F3.1   deg    e_phi0    [0.2/8] phi0 uncertainty
  38- 42 F5.1   km/s     Vrot    [6.6/211] Asymptotic rotation speed, V_rot_
  44- 47 F4.1   km/s   e_Vrot    [0.4/12.2] Vrot uncertainty
  49- 52 F4.1   arcsec   hrot    [0.1/38]? Rotation scale, h_rot_
  54- 56 F3.1   arcsec e_hrot    [0/5.1]? hrot uncertainty
  58- 60 A3     ---    f_hrot    Flag on hrot (2)
  62- 68 F7.1   km/s     Vsys    [224.8/12794] Systemic velocity
  70- 72 F3.1   km/s   e_Vsys    [0.3/4] Vsys uncertainty
  74- 78 F5.3   ---      Aphi    [0.015/0.4] Asymetry measure, A_{phi}_,
                                  defined in Section 5.3
  80- 84 F5.3   ---      Arc     [0.016/0.8]? Asymmetry measure, A_RC_, defined
                                  in Section 5.3
  86- 88 A3     ---      Note    Note (s) (3)
--------------------------------------------------------------------------------
Note (1): Flag as follows:
   l = "low" kinematic inclination; see Section 5
Note (2): Note as follows: 
   URC = For UGC 4458, we used the Courteau S. (1997AJ....114.2402C) empirical
          URC model to fit the rotation curve and derive a kinematic fit to
          the velocity field.
Note (3): Note as follows:
   1 = Two-armed spiral galaxy with no visual evidence of kinematic
        perturbations associated with arms.
   2 = Two-armed spiral galaxy with clear visual evidence of kinematic
        perturbations associated with arms.
   3 = Galaxy with evidence of minor interaction that perturbs the velocity
        field.
   4 = Galaxy with a misalignment between major and minor axes, which is
        evidence for a warp or strong spiral arm perturbations.
   5 = Patchy kinematic data makes visual inspection difficult.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table4.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label Explanations
--------------------------------------------------------------------------------
   1-  5 I5     ---     UGC   [16/12808] UGC galaxy identifier
   7- 11 F5.2   mag     Kmag  [8.62/11.93] 2MASS K band magnitude (4)
  13- 16 F4.2   mag   e_Kmag  [0.01/0.23] Uncertainty on the Kmag
  18- 22 F5.1   Mpc     Dist  [1.8/181.4] Distance (5)
  24- 26 F3.1   Mpc   e_Dist  [2.5/2.5] Dist uncertainty
  28- 32 F5.2   mag     Ak    [-0.03/0.06] K-band extinction
  34- 38 F5.2   mag     Kcor  [-0.1/0.01] K-correction
  40- 45 F6.2   mag     KMag  [-26.07/-16.09] Absolute K band magnitude
  47- 50 F4.2   mag   e_KMag  [0.02/1.95] KMag uncertainty
  52- 55 F4.1   deg     iiTF  [1.8/90] Inverse Tully-Fisher inclination, i_iTF_
  57- 59 F3.1   deg   e_iiTF  [0/4.3] iiTF uncertainty
--------------------------------------------------------------------------------
Note (4): As in DMS V (Bershady+ 2011ApJ...739L..47B), we measured total
    K-band magnitudes from Two Micron All Sky Survey images that we
    reprocessed to remove sky-background gradients as described in DMS VI
    (Martinsson+ 2013A&A...557A.130M).
    We also took advantage of J-, H-, and K-band images and used our own
    elliptical multiaperture photometry to estimate total magnitudes via
    an extrapolation method.
    See Section 6.1.2.
Note (5): To arrive at absolute K-band magnitudes for our samples, distances
    were generated by the Cosmicflows-3 Distance-Velocity Calculator
    (Kourkchi+ 2020AJ....159...67K) using the heliocentric velocities
    derived from our velocity field model fits and adopting
    H_0_=73km/s/Mpc consistent with the value used throughout this paper
    series. See Section 6.1.3.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table7.dat
--------------------------------------------------------------------------------
   Bytes Format Units             Label   Explanations
--------------------------------------------------------------------------------
  1-   5 I5     ---               UGC     [16/12808] UGC galaxy identifier
  7-   9 A3     ---               R       Run code (1)
      11 I1     ---               N       [1/5] Night during the observing run
      13 I1     ---               P       [1/4] SparsePak pointing
 15-  16 I2     ---               A       [1/82] SparsePak fiber aperture
      18 I1     ---               C       [1/9] Fitting code (2)
 20-  24 F5.1   arcsec            oRA     [-70.5/38.7] East-West offset of
                                           fiber aperture (3)
 26-  30 F5.1   arcsec            oDE     [-47/62.5] North-South offset of
                                           fiber aperture (3)
 32-  37 F6.1   10-10m            EWg     [-0.2/1167]? H{alpha} emission
                                           equivalent width from sum of Gaussian
                                           line parameters
 39-  45 F7.1   10-10m            EWsum   [-0.2/29060]? H{alpha} emission
                                           equivalent width from zeroth moment
                                           of line emission
 47-  52 F6.2   [mW/m2/arcsec2]   logmuN  [-16.6/-13] Log H{alpha} narrow
                                           component surface-brightness
                                           (flux within the 4.7" diameter fiber
                                           aperture); in erg/s/cm^2^/arcsec^2^
 54-  58 F5.2   [mW/m2/arcsec2] e_logmuN  [0/73] Uncertainty in logmuN
 60-  65 F6.1   km/s              VelN    [-382/348] Narrow component velocity
                                           relative to kinematic model galaxy
                                           barycenter
 67-  72 F6.1   km/s            e_VelN    [0/5579] Uncertainty in VelN
 74-  78 F5.1   km/s              sigmaN  [2.9/142] Narrow component velocity
                                           dispersion
 80-  84 F5.1   km/s            e_sigmaN  [0/999] Uncertainty in sigmaN
 86-  91 F6.2   [mW/m2/arcsec2]   logmuB  [-99/-13]? Log H{alpha} broad
                                           component surface-brightness;
                                           in erg/s/cm^2^/arcsec^2^
 93-  97 F5.2   [mW/m2/arcsec2] e_logmuB  [0/13.4]? Uncertainty in logmuB
 99- 104 F6.1   km/s              VelB    [-487/340]? Broad component velocity
                                           relative to the kinematic model
                                           galaxy barycenter
106- 111 F6.1   km/s            e_VelB    [0/1174]? Uncertainty in VelB
113- 117 F5.1   km/s              sigmaB  [0/888]? Broad component velocity
                                           dispersion
119- 123 F5.1   km/s            e_sigmaB  [0/999]? Uncertainty in sigmaB
--------------------------------------------------------------------------------
Note (1): SparsePak Run Log, in Table 1, as follows:
  -------------------------------
   RunID  UT start date   Nights
  -------------------------------
   P1     2001-05-06      2
   P2     2002-01-02      3.5
   P3     2002-03-24      3
   R1     2002-10-20      5
   R2     2002-12-08      2
   R3     2003-01-08      1.5
   R4     2003-03-14      2
   R5     2003-04-16      5
   R6     2003-05-18      3.5
   R7     2003-10-02      3
   R8     2004-03-28      3
   R9     2004-09-24      5
   R10    2005-03-27      2
   R11    2005-04-22      3
  -------------------------------
Note (2): Statistics of Halpha line fits, in Table 2, as follows:
  -----------------------------------------------
   Description                  Code   % of fits
  -----------------------------------------------
   Single                        1      43.3
   Double, symmetric wings       2       1.4
   Double, offset wings          3       6.9
   Double, distinct components   4       0.7
   Questionable single fit       7       4.3
   Questionable double fit       9       0.5
   No Line Fit                   0      31.6
   Bad Fit                      -1      11.3
  -----------------------------------------------
Note (3): From the kinematic model galaxy barycenter.
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

References:
    Bershady et al.     Paper I.      2010ApJ...716..198B   Cat. J/ApJ/716/198
    Bershady et al.     Paper II.     2010ApJ...716..234B
    Westfall et al.     Paper III.    2011ApJS..193...21W
    Westfall et al.     Paper IV.     2011ApJ...742...18W
    Bershady et al.     Paper V.      2011ApJ...739L..47B
    Martinsson et al.   Paper VI.     2013A&A...557A.130M
    Martinsson et al.   Paper VII.    2013A&A...557A.131M
    Westfall et al.     Paper VIII.   2014ApJ...785...43W
    Swaters et al.      Paper IX.     2014ApJ...797L..28S
    Martinsson et al.   Paper X.      2016A&A...585A..99M
    Swaters et al.      Paper XI.     2025ApJS..276...59S   This catalog

================================================================================
(End)                    Prepared by [AAS], Emmanuelle Perret [CDS] 31-Oct-2025
