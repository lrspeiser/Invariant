J/ApJ/716/198             The DiskMass survey. I.              (Bershady+, 2010)
================================================================================
The DiskMass survey.
I. Overview.
    Bershady M.A., Verheijen M.A.W., Swaters R.A., Andersen D.R.,
    Westfall K.B., Martinsson T.
   <Astrophys. J., 716, 198-233 (2010)>
   =2010ApJ...716..198B
================================================================================
ADC_Keywords: Galaxies, nearby ; Surveys
Keywords: dark matter - galaxies: evolution - galaxies: formation -
          galaxies: fundamental parameters - galaxies: halos -
          galaxies: kinematics and dynamics - galaxies: spiral -
          galaxies: stellar content - galaxies: structure

Abstract:
    We present a survey of the mass surface density of spiral disks,
    motivated by outstanding uncertainties in rotation-curve
    decompositions. Our method exploits integral-field spectroscopy to
    measure stellar and gas kinematics in nearly face-on galaxies sampled
    at 515, 660, and 860nm, using the custom-built SparsePak and PPak
    instruments. A two-tiered sample, selected from the UGC, includes 146
    nearly face-on galaxies, with B<14.7 and disk scale lengths between 
    10 and 20 arcsec, for which we have obtained H{alpha} velocity fields;
    and a representative 46 galaxy subset for which we have obtained
    stellar velocities and velocity dispersions. The survey is augmented
    by 4-70um Spitzer IRAC and MIPS photometry, ground-based UBVRIJHK
    photometry, and HI aperture-synthesis imaging. We outline the
    spectroscopic analysis protocol for deriving precise and accurate
    line-of-sight stellar velocity dispersions. Our key measurement is the
    dynamical disk-mass surface density. Star formation rates and
    kinematic and photometric regularity of galaxy disks are also central
    products of the study.

Description:
    H{alpha} kinematic data were taken with the SparsePak integral-field
    unit (IFU) on the WIYN 3.5m telescope. We observed 137 galaxies over
    13 runs from 2002 January to 2005 April totaling 41.5 nights, plus
    portions of two additional runs totaling 8 nights during SparsePak
    commissioning in 2001 May-June.

    Aperture-synthesis radio observations at 21cm have been obtained for
    43 galaxies with either the VLA (seven galaxies in 2005), the WSRT (20
    galaxies in 2007-2009, three overlapping with VLA), or the GMRT (19
    galaxies in 2008-2009, one overlapping with VLA and WSRT, two others
    overlapping with WSRT).

    Spitzer near- and mid-infrared (4.5, 8, 24, and 70um) images were
    taken of the majority of the Phase-B sample (40 galaxies).

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe         80        .   This file
table2.dat    128      231   DiskMass sample
--------------------------------------------------------------------------------

See also:
   VII/26 : Uppsala General Catalogue of Galaxies (UGC) (Nilson 1973)
   J/ApJS/166/505 : H{alpha} photometry of face-on galaxies (Andersen+, 2006)
   J/AJ/131/2035  : B and R magnitudes for spiral galaxies (Galaz+, 2006)
   J/A+A/442/137  : HI observations of WHISP disk galaxies (Noordermeer+, 2005)
   J/ApJ/633/844  : Stellar mass in disk-dominated galaxies (Pizagno+, 2005)
   J/A+AS/106/451 : Face-on disk galaxies photometry. I. (de Jong+, 1994)
   J/AJ/106/530   : Low surface brightness disk galaxies (Bothun+, 1993)

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units       Label  Explanations
--------------------------------------------------------------------------------
   1-  5  I5    ---         UGC    UGC number
   7-  8  I2    h           RAh    Hour of right ascension (J2000)
  10- 11  I2    min         RAm    Minute of right ascension (J2000)
  13- 16  F4.1  s           RAs    Second of right ascension (J2000)
      18  A1    ---         DE-    Sign of declination (J2000)
  19- 20  I2    deg         DEd    Degree of declination (J2000)
  22- 23  I2    arcmin      DEm    Arcminute of declination (J2000)
  25- 26  I2    arcsec      DEs    Arcsecond of declination (J2000)
  28- 33  A6    ---         Type   Hubble morphological type
      34  I1    ---       f_Type   [1/5]? additional classification (1)
  36- 40  I5    km/s        HRV    NED heliocentric velocity
  42- 46  F5.1  Mpc         Dist   NED distance (2)
  48- 51  F4.2  mag         AB     B-band Galactic extinction
  53- 56  F4.1  mag         Bmag   ? Apparent B-band magnitude from RC3, with
                                   corrections applied
  58- 60  F3.1  mag       e_Bmag   ? Bmag uncertainty
  62- 66  F5.2  mag         Ksmag  ? 2MASS Ks-band magnitude, with corrections
                                   applied
  68- 71  F4.2  mag       e_Ksmag  ? Ksmag uncertainty
  73- 77  F5.1  mag         KMag   ? Absolute K-band magnitude applying distance
                                   modulus and Galactic extinction corrections
  79- 81  F3.1  mag         B-K    ? Rest-frame B-K color index, corrected for
                                   Galactic extinction
  83- 85  F3.1  mag       e_B-K    ? B-K uncertainty
  87- 90  F4.1 mag/arcsec2  mu0    Estimated disk central surface brightness
                                   from POSS images, calibrated to R band
  92- 95  F4.1  arcsec      hR     Estimated disk radial scale length
  97-100  F4.1  arcsec      r23.5  Apparent isophotal radius where
                                   {mu}_R_=23.5mag/arcsec2
 102-107  A6    ---         Sel    Disk-mass Phase-A sample selection (3)
 109-111  A3    ---         Ha     H{alpha} IFU observations (D=DensePak or
                                   S=SparsePak)
     112  A1    ---       f_Ha     [g] incomplete H{alpha} observation (4)
 114-118  A5    ---         sigma  Stellar line-of-site velocity dispersion
                                   (IFS) observations ({sigma}^LOS^_*_) (5)
     119  A1    ---       n_sigma  [g] incomplete sigma observation (4)
     122  A1    ---         Spz    [y/ ] Spitzer/IRAC or MIPS imaging?
 124-128  A5    ---         HI     21cm aperture-synthesis observations (V: VLA,
                                   W: Westerbork and/or G: GMRT)
--------------------------------------------------------------------------------
Note (1): Flag as follows:
  1 = peculiar;
  2 = HII;
  3 = star-burst;
  4 = Liner;
  5 = AGN (Seyfert or BLAGN)
Note (2): Distance assuming H_0_=73km/s/Mpc using flow-corrected velocities.
Note (3): Selection as follows:
  p = pilot sample,
  4 = 4/4 (176 sources were agreed upon by all four reviewers as satisfactory
      targets, the 4/4 sample; see section 5.1 for further details),
  d = DSS.
  D = DSS sample objects (UGC 4036, 9610, 12224) have disk radial scale lengths
      slightly outside the nominal DSS selection due to the improved disk
      fitting described in Section 5.2.
Note (4):
  g = IFU observations in Columns 16 and 18 (H{alpha} or {sigma}^LOS^_*_) are in
      some way incomplete (depth or spatial coverage), but in many cases yield
      useful information.
Note (5): Code as follows:
  M = SparsePak MgI region;
  C = SparsePak CaII region;
  P = PPak MgI region.
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

References:
    Bershady et al.     Paper II.     2010ApJ...716..234B
    Westfall et al.     Paper III.    2011ApJS..193...21W
    Westfall et al.     Paper IV.     2011ApJ...742...18W
    Bershady et al.     Paper V.      2011ApJ...739L..47B
    Martinsson et al.   Paper VI.     2013A&A...557A.130M
    Martinsson et al.   Paper VII.    2013A&A...557A.131M
    Westfall et al.     Paper VIII.   2014ApJ...785...43W
    Swaters et al.      Paper IX.     2014ApJ...797L..28S
    Martinsson et al.   Paper X.      2016A&A...585A..99M
    Swaters et al.      Paper XI.     2025ApJS..276...59S   Cat. J/ApJS/276/59

================================================================================
(End)                                     Emmanuelle Perret [CDS]    24-May-2012
