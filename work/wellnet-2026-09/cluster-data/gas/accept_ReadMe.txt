J/ApJS/182/12          ICM entropy profiles (ACCEPT)          (Cavagnolo+, 2009)
================================================================================
Intracluster medium entropy profiles for a Chandra archival sample of
galaxy clusters.
    Cavagnolo K.W., Donahue M., Voit G.M., Sun M.
   <Astrophys. J. Suppl. Ser., 182, 12-32 (2009)>
   =2009ApJS..182...12C
================================================================================
ADC_Keywords: X-ray sources ; Clusters, galaxy ; Redshifts
Keywords: astronomical data bases: miscellaneous - cooling flows
          X-rays: galaxies: clusters - X-rays: general

Abstract:
    We present radial entropy profiles of the intracluster medium (ICM)
    for a collection of 239 clusters taken from the Chandra X-ray
    Observatory's Data Archive. We find that most ICM entropy profiles are
    well fitted by a model which is a power law at large radii and
    approaches a constant value at small radii:
    K(r)=K_0_+K_100_(r/100kpc)^{alpha}^, where K_0_ quantifies the typical
    excess of core entropy above the best-fitting power law found at
    larger radii. For completeness, we include previously unpublished
    optical spectroscopy of H{alpha} and [NII] emission lines discussed in
    Cavagnolo et al. (2008ApJ...683L.107C). All data and results
    associated with this work are publicly available via the project Web
    site (ACCEPT: Archive of Chandra Cluster Entropy Profile Tables).

Description:
    Our sample is collected from observations taken with the Chandra X-ray
    Observatory and which are publicly available in the CDA (Chandra Data
    Archive) as of 2008 August.

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe         80        .   This file
table1.dat     78      320   Summary of sample
table5.dat     96      960   Summary of entropy profile fits
--------------------------------------------------------------------------------

See also:
   B/chandra : The Chandra Archive Log (CXC, 1999-)
   http://www.pa.msu.edu/astro/MC2/accept : Home page of the project
   J/A+A/396/397 : Deep cluster survey in Chandra archival data (Boschin+, 2002)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 18  A18   ---     Name      Cluster name
  20- 24  I5    ---     ObsID     Chandra observation identifier
  26- 27  I2    h       RAh       Cluster Hour of Right Ascension (J2000)
  29- 30  I2    min     RAm       Cluster Minute of Right Ascension (J2000)
  32- 37  F6.3  s       RAs       Cluster Second of Right Ascension (J2000)
      39  A1    ---     DE-       Cluster Sign of the Declination (J2000)
  40- 41  I2    deg     DEd       Cluster Degree of Declination (J2000)
  43- 44  I2    arcmin  DEm       Cluster Arcminute of Declination (J2000)
  46- 50  F5.2  arcsec  DEs       Cluster Arcsecond of Declination (J2000)
  52- 56  F5.1  ks      Exp       Exposure time
  58- 59  A2    ---     ACIS      CCD location of cluster center
  61- 66  F6.4  ---     z         Redshift
  68- 72  F5.2  keV     <kT>      Average cluster temperature
  74- 78  A5    ---     Notes     Assigned note(s) (1)
--------------------------------------------------------------------------------
Note (1): Notes as follows:
    a = Clusters analyzed using the best-fit {beta}-model for the surface
        brightness profiles (discussed in Sect. 3.2);
    b = Clusters with complex surface brightness of which only the central
        regions were used in fitting K(r);
    c = Clusters only used during analysis of the HIFLUGCS sub-sample
        (discussed in Sect. 5.4);
    d = Clusters with central AGN removed during analysis
        (discussed in Sect. 3.5);
    e = Clusters with central compact source removed during analysis
        (discussed in Sect. 3.5);
    f = Clusters with central bin ignored during fitting
        (discussed in Sect. 3.5).
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table5.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label  Explanations
--------------------------------------------------------------------------------
   1- 18  A18   ---       Name   Cluster name
  20- 23  A4    ---       Meth   ?=- Method of T_X_ interpolation in inner
                                     region: flat or ext (linear extrapolation)
  25- 26  I2    ---       Nbin   ? Number of radial bins in fit
  28- 31  F4.2  Mpc       rmax   ? Maximum radius for fit
  33- 37  F5.1  keV.cm2   K0     Best-fit core entropy (K=Tx.n_e_^-2/3^)
  39- 44  F6.2  keV.cm2 e_K0     ? Uncertainty in K0
  46- 50  F5.1  ---       sigK0  ? Number of sigma K_0_ is away from zero
  52- 58  F7.1  keV.cm2   K100   Best-fit entropy at 100kpc
  60- 65  F6.1  keV.cm2 e_K100   Uncertainty in K100
  67- 70  F4.2  ---       alpha  Best-fit power-law index
  72- 75  F4.2  ---     e_alpha  Uncertainty in alpha
  77- 78  I2    ---       DOF    Degrees of Freedom in fit
  80- 86  F7.2  ---       chi2   The {chi}^2^ of best-fit model
  88- 96  A9    ---       p-val  Probability of worse fit given chi2 and DOF
--------------------------------------------------------------------------------

History:
    From electronic version of the journal
================================================================================
(End)                 Greg Schwarz [AAS], Emmanuelle Perret [CDS]    17-Nov-2009
