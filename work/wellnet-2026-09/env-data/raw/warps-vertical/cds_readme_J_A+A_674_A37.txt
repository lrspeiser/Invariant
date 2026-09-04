J/A+A/674/A37   Gaia DR3. MW asymmetric disc mapping         (Gaia Coll.+, 2023)
================================================================================
Gaia Data Release 3: Mapping the asymmetric disc of the Milky Way.
    Gaia Collaboration, Drimmel R., Romero-Gomez M., Chemin L., Ramos P.,
    Poggio E., Ripepi V., et al.
    <Astron. Astrophys. 674, A37 (2023)>
    =2023A&A...674..A37G        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Milky Way
Mission_Name: Gaia
Keywords: Galaxy: kinematics and dynamics - Galaxy: structure - Galaxy: disk -
          Galaxy: bulge - catalogs

Abstract:
    With the most recent Gaia data release the number of sources with
    complete 6D phase space information (position and velocity) has
    increased to well over 33 million stars, while stellar astrophysical
    parameters are provided for more than 470 million sources, in addition
    to the identification of over 11 million variable stars.

    Using the astrophysical parameters and variability classifications
    provided in Gaia DR3, we select various stellar populations to explore
    and identify non-axisymmetric features in the disc of the Milky Way in
    both configuration and velocity space.

    Using more about 580 thousand sources identified as hot OB stars,
    together with 988 known open clusters younger than 100 million years,
    we map the spiral structure associated with star formation 4-5 kpc
    from the Sun. We select over 2800 Classical Cepheids younger than 200
    million years, which show spiral features extending as far as 10 kpc
    from the Sun in the outer disc. We also identify more than 8.7 million
    sources on the red giant branch (RGB), of which 5.7 million have
    line-of-sight velocities, allowing the velocity field of the Milky Way
    to be mapped as far as 8 kpc from the Sun, including the inner disc.

    The spiral structure revealed by the young populations is consistent
    with recent results using Gaia DR3 astrometry and source lists based
    on NIR photometry, showing the Local (Orion) arm to be at least 8 kpc
    long, and an outer arm consistent with what is seen in HI surveys,
    which seems to be a continuation of the Perseus arm into the third
    quadrant. Meanwhile, the subset of RGB stars with velocities clearly
    reveals the large scale kinematic signature of the bar in the inner
    disc, as well as evidence of streaming motions in the outer disc that
    might be associated with spiral arms or bar resonances. A local
    comparison of the velocity field of the OB stars reveals both
    similarities and differences with the RGB sample. This cursory study
    of Gaia DR3 data shows there is a rich bounty of kinematic information
    to be explored more deeply, which will undoubtedly lead us to an
    understanding of the dynamical nature of the Milky Way's
    non-axisymmetric structures.

Description:
    Table 1: Selected parameters for the Open Clusters
    sample used in the paper.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       115     2531   Selected parameters for the open clusters sample
                                 used in the paper
table2.dat        95     3306   Selected parameters for the classical Cepheid
                                 sample used in the paper
vpob.dat         108       18   Velocity profiles of the OB stars
                                 from Fig. 17 of the paper
vprgb.dat        108       68   Velocity profiles of the RGB stars
                                 from Fig. 17 of the paper
list.dat          83       18   List of fits maps
fits/*             .       18   Individual fits maps
--------------------------------------------------------------------------------

See also:
   I/355 : Gaia DR3 Part 1. Main source (Gaia Collaboration, 2022)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units      Label   Explanations
--------------------------------------------------------------------------------
   1- 17  A17   ---        Cluster Cluster name
  19- 25  F7.3  deg        GLON    Galactic longitude
  27- 33  F7.3  deg        GLAT    Galactic latitude
  35- 38  I4    ---        Nmemb   Number of members
  40- 46  F7.3  mas/yr     pmRA    Median proper motion along RA
  48- 53  F6.3  mas/yr   e_pmRA    Proper motion dispersion along RA
  55- 61  F7.3  mas/yr     pmDE    Median proper motion along Dec
  63- 68  F6.3  mas/yr   e_pmDE    Proper motion dispersion along Dec
  70- 75  F6.3  mas        Plx     Median parallax
  77- 81  F5.3  mas      e_Plx     Parallax dispersion
  83- 91  F9.3  km/s       RV      ?=-1000 median radial velocity of members
  93-101  F9.3  km/s     e_RV      ?=-1000 uncertainty on median radial velocity
 103-106  I4    ---      o_RV      Number of stars used to compute medianRV
 108-111  F4.2  [yr]       Age     Age of the cluster
 113-115  I3    ---        nmag0   Number of stars with absolute magnitude<0
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 19  I19   ---     GaiaDR3   Gaia EDR3 identification source_id
  21- 29  F9.5  deg     GLON      Galactic longitude
  31- 39  F9.5  deg     GLAT      Galactic latitude
  43- 47  F5.2  kpc     Dist      Distance
  49- 54  F6.3  mag     mu        Distance modulus
  57- 61  F5.3  mag   e_mu        Uncertainty on the distance modulus
  63- 67  F5.2  [-]     [Fe/H]    Metallicity
  69- 72  F4.2  [-]   e_[Fe/H]    Uncertainty on the metallicity
      74  I1    ---     Flag      [0/1] Metallicity flag (1)
  76- 83  A8    ---     Ref       Provenance of the classical Cepheid (2)
  85- 89  F5.3  [Gyr]   logAge    Logarithm of the age
  91- 95  F5.3  [Gyr] e_logAge    Uncertainty on the logarithm of the age
--------------------------------------------------------------------------------
Note (1): Flag as follows:
   0 = the metallicity was taken from Gaia DR3 astrophysical parameters
   1 = the metallicity was calculated from the the metallicity gradient of the
        Galactic disc (see text fro details)
Note (2): Provenance of the Classical Cepheid as follows:
  Gaia_DR3 = the star is included in the Gaia DR3 vari_cepheids catalogue
  P21  = objects taken from Pietrukowicz et al. (2021ApJ...914..127I) catalogue
  Inno = objects taken from Inno et al. (2021AcA....71..205I) catalogue
--------------------------------------------------------------------------------

Byte-by-byte Description of file: vpob.dat vprgb.dat
--------------------------------------------------------------------------------
   Bytes Format Units Label       Explanations
--------------------------------------------------------------------------------
   1-  4  F4.1  kpc   Radius      Galactocentric radius
   7- 13  F7.3  km/s  vphimin     Azimuthal velocity at 16th percentile
  16- 22  F7.3  km/s  vphi        Azimuthal velocity at 50th percentile
  25- 31  F7.3  km/s  vphimax     Azimuthal velocity at 84th percentile
  34- 39  F6.3  km/s  sigvRmin    Radial velocity dispersion at 16th percentile
  42- 47  F6.3  km/s  sigvR       Radial velocity dispersion at 50th percentile
  50- 55  F6.3  km/s  sigvRmax    Radial velocity dispersion at 84th percentile
  58- 63  F6.3  km/s  sigvphimin  Azimuthal velocity dispersion
                                   at 16th percentile
  66- 71  F6.3  km/s  sigvphi     Azimuthal velocity dispersion
                                   at 50th percentile
  74- 79  F6.3  km/s  sigvphimax  Azimuthal velocity dispersion
                                   at 84th percentile
  82- 87  F6.3  km/s  sigvZmin    Vertical velocity dispersion
                                   at 16th percentile
  90- 95  F6.3  km/s  sigvZ       Vertical velocity dispersion
                                   at 50th percentile
  98-103  F6.3  km/s  sigvZmax    Vertical velocity dispersion
                                   at 84th percentile
 106-108  I3    ----  npix        Number of pixel in the radial bin of the
                                   velocity map
--------------------------------------------------------------------------------

Byte-by-byte Description of file: list.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  3  I3    Kibyte  size      Size of FITS file
   5- 31  A27   ---     FileName  Name of FITS file, in subdirectory fits
  33- 83  A51   ---     Title     Title of the FITS file
--------------------------------------------------------------------------------

Acknowledgements:
    Vincenzo Ripepi, vincenzo.ripepi(at)inaf.it
    Laurent Chemin, astro.chemin(at)gmail.com

================================================================================
(End)                                        Patricia Vannier [CDS]  16-Jun-2022
