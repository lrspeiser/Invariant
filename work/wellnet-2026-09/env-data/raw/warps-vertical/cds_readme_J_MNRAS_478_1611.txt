J/MNRAS/478/1611    Local Volume H I Survey (LVHIS)          (Koribalski+, 2018)
================================================================================
The Local Volume H I Survey (LVHIS).
    Koribalski B.S., Wang J., Kamphuis P., Westmeier T., Staveley-Smith L.,
    Oh S.-H., Lopez-Sanchez A.R., Wong O.I., Ott J., De Blok W.J.G., Shao L.
   <Mon. Not. R. Astron. Soc., 478, 1611-1648 (2018)>
   =2018MNRAS.478.1611K    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Surveys ; H I data ; Galaxies, radio
Keywords: surveys - galaxies: dwarf - galaxies: kinematics and dynamics -
          galaxies: spiral - galaxies: structure - radio lines: galaxies

Abstract:
    The 'Local Volume HI Survey' (LVHIS) comprises deep HI spectral line
    and 20-cm radio continuum observations of 82 nearby, gas-rich
    galaxies, supplemented by multiwavelength images. Our sample consists
    of all galaxies with Local Group velocities v_LG_<550km/s or
    distances D<10Mpc that are detected in the HI Parkes All Sky
    Survey (HIPASS). Using full synthesis observations in at least three
    configurations of the Australia Telescope Compact Array (ATCA), we
    obtain detailed HI maps for a complete sample of gas-rich galaxies
    with {delta}~-30{deg}. Here we present a comprehensive LVHIS
    galaxy atlas, including the overall gas distribution, mean velocity
    field, velocity dispersion, and position-velocity diagrams, together
    with a homogeneous set of measured and derived galaxy properties. Our
    primary goal is to investigate the HI morphologies, kinematics, and
    environment at high resolution and sensitivity. LVHIS galaxies
    represent a wide range of morphologies and sizes; our measured HI
    masses range from ~10^7^ to 10^10^M_{sun}_, based on
    independent distance estimates.

Description:
    The Local Volume HI Survey (LVHIS) consists of all galaxies with
    Local Group velocities vLG<550km/s or distances D<10Mpc
    that are detected in the HIPASS at declinations {delta}<~-30{deg}.
    We present the results of deep ATCA HI spectral line observations
    of a complete sample of 82 nearby, gas-rich galaxies, including a
    comprehensive HI atlas (see the Appendix) and on-line data base
    (www.atnf.csiro.au/research/LVHIS)

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table2.dat       108       82   Optical properties of LVHIS galaxies
table4.dat        93       82   HIPASS properties of the LVHIS galaxies
table6.dat        97       82   ATCA HI properties of LVHIS galaxies
table7.dat        96       15   ATCA HI properties for galaxies newly discovered
                                 in HIPASS and here
table8.dat        46       82   Derived properties of the LVHIS galaxies
table9.dat       102       47   ATCA HI kinematic properties of LVHIS galaxies
--------------------------------------------------------------------------------

See also:
  VIII/73 : HI Parkes All Sky Survey Catalogue (HIPASS) (Meyer+, 2004)

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 15  A15   ---     HIPASS    HIPASS name  (HIPASS JHHMM+DD)
      16  A1    ---   m_HIPASS    [A] Multiliity index on HIPASS
  18- 29  A12   --      OName     Optical galaxy name
      31  A1    ---   n_OName     [1] Note on OName (1)
  33- 36  F4.2  Mpc     D         Best available galaxy distance (2)
  38- 46  A9    ---     MType     Morphological type
  48- 52  F5.3  mag     AB        B-band extinction, fron
                                   Schlafly & Finkbeiner (2011ApJ...737..103S)
  54- 58  F5.2  mag     BTmag     ?=- BT magnitude from Lauberts & Valentijn
                                   (Cat. VII/115)
      59  A1    ---   n_BTmag     [)] Uncertainty flag on BTmag
  61- 65  F5.2  [Lsun]  logLB     ?=- Luminosity in B Band
      66  A1    ---   u_logLB     [)] Uncertainty flag on logLB
  68- 72  F5.2  mag     RTmag     ?=- RT magnitude from Lauberts & Valentijn
                                   (Cat. VII/115)
      73  A1    ---   u_RTmag     [)] Uncertainty flag on RTmag
  75- 79  F5.2  mag     B-R       ?=- B-R colour index
      80  A1    ---   u_B-R       [)] Uncertainty flag on B-R
  82- 85  I4    Mpc     Dopt      ?=- B-band diameter at 25.5mag/arcsec^2^
      86  A1    ---   u_Dopt      [)] Uncertainty flag on Dopt
  88- 89  I2    deg     i         ?=- Inclination angle, from
                                   Lauberts, 1982, Cat. VII/34
      90  A1    ---   u_i         [)] Uncertainty flag on i
  92- 94  I3    deg     PA        ?=- Position angle, from
                                   Lauberts, 1982, Cat. VII/34
      95  A1    ---   u_PA        [)] Uncertainty flag on PA
  96-105  A10   ---     Group     Galaxy subgroup
 107-108  I2    ---     Ncln      Number of close neighbours (within 300arcmin
                                   and vsys<800km/s)
--------------------------------------------------------------------------------
Note (1): Note as follows:
   1 = The optical properties listed are for ESO252-IG001 NED01
Note (2): Tully-Fisher (TF), Hubble (Ho), and membership (mem) distances are
   here given to one decimal accuracy, while TRGB and Cepheid distances are
   given to two decimal points.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table4.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 15  A15   ---       HIPASS    HIPASS name  (HIPASS JHHMM+DD)
      16  A1    ---     m_HIPASS    [A] Multiliity index on HIPASS
  18- 29  A12   --        OName     Optical galaxy name
  34- 36  I3    km/s      vLG       Local Group velocity
      37  A1    ---     u_vLG       [)] Uncertainty flag on vLG
  39- 44  F6.1  Jy.km/s   FHI       ?=- HI flux density
  47- 51  F5.1  Jy.km/s e_FHI       ?=- HI flux density error
  54- 57  F4.2  [Msun]    logMHI    ?=- HI mass
  59- 61  I3    km/s      HV        ?=- Hi systemic velocity in the optical,
                                     heliocentric velocity frame
      62  A1    ---     u_HV        [)] Uncertainty flag on HV
  64- 66  I3    km/s      W50       ?=- HI velocity width at 50% of the
                                     HI peak flux
      67  A1    ---     u_W50       [?] Uncertainty flag on W50
  69- 71  I3    km/s      W20       ?=- HI velocity width at 20% of the
                                     HI peak flux
  73- 83  A11   ---       Cat       HIPASS catalogue (1)
  85- 93  A9    ---       Notes     Notes (2)
--------------------------------------------------------------------------------
Note (1): HIPASS catalogues as follows:
   B99    = Banks et al. (1999ApJ...524..612B)
   HIDEEP = Minchin et al. (2003MNRAS.346..787M, Cat. J/MNRAS/346/787)
   BGC    = Koribalski et al. (2004AJ....128...16K, Cat. J/AJ/128/16)
   HICAT  = Meyer et al. (2004MNRAS.350.1195M, Cat. VIII/73)
Note (2): Notes as follows:
   e = extended
   c = confused
   r = severe baseline ripple
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table6.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 15  A15   ---       HIPASS    HIPASS name (HIPASS JHHMM+DDA)
      16  A1    ---     m_HIPASS    [A] Multiliity index on HIPASS
  18- 29  A12   --        OName     Optical galaxy name
  31- 36  F6.1  Jy.km/s   FHI       HI flux density
      37  A1    ---     u_FHI       [)] Uncertainty flag on FHI
  39- 42  F4.2  [Msun]    logMHI    HI mass
      43  A1    ---     u_logMHI    [)] Uncertainty flag on logMHI
  45- 48  I4    arcsec    RHI       ?=- HI radius
      49  A1    ---     u_RHI       [)] Uncertainty flag on RHI
  51- 52  I2    deg       i         ?=- Inclination angle
  54- 56  I3    deg       PA        ?=- Position angle
  58- 63  F6.1  Jy.km/s   FHI*      ?=- HI flux density enclosed within RHI
  65- 69  F5.2 Msun/Lsun  MHI/LB    ?=- HI mass to B luminosity ratio
      70  A1    ---     u_MHI/LB    [)] Uncertainty flag on MHI/LB
  72- 75  F4.2  ---       DHI/Dopt  ?=- HI to optical diameter ratio
  77- 83  A7    ---       Fig       Figure number in the paper
  84- 97  A14   ---       Notes     Notes (1)
--------------------------------------------------------------------------------
Note (1): Notes as follows:
   (1) = the HI distribution is unresolved
   (2) = the HI distribution is poorly resolved (DHI<2Bmaj), ie. DHI and
          DHI/Dopt are upper limits, and i and PA may differ significantly
          from the measured values
   (3) = FHI is a lower limit due to significant HI absorption
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table7.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 19  A19   ---     Name      Galaxy name
                                   (HIPASS JHHMM+DD or ATCA JHHMMSSs+DDMMSS)
  21- 52  A32   ---     Notes     Nores
      53  A1    ---   n_Notes     [1] Note on Notes (1)
  55- 56  I2    h       RAh       Right ascension (J2000)
  58- 59  I2    min     RAm       Right ascension (J2000)
  61- 64  F4.1  s       RAs       Right ascension (J2000)
      66  A1    ---     DE-       Declination sign (J2000)
  67- 68  I2    deg     DEd       Declination (J2000)
  70- 71  I2    arcmin  DEm       Declination (J2000)
  73- 76  F4.1  arcsec  DEs       Declination (J2000)
  78- 81  F4.2  Jy.km/s FHI       HI flux density
  83- 92  A10   ---     HIdim     HI dimensions (Gaussian fit)
  94- 96  I3    deg     PA        []? Position angle
--------------------------------------------------------------------------------
Note (1): Note as follows:
   1 = HI detected galaxy in ESO252-IG001 NED01
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table8.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label        Explanations
--------------------------------------------------------------------------------
   1- 15  A15   ---     HIPASS       HIPASS name (HIPASS JHHMM+DD)
  17- 28  A12   --      OName        Optical galaxy name
  32- 34  I3    km/s    vrot         ?=- Rotational velocity
  36- 40  F5.2  [Msun]  logMdyn      ?=- Dynamical mass
  42- 46  F5.2  [-]     logMHI/Mdyn  ?=- HI to dynamical mass ratio
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table9.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 15  A15   ---     HIPASS    HIPASS name (HIPASS JHHMM+DDA)
  17- 28  A12   --      OName     Optical galaxy name
  32- 36  F5.1  km/s    vrot      Rotational velocity near the maximum
                                   fitted HI radius
  40- 43  F4.1  kpc     Rmax      Radius of the fitted HI disc
  46- 49  F4.1  deg     i         Fitted inclination or lower value of fitted
                                   inclination interval over the fitted HI disk
      50  A1    ---     ---       [-]
  51- 53  I3    ---     imax      ? Upper value of fitted inclination interval
      54  A1    ---   u_i         [)] Uncertainty flag on i
      55  A1    ---   l_PA        [~] Limit flag on PA
  56- 60  F5.1  deg     PA        Position angle or lower value of position
                                   angle interval over the fitted HI disk
      61  A1    ---     ---       [-]
  62- 64  I3    deg     PAmax     ? Upper value of position angle interval
  66- 70  F5.2  [Msun]  logMdyn   Dynamical mass (Mdyn=2.31x10^5^vrot^2^Rmax)
  72- 76  F5.3  ---     MHI/Mdyn  HI to dynamical mass ratio
      78  A1    ---   l_Nbeam     [~< ] Limit flag on Nbeam
  79- 80  I2    ---     Nbeam     ? Number of of resolution elements across
                                   the HI disc major axis
      81  A1    ---   n_Nbeam     [+?] Note on Nbeam
  83-102  A20   ---     Notes     Notes (1)
--------------------------------------------------------------------------------
Note (1): References as follows:
      C2000 = Cote Carignan & Freeman (2000AJ....120.3027C)
      E2011 = Elson, de Blok & Kraan-Korteweg (2011MNRAS.415..323E)
      K2011 = Kreckel et al. (2011AJ....141..204K)
      K2012 = Kirby et al. (2012MNRAS.420.2924K)
      O2017 = Oh et al. (2018MNRAS.473.3256O)
      S2010 = Struve et al. (2010A&A...515A..67S)
      W2011 = Westmeier, Braun & Koribalski (2011MNRAS.410.2217W)
      W2013 = Westmeier, Koribalski & Braun (2013MNRAS.434.3511W)
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

================================================================================
(End)                                      Patricia Vannier [CDS]    30-Jun-2021
