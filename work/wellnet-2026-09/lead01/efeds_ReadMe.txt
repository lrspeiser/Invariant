J/A+A/661/A7        X-ray properties of eFEDS clusters and groups (Bahar+, 2022)
================================================================================
The eROSITA Final Equatorial-Depth Survey (eFEDS).
X-ray properties and scaling relations of galaxy clusters and groups.
    Bahar Y.E., Bulbul E., Clerc N., Ghirardini V., Liu A., Nandra K.,
    Pacaud F., Chiu I-N., Comparat J., Ider-Chitham J., Klein M., Liu T.,
    Merloni A., Migkas K., Okabe N., Ramos-Ceja M.E., Reiprich T.H.,
    Sanders J.S., Schrabback T.
    <Astron. Astrophys. 661, A7 (2022)>
    =2022A&A...661A...7B        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Clusters, galaxy ; Surveys ; X-ray sources
Keywords: galaxies: clusters: general - galaxies: groups: general -
          galaxies: clusters: intracluster medium - X-rays: galaxies: clusters

Abstract:
    Scaling relations link the physical properties of clusters at cosmic
    scales. They are used to probe the evolution of large-scale structure,
    estimate observables of clusters, and constrain cosmological
    parameters through cluster counts. We investigate the scaling
    relations between X-ray observables of the clusters detected in the
    eFEDS field using Spectrum-Roentgen-Gamma/eROSITA observations taking
    into account the selection effects and the distributions of
    observables with cosmic time. We extract X-ray observables (Lx, Lbol,
    T, Mgas, Yx) within R500 for the sample of 542 clusters in the eFEDS
    field. By applying detection and extent likelihood cuts, we construct
    a subsample of 265 clusters with a contamination level of <10%
    (including AGNs and spurious fluctuations) to be used in our scaling
    relations analysis. The selection function based on the
    state-of-the-art simulations of the eROSITA sky is fully accounted for
    in our work. We provide the X-ray observables in the core- included
    <R500 and core-excised 0.15*R500-R500 apertures for 542 galaxy
    clusters and groups detected in the eFEDS field. Additionally, we
    present our best-fit results for the normalization, slope, redshift
    evolution, and intrinsic scatter parameters of the X-ray scaling
    relations between Lx-T, Lx- Mgas, Lx-Yx, Lbol-T, Lbol-Mgas, Lbol-Yx,
    and Mgas-T. We find that the best-fit slopes significantly deviate
    from the self-similar model at a >4sigma confidence level, but our
    results are nevertheless in good agreement with the simulations
    including non-gravitational physics, and the recent results that take
    into account selection effects. The strong deviations we find from the
    self-similar scenario indicate that the non-gravitational effects play
    an important role in shaping the observed physical state of clusters.
    This work extends the scaling relations to the low-mass,
    low-luminosity galaxy cluster and group regime using eFEDS
    observations, demonstrating the ability of eROSITA to measure emission
    from the intracluster medium out to R500 with survey-depth exposures
    and constrain the scaling relations in a wide mass-luminosity-redshift
    range.

Description:
    This catalog contains X-ray properties and best-fit electron density
    model parameters of 542 galaxy clusters and groups detected in the
    eFEDS field. Table of X-ray properties includes temperature, soft band
    (0.5-2.0keV) X-ray luminosity, bolometric (0.01-100keV) X-ray
    luminosity, gass mass and Yx measurements. All the properties are
    measured within apertures of R500. Additionally temperature, soft band
    X-ray luminosity and bolometric X-ray luminosity are measured within
    core-excluded apertures(0.15*R500-R500). X-ray observable
    measurements <2sigma are presented as 2sigma upper limits except T500
    and Tcex500. Table of best-fit electron density model parameters are
    obtained using Vikhlinin et al. model (2006ApJ...640..691V).

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       109      542   Electron density model parameters of
                                 eFEDS clusters
table2.dat       279      542   X-ray observable measurements of eFEDS clusters
--------------------------------------------------------------------------------

See also:
 J/A+A/661/A1  : The eFEDS X-ray catalogs (V7.4) (Brunner+, 2022)
 J/A+A/661/A2  : Galaxy clusters and groups in eFEDS (Liu+, 2022)
 J/A+A/661/A3  : eFEDS counterparts to point-like sources (Salvato+, 2022)
 J/A+A/661/A4  : eFEDS optical cluster and group catalog (Klein+, 2022)
 J/A+A/661/A5  : The eFEDS AGN catalog (Liu+, 2022)
 J/A+A/661/A8  : eFEDS catalogue of variable X-ray sources (Boller+, 2022)
 J/A+A/661/A10 : eFEDS galaxy clusters and groups in disguise (Bulbul+, 2022)
 J/A+A/661/A23 : Exoplanet X-ray irradiation & evaporation rates (Foster+, 2022)
 J/A+A/661/A24 : Corona-chromosphere connection (Fuhrmeister+, 2022)
 J/A+A/661/A29 : First eROSITA study of nearby M dwarfs (Magaudda+, 2022)
 J/A+A/661/A34 : eta Cha cluster eROSITA X-ray scan (Robrade+, 2022)
 J/A+A/661/A35 : eROSITA study of 47 Tuc globular cluster (Saeedi+, 2022)
 J/A+A/661/A38 : SRG/ART-XC 1st year all-sky X-ray survey (Pavlinsky+, 2022)
 J/A+A/661/A40 : eRASS1 X-ray sources Sco-Cen members (Schmitt+, 2022)
 J/A+A/661/A44 : A first eROSITA view of ultracool dwarfs (Stelzer+, 2022)

    https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/ :
     eROSITA-DE early data release catalogues

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 15  A15   ---       ID        Cluster ID
  17- 21  I5    ---       ID-SRC    Unique source ID
  23- 28  F6.3 10-7cm-6   n0        n0 parameter of Vikhlinin06 model (1)
  30- 35  F6.3 10-7cm-6 e_n0        Minus side error bar on n0
  37- 43  F7.3 10-7cm-6 E_n0        Plus side error bar on n0
  45- 49  F5.1  arcsec    rs        rs parameter of Vikhlinin06 model (1)
  51- 55  F5.1  arcsec  e_rs        Minus side error bar on rs
  57- 61  F5.1  arcsec  E_rs        Plus side error bar on rs
  63- 66  F4.2  ---       epsilon   epsilon parameter of Vikhlinin06 model (1)
  68- 71  F4.2  ---     e_epsilon   Minus side error bar on epsilon
  73- 76  F4.2  ---     E_epsilon   Plus side error bar on epsilon
  78- 82  F5.3  ---       beta      beta parameter of Vikhlinin06 model (1)
  84- 88  F5.3  ---     e_beta      Minus side error bar on beta
  90- 94  F5.3  ---     E_beta      Plus side error bar on beta
  96- 99  F4.2  ---       alpha     alpha parameter of Vikhlinin06 model (1)
 101-104  F4.2  ---     e_alpha     Minus side error bar on alpha
 106-109  F4.2  ---     E_alpha     Plus side error bar on alpha
--------------------------------------------------------------------------------
Note (1): Vikhlinin et al. (2006ApJ...640..691V).
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units          Label      Explanations
--------------------------------------------------------------------------------
   1- 15  A15   ---            ID         Cluster ID (HHMMSS.s+DDMMSS)
  17- 21  I5    ---            ID-SRC     Unique source ID
  23- 30  F8.4  deg            RAdeg      Right ascension (J2000.0)
  32- 38  F7.4  deg            DEdeg      Declination (J2000.0)
  40- 44  F5.1  ---            ExtLike    Extent likelihood
  46- 51  F6.1  ---            DetLike    Detection likelihood
  53- 57  F5.3  ---            z          Redshift
  59- 64  F6.3  arcmin         R500       R500
  66- 71  F6.3  keV            T500       Temperature within R500
  73- 78  F6.3  keV          e_T500       Minus side error bar on T500
  80- 85  F6.3  keV          E_T500       Plus side error bar on T500
      87  A1    ---          l_Lx500      Limit flag on Lx500
  88- 94  F7.3  10+35W         Lx500      Soft band lumin within R500
                                           (in 10^42^erg/s)
  96-102  F7.3  10+35W       e_Lx500      ? Minus side error bar on Lx500
 104-111  F8.3  10+35W       E_Lx500      ? Minus side error bar on Lx500
     113  A1    ---          l_Lbol500    Limit flag on Lbol500
 114-121  F8.3  10+35W         Lbol500    Bolometric lumin within R500
 123-130  F8.3  10+35W       e_Lbol500    ? Minus side error bar on Lbol500
 132-139  F8.3  10+35W       E_Lbol500    ? Plus side error bar on Lbol500
     141  A1    ---          l_Mgas500    Limit flag on Mgas500
 142-148  F7.4 10+12Msun       Mgas500    Gas mass within R500
 150-157  F8.4 10+12Msun     e_Mgas500    ? Minus side error bar on Mgas500
 159-166  F8.4 10+12Msun     E_Mgas500    ? Plus side error bar on Mgas500
     168  A1    ---          l_Yx500      Limit flag on Yx500
 169-177  F9.4 10+12keV.Msun   Yx500      Yx (gas mass times mean X-ray spectral
                                           temperature) within R500
 179-187  F9.4 10+12keV.Msun e_Yx500      ? Minus side error bar on Yx500
 189-197  F9.4 10+12keV.Msun E_Yx500      ? Plus side error bar on Yx500
 199-204  F6.3  keV            Tcex500    Temperature within 0.15*R500-R500
 206-211  F6.3  keV          e_Tcex500    Minus side error bar on Tcex500
 213-218  F6.3  keV          E_Tcex500    Plus side error bar on Tcex500
     220  A1    ---          l_Lxcex500   Limit flag on Lxcex500
 221-227  F7.3  10+35W         Lxcex500   Soft band lumin within 0.15*R500-R500
 229-235  F7.3  10+35W       e_Lxcex500   ? Minus side error bar on Lxcex500
 237-244  F8.3  10+35W       E_Lxcex500   ? Plus side error bar on Lxcex500
     246  A1    ---          l_Lbolcex500 Limit flag on Lbolcex500
 247-257  F11.3 10+35W         Lbolcex500 Bolometric lumin within 0.15*R500-R500
 259-265  F7.2  10+35W       e_Lbolcex500 ? Minus side error bar on Lbolcex500
 267-274  F8.3  10+35W       E_Lbolcex500 ? Plus side error bar on Lbolcex500
 276-279  I4    s              Texp       Vignetted exposure at cluster center
--------------------------------------------------------------------------------

Acknowledgements:
    Y. Emre Bahar, ebahar(at)mpe.mpg.de

References:
    The Early Data Release of eROSITA and Mikhail Pavlinsky ART-XC on the
    SRG mission, 2022, A&A, 661, A1-A46

================================================================================
(End) Y. Emre Bahar [MPE, Germany], Patricia Vannier [CDS]           21-Jan-2022
