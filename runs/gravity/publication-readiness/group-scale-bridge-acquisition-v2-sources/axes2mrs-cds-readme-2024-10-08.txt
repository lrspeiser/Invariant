J/A+A/690/A212    AXES-2MRS extended X-ray galaxy groups catalog (Khalil+, 2024)
================================================================================
AXES-2MRS: A new all-sky catalogue of extended X-ray galaxy groups.
    Khalil H., Finoguenov A., Tempel E., Mamon G.A.
    <Astron. Astrophys. 690, A212 (2024)>
    =2024A&A...690A.212K        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Clusters, galaxy ; X-ray sources
Keywords: galaxies: clusters: general -
          galaxies: clusters: intracluster medium -
          galaxies: groups: general - large-scale structure of Universe

Abstract:
    Understanding baryonic physics at the galaxy-group level is a
    prerequisite for cosmological studies of the large-scale structure.
    One poorly understood aspect of galaxy groups is related to the
    properties of their hot intragroup medium. The well-studied X-ray
    groups have strong cool cores by which they were selected, so
    expanding the selection of groups is currently an important avenue in
    uncovering the diversity within the galaxy group population. We
    present a new all-sky catalogue of X-ray-detected groups (AXES-2MRS)
    based on the identification of large X-ray sources found in the ROSAT
    All-Sky Survey (RASS) with the Two Micron Redshift Survey (2MRS)
    Bayesian Group Catalogue. We studied the basic properties of these
    galaxy groups to gain insights into the effect of different group
    selections on the properties. In addition to X-ray luminosity from
    shallow survey data of RASS, we obtained detailed X-ray properties of
    the groups by matching the AXES-2MRS catalogue to archival X-ray
    observations by XMM-Newton and complemented this by adding the
    published XMM-Newton results on galaxy clusters in our catalogue. We
    analysed temperature and density to the lowest overdensity accessible
    by the data, obtaining hydrostatic mass estimates at a uniform
    overdensity of 10000 times the critical, and comparing them to the
    velocity dispersions of the groups. We explored the relationship
    between X-ray and optical properties of AXES-2MRS groups through the
    sigma_v_-L_X_, sigma_v-kT, kT-L_X_, {sigma}_v-M_, and c_200_-L_X_
    scaling relations. We find a large spread in the central mass
    M_10000_, measured by XMM-Newton, to virial mass M_200_, inferred by
    the velocity dispersion, ratios for galaxy groups. This can either
    indicate that large non-thermal pressure of galaxy groups affects our
    X-ray mass measurements or the effect of a diversity of halo
    concentrations on the X-ray properties of galaxy groups. Previous
    catalogues based on detecting the peak of the X-ray emission
    preferentially sample the high-concentration groups. In contrast, our
    new catalogue uncovered many low-concentration groups, completely
    revising our understanding of X-ray groups.

Description:
    We present a new catalogue of AXES- MRS X-ray galaxy groups that has a
    selection based on the baryonic content at M500.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
axes2mrs.dat     147      558   AXES-2MRS catalogue (table C1)
--------------------------------------------------------------------------------

See also:
 J/A+A/618/A81 : Bayesian group finder applied to the 2MRS data (Tempel+, 2018)

Byte-by-byte Description of file: axes2mrs.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label       Explanations
--------------------------------------------------------------------------------
   1-  8  I8    ---     AXES        Extended X-ray source ID in the AXES
                                     catalogue (GROUP_ID)
  10- 13  I4    ---     Group       2MRS group identification number from
                                     Tempel et al. (2018A&A...618A..81T,
                                     Cat. J/A+A/618/A81) (AXES_ID)
  15- 23  F9.5  deg     RAdeg       X-ray detection right ascension (J2000) (RA)
  25- 33  F9.5  deg     DEdeg       X-ray detection declination (J2000) (DEC)
  35- 37  I3    ---     Nmemb       Number of spectroscopic members in
                                     2MRS group catalogue (NMEM)
  40- 42  I3    ---     NmembClean  ?=-99 Number of spectroscopic members after
                                     the cleaning (NMEM_CLEAN)
  44- 49  F6.4  ---     zsp         2MRS group redshift (ZSPEC)
  51- 60  F10.6 ---     zspClean    Group redshift, assigned using median value
                                     of clean members (ZSPEC_CLEAN)
  62- 72  F11.6 km/s    CluvDispGap Gapper estimate of the cluster velocity
                                     dispersion (CLUVDISP_GAP)
  74- 75  A2    ---     Gaussianity [G NA NG ] Gaussianity, based on the
                                     substructure analysis (GAUSSIANITY) (1)
  77- 88  E12.6 10-7W   LX(0.1-2.4) Luminosity in the (0.1-2.4)keV band of the
                                     cluster, aperture R500c (LX0124)
  90-101  E12.6 10-7W e_LX(0.1-2.4) Uncertainty on LX(0.1-2.4) (ELX)
 103-114  E12.6 mW/m2   F(0.5-2.0)  Galaxy cluster X-ray flux in the
                                     0.5-2.0keV band (FLUX520)
 116-127  E12.6 mW/m2 e_F(0.5-2.0)  Uncertainty on F(0.5-2.0) (EFLUX052)
 129-136  F8.5  arcmin  RE          Apparent radial extent of X-ray emission
                                     (R_E)
 138-147  F10.6 arcmin  R500        R500 radius (R_500)
--------------------------------------------------------------------------------
Note (1): Gaussianity flag as follows:
           G = Gaussian
          NG = non-Gaussian
          NA = not analyzed
--------------------------------------------------------------------------------

Acknowledgements:
     Alexis Finoguenov, alexis.finoguenov(at)helsinki.fi

================================================================================
(End)                                        Patricia Vannier [CDS]  16-Sep-2024
