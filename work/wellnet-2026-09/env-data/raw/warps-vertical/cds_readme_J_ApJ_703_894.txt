J/ApJ/703/894       PNe in 5 nearby galaxies                 (Herrmann+, 2009)
================================================================================
Planetary nebulae in face-on spiral galaxies.
II. Planetary nebula spectroscopy.
    Herrmann K.A., Ciardullo R.
   <Astrophys. J. 703, 894 (2009)>
   =2009ApJ...703..894H
================================================================================
ADC_Keywords: Galaxies, nearby ; Planetary nebulae ; Spectroscopy
Keywords: galaxies: individual (IC 342, NGC 628, NGC 5236, NGC 4736, NGC 5457) -
          galaxies: kinematics and dynamics - galaxies: spiral -
          planetary nebulae: general

Abstract:
    As the second step in our investigation of the mass-to-light ratio of
    spiral disks, we present the results of a spectroscopic survey of
    planetary nebulae (PNe) in five nearby, low-inclination galaxies:
    IC 342, M74 (NGC 628), M83 (NGC 5236), M94 (NGC 4736), and M101
    (NGC 5457). Using 50 setups of the WIYN/Hydra and Blanco/Hydra
    spectrographs, and 25 observations with the Hobby-Eberly Telescope's
    Medium Resolution Spectrograph, we determine the radial velocities of
    99, 102, 162, 127, and 48 PNe, respectively, to a precision better
    than 15km/s. Although the main purpose of this data set is to
    facilitate dynamical mass measurements throughout the inner and outer
    disks of large spiral galaxies, our spectroscopy has other uses as
    well. Here, we co-add these spectra to show that, to first order, the
    [OIII] and Balmer line ratios of PNe vary little over the top ~1.5mag
    of the PN luminosity function.

Description:
    Our Hydra spectroscopy in the north was performed with the 3.5m WIYN
    telescope at Kitt Peak during six separate runs between 2003 March and
    2007 November. For our southern (M83) observations, we used the
    version of Hydra on the CTIO 4m Blanco telescope. Finally, to
    supplement our Hydra observations, we targeted some of the M101 PNe
    with the Medium Resolution Spectrograph of the queue-scheduled HET.

Objects:
    -----------------------------------------------
       RA   (2000)   DE      Designation(s)
    -----------------------------------------------
    03 46 49.1   +68 05 47   IC 342 = UGC 2847
    14 03 12.5   +54 20 53   M101   = NGC 5457
    01 36 41.8   +15 47 00   M74    = NGC 628
    13 37 00.8   -29 51 59   M83    = NGC 5236
    12 50 52.6   +41 07 09   M94    = NGC 4736
    -----------------------------------------------

File Summary:
--------------------------------------------------------------------------------
 FileName  Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe        80        .   This file
table4.dat   131      774   Planetary nebula identifications
table6.dat    52       19   NGC 5068 planetary nebula candidates
table7.dat    47       71   NGC 6946 planetary nebula candidates
IC342/*        .      111   Spectra of IC 342 planetary nebulae, as fits files
M74/*          .      118   Spectra of M 74 planetary nebulae, as fits files
M83/*          .      209   Spectra of M 83 planetary nebulae, as fits files
M94/*          .      138   Spectra of M 94 planetary nebulae, as fits files
M101/*         .       64   Spectra of M 101 planetary nebulae, as fits files
-------------------------------------------------------------------------------

See also:
          V/127  : MASH Catalogues of Planetary Nebulae (Parker+ 2006-2008)
  J/ApJS/38/351  : M31 planetary nebulae (Ford+, 1978)
  J/ApJ/479/231  : PNe in M51, M96 + M101 (Feldmeier+ 1997)
  J/ApJ/492/62   : M87 planetary nebulae (Ciardullo+, 1998)
  J/A+A/336/667  : PN abundances in five galaxies (Stasinska+ 1998)
  J/ApJ/577/31   : PNe in six galaxies (Ciardullo+, 2002)
  J/ApJ/635/290  : PNe and stellar kinematics in NGC1344 (Teodorescu+, 2005)
  J/AJ/131/2089  : Planetary nebulae in NGC 3379 and NGC 3384 (Sluis+, 2006)
  J/ApJ/657/76   : Planetary nebula candidates in 3 galaxies (Feldmeier+, 2007)
  J/ApJ/664/257  : NGC 3379 planetary nebula catalog (Douglas+, 2007)
  J/ApJS/175/522 : Catalog of PNe in NGC 4697 (Mendez+, 2008)
  J/A+A/495/447  : PN and HII regions in NGC6822 (Hernandez-Martinez+, 2009)

Byte-by-byte Description of file: table[467].dat
--------------------------------------------------------------------------------
   Bytes Format Units  Label  Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---    ID     Source identification (1)
  12- 13  I2    h      RAh    Hour of Right Ascension (J2000)
  15- 16  I2    min    RAm    Minute of Right Ascension (J2000)
  18- 22  F5.2  s      RAs    Second of Right Ascension (J2000)
      24  A1    ---    DE-    Sign of the Declination (J2000)
  25- 26  I2    deg    DEd    Degree of Declination (J2000)
  28- 29  I2    arcmin DEm    Arcminute of Declination (J2000)
  31- 34  F4.1  arcsec DEs    Arcsecond of Declination (J2000)
  36- 40  F5.2  mag    m5007  The [O III] (5007) emission line magnitude
      42  A1    ---  l_R      Limit flag on R
  44- 47  F4.2  ---    R      ? I({lambda}5007)_0_/I(H{alpha}+[N II])_0_ ratio
  49- 52  F4.2  ---  e_R      ? The 1{sigma} error in R
  54- 57  A4    ---  n_R      ? Type of R value given (L, Phot, Spect)
      59  I1    ---    Ntrg   ? Number of Hydra+MRS setups where PN was targeted
      61  I1    ---    Ndet   ? Number of Hydra+MRS setups where PN was detected
  63- 68  F6.1  km/s   RV     ? Radial velocity
  70- 73  F4.1  km/s e_RV     ? The 1{sigma} error in RV
  75- 77  A3    ---    Rem    Remarks (2)
  79-104  A26   ---    File   Name of the FITS file of the spectrum
 106-131  A26   ---    File2  Name of the second FITS file of the spectrum
--------------------------------------------------------------------------------
Note (1): Names as IC 342-NNN, M101-NNN, M74-NNN, M83-NNN and M94-NNN
Note (2): Notes as follows:
      C = Emission line is a blend of multiple components, dominated by a
          low-excitation contaminant
      D = PN first measured by Douglas et al. (2000MNRAS.316..795D)
      H = Spectrum likely that of a nearby H II region
      P = Emission line is a blend of multiple components, dominated by the PN
      S = Emission line blend: velocity obtained by subtracting off the 
          low excitation component
      U = Not part of our analysis: {sigma}_v_>15km/s
      X = Velocity and uncertainty derived from xcsao
--------------------------------------------------------------------------------

Acknowledgements:
   Kim Herrmann, herrmann(at)lowell.edu
================================================================================
(End)                                     Francois Ochsenbein [CDS]  18-Aug-2009
