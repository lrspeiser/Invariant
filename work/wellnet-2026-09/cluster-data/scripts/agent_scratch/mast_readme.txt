This readme refers to the MAST HLSP: HFF-DeepSpace
MAST webpage:  https://archive.stsci.edu/hlsp/hff-deepspace
Refer to this HLSP with DOI:  TBD

#Introduction 

This HLSP presents the data release of HFF-DeepSpace, a program funded by
NSF and STScI to produce multi-wavelength photometric catalogs of the 12
cluster and parallel field data of the Hubble Frontier Fields, and catalogs of
photometric redshifts, stellar population properties, and lensing magnification
factors. 

Full details of the image processing, matching, and photometry can be found in
the catalog release paper by H. Shipley et al. (2018). Please cite this paper
when using the catalogs of high-level science products in this HFF-DeepSpace
release. The original data release was made available at the following URL,
which also provides additional details and data access options:
 http://cosmos.phy.tufts.edu/~danilo/HFF/Home.html

HFF-DeepSpace is supported by the National Science Foundation under grant
No. 1513473 and by grant HST-RA-14302 awarded by the Space Telescope Science
Institute.

When using products from HFF-DeepSpace, please include the following
acknowledgement: ``This work is based on data and catalog products from
HFF-DeepSpace, funded by the National Science Foundation and Space Telescope
Science Institute (operated by the Association of Universities for Research
in Astronomy, Inc., under NASA contract NAS5-26555).'' and cite Shipley et al.
(2018).

#Data Products

Data files are hosted at the parent directory: https://archive.stsci.edu/hlsps/hff-deepspace/

===============================================================================

- hlsp_hff-deepspace_hst_acs-wfc3_all_multi_v1_eazy-files.tar.gz

  This file contains the template set, the prior, the template error function,
  and the filter file adopted in EAZY to derive the photometry redshifts and
  the rest-frame luminosities. 

===============================================================================

- hlsp_hff-deepspace_hst_acs-wfc3_all_multi_v1_fast-files.tar.gz

  This file contains the FAST translate and parameter files used in FAST to
  derive the stellar population properties (e.g., stellar mass).

===============================================================================

For each pointing (ABELL2744, MACS0416, MACS0717, MACS1149, ABELLS1063, and
ABELL370; cluster and parallel), we release the following, where an example of {root} 
for the cluster pointing of Abell1063 is
"hlsp_hff-deepspace_hst_acs-wfc3_abell1063_" and for the parallel field
is is "hlsp_hff-deepspace_hst_acs-wfc3_abell1063-par_".

1. {root}_catalogs.tar [200-400 MB depending on the field] contains:

   i. hffds_macs1149clu_v3.9.cat	ASCII file with the photometry catalog
   ii. hffds_macs1149clu_v3.9.cat.save	IDL binary with the photometry catalog

   Fluxes are total f_nu, with m_AB=-2.5*alog10(f)+25
   Fluxes have been corrected for MW extinction and ZP calibrated
   use_phot=1 is the minimum quality cut to apply - additional quality cuts
   should be applied depending on the scientific application of the catalogs.

   Depending on the field, the photometry is given in a combination of the
   following filters: f225w, f275w, 336w, f390w, f435w, f475w, f555w, f606w,
   f625w, f775w, f814w, f850lp, f105w, f110w, f125w, f140w, f160w, Ks, ch1,
   ch2, ch3, ch4. 

   iii. macs1149_bcgs_out_cat_data.dat	table with filters/images information
   
   iv. EAZY output directory, including photometric redshifts and rest-frame
       luminosities in the following filters: Johnson-Cousins B and R;
       Johnson-Morgan B; Johnson U, B and V; SDSS ugriz; 2MASS J, H and K;
       UV 1600 and 2800; Tophat 1400, 1700, 2200, 2700 and 2800.

       The EAZY output .zout contains "z_peak", preferred choice of photometric
       redshift. Rest-frame luminosities (in the EAZY output files *.rf) are
       calculated at the spectroscopic redshift when available or at z_peak.

       Rest-frame absolute magnitudes from the LXXX luminosities in the .rf
       files are calculated using:
       M_XXX(AB) = -2.5*alog10(LXXX)+25-DM
       where DM is the distance modulus also listed in the .rf file 

   v. FAST output directory, including stellar population properties and best
      fit models. The stellar population properties (stellar mass, star
      formation rate, stellar age, extinction, tau, and metallicity) are listed
      in the .fout file. The best-fit FAST models can be taken from
      BEST_FITS.zip (ASCII files) or from {field}_v3.9_FASTseds.save (IDL
      binary format). 


   vi. Gravitational lensing magnification factors for all detected sources in
       the cluster pointings. The file {field}_errmodels.lmf contains the
       best, average, median, and lower/upper 68%-iles of the lensing factors
       from all publicly released lensing models.  The file
       {field}_errzphot.lmf contains upper/lower 68%-ile values of the lensing
       factors including photometric redshift uncertainties. 

2. {root}_misc.tar [1.9-4.7 GB depending on the field] contains the following directories:

   i. completeness
      This directory provides the point-source completenesss curve as a
      function of magnitudes in the F160W and F814W bands. The completeness
      curves are provided in the ASCII files {field}_{filter}_completeness.dat.
      For the completeness curves, mag is in AB; "no" means not allowing overlap
      of injected stars with detected sources; "overlap" means allowing overlap.
      The images were divided in three regions, which are specified in the
      associated maps {field}_{filter}_reg_map.fits. The completeness were
      derived by allowing injected stars to overlap/not overlap with the
      detected sources. Therefore, the completeness curves derived by allowing
      for overlap include the effect of blending.
      Example file name extensions:
      _completeness.dat
      _reg-map.fits

   ii. images/bcgs_models
       This directory provides the stacked models of all bright cluster galaxies
       modeled out from the original images.
       Example file name extensions:
       _bcgs-model.fits

   iii. images/bcgs_out
       	This directory provide the images after subtraction of the bright
	cluster galaxies modeled out from the original images. Images are also
 	background subtracted, cosmic ray cleaned, and weight masked. Weight
 	images are identical to original weight images other than the applied 
	weight mask.
	Example file name extensions:
	_bcgs-out-drz.fits
	_bcgs-out-wht.fits

   iv. images/detection
       This directory includes the F814W+F105W+F125W+F140W+F160W detection
       image. We also provide the weighted mean of the images of the filters
       used to construct the detection image, as well as the corresponding
       weight, variance, error, and mask images. The detection_img =
       weighted_mean / error. For the binary weight mask, a weight <= 1 = 0,
       whereas a weight > 1 = 1.
       Example file name extensions:
       _bcgs-out-det.fits
       _bcgs-out-det-err.fits
       _bcgs-out-det-mask.fits
       _bcgs-out-det-var.fits
       _bcgs-out-det-mean.fits
       _bcgs-out-det-wht.fits

   v. images/original
       This directory provides the original image files, as well as a table
       with the FWHMs of the individual images.
       Example file name extensions:
       _drz.fits
       _wht.fits
       _fwhm.lis
       For some filters, there are multiple epochs.  Science images were combined
        by taking the weighted mean of the background subtracted science images
	for each epoch. Weight images were combined by coadding the weight images
	of each epoch. In these cases, the file names for the drz and wht files are:
       *<target>-epoch1_*_drz.fits (for the first epoch)
       *<target>-epoch2_*_drz.fits (for the second epoch)
       *<target>-epoch1_*_wht.fits (for the first epoch)
       *<target>-epoch2_*_wht.fits (for the second epoch)
       *_bkg-drz.fits (for the epoch-combined)
       *_bkg-wht.fits (for the epoch-combined)

   vi. images/psf_matched
       This directory provides the PSF-matched, bright cluster galaxies
       subtracted, cosmic ray cleaned science images.
       Example file name extensions:
       _psf-bkg-drz.fits

   vii. kernels
   	This directory provides the kernel derived to PSF match all HST images
	to the F160W, the F160W to the K-band image, the F814W to the K-band
	image, the F160W to the IIRAC images, and the F814W to the IRAC images.
	The directory also provides figures showing the quality of the
	PSF-matching process. The convolution kernels are 69x69 pixels, except
	for the IRAC which are 17x17 pixels, and used I. Labbé's deconvolution
	code, which fits a series of Gaussian-weighted Hermite polynomials to
	the Fourier transform of the stacked stars, to find the kernel that
	convolves each filter's PSF to match the f160w PSF.
	The curve of growth plots show the PSFs before and after being convolved
	using their respective kernels.
	Example file name extensions:
	_kernel-<ref-filter>.fits (example, _kernel-f160w.fits)
	_cog-<cogtype>.eps (example, _cog-clash.eps)
	

   viii. photometry
   	 This directory provides the Source Extractor file used to run SExtractor in
	 dual mode. The directory provides the output .cat files, as well as
	 the segmentation map and the detected-objected subtracted detection
	 image. Source Extractor outputs are available for each filter and the
	 detection image.
	 Example file name extensions:
	 _bcgs-out-det.cat
	 _bcgs-out-det.sex
	 _bcgs-out-det-seg.fits
	 _bcgs-out-det-objsub.fits
	 _bcgs-out.sex
	 _bcgs-out.cat

   ix. star_psfs
       This directory provides the constructed stars in each filter used to
       construct the kernels for the PSF-matching process. Also provided is
       the growth curve of the F160W PSF. A grid of 69x69 pixels is used,
       except for Spitzer that uses a 17x17 pixel grid. The PSFs are the
       stacked weighted mean of a number of point sources across the image.
       Example file name extensions:
       _bcgs-out-psf.fits
       _bcgs-out-cog.dat
       _bcgs-out-cog.eps

3. {root}_rgbs.tar [100-150 MB depending on the field] contains RGB color images of the fields.
   Color images are constructed using both the original images and those from
   which the bright cluster galaxies have been subtracted. Color images are
   constructed using F435W+F606W+F814W and F814W+F125W+F160W. 
