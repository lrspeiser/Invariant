# Metadata-only foreground hypothesis audit

Read the actual existing bound NGC2976 THINGS cube header, never its science
arrays. Derive zero-based stored channels10/20/30 and neighboring21 through
Astropy spectral WCS; compare with an independent calculation appropriate to
the actual convention. Do not assume FELO-HEL: inspect CTYPE and VELREF first.
Header exploratory reads already occurred before this scope document; this is
not a retrospective preregistration of metadata discovery.

Read primary public papers only: THINGS0810.2125 and Sorgho et al.1903.03767,
published as MNRAS486,504,doi10.1093/mnras/stz696. Document rest-frame and Doppler
convention separately. A numerical foreground interval match is forbidden
unless both are established. No channel deletion, masking, fitting or response
scores. Published systemic metadata is contextual, not a fitting target.

Controls: spectral extraction must preserve FITS index origin; independent
actual-radio linear mapping agrees1e-9m/s, world/pixel roundtrip1e-10pixel;
radio-to-frequency-to-optical reparameterization must agree1e-7km/s with the
analytic expression. It is not a reference-frame transformation.

Preserve initial exploratory probe limitation: the cube has four WCS axes
(including singleton Stokes); a three-coordinate all_pix2world call failed
before any coordinate result. The corrected method extracts WCS.spectral.
No science arrays were opened by either probe.

No new observational raw files; remote paper bytes may be read in memory and
hashed as source evidence. Publication stores only metadata, URLs, hashes and
short paraphrases. SOURCE_BLOCKED remains for foreground attribution and
observational gravity likelihoods if the reference-frame evidence is incomplete.
