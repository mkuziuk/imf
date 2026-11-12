# Original AS222 papers

Downloaded from Oxford Academic on 13 October 2026 and read before implementing
the comparison. The PDFs retain their publisher watermarks.

| File | Publication | Publisher record |
| --- | --- | --- |
| `hardle_1987_as222.pdf` | W. Härdle, *Resistant Smoothing Using the Fast Fourier Transform*, Applied Statistics 36, 104-111, 1987 | https://academic.oup.com/jrsssc/article/36/1/104/6985332 |
| `dusoir_1988_as_r73.pdf` | Tony Dusoir, *Remark AS R73*, Applied Statistics 37, 316-317, 1988 | https://academic.oup.com/jrsssc/article/37/2/316/6985480 |

AS222 equation 2 specifies the pilot and observation-centered score correction.
Its numerical-method section specifies Gaussian smoothing and its first page
specifies Huber's score. Page 106 explicitly distinguishes observation-centered
residuals from target-centered residuals. AS R73 corrects the retained density
transforms and the FFT length initialization. The new Python comparator uses
explicit kernel spectra and inverse-transform lengths, without the Fortran cache.

The five-sine experiment remains the supplied report's adapted estimator, with
Gaussian scores and Epanechnikov weights. It is not the original paper's random-
design, contaminated-noise experiment. See `experiments/pimf-as222/README.md` for
the reproducible comparison and `output/pimf-as222/results.json` for PDF hashes.
