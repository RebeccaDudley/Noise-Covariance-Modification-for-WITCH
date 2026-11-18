# Noise-Covariance-Modification-for-WITCH

## ACT split and coadd cutouts

The checked-in A3016–A3017 cutouts use the [ACT DR6.02 standard maps](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_maps_standard_get.html).
Use the `AA_night` four-way `map_srcfree` and `ivar` products for the matching
array, frequency, and split or coadd. WITCH uses the Stokes I plane of the
source-subtracted maps.

The box below covers the cluster and background regions in both the standard
and generated A3016–A3017 covariance configs. Use the same box for every product.
Keep the output filenames in `base_A3016-3017.yaml` when regenerating its inputs.

```sh
python generate_act_cutouts.py --box 35.90 -42.49 36.78 -41.59 --temperature-only \
  --product /path/to/act_dr6.02_std_AA_night_pa5_f090_4way_coadd_map_srcfree.fits \
    "ACT_splits+coadds/pa5_f090/act_cut_cmb_night_pa5_f090_3pass_4way_coadd_map_srcfree.fits" \
  --product /path/to/act_dr6.02_std_AA_night_pa5_f090_4way_coadd_ivar.fits \
    "ACT_splits+coadds/pa5_f090/act_cut_cmb_night_pa5_f090_3pass_4way_coadd_ivar.fits"
```

Add `--overwrite` to replace existing cutouts. The script crops on the source
pixel grid and writes the adjusted celestial WCS. A thumbnail whose FITS
footprint is centered on (0°, 0°) cannot be used as the source for this field.

Regenerate the four configured covariance arrays after replacing the inputs.
The cluster box sets the tile dimensions in pixels using the first covariance
dataset's coadd grid. Background regions use their configured sky centers and
the same pixel dimensions as the cluster. All input maps must share that pixel grid.

```sh
python cov_maker.py --config covariance_A3016-3017.yaml --cov covariances
```
