import numpy as np
from pixell import enmap
#from glob import glob
import warnings
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

import utils as ut
from map_geometry import (
    covariance_region_geometries,
    require_covariance_coverage,
    sample_covariance_regions,
)
import os

warnings.filterwarnings('ignore')


def radial_power_scale(auto_power, ell):
    """Estimate the positive radial power profile of an auto-spectrum."""
    nonzero = ell > 0
    if not np.any(nonzero):
        return np.ones_like(ell)

    radii = ell[nonzero]
    rings = np.rint(radii / radii.min()).astype(int)
    counts = np.bincount(rings)
    radius_sum = np.bincount(rings, weights=radii)
    power_sum = np.bincount(rings, weights=np.real(auto_power[nonzero]))
    usable = (counts > 0) & np.isfinite(power_sum) & (power_sum > 0)
    if not np.any(usable):
        return np.ones_like(ell)

    log_radius = np.log(radius_sum[usable] / counts[usable])
    log_power = np.log(power_sum[usable] / counts[usable])
    return np.exp(np.interp(np.log(np.maximum(ell, radii.min())), log_radius, log_power))


def smooth_total_covariances(spectra, ell, sigma):
    """Smooth Fourier pixels after dividing out each band's radial power."""
    if not sigma:
        return spectra

    scales = {
        a: np.sqrt(radial_power_scale(spectrum, ell))
        for (a, b), spectrum in spectra.items()
        if a == b
    }
    keep = ell > 0
    weight = gaussian_filter(keep.astype(float), sigma, mode="wrap")
    result = {}
    for (a, b), spectrum in spectra.items():
        scale = scales[a] * scales[b]
        normalized = np.where(keep, spectrum / scale, 0)
        averaged = gaussian_filter(normalized, sigma, mode="wrap")
        np.divide(averaged, weight, out=averaged, where=weight > 0)
        result[a, b] = np.where(keep, averaged * scale, spectrum)
    return result


# COVARIANCE
def get_covariance_files(dataset_cfg, repo_dir):
    """
    Resolve the exact coadd, split, and inverse-variance files
    specified in the dataset's covariance configuration.
    """

    cov_cfg = dataset_cfg["covariance"]

    data_dir = cov_cfg["data_dir"]

    if not os.path.isabs(data_dir):
        data_dir = os.path.join(repo_dir, data_dir)

    data_dir = os.path.abspath(data_dir)

    # Coadd products
    coadd_map = os.path.join(
        data_dir,
        cov_cfg["coadd"]["map"],
    )

    coadd_ivar = os.path.join(
        data_dir,
        cov_cfg["coadd"]["ivar"],
    )

    # Split products
    split_maps = [
        os.path.join(data_dir, split["map"])
        for split in cov_cfg["splits"]
    ]

    split_ivars = [
        os.path.join(data_dir, split["ivar"])
        for split in cov_cfg["splits"]
    ]

    # Basic validation
    if len(split_maps) != len(split_ivars):
        raise ValueError(
            "Number of split maps does not match number "
            "of split inverse-variance maps."
        )

    if len(split_maps) < 2:
        raise ValueError(
            "At least two splits are required to compute "
            "the noise covariance."
        )

    files_to_check = (
        [coadd_map, coadd_ivar]
        + split_maps
        + split_ivars
    )

    for fname in files_to_check:
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"Covariance input file does not exist:\n{fname}"
            )

    return {
        "coadd_map": coadd_map,
        "coadd_ivar": coadd_ivar,
        "split_maps": split_maps,
        "split_ivars": split_ivars,
    }

def read_region_map(filename, geometry):
    return ut.imap_dim_check(enmap.read_map(filename, geometry=geometry))


def noise_covariance_enabled(ds1, ds2, noise_type):
    if noise_type == "all":
        return True
    if noise_type == "inst":
        return ds1["instrument"] == ds2["instrument"]
    if noise_type == "array":
        return ds1["array"] == ds2["array"]
    if noise_type == "array_freq":
        return ds1["array"] == ds2["array"] and str(ds1["frequency"]) == str(ds2["frequency"])
    raise ValueError("Invalid noise covariance type.")


def get_covariance(ds1,
                   ds2,
                   data_wcs,
                   repo_dir,
                   cf, 
                   debug_pix_noise=(False, False),
                   region_geometries=None):

    freq1 = str(ds1["frequency"])
    array1 = ds1["array"]
    
    freq2 = str(ds2["frequency"])
    array2 = ds2["array"]

    files1 = get_covariance_files(ds1,repo_dir,)
    files2 = get_covariance_files(ds2,repo_dir,)

    coadd1 = files1["coadd_map"]
    coadd2 = files2["coadd_map"]

    coadd_ivar1 = files1["coadd_ivar"]
    coadd_ivar2 = files2["coadd_ivar"]

    splits1 = files1["split_maps"]
    splits2 = files2["split_maps"]

    ivars1 = files1["split_ivars"]
    ivars2 = files2["split_ivars"]

    if region_geometries is None:
        sky_regions = ut.get_all_regions(cf["covariance_regions"])
        region_geometries = covariance_region_geometries(coadd1, sky_regions)
        paths = {coadd1, coadd2, coadd_ivar1, coadd_ivar2, *splits1, *splits2, *ivars1, *ivars2}
        for path in sorted(paths):
            require_covariance_coverage(path, region_geometries)

    # We want the coadd maps to be in Jy/sr
    # We want the ivar maps to be in 1 / (Jy/sr)^2
    flux_factor1 = ut.flux_factor(array1, freq1) # Jy/uK/sr
    flux_factor2 = ut.flux_factor(array2, freq2) # Jy/uK/sr

    noise_type_bool = noise_covariance_enabled(ds1, ds2, cf["noise_type"])
    max_split_number = min(len(splits1), len(splits2))
    all_regions_npsd, all_regions_spsd = [], []

    if cf['apply_region_weight']:
        print("Applying region weights.")
        coadd1_ivars = [read_region_map(coadd_ivar1, region) / flux_factor1**2 for region in region_geometries]
        coadd2_ivars = [read_region_map(coadd_ivar2, region) / flux_factor2**2 for region in region_geometries]
        sum_coadd1_ivar = np.sum(coadd1_ivars[:-1], axis=0)
        sum_coadd2_ivar = np.sum(coadd2_ivars[:-1], axis=0)

    for i, region in enumerate(region_geometries):
        coadd1_map = read_region_map(coadd1, region) * flux_factor1
        coadd2_map = read_region_map(coadd2, region) * flux_factor2

        if cf['apply_region_weight']:
            region_weight = np.mean(
                coadd1_ivars[i] * coadd2_ivars[i] / (sum_coadd1_ivar * sum_coadd2_ivar)
            )

        ivars1_list, ivars2_list = [], []
        
        for j in range(len(ivars1)):
            ivar_map1 = read_region_map(ivars1[j], region) / flux_factor1**2
            ivars1_list.append(ivar_map1)

        for j in range(len(ivars2)):
            ivar_map2 = read_region_map(ivars2[j], region) / flux_factor2**2
            ivars2_list.append(ivar_map2)

        sum_ivar1 = np.sum(ivars1_list, axis=0)
        sum_ivar2 = np.sum(ivars2_list, axis=0)

        # NOISE COVARIANCE
        noise_all_splits_loop = []
       
        if noise_type_bool:
            for idx_split in range(max_split_number):
                
                split1 = read_region_map(splits1[idx_split], region) * flux_factor1
                split2 = read_region_map(splits2[idx_split], region) * flux_factor2

                ivar1 = read_region_map(ivars1[idx_split], region) / flux_factor1**2
                ivar2 = read_region_map(ivars2[idx_split], region) / flux_factor2**2
                ################################

                diff1 = split1 - coadd1_map
                diff2 = split2 - coadd2_map

                ivar1_norm = ivar1 / sum_ivar1
                ivar2_norm = ivar2 / sum_ivar2

                diff1_norm = diff1 * ivar1_norm
                diff2_norm = diff2 * ivar2_norm

                apod_mask1 = (enmap.apod(diff1_norm*0+1, cf['apod_pix']))
                diff1_apod = diff1_norm * apod_mask1

                apod_mask2 = (enmap.apod(diff2_norm*0+1, cf['apod_pix']))
                diff2_apod = diff2_norm * apod_mask2

                diff1_fft = np.fft.fft2(diff1_apod)
                diff2_fft = np.fft.fft2(diff2_apod)

                if debug_pix_noise[0] == True:
                    print("Debug: add artificial noise to the high ell region in diff1.")
                    target = diff1_fft
                    modlmap = enmap.modlmap(coadd1_map.shape, coadd1_map.wcs)
                    m1 = np.logical_and(modlmap >= 5000, modlmap <= 6000)
                    m2 = modlmap >= 6000
                    mean_var = np.mean(target[m1]).real
                    target[m2] = mean_var * 1000 + 0.j

                if debug_pix_noise[1] == True:
                    print("Debug: add artificial noise to the high ell region in diff2.")
                    target = diff2_fft
                    modlmap = enmap.modlmap(coadd1_map.shape, coadd1_map.wcs)
                    m1 = np.logical_and(modlmap >= 5000, modlmap <= 6000)
                    m2 = modlmap >= 6000
                    mean_var = np.mean(target[m1]).real
                    target[m2] = mean_var * 1000 + 0.j

                weight = np.mean(ivar1_norm * apod_mask1 * ivar2_norm * apod_mask2)

                npsd_region = diff1_fft * np.conjugate(diff2_fft) / weight

                noise_all_splits_loop.append(npsd_region)

            npsd = (1/(max_split_number*(max_split_number-1))) * np.sum(noise_all_splits_loop, axis=0)

            if cf['apply_region_weight']:
                npsd *= region_weight
            all_regions_npsd.append(enmap.ndmap(np.array(npsd), wcs=data_wcs))
        
        else:
            npsd = np.zeros_like(coadd1_map)
            all_regions_npsd.append(enmap.ndmap(np.array(npsd), wcs=data_wcs))

        # SIGNAL COVARIANCE
        apod_mask1 = (enmap.apod(coadd1_map*0+1, cf['apod_pix']))
        coadd1_map_apod = coadd1_map * apod_mask1
        coadd1_map_norm = np.fft.fft2(coadd1_map_apod) 

        apod_mask2 = (enmap.apod(coadd2_map*0+1, cf['apod_pix']))
        coadd2_map_apod = coadd2_map * apod_mask2
        coadd2_map_norm = np.fft.fft2(coadd2_map_apod) 

        spsd = coadd1_map_norm * np.conjugate(coadd2_map_norm) / np.mean(apod_mask1 * apod_mask2)

        if cf['apply_region_weight']:
            spsd *= region_weight

        # Subtract the noise covariance from the signal cov from each region
        if (noise_type_bool): 
            spsd -= npsd
        all_regions_spsd.append(enmap.ndmap(np.array(spsd), wcs=data_wcs))

    # average over all except the last region
    if cf['apply_region_weight']:
        mean_spsd = np.sum(all_regions_spsd[:-1], axis=0)
        mean_npsd = np.sum(all_regions_npsd[:-1], axis=0)
        if cf['use_cluster_only_for_noise']:
            mean_npsd = all_regions_npsd[-1]
    else:
        mean_spsd = np.mean(all_regions_spsd[:-1], axis=0)
        mean_npsd = np.mean(all_regions_npsd[:-1], axis=0)
        if cf['use_cluster_only_for_noise']:
            mean_npsd = all_regions_npsd[-1]

    mean_spsd = enmap.ndmap(np.array(mean_spsd), wcs=data_wcs)

    if cf['rad_avg_noise']:
        mean_npsd = enmap.ndmap(np.array(mean_npsd), wcs=data_wcs)
        real_rad_npsd_cluster = ut.rad_avg_2d(image=np.real(mean_npsd), bsize=cf['rad_avg_bsize'])
        imag_rad_npsd_cluster = ut.rad_avg_2d(image=np.imag(mean_npsd), bsize=cf['rad_avg_bsize'])
        mean_npsd = real_rad_npsd_cluster + 1.j*imag_rad_npsd_cluster

    if cf['rad_avg_signal']:
        mean_spsd = enmap.ndmap(np.array(mean_spsd), wcs=data_wcs)
        real_rad_spsd_cluster = ut.rad_avg_2d(image=np.real(mean_spsd), bsize=cf['rad_avg_bsize'])
        imag_rad_spsd_cluster = ut.rad_avg_2d(image=np.imag(mean_spsd), bsize=cf['rad_avg_bsize'])
        mean_spsd = real_rad_spsd_cluster + 1.j*imag_rad_spsd_cluster

    if cf['smooth_noise']:
        mean_npsd = gaussian_filter(input=mean_npsd, sigma=cf['smooth_noise_pix'])

    if cf['smooth_signal']:
        mean_spsd = gaussian_filter(input=mean_spsd, sigma=cf['smooth_signal_pix'])
    
    mean_tpsd_cluster = mean_spsd + mean_npsd

    if cf['rad_avg_total']:
        mean_tpsd_cluster = enmap.ndmap(np.array(mean_tpsd_cluster), wcs=data_wcs)
        real_rad_tpsd_cluster = ut.rad_avg_2d(image=np.real(mean_tpsd_cluster), 
                                              bsize=cf['rad_avg_bsize'])
        imag_rad_tpsd_cluster = ut.rad_avg_2d(image=np.imag(mean_tpsd_cluster), 
                                              bsize=cf['rad_avg_bsize'])
        mean_tpsd_cluster = real_rad_tpsd_cluster + 1.j*imag_rad_tpsd_cluster

    if cf['smooth_total']:
        print(f"Smoothing tpsd: {cf['smooth_total_pix']} pix.")
        mean_tpsd_cluster = gaussian_filter(input=mean_tpsd_cluster, sigma=cf['smooth_total_pix'])
    
    # mean_tpsd_cluster = np.abs(mean_tpsd_cluster)
    mean_tpsd_cluster = enmap.ndmap(np.array(mean_tpsd_cluster), wcs=data_wcs)
    mean_npsd = enmap.ndmap(np.array(mean_npsd), wcs=data_wcs)
    mean_spsd = enmap.ndmap(np.array(mean_spsd), wcs=data_wcs)

    return mean_tpsd_cluster, mean_npsd, mean_spsd, all_regions_npsd, all_regions_spsd

def get_covariance_sample(ds1, ds2, data_wcs, repo_dir, cf):
    """Estimate signal from background tiles and noise from shifted cluster tiles."""
    files1 = get_covariance_files(ds1, repo_dir)
    files2 = get_covariance_files(ds2, repo_dir)
    sky_regions = ut.get_all_regions(cf["covariance_regions"])
    regions = covariance_region_geometries(files1["coadd_map"], sky_regions)
    filenames = sorted({
        path
        for files in (files1, files2)
        for path in [files["coadd_map"], *files["split_maps"], *files["split_ivars"]]
    })
    for filename in filenames:
        require_covariance_coverage(filename, regions)

    sampled_regions = []
    if noise_covariance_enabled(ds1, ds2, cf["noise_type"]):
        sampled_regions = sample_covariance_regions(regions[-1], filenames, count=100)

    # Apply postprocessing once, after averaging the intended set of tiles.
    raw_config = dict(
        cf, apply_region_weight=False, use_cluster_only_for_noise=False,
        rad_avg_noise=False, rad_avg_signal=False, rad_avg_total=False,
        smooth_noise=False, smooth_signal=False, smooth_total=False,
    )
    _, _, _, noise, signal = get_covariance(
        ds1, ds2, data_wcs, repo_dir, raw_config,
        region_geometries=regions + sampled_regions,
    )
    mean_signal = enmap.ndmap(np.mean(signal[:len(regions) - 1], axis=0), wcs=data_wcs)
    mean_noise = enmap.ndmap(
        np.mean(noise[len(regions):], axis=0) if sampled_regions else noise[-1],
        wcs=data_wcs,
    )

    if cf['rad_avg_signal']:
        real = ut.rad_avg_2d(image=np.real(mean_signal), bsize=cf['rad_avg_bsize'])
        imag = ut.rad_avg_2d(image=np.imag(mean_signal), bsize=cf['rad_avg_bsize'])
        mean_signal = real + 1j * imag
    if cf['smooth_signal']:
        mean_signal = gaussian_filter(mean_signal, sigma=cf['smooth_signal_pix'])
    if cf['smooth_noise']:
        mean_noise = gaussian_filter(mean_noise, sigma=cf['smooth_noise_pix'])

    total = enmap.ndmap(mean_signal + mean_noise, wcs=data_wcs)
    if cf['rad_avg_total']:
        real = ut.rad_avg_2d(image=np.real(total), bsize=cf['rad_avg_bsize'])
        imag = ut.rad_avg_2d(image=np.imag(total), bsize=cf['rad_avg_bsize'])
        total = real + 1j * imag
    if cf['smooth_total']:
        total = gaussian_filter(total, sigma=cf['smooth_total_pix'])

    return enmap.ndmap(total, wcs=data_wcs), mean_noise, mean_signal


def func(l, l_knee, alpha, white_noise, eps=1):
    return ( (l_knee / (l+eps) )**-alpha + 1 ) * white_noise**2.

def smoothing(l_npsd, 
              b_npsd, 
              twoD_npsd_orig, 
              gauss_smooth_sigma, 
              mask_value, 
              geometry):
    # 1) 2D noise power spectrum in Fourier space (real and imag part): 2d_psd_orig
    # 2) Make a binned version which in 1D for the abs value (2 arrays, psd, and ell)
    # 3) Fit a functional form to the array above (3 parameters: alpha, knee, and white noise)
    # 4) 2D projection of the fitted function form: 2d_psd_fit
    # 5) 2d_residual = 2d_psd_orig / 2d_psd_fit
    # 6) 2d_residual_smooth = SMOOTH(2d_residual)
    # 7) 2d_psd_final = 2d_psd_fit * 2d_residual_smooth
    
    # Step 3: Fit a functional form to the array above (3 parameters: alpha, knee, and white noise)
    x = l_npsd
    y = b_npsd
    mask = x >= mask_value

    x_filtered = x[mask]
    y_filtered = y[mask]

    # Initial guess for parameters
    # Find the average of values between l = 5000 and l = 10000
    white_noise = np.mean(y_filtered[(x_filtered > 5000) & (x_filtered < 10000)])

    p0 = [2000, -3, np.sqrt(white_noise)]  # initial guess for l_knee, alpha, white_noise

    # Fit the curve using filtered data
    params, _ = curve_fit(func, x_filtered, y_filtered, sigma=y_filtered**0.5, p0=p0)

    #print("Fitted parameters: ", params)

    # Step 4: 2D projection of the fitted function form: 2d_psd_fit
    modlmap = enmap.modlmap(*geometry)

    twoD_npsd_fit = func(modlmap, *params)

    # Step 5: 2d_residual = 2d_psd_orig / 2d_psd_fit
    twoD_residual = twoD_npsd_orig / twoD_npsd_fit

    # Step 6: 2d_residual_smooth = SMOOTH(2d_residual)
    if gauss_smooth_sigma > 0:
        twoD_residual_smooth = gaussian_filter(input=twoD_residual, sigma=gauss_smooth_sigma)
    else:
        twoD_residual_smooth = twoD_residual

    # Step 7: 2d_psd_final = 2d_psd_fit * 2d_residual_smooth
    twoD_npsd_smooth = twoD_npsd_fit * twoD_residual_smooth

    return twoD_npsd_smooth


def fit_one_over_f(l_npsd, 
                   b_npsd, 
                   geometry,
                   fit_lmin, 
                   fit_range = [6000, 8000]):
    # 1) 2D noise power spectrum in Fourier space (real and imag part): 2d_psd_orig
    # 2) Make a binned version which in 1D for the abs value (2 arrays, psd, and ell)
    # 3) Fit a functional form to the array above (3 parameters: alpha, knee, and white noise)
    # 4) 2D projection of the fitted function form: 2d_psd_fit
    
    # Step 3: Fit a functional form to the array above (3 parameters: alpha, knee, and white noise)
    x = l_npsd
    y = b_npsd
    mask = x >= fit_lmin

    x_filtered = x[mask]
    y_filtered = y[mask]

    # Initial guess for parameters
    # Find the average of values between l = 5000 and l = 10000
    white_noise = np.mean(y_filtered[(x_filtered > fit_range[0]) & (x_filtered < fit_range[1])])

    p0 = [2000, -3, np.sqrt(white_noise)]  # initial guess for l_knee, alpha, white_noise

    # Fit the curve using filtered data
    params, _ = curve_fit(func, x_filtered, y_filtered, sigma=y_filtered**0.5, p0=p0)

    #print("Fitted parameters: ", params)

    # Step 4: 2D projection of the fitted function form: 2d_psd_fit
    modlmap = enmap.modlmap(*geometry)

    twoD_npsd_fit = func(modlmap, *params)

    return twoD_npsd_fit
