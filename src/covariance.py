import numpy as np
from pixell import enmap
#from glob import glob
import warnings
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

import utils as ut
import os

warnings.filterwarnings('ignore')


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

##### new stuff for background cutouts to be same size:#####
def center_crop_square(imap, target_side):
    """
    Center-crop a 2D enmap to a square of shape
    (target_side, target_side).
    """

    ny, nx = imap.shape[-2:]

    if ny < target_side or nx < target_side:
        raise ValueError(
            f"Map shape {imap.shape} is smaller than "
            f"requested square size {target_side}"
        )

    y0 = (ny - target_side) // 2
    x0 = (nx - target_side) // 2

    return imap[
        y0:y0 + target_side,
        x0:x0 + target_side,
    ]

def read_square_map(fname, region, target_side):
    """
    Read a region from a map and center-crop it to the
    common square covariance geometry.
    """

    imap = ut.imap_dim_check(
        enmap.read_map(
            fname,
            box=region,
        )
    )

    return center_crop_square(
        imap,
        target_side,
    )
###############################################

def get_covariance(ds1,
                   ds2,
                   data_wcs,
                   repo_dir,
                   cf, 
                   debug_pix_noise=(False, False)):

    freq1 = str(ds1["frequency"])
    array1 = ds1["array"]
    inst1 = ds1["instrument"]
    
    freq2 = str(ds2["frequency"])
    array2 = ds2["array"]
    inst2 = ds2["instrument"]

    region_config = cf["covariance_regions"]
    regions = ut.get_all_regions(region_config)
    
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

    # We want the coadd maps to be in Jy/sr
    # We want the ivar maps to be in 1 / (Jy/sr)^2
    flux_factor1 = ut.flux_factor(array1, freq1) # Jy/uK/sr
    flux_factor2 = ut.flux_factor(array2, freq2) # Jy/uK/sr

    all_regions_npsd, all_regions_spsd = [], []

    if cf['apply_region_weight']:
        print("Applying region weights.")
        # computing weights per region for combining the noise power spectra across regions
        coadd1_ivar_list, coadd2_ivar_list = [], []
        for region in regions:
            coadd1_ivar = ut.imap_dim_check(enmap.read_map(coadd_ivar1, box=region)) / flux_factor1**2
            coadd2_ivar = ut.imap_dim_check(enmap.read_map(coadd_ivar2, box=region)) / flux_factor2**2
            coadd1_ivar_list.append(coadd1_ivar)
            coadd2_ivar_list.append(coadd2_ivar)
        sum_coadd1_ivar = np.sum(coadd1_ivar_list[:-1], axis=0)
        sum_coadd2_ivar = np.sum(coadd2_ivar_list[:-1], axis=0)

    # Loop over all regions (now regular noise and signal cov calculations)
    for _, region in enumerate(regions):

        coadd1_map = ut.imap_dim_check(enmap.read_map(coadd_ivar1, box=region)) * flux_factor1
        coadd2_map = ut.imap_dim_check(enmap.read_map(coadd_ivar2, box=region)) * flux_factor2

        if cf['apply_region_weight']:
            coadd1_ivar = ut.imap_dim_check(enmap.read_map(coadd_ivar1, box=region)) / flux_factor1**2
            coadd2_ivar = ut.imap_dim_check(enmap.read_map(coadd_ivar2, box=region)) / flux_factor2**2
            region_weight = np.mean(coadd1_ivar * coadd2_ivar / (sum_coadd1_ivar*sum_coadd2_ivar))

        ivars1_list, ivars2_list = [], []
        
        for j in range(len(ivars1)):
            ivar_map1 = ut.imap_dim_check(enmap.read_map(ivars1[j], box=region)) / flux_factor1**2
            ivars1_list.append(ivar_map1)

        for j in range(len(ivars2)):
            ivar_map2 = ut.imap_dim_check(enmap.read_map(ivars2[j], box=region)) / flux_factor2**2
            ivars2_list.append(ivar_map2)

        sum_ivar1 = np.sum(ivars1_list, axis=0)
        sum_ivar2 = np.sum(ivars2_list, axis=0)

        # NOISE COVARIANCE
        noise_all_splits_loop = []
       
        if cf['noise_type'] == 'all':  # Calc noise cov for all arrays
            noise_type_bool = True
            max_split_number = min(len(splits1), len(splits2))
        
        elif cf['noise_type'] == 'inst':  # Calc noise cov for same instruments only
            noise_type_bool = (inst1 == inst2)
            max_split_number = min(len(splits1), len(splits2))      
        
        elif cf['noise_type'] == 'array':  # Calc noise cov for same arrays only
            noise_type_bool = (array1 == array2)
            max_split_number = min(len(splits1), len(splits2))
        
        elif cf['noise_type'] == 'array_freq':  # Calc noise cov for same arrays only
            noise_type_bool = (array1 == array2) and (freq1 == freq2)
            max_split_number = min(len(splits1), len(splits2))     

        else: 
            raise ValueError("Invalid noise covariance type.")

        if noise_type_bool:
            for idx_split in range(max_split_number):
                
                split1 = ut.imap_dim_check(enmap.read_map(splits1[idx_split], box=region)) * flux_factor1
                split2 = ut.imap_dim_check(enmap.read_map(splits2[idx_split], box=region)) * flux_factor2

                ivar1 = ut.imap_dim_check(enmap.read_map(ivars1[idx_split], box=region)) / flux_factor1**2
                ivar2 = ut.imap_dim_check(enmap.read_map(ivars2[idx_split], box=region)) / flux_factor2**2

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

def get_covariance_sample(ds1,
                          ds2,
                          data_wcs,
                          repo_dir,
                          cf,):

    files1 = get_covariance_files(ds1, repo_dir)
    files2 = get_covariance_files(ds2, repo_dir)

    regions = ut.get_all_regions(cf["covariance_regions"])
    
    data_str_flag = sim_check(cf)
    
    # Exact file paths from YAML
    splits1 = files1["split_maps"]
    splits2 = files2["split_maps"]

    coadd1 = files1["coadd_map"]
    coadd2 = files2["coadd_map"]

    ivars1 = files1["split_ivars"]
    ivars2 = files2["split_ivars"]

    coadd_ivar1 = files1["coadd_ivar"]
    coadd_ivar2 = files2["coadd_ivar"]

    # We want the coadd maps to be in Jy/sr
    # We want the ivar maps to be in 1 / (Jy/sr)^2
    flux_factor1 = ut.flux_factor(array1, freq1) # Jy/uK/sr
    flux_factor2 = ut.flux_factor(array2, freq2) # Jy/uK/sr

    if len(coadd1) == 0 or len(coadd2) == 0: 
        raise ValueError("Length of coadded maps is zero.")

    all_regions_npsd, all_regions_spsd = [], []

    noise_type_bool = False
    max_split_number = min(len(splits1), len(splits2))

    # Loop over all regions (now regular noise and signal cov calculations)
    for _, region in enumerate(regions):

        coadd1_map = ut.imap_dim_check(enmap.read_map(coadd1, box=region)) * flux_factor1
        coadd2_map = ut.imap_dim_check(enmap.read_map(coadd2, box=region)) * flux_factor2

        ivars1_list, ivars2_list = [], []
        
        for j in range(len(ivars1)):
            ivar_map1 = ut.imap_dim_check(enmap.read_map(ivars1[j], box=region)) / flux_factor1**2
            ivars1_list.append(ivar_map1)

        for j in range(len(ivars2)):
            ivar_map2 = ut.imap_dim_check(enmap.read_map(ivars2[j], box=region)) / flux_factor2**2
            ivars2_list.append(ivar_map2)

        sum_ivar1 = np.sum(ivars1_list, axis=0)
        sum_ivar2 = np.sum(ivars2_list, axis=0)

        # NOISE COVARIANCE
        noise_all_splits_loop = []
       
        if cf['noise_type'] == 'all':  # Calc noise cov for all arrays
            noise_type_bool = True
        elif cf['noise_type'] == 'inst':  # Calc noise cov for same instruments only
            noise_type_bool = (inst1 == inst2)
        elif cf['noise_type'] == 'array':  # Calc noise cov for same arrays only
            noise_type_bool = (array1 == array2)
        elif cf['noise_type'] == 'array_freq':  # Calc noise cov for same arrays only
            noise_type_bool = (array1 == array2) and (freq1 == freq2)
        else: 
            raise ValueError("Invalid noise covariance type.")

        if noise_type_bool:
            for idx_split in range(max_split_number):
                
                split1 = ut.imap_dim_check(enmap.read_map(splits1[idx_split], box=region)) * flux_factor1
                split2 = ut.imap_dim_check(enmap.read_map(splits2[idx_split], box=region)) * flux_factor2

                ivar1 = ut.imap_dim_check(enmap.read_map(ivars1[idx_split], box=region)) / flux_factor1**2
                ivar2 = ut.imap_dim_check(enmap.read_map(ivars2[idx_split], box=region)) / flux_factor2**2

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

                weight = np.mean(ivar1_norm * apod_mask1 * ivar2_norm * apod_mask2)

                npsd_region = diff1_fft * np.conjugate(diff2_fft) / weight

                noise_all_splits_loop.append(npsd_region)

            npsd = (1/(max_split_number*(max_split_number-1))) * np.sum(noise_all_splits_loop, axis=0)

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

        # Subtract the noise covariance from the signal cov from each region
        if (noise_type_bool): 
            spsd -= npsd
        
        all_regions_spsd.append(enmap.ndmap(np.array(spsd), wcs=data_wcs))

    # Noise covariance over the central region with random shifts
    # This is to get a better estimate of the noise covariance
    npsd_list = []

    if noise_type_bool:
        iterations = 100
        play_degrees = 0.5

        for idx in range(iterations):
            cluster_cfg = cf["covariance_regions"]["cluster"]
            region_center_ra  = cluster_cfg["region_center_ra"]  + np.random.uniform(-play_degrees, play_degrees)
            region_center_dec = cluster_cfg["region_center_dec"] + np.random.uniform(-play_degrees, play_degrees)
            region_width      = cluster_cfg["region_width"]

            region = ut.get_region(region_center_ra=region_center_ra,
                                   region_center_dec=region_center_dec,
                                   region_width=region_width)    

            coadd1_map = ut.imap_dim_check(enmap.read_map(coadd1[0], box=region)) * flux_factor1
            coadd2_map = ut.imap_dim_check(enmap.read_map(coadd2[0], box=region)) * flux_factor2

            ivars1_list, ivars2_list = [], []

            for j in range(len(ivars1)):
                ivar_map1 = ut.imap_dim_check(enmap.read_map(ivars1[j], box=region)) / flux_factor1**2
                ivars1_list.append(ivar_map1)

            for j in range(len(ivars2)):
                ivar_map2 = ut.imap_dim_check(enmap.read_map(ivars2[j], box=region)) / flux_factor2**2
                ivars2_list.append(ivar_map2)

            sum_ivar1 = np.sum(ivars1_list, axis=0)
            sum_ivar2 = np.sum(ivars2_list, axis=0)

            noise_all_splits_loop = []

            for idx_split in range(max_split_number):
                    
                    split1 = ut.imap_dim_check(enmap.read_map(splits1[idx_split], 
                                                              box=region)) * flux_factor1
                    split2 = ut.imap_dim_check(enmap.read_map(splits2[idx_split], 
                                                              box=region)) * flux_factor2
    
                    ivar1 = ut.imap_dim_check(enmap.read_map(ivars1[idx_split], 
                                                             box=region)) / flux_factor1**2
                    ivar2 = ut.imap_dim_check(enmap.read_map(ivars2[idx_split], 
                                                             box=region)) / flux_factor2**2
    
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
    
                    weight = np.mean(ivar1_norm * apod_mask1 * ivar2_norm * apod_mask2)
    
                    npsd_region = diff1_fft * np.conjugate(diff2_fft) / weight

                    noise_all_splits_loop.append(npsd_region)
            
            npsd = (1/(max_split_number*(max_split_number-1))) * np.sum(noise_all_splits_loop, axis=0)
            npsd = enmap.ndmap(np.array(npsd), wcs=data_wcs)
            npsd_list.append(npsd)

    else:
        npsd = np.zeros_like(coadd1_map)
        npsd_list.append(enmap.ndmap(np.array(npsd), wcs=data_wcs))
    
    mean_npsd_iterators = np.mean(npsd_list, axis=0)
    mean_npsd_iterators = enmap.ndmap(np.array(mean_npsd_iterators), wcs=data_wcs)

    mean_spsd = np.mean(all_regions_spsd[:-1], axis=0)
    mean_spsd = enmap.ndmap(np.array(mean_spsd), wcs=data_wcs)

    mean_npsd = np.mean(all_regions_npsd, axis=0)
    mean_npsd = enmap.ndmap(np.array(mean_npsd), wcs=data_wcs)

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
        # mean_npsd = gaussian_filter(input=mean_npsd, sigma=cf['smooth_noise_pix'])
        mean_npsd_iterators = gaussian_filter(input=mean_npsd_iterators, sigma=cf['smooth_noise_pix'])

    if cf['smooth_signal']:
        mean_spsd = gaussian_filter(input=mean_spsd, sigma=cf['smooth_signal_pix'])
    
    # mean_tpsd_cluster = mean_spsd + mean_npsd
    mean_tpsd_cluster = mean_spsd + mean_npsd_iterators

    if cf['rad_avg_total']:
        mean_tpsd_cluster = enmap.ndmap(np.array(mean_tpsd_cluster), wcs=data_wcs)
        real_rad_tpsd_cluster = ut.rad_avg_2d(image=np.real(mean_tpsd_cluster), 
                                              bsize=cf['rad_avg_bsize'])
        imag_rad_tpsd_cluster = ut.rad_avg_2d(image=np.imag(mean_tpsd_cluster), 
                                              bsize=cf['rad_avg_bsize'])
        mean_tpsd_cluster = real_rad_tpsd_cluster + 1.j*imag_rad_tpsd_cluster

    if cf['smooth_total']:
        mean_tpsd_cluster = gaussian_filter(input=mean_tpsd_cluster, sigma=cf['smooth_total_pix'])

    # Test
   # mean_tpsd_cluster = np.abs(mean_tpsd_cluster)
    
    mean_tpsd_cluster = enmap.ndmap(np.array(mean_tpsd_cluster), wcs=data_wcs)
    
    return mean_tpsd_cluster, mean_npsd_iterators, mean_spsd

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
