import numpy as np
from pixell import enmap
from glob import glob
import warnings
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

import utils as ut

warnings.filterwarnings('ignore')

def sim_check(cf):
    """
    Check if the data is simulated or real.

    Parameters
    ----------
    cf : dict
        Configuration dictionary.
    
    Returns
    -------
    data_str_flag : str
        String flag for the data type.
    """
    if cf['system_type'] == 'sim':
        data_str_flag = "_srcfree_model"
    elif cf['system_type'] == 'real':
        data_str_flag = "_srcfree"
    elif cf['system_type'] == 'real_with_sources':
        data_str_flag = ""
    else:
        raise ValueError("Undefined system type.")
    return data_str_flag

# COVARIANCE
def get_covariance(freq1, 
                   freq2, 
                   array1, 
                   array2, 
                   inst1, 
                   inst2,
                   data_wcs, 
                   data_dir1, 
                   data_dir2,
                   cf, 
                   debug_pix_noise=(False, False)):

    regions = ut.get_all_regions(region_center_ra=cf['region_center_ra'],
                                 region_center_dec=cf['region_center_dec'],
                                 region_width=cf['region_width'])
    
    data_str_flag = sim_check(cf)
    
    # File names
    splits1_str = f"{data_dir1}*{array1}*{freq1}*set*map{data_str_flag}.fits"
    splits2_str = f"{data_dir2}*{array2}*{freq2}*set*map{data_str_flag}.fits"

    splits1 = np.sort(glob(splits1_str))
    splits2 = np.sort(glob(splits2_str))

    coadd1_str = f"{data_dir1}*{array1}*{freq1}*coadd*map{data_str_flag}.fits"
    coadd2_str = f"{data_dir2}*{array2}*{freq2}*coadd*map{data_str_flag}.fits"

    coadd1 = np.sort(glob(coadd1_str))
    coadd2 = np.sort(glob(coadd2_str))

    ivars1_str = f"{data_dir1}*{array1}*{freq1}*set*ivar.fits"
    ivars2_str = f"{data_dir2}*{array2}*{freq2}*set*ivar.fits"

    ivars1 = np.sort(glob(ivars1_str))
    ivars2 = np.sort(glob(ivars2_str))

    # We want the coadd maps to be in Jy/sr
    # We want the ivar maps to be in 1 / (Jy/sr)^2
    flux_factor1 = ut.flux_factor(array1, freq1) # Jy/uK/sr
    flux_factor2 = ut.flux_factor(array2, freq2) # Jy/uK/sr

    if len(coadd1) == 0 or len(coadd2) == 0: 
        raise ValueError("Length of coadded maps is zero.")

    all_regions_npsd, all_regions_spsd = [], []

    if cf['apply_region_weight']:
        print("Applying region weights.")
        ivarc1_str = f"{data_dir1}*{array1}*{freq1}*coadd*ivar.fits"
        ivarc2_str = f"{data_dir2}*{array2}*{freq2}*coadd*ivar.fits"
        ivarc1 = np.sort(glob(ivarc1_str))
        ivarc2 = np.sort(glob(ivarc2_str))
        # computing weights per region for combining the noise power spectra across regions
        coadd1_ivar_list, coadd2_ivar_list = [], []
        for _, region in enumerate(regions):
            coadd1_ivar = ut.imap_dim_check(enmap.read_map(ivarc1[0], box=region)) / flux_factor1**2
            coadd2_ivar = ut.imap_dim_check(enmap.read_map(ivarc2[0], box=region)) / flux_factor2**2
            coadd1_ivar_list.append(coadd1_ivar)
            coadd2_ivar_list.append(coadd2_ivar)
        sum_coadd1_ivar = np.sum(coadd1_ivar_list[:-1], axis=0)
        sum_coadd2_ivar = np.sum(coadd2_ivar_list[:-1], axis=0)

    # Loop over all regions (now regular noise and signal cov calculations)
    for _, region in enumerate(regions):

        coadd1_map = ut.imap_dim_check(enmap.read_map(coadd1[0], box=region)) * flux_factor1
        coadd2_map = ut.imap_dim_check(enmap.read_map(coadd2[0], box=region)) * flux_factor2

        if cf['apply_region_weight']:
            coadd1_ivar = ut.imap_dim_check(enmap.read_map(ivarc1[0], box=region)) / flux_factor1**2
            coadd2_ivar = ut.imap_dim_check(enmap.read_map(ivarc2[0], box=region)) / flux_factor2**2
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

def get_covariance_sample(freq1, 
                            freq2, 
                            array1, 
                            array2, 
                            inst1, 
                            inst2,
                            data_wcs, 
                            data_dir1, 
                            data_dir2,
                            cf):

    regions = ut.get_all_regions(region_center_ra=cf['region_center_ra'],
                                 region_center_dec=cf['region_center_dec'],
                                 region_width=cf['region_width'])
    
    data_str_flag = sim_check(cf)
    
    # File names
    splits1_str = f"{data_dir1}*{array1}*{freq1}*set*map{data_str_flag}.fits"
    splits2_str = f"{data_dir2}*{array2}*{freq2}*set*map{data_str_flag}.fits"

    splits1 = np.sort(glob(splits1_str))
    splits2 = np.sort(glob(splits2_str))

    coadd1_str = f"{data_dir1}*{array1}*{freq1}*coadd*map{data_str_flag}.fits"
    coadd2_str = f"{data_dir2}*{array2}*{freq2}*coadd*map{data_str_flag}.fits"

    coadd1 = np.sort(glob(coadd1_str))
    coadd2 = np.sort(glob(coadd2_str))

    ivars1_str = f"{data_dir1}*{array1}*{freq1}*set*ivar.fits"
    ivars2_str = f"{data_dir2}*{array2}*{freq2}*set*ivar.fits"

    ivars1 = np.sort(glob(ivars1_str))
    ivars2 = np.sort(glob(ivars2_str))

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
            region_center_ra =  cf['region_center_ra'] + np.random.uniform(-play_degrees, play_degrees)
            region_center_dec = cf['region_center_dec'] + np.random.uniform(-play_degrees, play_degrees)
            region_width = cf['region_width']

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
