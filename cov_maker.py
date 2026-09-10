# %%
# Env
import dotenv
dotenv.load_dotenv()

import sys, os
dir_head = os.environ.get("REPO_DIR", ".")
sys.path.insert(0, os.path.join(dir_head, "src"))

import numpy as np
from pixell import enmap

import warnings
import itertools
from scipy.ndimage import gaussian_filter
import multiprocessing
from functools import partial

warnings.filterwarnings("ignore")

# Files from the src dir in git
import utils as ut
import covariance as cov

import matplotlib.pyplot as plt


def process_combo_cov(combo, 
                      config_data,
                      geometry,
                      cov_dir,
                      plot_1d=False, 
                      plot_2d=False, 
                      save_cov=True):
    freq1 = combo[0].split('_')[0]
    array1 = combo[0].split('_')[1]
    inst1 = combo[0].split('_')[2]
    scan1 = combo[0].split('_')[3]

    freq2 = combo[1].split('_')[0]
    array2 = combo[1].split('_')[1]
    inst2 = combo[1].split('_')[2]
    scan2 = combo[1].split('_')[3]

    if inst1 == 'act':
        if scan1 == 'dr6v2':
            data_dir1 = os.getenv("ACT_DATADIR")
        elif scan1 == 'dr6v4':
            #data_dir1 = "/fs/lustre/scratch/agill/dr6v4/"
            data_dir1 = "/home/gill/research/ACT/bridge/dr6v4/"
        else: 
            raise ValueError("Invalid scan type")
    
    elif inst1 == 'planck':
        data_dir1 = os.getenv("PLANCK_DATADIR")
    
    if inst2 == 'act':
        if scan2 == 'dr6v2':
            data_dir2 = os.getenv("ACT_DATADIR")
        elif scan2 == 'dr6v4':
            #data_dir2 = "/fs/lustre/scratch/agill/dr6v4/"
            data_dir2 = "/home/gill/research/ACT/bridge/dr6v4/"
        else: 
            raise ValueError("Invalid scan type")
    
    elif inst2 == 'planck':
        data_dir2 = os.getenv("PLANCK_DATADIR")

    print("Calculating covariance for: {}".format(combo))

    data_shape, data_wcs = geometry
    debug_pix_noise = (False, False)

    mean_tpsd, mean_npsd, mean_spsd, all_regions_npsd, all_regions_spsd = cov.get_covariance(freq1=freq1, 
                                                                                             freq2=freq2,
                                                                                             array1=array1, 
                                                                                             array2=array2,
                                                                                             inst1=inst1, 
                                                                                             inst2=inst2,
                                                                                             data_dir1=data_dir1,
                                                                                             data_dir2=data_dir2,
                                                                                             data_wcs=geometry[1],
                                                                                             cf=config_data,
                                                                                             debug_pix_noise=debug_pix_noise,)

    # apply ell-dependent scaling to the pa4-pa5 noise power spectrum
    if config_data['cluster_region_noise_rescale'] and (combo[0] == combo[1] and (array1 in ['pa4', 'pa5'])):
        print(f"Rescaling cluster region noise for {combo[0]}")
        modlmap = enmap.modlmap(data_shape, data_wcs)

        # two ways of computing the average 1d noise power spectrum:
        # 1. compute 1d noise power spectrum for each region and average them
        # 2. average the 2d noise power spectrum over all regions and compute 1d noise power spectrum
        # they should be the same for auto cases, when all covariances are 
        # b_avg = np.mean(b_regions[:-1], axis=0)
        # l_avg = np.mean(l_regions[:-1], axis=0)

        mean_npsd_regions = enmap.ndmap(np.mean(np.abs(all_regions_npsd[:-1]), axis=0), wcs=geometry[1])
        mean_spsd_regions = enmap.ndmap(np.mean(np.abs(all_regions_spsd[:-1]), axis=0), wcs=geometry[1])
        mean_tpsd_regions = mean_spsd_regions + mean_npsd_regions
        tpsd_cluster = all_regions_npsd[-1] + all_regions_spsd[-1]

        if config_data['rescale_by_transfer']:
            binsize = 200
            b_avg, l_avg = enmap.lbin(map=np.abs(mean_npsd_regions), bsize=binsize)
            b_cluster, l_cluster = enmap.lbin(map=np.abs(all_regions_npsd[-1]), bsize=binsize)

            # 2D fit over all regions except the center
            npsd_2d_regions_fit = cov.fit_one_over_f(l_npsd=l_avg,
                                                    b_npsd=b_avg,
                                                    fit_lmin=500,
                                                    geometry=geometry)
        
            # # 2D fit over the center region
            npsd_center_fit = cov.fit_one_over_f(l_npsd=l_cluster,
                                                b_npsd=b_cluster,
                                                fit_lmin=500,
                                                geometry=geometry)
        
            scale_factor = npsd_center_fit / npsd_2d_regions_fit
            print("Rescaling noise power spectrum by transfer function")
            print(f"{tuple(combo)}: {np.median(scale_factor)}")
            lmin = config_data.get('cluster_region_noise_rescale_lmin', 3000)
            print(f"Rescaling noise power spectrum from ell={lmin}")
            m = modlmap > lmin
            if config_data['apply_transfer_to_tpsd']:
                mean_tpsd = mean_tpsd_regions
                mean_tpsd[m] *= scale_factor[m]
            else:
                mean_npsd = mean_npsd_regions
                mean_npsd[m] *= scale_factor[m]
                mean_tpsd = mean_spsd_regions + mean_npsd
        else:
            print("Rescaling noise power spectrum by number")
            # range we are measuring scaling factor (l>4000)
            m = (modlmap >= 6000) * (modlmap <= 8000)
            # scale_factor = np.median(np.abs(tpsd_cluster[m])/np.abs(mean_tpsd_regions[m]))
            fit_by_tpsd = False
            fit_by_npsd = True
            if fit_by_tpsd:
                scale_factor = np.median(np.abs(tpsd_cluster[m])/np.abs(mean_tpsd_regions[m]))
            elif fit_by_npsd:
                scale_factor = np.mean(np.abs(all_regions_npsd[-1][m])/np.abs(mean_npsd_regions[m]))
            print(f"{tuple(combo)}: {scale_factor}")
            lmin = config_data.get('cluster_region_noise_rescale_lmin', 3000)
            print(f"Rescaling noise power spectrum from ell={lmin}")
            m = modlmap > lmin
            if config_data['apply_transfer_to_tpsd']:
                mean_tpsd = mean_tpsd_regions
                mean_tpsd[m] *= scale_factor
            else:
                mean_npsd = mean_npsd_regions
                mean_npsd[m] *= scale_factor
                mean_tpsd = mean_spsd_regions + mean_npsd

        if config_data['smooth_total']:
            print("Smoothing total power spectrum")
            mean_tpsd = gaussian_filter(input=mean_tpsd, sigma=config_data['smooth_total_pix'])
            mean_tpsd = enmap.ndmap(np.array(mean_tpsd), wcs=data_wcs)

    # rescale planck noise level below pixel scale such that ACT will dominate in those scales
    if config_data['whiten_planck']:
        if (inst1 == 'planck' and (combo[0] == combo[1])):
            modlmap = enmap.modlmap(data_shape, data_wcs)
            # 1. taking mean power from tpsd in the range 5000-6000
            if float(freq1) >= 80:
                limits = [5000, 6000]
            else:
                limits = [2000, 3000]
            m = np.logical_and(modlmap >= limits[0], modlmap <= limits[1])
            mean_var = np.mean(np.abs(mean_tpsd[m]))
            m2 = modlmap>limits[1]
            mean_tpsd[m2] = mean_var
            mean_tpsd = mean_tpsd + 0j
            print(f"whitening {combo} large ell noise with {mean_var:.2e}")

    if config_data['whiten_act']:
        if (inst1 == 'act' and (combo[0] == combo[1])):
            modlmap = enmap.modlmap(data_shape, data_wcs)
            # 1. taking mean power from tpsd in the range 5000-6000
            limits = [6000, 8000]
            m = np.logical_and(modlmap >= limits[0], modlmap <= limits[1])
            mean_var = np.mean(np.abs(mean_tpsd[m]))
            m = modlmap>limits[0]
            mean_tpsd[m] = mean_var
            mean_tpsd = mean_tpsd + 0j
            print(f"whitening {combo} large ell noise with {mean_var:.2e}")

    if plot_1d:
        binsize = 200
        b_npsd, l_npsd = enmap.lbin(map=np.abs(mean_npsd), bsize=binsize)
        b_tpsd, l_tpsd = enmap.lbin(map=np.abs(mean_tpsd), bsize=binsize)
        b_spsd, l_spsd = enmap.lbin(map=np.abs(mean_spsd), bsize=binsize)

        plt.figure(figsize=(7, 3))
        plt.loglog(l_npsd, b_npsd, label="Noise")
        plt.loglog(l_tpsd, b_tpsd, label="Total")
        plt.loglog(l_spsd, b_spsd, label="Signal")
        plt.title("Absolute value of the power spectrum", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid()

    if plot_2d:
        ut.plot_image(image=np.fft.fftshift(np.abs(mean_tpsd)), 
                    title="Total power spectrum",
                    interval_type='zscale')

    if save_cov:
        ofile = f"{cov_dir}/cov_{freq1}_{array1}_{inst1}_{scan1}_{freq2}_{array2}_{inst2}_{scan2}.npy"
        print(f"Saving covariance to {ofile}")
        np.save(file=ofile, arr=mean_tpsd)


def main(config_data_fname, cov_dir=os.environ['COV_DIR'], save_cov=True):
    config_data = ut.get_config_file(config_data_fname)

    cluster_region = ut.get_region(region_center_ra=config_data['region_center_ra'], 
                                region_center_dec=config_data['region_center_dec'],
                                region_width=config_data['region_width'])

    fname = os.getenv("ACT_DATADIR") + "act_cut_dr6v2_pa6_f150_4way_coadd_map_srcfree.fits"

    # plot_1d = 0
    # plot_2d = 0

    data_ref = ut.imap_dim_check(enmap.read_map(fname=fname, box=cluster_region))
    data_shape, data_wcs = data_ref.shape, data_ref.wcs

    combos = list(itertools.product(config_data['data'], config_data['data']))
    num_processes = multiprocessing.cpu_count()
    print(f"Number of processes: {num_processes}")

    with multiprocessing.Pool(processes=num_processes) as pool:
        fun = partial(process_combo_cov, 
                      config_data=config_data,
                      cov_dir=cov_dir,
                      geometry=(data_shape, data_wcs),
                      plot_1d=False, 
                      plot_2d=False, 
                      save_cov=save_cov)
        pool.map(fun, combos)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Make covariance matrices')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--cov', type=str, default=os.environ['COV_DIR'], help='Path to the directory to save the covariances')
    parser.add_argument('--clear', action='store_true', help='Clear the directory before saving the covariances')
    args = parser.parse_args()
    if not os.path.exists(args.cov):
        os.makedirs(args.cov)
    main(args.config, args.cov)