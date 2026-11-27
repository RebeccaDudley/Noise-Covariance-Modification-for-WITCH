# %%
# Env
import dotenv
dotenv.load_dotenv()

import sys, os
dir_head = os.path.abspath(os.environ.get("REPO_DIR", "."))
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
from map_geometry import covariance_region_geometries, require_covariance_coverage

import matplotlib.pyplot as plt


def process_combo_cov(combo, 
                      config_data,
                      geometry,
                      cov_dir,
                      plot_1d=False, 
                      plot_2d=False, 
                      save_cov=True,
                      region_geometries=None,
                      save_reverse=False,
                      return_spectra=False):
    
    datasets = config_data["datasets"]

    name1, name2 = combo

    ds1 = datasets[name1]
    ds2 = datasets[name2]

    array1 = ds1["array"]

    print(f"Calculating covariance for: {combo}")

    data_shape, data_wcs = geometry
    debug_pix_noise = (False, False)

    mean_tpsd, mean_npsd, mean_spsd, all_regions_npsd, all_regions_spsd = cov.get_covariance(ds1=ds1,
                                                                                             ds2=ds2,
                                                                                             data_wcs=geometry[1],
                                                                                             repo_dir=dir_head,
                                                                                             cf=config_data,
                                                                                             debug_pix_noise=debug_pix_noise,
                                                                                             region_geometries=region_geometries)

    # Save the component estimates before the writer's optional rescaling.
    signal_covariance = mean_spsd
    noise_covariance = mean_npsd

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

    mean_tpsd = whiten_total(mean_tpsd, config_data, geometry, ds1, combo)

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

    spectra = {
        "cov": mean_tpsd,
        "signal_cov": signal_covariance,
        "noise_cov": noise_covariance,
    }
    if save_cov:
        save_pair_covariances(combo, config_data, cov_dir, spectra, save_reverse)
    if return_spectra:
        return spectra


def whiten_total(spectrum, config, geometry, dataset, combo):
    """Apply the configured high-ell whitening to an auto-spectrum."""
    instrument = dataset["instrument"]
    if combo[0] != combo[1] or not config.get(f"whiten_{instrument}", False):
        return spectrum
    if instrument not in ("act", "planck"):
        return spectrum

    ell = enmap.modlmap(*geometry)
    limits = (6000, 8000)
    if instrument == "planck":
        limits = (2000, 3000) if float(dataset["frequency"]) < 80 else (5000, 6000)
    low, high = limits
    power = np.mean(np.abs(spectrum[(ell >= low) & (ell <= high)]))
    spectrum = spectrum.copy()
    spectrum[ell > (low if instrument == "act" else high)] = power
    print(f"whitening {combo} large ell noise with {power:.2e}")
    return spectrum + 0j


def save_pair_covariances(combo, config, cov_dir, spectra, save_reverse):
    labels = [
        "{frequency}_{array}_{instrument}_{scan}".format(**config["datasets"][name])
        for name in combo
    ]
    for prefix, spectrum in spectra.items():
        output = os.path.join(cov_dir, f"{prefix}_{labels[0]}_{labels[1]}.npy")
        print(f"Saving covariance to {output}")
        np.save(output, spectrum)

        if save_reverse and combo[0] != combo[1]:
            reverse = os.path.join(cov_dir, f"{prefix}_{labels[1]}_{labels[0]}.npy")
            print(f"Saving covariance to {reverse}")
            np.save(reverse, np.conjugate(spectrum))


def main(config_data_fname, cov_dir=None, save_cov=True):
    config_data = ut.get_config_file(config_data_fname)

    region_config = config_data["covariance_regions"]
    regions = ut.get_all_regions(region_config)
    dataset_names = config_data["covariance_datasets"]
    datasets = config_data["datasets"]
    dataset_files = [
        cov.get_covariance_files(datasets[name], dir_head) for name in dataset_names
    ]
    fname = dataset_files[0]["coadd_map"]
    print(f"Using reference map for geometry: {fname}")
    region_geometries = covariance_region_geometries(fname, regions)
    geometry = region_geometries[-1]
    print(f"Using {len(region_geometries)} covariance tiles with shape {geometry[0]}")

    for files in dataset_files:
        paths = [
            files["coadd_map"], files["coadd_ivar"],
            *files["split_maps"], *files["split_ivars"],
        ]
        for path in paths:
            require_covariance_coverage(path, region_geometries)

    combos = list(itertools.combinations_with_replacement(dataset_names, 2))

    for combo in combos:
        print(f"  {combo}")
  
    num_processes = multiprocessing.cpu_count()
    print(f"Number of processes: {num_processes}")

    smooth_total = config_data["smooth_total"]
    worker_config = config_data
    if smooth_total:
        worker_config = dict(config_data, smooth_total=False, whiten_act=False, whiten_planck=False)

    with multiprocessing.Pool(processes=num_processes) as pool:
        fun = partial(process_combo_cov, 
                      config_data=worker_config,
                      cov_dir=cov_dir,
                      geometry=geometry,
                      region_geometries=region_geometries,
                      plot_1d=False, 
                      plot_2d=False, 
                      save_cov=save_cov and not smooth_total,
                      save_reverse=True,
                      return_spectra=smooth_total)
        pairs = pool.map(fun, combos)

    if smooth_total:
        totals = cov.smooth_total_covariances(
            {pair: spectra["cov"] for pair, spectra in zip(combos, pairs)},
            enmap.modlmap(*geometry),
            config_data["smooth_total_pix"],
        )
        if save_cov:
            for pair, spectra in zip(combos, pairs):
                spectra["cov"] = whiten_total(totals[pair], config_data, geometry, datasets[pair[0]], pair)
                save_pair_covariances(pair, config_data, cov_dir, spectra, save_reverse=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Make covariance matrices')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--cov', type=str, default=os.environ.get('COV_DIR', "./covariances"), help='Path to the directory to save the covariances')
    parser.add_argument('--clear', action='store_true', help='Clear the directory before saving the covariances')
    args = parser.parse_args()
    if not os.path.exists(args.cov):
        os.makedirs(args.cov)
    main(args.config, args.cov)
