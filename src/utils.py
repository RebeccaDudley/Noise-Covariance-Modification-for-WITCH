import os 
import matplotlib.pyplot as plt
from pixell import enmap, uharm
import warnings
import glob
import numpy as np
import yaml

from numba import njit, jit

from astropy import units as u
from astropy import constants as const, units as u
from astropy.cosmology import Planck18

#import bandpass as bp

plt.rcParams.update({'font.size': 18})
warnings.filterwarnings('ignore')

dire_base = os.getenv("DIRE_BASE")

# Absolute photometric calibration factors
cal_dict = {'dr6_pa4_f150': {'multipole_range': [1250, 1800],
  'ref_array': 'Planck_f143',
  'calibs': [1.0072175262677963, 0.0023154207035651023]},
 'dr6_pa4_f220': {'multipole_range': [1500, 2000],
  'ref_array': 'Planck_f217',
  'calibs': [1.0340185610989223, 0.01626333139583416]},
 'dr6_pa5_f098': {'multipole_range': [800, 1100],
  'ref_array': 'Planck_f100',
  'calibs': [1.0218836321586204, 0.0011691303344386838]},
 'dr6_pa5_f150': {'multipole_range': [1250, 1800],
  'ref_array': 'Planck_f143',
  'calibs': [0.9876564792184743, 0.001704546067935355]},
 'dr6_pa6_f098': {'multipole_range': [800, 1100],
  'ref_array': 'Planck_f100',
  'calibs': [1.0181628559678786, 0.0013096342890986763]},
 'dr6_pa6_f150': {'multipole_range': [1250, 1800],
  'ref_array': 'Planck_f143',
  'calibs': [0.9699498005405339, 0.002141045797557799]}}

def coord_from_ref(ref_ra, ref_dec, des_ra, des_dec, delta):
    """
    Return the coordinates relative to the center of the image.

    Parameters
    ----------
    ref_ra : float
        Right ascension of the reference pixel.
    ref_dec : float
        Declination of the reference pixel.
    des_ra : float
        Right ascension of the desired pixel.
    des_dec : float 
        Declination of the desired pixel.
    delta : float
        Plus or minus delta degrees from the desired coordinates.

    Returns
    -------
    tuple
        Tuple of the coordinates relative to the center of the image.
    """
    new_ra = des_ra - ref_ra
    new_dec = des_dec - ref_dec

    print(f"New RA: {new_ra} degrees, New Dec: {new_dec} degrees")
    print(f"New RA minus: {new_ra - delta} degrees, New RA plus: {new_ra + delta} degrees")
    print(f"New Dec minus: {new_dec - delta} degrees, New Dec plus: {new_dec + delta} degrees\n")

def get_coord_pix(data_wcs, cf):
    c1_ra_min_pix = (data_wcs.celestial.wcs_world2pix(cf['c1_ra_min'], 
                                                      cf['c1_dec_min'], 0)[0])
    c1_ra_max_pix = (data_wcs.celestial.wcs_world2pix(cf['c1_ra_max'], 
                                                      cf['c1_dec_max'], 0)[0])
    
    c1_dec_min_pix = (data_wcs.celestial.wcs_world2pix(cf['c1_ra_min'], 
                                                       cf['c1_dec_min'], 0)[1])
    c1_dec_max_pix = (data_wcs.celestial.wcs_world2pix(cf['c1_ra_max'], 
                                                       cf['c1_dec_max'], 0)[1])
    
    return c1_ra_min_pix, c1_ra_max_pix, c1_dec_min_pix, c1_dec_max_pix

def load_combos():
    combos = ['150_pa4_act', 
            '220_pa4_act', 
            '150_pa5_act',
            '98_pa5_act',
            '98_pa6_act',
            '150_pa6_act',
            '30_npipe_planck',
            '44_npipe_planck',
            '70_npipe_planck',
            '100_npipe_planck',
            '143_npipe_planck',
            '217_npipe_planck',
            '353_npipe_planck',
            '545_npipe_planck',
            '857_npipe_planck']
    return combos

@njit
def r_grid(xgrid, ygrid, ra_pix, dec_pix, theta_cluster, e):
    """
    Get r grid, given xgrid and ygrid, 
    position angle, and ellipticity
    """
    xprime = ((xgrid - (ra_pix))*np.cos(theta_cluster) -
                (ygrid - (dec_pix))*np.sin(theta_cluster))
    yprime = ((xgrid - (ra_pix))*np.sin(theta_cluster) +
                (ygrid - (dec_pix))*np.cos(theta_cluster))

    r = np.sqrt(xprime**2. + (yprime * e)**2.)
    return r

'''
def get_freq_bandpass(array, freq_str):
    x_array = bp.x_dict[f"x_{array}_{freq_str}"]
    freq_array = bp.freq_dict[f"freq_{array}_{freq_str}"] 
    del_freq = bp.del_freq_dict[f"del_freq_{array}_{freq_str}"]
    return x_array, freq_array, del_freq

def get_bandpass(array, freq_str):
    """
    Get bandpass and bandpass integrated
    """
    bp_str = f"passband_{array}_{freq_str}"
    bp_int_str = f"passband_{array}_{freq_str}_int"
    
    bandpass_int = bp.passband_int_dict[bp_int_str]
    bandpass = bp.passband_dict[bp_str]

    return bandpass, bandpass_int
'''

def get_freq_str(frequency):
    if float(frequency) < 100:
        frequency_tmp = str(int(frequency))
        freq_str = f"0{frequency_tmp}"
    else: 
        freq_str = str(int(frequency))
    return freq_str

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

@jit(nopython=True, parallel=False)
def bandpass_integrate(bandpass, bandpass_int, I, freq_array, del_freq):
    numerator = np.trapz(y=bandpass * I,
                            x=freq_array, 
                            dx=del_freq)
    
    sz_signal = numerator / (bandpass_int) # SZ model
    return sz_signal

def get_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # add some environment variables to config_data for backward compatibility
    # config_data["act_data_dir"] = os.environ["ACT_DATADIR"]
    # config_data["planck_data_dir"] = os.environ["PLANCK_DATADIR"]

    return config_data

def save_config_file(config_file, config_data):
    with open(config_file, 'w') as stream:
        try:
            yaml.dump(config_data, stream)
        except yaml.YAMLError as exc:
            print(exc)

def get_rho_crit(H_z):
    """
    Get critical density of 
    the Universe given Hubble parameter.
    """
    return (3*H_z**2) / (8*np.pi*const.G)

def get_r200(z, mass):
    """
    Get r_200 [pixels] 
    given redshift and mass.
    """
    overdense = 200
    pixel_scale = 0.5 # arcmin / pixel

    H_z = Planck18.H(z=z)
    D_A = Planck18.angular_diameter_distance(z=z)

    rho_crit = get_rho_crit(H_z=H_z)

    r = ((3*mass*10**(14)*u.solMass) / (4*np.pi*overdense*rho_crit))**(1/3)

    r_Mpc = r.to(u.Mpc)
    
    r200_arcmin = (np.arctan(r_Mpc/D_A.to(u.Mpc))).to(u.arcmin)

    r200_pix = r200_arcmin / pixel_scale

    return r200_pix.value

def get_r500(z, mass):
    """
    Get r_500 [pixels] 
    given redshift and mass.
    """
    overdense = 500
    pixel_scale = 0.5 # arcmin / pixel

    H_z = Planck18.H(z=z)
    D_A = Planck18.angular_diameter_distance(z=z)

    rho_crit = get_rho_crit(H_z=H_z)

    r = ((3*mass*10**(14)*u.solMass) / (4*np.pi*overdense*rho_crit))**(1/3)

    r_Mpc = r.to(u.Mpc)
    
    r500_arcmin = (np.arctan(r_Mpc/D_A.to(u.Mpc))).to(u.arcmin)

    r500_pix = r500_arcmin / pixel_scale

    return r500_pix.value

''' 
def flux_factor(array, freq):
    """
    Returns Jy/K/sr
    """
    if float(freq) < 100:
        frequency_tmp = str(int(freq))
        freq_str = f"0{frequency_tmp}"
    else: 
        freq_str = str(int(freq))
        
    Tcmb = 2.725 * u.K
    freq_array = bp.freq_dict[f"freq_{array}_{freq_str}"] * u.GHz
    del_freq =  ((np.max(freq_array) - np.min(freq_array))  / len(freq_array))
    bp_str = f"passband_{array}_{freq_str}"
    bp_int_str = f"passband_{array}_{freq_str}_int"
    bandpass_int = bp.passband_int_dict[bp_int_str]
    bandpass = bp.passband_dict[bp_str]
    
    xcmb = (const.h * freq_array.to(1/u.s)) / (const.k_B * Tcmb)

    dplanck = ( ((2*const.k_B**3*Tcmb**2*xcmb**4*np.exp(xcmb)) 
               / ((const.c*const.h)**2 * (np.exp(xcmb)-1)**2.)).to(u.Jansky/u.K) / u.steradian )
    
    flux_factor = np.trapz(y=bandpass*dplanck, x=freq_array, dx=del_freq) / (bandpass_int * u.GHz)

    return flux_factor.to(u.Jy/u.microKelvin/u.steradian).value
'''
    
def rad_avg_2d(image, bsize):
    bl, l = enmap.lbin(map=image, bsize=bsize)
    return np.interp(x=image.modlmap(), xp=l, fp=bl)

def get_all_centers(region_center_ra, region_center_dec, fit_radius):
    """
    Takes in the center of the cluster and the fit radius
    and returns the centers of the 8 regions and the center
    of the cluster.

    Parameters
    ----------
    region_center_ra : float in degrees
        Right ascension of the center of the cluster.
    region_center_dec : float
        Declination of the center of the cluster.
    fit_radius : float
        Radius of the cluster in degrees.
    
    Returns
    -------
    list
        List of the centers of the 8 regions and the center
        of the cluster in radians.
    """
    
    region_center_ra = np.deg2rad(region_center_ra)
    region_center_dec = np.deg2rad(region_center_dec)
    box_size = np.deg2rad(2 * fit_radius)

    cluster_center = [region_center_dec, region_center_ra]

    center_0 = [region_center_dec, region_center_ra-box_size]
    center_1 = [region_center_dec + box_size, region_center_ra - box_size]
    center_2 = [region_center_dec + box_size, region_center_ra]
    center_3 = [region_center_dec + box_size, region_center_ra + box_size]
    center_4 = [region_center_dec, region_center_ra + box_size]
    center_5 = [region_center_dec-box_size, region_center_ra+box_size]
    center_6 = [region_center_dec-box_size, region_center_ra]
    center_7 = [region_center_dec-box_size, region_center_ra-box_size]

    return [center_0, center_1, center_2, 
            center_3, center_4, center_5, 
            center_6, center_7, cluster_center]

###formatting for cluster and background region info for the get_all_regions function below:###
#cluster = [36.342, -42.008, 0.20]

#background_regions = [
#    [36.30, -41.80, 0.20],
#    [36.55, -41.95, 0.30],
#    [36.60, -42.20, 0.15],
#    [36.25, -42.35, 0.25],
#]

def make_region(center_ra, center_dec, width):

    center_ra = np.deg2rad(center_ra)
    center_dec = np.deg2rad(center_dec)
    width = np.deg2rad(width)

    return [
        [center_dec - width/2, center_ra - width/2],
        [center_dec + width/2, center_ra + width/2]
    ]


def get_all_regions(cluster_definition, background_definitions):

    cluster_reg = make_region(*cluster_definition)

    regions = [
        make_region(*region)
        for region in background_definitions
    ]

    return regions + [cluster_reg]



def imap_dim_check(imap):
    if imap.ndim > 2:
        return imap[0]
    elif imap.ndim == 2:
        return imap
    else:
        raise ValueError("Invalid number of dimensions of the map.")

def get_region(region_center_ra, region_center_dec, region_width):
    region_center_ra = np.deg2rad(region_center_ra)
    region_center_dec = np.deg2rad(region_center_dec)
    region_width = np.deg2rad(region_width)
    reg = [[region_center_dec-region_width/2, region_center_ra-region_width/2], 
           [region_center_dec+region_width/2, region_center_ra+region_width/2]]
    return reg

def freq_to_x(frequency):
    frequency *= u.GHz
    return (const.h * frequency.to(u.s**-1) / (const.k_B * 2.725 * u.K)).value

def gaussian_2d_function(x=0, y=0, x0=0, y0=0, sx=1, sy=1):
    return np.exp(-((x - x0)**2. / (2. * sx**2.) + (y - y0)**2. / (2. * sy**2.)))

def make_2d_gauss(freq, data_shape, inst):
    img_size_x_pix = data_shape[1]
    img_size_y_pix = data_shape[0]
    freq = float(freq)

    if inst == 'planck':
        plate_scale = 0.5

        if freq == 30:
            beam_FWHM_arcmin = 32.408
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 44:
            beam_FWHM_arcmin = 27.1
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 70:
            beam_FWHM_arcmin = 13.315
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 100:
            beam_FWHM_arcmin = 9.682
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 143:
            beam_FWHM_arcmin = 7.303
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 217:
            beam_FWHM_arcmin = 5.021
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 353:
            beam_FWHM_arcmin = 4.944
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 545:
            beam_FWHM_arcmin = 4.831
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        elif freq == 857:
            beam_FWHM_arcmin = 4.638
            beam_FWHM_pix = beam_FWHM_arcmin / plate_scale
        else:
            raise ValueError(f"Invalid frequency: {freq}")
    else:
        raise ValueError("Invalid instrument")

    x_stddev_pix = beam_FWHM_pix / (2*np.sqrt(2*np.log(2)))
    y_stddev_pix = beam_FWHM_pix / (2*np.sqrt(2*np.log(2)))

    xgrid, ygrid = np.meshgrid(np.arange(0, img_size_x_pix, 1), np.arange(0, img_size_y_pix, 1))

    gaussian_2d_profile = gaussian_2d_function(x=xgrid, y=ygrid,
                                               x0=img_size_x_pix/2, 
                                               y0=img_size_y_pix/2,
                                               sx=x_stddev_pix, sy=y_stddev_pix)
    gaussian_2d_profile = abs(np.fft.fft2(gaussian_2d_profile))
    gaussian_2d_profile /= np.max(gaussian_2d_profile)
    return gaussian_2d_profile

def get_2d_beam(data_shape, freq, array, inst, data_wcs, version="dr6v2"):

    if inst == 'act':
        
        if version == "dr6v3":
            beam_dir = f"{dire_base}/beams/20220817_beams/"
        
        elif version == "dr6v2":
            beam_dir = f"{dire_base}/beams/20210913_beams/"
        
        elif version == "dr6v4":
            beam_dir = f"{dire_base}/beams/20230130_beams/"
        
        file = glob.glob(beam_dir+"*coadd*"+array+"*" +
                         str(freq)+"*tform_jitter_cmb.txt")
        
        if len(file) == 0:
            raise ValueError("Invalid array and frequency for ACT.")
        
        beam_fft = np.loadtxt(file[0])
        
        b_ell = beam_fft[0:, 1]
        b_ell /= b_ell[0]

        uht = uharm.UHT(data_shape, data_wcs, mode="flat")
        
        beam_2d_fft = uht.lprof2hprof(b_ell)

    elif inst == 'planck':
        beam_2d_fft = make_2d_gauss(freq=freq, data_shape=data_shape,inst=inst)
    
    else:
        raise ValueError("Invalid instrument.")
    
    return beam_2d_fft

def src_sub_act(dire):  
    """
    Source subtract ACT data, given direc
    """
    arrays = ['*pa4*', '*pa5*', '*pa6*']
    freqs = ['*30*', '*40*', '*98*', '*150*', '*220*']
    
    for array in arrays:
        for freq in freqs:
            temp_maps = np.sort(glob.glob(dire + array + freq + "*map.fits"))
            src_maps = np.sort(glob.glob(dire + array + freq + "*srcs.fits"))
            
            if len(temp_maps) > 0:
                
                for i in range(len(temp_maps)):
                    temp_map = imap_dim_check(enmap.read_map(temp_maps[i]))
                    src_map = imap_dim_check(enmap.read_map(src_maps[i]))

                    diff = temp_map - src_map

                    outname = temp_maps[i][:-5] + "_srcfree.fits"

                    #if 'coadd' in outname: 
                        #plot_image(diff, title=freq+array)

                    enmap.write_map(fname=outname, 
                                    emap=diff,
                                    fmt="fits")
                    
def coadd_act(nways, dire, wcs_ref, obstype, plot_coadd=True, use_srcfree_maps=True):
    """
    Makes a coadded map of ACT data.
    nways: number of splits [int]
    dire: directory [str]
    wcs_ref: reference file for wcs [str]
    """
    freqs = ['*098*', '*150*', '*220*']
    arrays = ['*pa4*', '*pa5*', '*pa6*']

    ref_wcs = enmap.read_map(wcs_ref).wcs

    if use_srcfree_maps:
        data_str = "_srcfree"
    else:
        data_str = ""

    for freq in freqs:
        for array in arrays:
            exist_check = (glob.glob(dire+array+freq+"*set0*map*fits*"))

            if len(exist_check) == 0: 
                continue
            
            if nways == 8:
                split0_str = np.sort(glob.glob(dire+array+freq+"*set0*map"+data_str+".fits*"))[0]
                split1_str = np.sort(glob.glob(dire+array+freq+"*set1*map"+data_str+".fits*"))[0]
                split2_str = np.sort(glob.glob(dire+array+freq+"*set2*map"+data_str+".fits*"))[0]
                split3_str = np.sort(glob.glob(dire+array+freq+"*set3*map"+data_str+".fits*"))[0]
                split4_str = np.sort(glob.glob(dire+array+freq+"*set4*map"+data_str+".fits*"))[0]
                split5_str = np.sort(glob.glob(dire+array+freq+"*set5*map"+data_str+".fits*"))[0]
                split6_str = np.sort(glob.glob(dire+array+freq+"*set6*map"+data_str+".fits*"))[0]
                split7_str = np.sort(glob.glob(dire+array+freq+"*set7*map"+data_str+".fits*"))[0]

                ivar0_str = np.sort(glob.glob(dire+array+freq+"*set0*ivar.fits*"))[0]
                ivar1_str = np.sort(glob.glob(dire+array+freq+"*set1*ivar.fits*"))[0]
                ivar2_str = np.sort(glob.glob(dire+array+freq+"*set2*ivar.fits*"))[0]
                ivar3_str = np.sort(glob.glob(dire+array+freq+"*set3*ivar.fits*"))[0]
                ivar4_str = np.sort(glob.glob(dire+array+freq+"*set4*ivar.fits*"))[0]
                ivar5_str = np.sort(glob.glob(dire+array+freq+"*set5*ivar.fits*"))[0]
                ivar6_str = np.sort(glob.glob(dire+array+freq+"*set6*ivar.fits*"))[0]
                ivar7_str = np.sort(glob.glob(dire+array+freq+"*set7*ivar.fits*"))[0]

                # Put ut.imap_dim_check in front of enmap.read_fits
                # to check if the dimensions are the same

                split0 = imap_dim_check(enmap.read_fits(split0_str))
                split1 = imap_dim_check(enmap.read_fits(split1_str))
                split2 = imap_dim_check(enmap.read_fits(split2_str))
                split3 = imap_dim_check(enmap.read_fits(split3_str))
                split4 = imap_dim_check(enmap.read_fits(split4_str))
                split5 = imap_dim_check(enmap.read_fits(split5_str))
                split6 = imap_dim_check(enmap.read_fits(split6_str))
                split7 = imap_dim_check(enmap.read_fits(split7_str))

                ivar0 = imap_dim_check(enmap.read_fits(ivar0_str))
                ivar1 = imap_dim_check(enmap.read_fits(ivar1_str))
                ivar2 = imap_dim_check(enmap.read_fits(ivar2_str))
                ivar3 = imap_dim_check(enmap.read_fits(ivar3_str))
                ivar4 = imap_dim_check(enmap.read_fits(ivar4_str))
                ivar5 = imap_dim_check(enmap.read_fits(ivar5_str))
                ivar6 = imap_dim_check(enmap.read_fits(ivar6_str))
                ivar7 = imap_dim_check(enmap.read_fits(ivar7_str))
                                
                splits = np.array([split0, 
                                   split1,
                                   split2,
                                   split3,
                                   split4,
                                   split5,
                                   split6,
                                   split7])

                ivars = np.array([ivar0, 
                                  ivar1,
                                  ivar2,
                                  ivar3,
                                  ivar4,
                                  ivar5,
                                  ivar6,
                                  ivar7])

                coadd_sum = np.sum(splits * ivars, axis=0) / np.sum(ivars, axis=0)
                coadd_sum = enmap.ndmap(arr=coadd_sum, wcs=ref_wcs)
                
                fname_coadd = (dire+ "act_cut_cmb_" + obstype + '_' + array[1:4] 
                               + '_f' + freq[1:4] + f"_8way_coadd_map{data_str}.fits")
                
                print(fname_coadd)
                
                enmap.write_fits(fname_coadd, coadd_sum)
                
                if plot_coadd:
                    plot_image(coadd_sum, title=freq + array, interval_type='percentile')
        
            elif nways == 4:
                split0_str = np.sort(glob.glob(dire+array+freq+"*set0*map"+data_str+".fits*"))[0]
                split1_str = np.sort(glob.glob(dire+array+freq+"*set1*map"+data_str+".fits*"))[0]
                split2_str = np.sort(glob.glob(dire+array+freq+"*set2*map"+data_str+".fits*"))[0]
                split3_str = np.sort(glob.glob(dire+array+freq+"*set3*map"+data_str+".fits*"))[0]
                
                ivar0_str = np.sort(glob.glob(dire+array+freq+"*set0*ivar.fits*"))[0]
                ivar1_str = np.sort(glob.glob(dire+array+freq+"*set1*ivar.fits*"))[0]
                ivar2_str = np.sort(glob.glob(dire+array+freq+"*set2*ivar.fits*"))[0]
                ivar3_str = np.sort(glob.glob(dire+array+freq+"*set3*ivar.fits*"))[0]

                split0 = imap_dim_check(enmap.read_fits(split0_str))
                split1 = imap_dim_check(enmap.read_fits(split1_str))
                split2 = imap_dim_check(enmap.read_fits(split2_str))
                split3 = imap_dim_check(enmap.read_fits(split3_str))

                ivar0 = imap_dim_check(enmap.read_fits(ivar0_str))
                ivar1 = imap_dim_check(enmap.read_fits(ivar1_str))
                ivar2 = imap_dim_check(enmap.read_fits(ivar2_str))
                ivar3 = imap_dim_check(enmap.read_fits(ivar3_str))
                
                splits = np.array([split0, 
                                   split1,
                                   split2,
                                   split3])

                ivars = np.array([ivar0, 
                                  ivar1,
                                  ivar2,
                                  ivar3])

                coadd_sum = np.sum(splits * ivars, axis=0) / np.sum(ivars, axis=0)
                coadd_sum = enmap.ndmap(arr=coadd_sum, wcs=ref_wcs)
                
                fname_coadd = (dire+ "act_cut_cmb_" + obstype + '_' + array[1:4] 
                               + '_f' + freq[1:4] + "_4way_coadd_map" + data_str + ".fits")
                print(fname_coadd)
                enmap.write_fits(fname_coadd, coadd_sum)
                
                if plot_coadd:
                    plot_image(coadd_sum, title=freq + array, interval_type='percentile')
                
            else: 
                raise ValueError("Invalid number of ways/splits.")

def check_real_covariance(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False, "Matrix is not square"

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
       return False, "Matrix is not symmetric"

    # Check if the matrix is positive semidefinite
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(eigenvalues) < 0:
        return False, "Matrix is not positive semidefinite"

    return True, "Matrix is a valid real covariance matrix"

def check_complex_covariance(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False, "Matrix is not square"

    # Check if the matrix is Hermitian
    if not np.allclose(matrix, np.conjugate(matrix.T)):
        return False, "Matrix is not Hermitian"

    # Check if the matrix is positive semidefinite
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(np.real(eigenvalues) < 0):
        return False, "Matrix is not positive semidefinite"

    return True, "Matrix is a valid complex covariance matrix"

def plot_image(image,
               cmap='viridis',
               title='',
               interval_type='simple_norm',
               stretch='linear',
               percentile=99,
               cbar_label=r'Jansky per steradian',
               xlabel='Pixel',
               ylabel='Pixel',
               projection=None,
               modlmap=None,
               save=False,
               save_fname='img.png',
               show=True):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from astropy.visualization import (ZScaleInterval,
                                       PercentileInterval,
                                       MinMaxInterval,
                                       ImageNormalize,
                                       simple_norm)

    fig = plt.figure(figsize=(5, 5))

    if projection is not None:
        ax = fig.add_subplot(111, projection=projection)
        xlabel = ""
        ylabel = ""

    else:
        ax = fig.add_subplot(1, 1, 1)

    if interval_type == 'zscale':
        norm = ImageNormalize(image, interval=ZScaleInterval())

    elif interval_type == 'percentile':
        norm = ImageNormalize(image,
                              interval=PercentileInterval(percentile))
    elif interval_type == 'minmax':
        norm = ImageNormalize(image,
                              interval=MinMaxInterval())
    elif interval_type == 'simple_norm':
        norm = simple_norm(image, stretch)

    if modlmap is not None:
        hor_min = 0
        hor_max = np.max(modlmap[0, :int(image.shape[0]/2)])
        ver_min = 0
        ver_max = np.max(modlmap[0, :int(image.shape[0]/2)])
        extent = [-hor_max, hor_max, -ver_max, ver_max]

        im = ax.imshow(image,
                       cmap=cmap,
                       interpolation='none',
                       origin='upper',
                       norm=norm,
                       extent=extent)
    else:
        im = ax.imshow(np.fliplr(image),
                       cmap=cmap,
                       interpolation='none',
                       origin='lower',
                       norm=norm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.5)

    plt.colorbar(im, cax=cax, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)

    if save:
        plt.savefig(save_fname)
    if show:
        plt.show()
    plt.close()
    return None

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i
