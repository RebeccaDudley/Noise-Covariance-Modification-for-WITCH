import numpy as np
from pixell import enmap, wcsutils


def require_sky_coverage(filename, regions):
    """Reject FITS inputs whose celestial footprint misses a requested box."""
    shape, wcs = enmap.read_map_geometry(str(filename))
    ra_axis, dec_axis = wcs.wcs.ctype
    if not (ra_axis.startswith("RA---") and dec_axis.startswith("DEC--")):
        raise ValueError(f"{filename} has no equatorial RA/Dec FITS WCS.")

    ny, nx = shape[-2:]
    wraps_ra = wcsutils.is_cyl(wcs) and nx >= round(360 / abs(wcs.wcs.cdelt[0]))
    for (south, west), (north, east) in regions:
        corners = [[south, south, north, north], [west, east, west, east]]
        y, x = enmap.sky2pix(shape, wcs, corners, bcheck=True)
        inside_y = np.all(np.isfinite(y) & (y >= -0.5) & (y <= ny - 0.5))
        inside_x = np.all(np.isfinite(x)) and (
            wraps_ra or np.all((x >= -0.5) & (x <= nx - 0.5))
        )
        if not (inside_y and inside_x):
            ra, dec = np.rad2deg([(west + east) / 2, (south + north) / 2])
            raise ValueError(
                f"{filename} does not cover the full requested sky box "
                f"centered at RA={ra:.6f} deg, Dec={dec:.6f} deg. "
                "Check the FITS WCS or regenerate the cutout from a sky-referenced map."
            )
