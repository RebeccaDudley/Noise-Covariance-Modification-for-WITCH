import numpy as np
from pixell import enmap, wcsutils


def covariance_region_geometries(filename, sky_regions):
    """Define odd-sized native-grid tiles using the cluster region's footprint."""
    shape, wcs = enmap.read_map_geometry(filename)
    shape = shape[-2:]
    require_sky_coverage(filename, sky_regions[-1:])
    bounds = enmap.subinds(shape, wcs, sky_regions[-1], cap=False)
    # Reverse-axis slices stop one pixel before the lowest included pixel.
    reverse_axes = bounds[2] < 0
    cluster_start = np.minimum(bounds[0], bounds[1]) + reverse_axes
    box_shape = np.abs(bounds[1] - bounds[0])
    if np.any(box_shape == 0):
        raise ValueError("The cluster covariance region must cover at least one pixel per axis.")

    # Trim one pixel from each even axis so the tile has a central pixel.
    tile_shape = box_shape - (box_shape % 2 == 0)
    half_shape = tile_shape // 2
    starts = []
    for box in sky_regions:
        center = enmap.sky2pix(shape, wcs, np.mean(box, axis=0))
        if not np.isfinite(center).all():
            raise ValueError("A covariance region has an invalid sky center.")
        starts.append(np.rint(center).astype(int) - half_shape)
    starts[-1] = np.clip(starts[-1], cluster_start, cluster_start + box_shape - tile_shape)

    return [
        enmap.slice_geometry(
            shape, wcs,
            tuple(slice(int(lo), int(lo + size)) for lo, size in zip(start, tile_shape)),
            nowrap=True,
        )
        for start in starts
    ]


def require_covariance_coverage(filename, regions):
    """Validate that each tile can be copied from the input without padding."""
    shape, wcs = enmap.read_map_geometry(filename)
    pixel_period = round(360 / abs(wcs.wcs.cdelt[0])) if wcsutils.is_cyl(wcs) else 0
    wraps_ra = pixel_period > 0 and shape[-1] >= pixel_period
    for index, (tile_shape, tile_wcs) in enumerate(regions):
        if not wcsutils.is_compatible(wcs, tile_wcs):
            raise ValueError(f"{filename} is not aligned to the reference covariance pixel grid.")
        bounds = enmap.pixbox_of(wcs, tile_shape, tile_wcs)
        inside = (bounds[0] >= 0) & (bounds[1] <= shape[-2:])
        if not inside[0] or not (wraps_ra or inside[1]):
            raise ValueError(
                f"{filename} does not cover covariance tile {index + 1}. "
                "Check the FITS WCS or regenerate a larger cutout."
            )


def sample_covariance_regions(cluster_geometry, filenames, count):
    """Sample cluster-sized tiles within the shared FITS pixel footprint."""
    tile_shape, tile_wcs = cluster_geometry
    # Preserve the original search radius of half a degree along each axis.
    max_offset = np.floor(0.5 / np.abs(tile_wcs.wcs.cdelt[::-1])).astype(int)
    lower, upper = -max_offset, max_offset.copy()
    for filename in filenames:
        shape, wcs = enmap.read_map_geometry(filename)
        start, stop = enmap.pixbox_of(wcs, tile_shape, tile_wcs)
        lower = np.maximum(lower, -start)
        upper = np.minimum(upper, np.array(shape[-2:]) - stop)

    if np.any(lower > upper):
        raise ValueError("No sampled covariance tile fits inside all input FITS maps.")

    regions = []
    for offset in np.random.randint(lower, upper + 1, size=(count, 2)):
        wcs = tile_wcs.deepcopy()
        wcs.wcs.crpix -= offset[::-1]
        regions.append((tile_shape, wcs))
    return regions


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
