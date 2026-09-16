import os
import yaml
from astropy.io import fits
from astropy.wcs import WCS


def load_map_geometry(map_file):
    """
    Read the FITS map and return its celestial WCS and dimensions.
    """

    with fits.open(map_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    wcs = WCS(header).celestial

    ny, nx = data.shape[-2:]

    return wcs, nx, ny


def fractional_position_to_pixel(x_frac, y_frac, nx, ny):
    """
    Convert a position expressed relative to the map into pixel coordinates.

    Definitions
    -----------
    x_frac = 0.0 -> left edge
    x_frac = 1.0 -> right edge

    y_frac = 0.0 -> bottom edge
    y_frac = 1.0 -> top edge
    """

    x_pix = x_frac * (nx - 1)
    y_pix = y_frac * (ny - 1)

    return x_pix, y_pix


def pixel_to_radec(wcs, x_pix, y_pix):
    """
    Convert pixel coordinates to RA/Dec using the FITS WCS.
    """

    sky = wcs.pixel_to_world(x_pix, y_pix)

    ra = sky.ra.deg
    dec = sky.dec.deg

    return float(ra), float(dec)


def width_fraction_to_degrees(width_frac, nx, ny, wcs):
    """
    Convert a box width expressed as a fraction of the map width
    into degrees.

    Assumes approximately square pixels.
    """

    # Pixel scale from the WCS
    pixel_scale_x = abs(wcs.pixel_scale_matrix[0, 0])
    pixel_scale_y = abs(wcs.pixel_scale_matrix[1, 1])

    # Use average scale in case the two are very slightly different
    pixel_scale = (pixel_scale_x + pixel_scale_y) / 2

    # Width expressed relative to the x dimension of the map
    width_pixels = width_frac * nx

    width_deg = width_pixels * pixel_scale

    return float(width_deg)


def make_region(box, wcs, nx, ny):
    """
    Convert one map-relative box definition into the RA/Dec form
    required by covariance_regions.
    """

    x_pix, y_pix = fractional_position_to_pixel(
        box["x_frac"],
        box["y_frac"],
        nx,
        ny,
    )

    ra, dec = pixel_to_radec(
        wcs,
        x_pix,
        y_pix,
    )

    width = width_fraction_to_degrees(
        box["width_frac"],
        nx,
        ny,
        wcs,
    )

    return {
        "ra": round(ra, 8),
        "dec": round(dec, 8),
        "width": round(width, 8),
    }


def make_covariance_yaml(
    map_file,
    template_yaml,
    output_yaml,
    cluster,
    background_boxes,
):
    """
    Build a covariance YAML file using regions defined relative
    to the original FITS map.
    """

    # ------------------------------------------------------------
    # 1. Load map geometry
    # ------------------------------------------------------------

    wcs, nx, ny = load_map_geometry(map_file)

    print(f"Map dimensions: {nx} x {ny}")

    # ------------------------------------------------------------
    # 2. Convert cluster location
    # ------------------------------------------------------------

    cluster_region = make_region(
        cluster,
        wcs,
        nx,
        ny,
    )

    # ------------------------------------------------------------
    # 3. Convert all background boxes
    # ------------------------------------------------------------

    background_regions = []

    for box in background_boxes:
        region = make_region(
            box,
            wcs,
            nx,
            ny,
        )

        background_regions.append(region)

    # ------------------------------------------------------------
    # 4. Load existing covariance YAML
    # ------------------------------------------------------------

    with open(template_yaml, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------
    # 5. Replace covariance-region information
    # ------------------------------------------------------------

    config["covariance_regions"] = {
        "cluster": {
            "ra": cluster_region["ra"],
            "dec": cluster_region["dec"],
            "width": cluster_region["width"],

            # Keep these because your current covariance config
            # contains them.
            "region_center_ra": cluster_region["ra"],
            "region_center_dec": cluster_region["dec"],
            "region_width": cluster_region["width"],
        },

        "background": background_regions,
    }

    # ------------------------------------------------------------
    # 6. Write new YAML
    # ------------------------------------------------------------

    with open(output_yaml, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    print()
    print(f"Created: {output_yaml}")

    print()
    print("Cluster:")
    print(cluster_region)

    print()
    print("Background regions:")

    for i, region in enumerate(background_regions):
        print(f"  Region {i}: {region}")


# ================================================================
# USER SETTINGS
# ================================================================

map_file = "A3016-3017_map.fits"

template_yaml = "covariance_A3016-3017.yaml"

output_yaml = "covariance_A3016-3017_generated.yaml"


# ----------------------------------------------------------------
# CLUSTER REGION
# ----------------------------------------------------------------
#
# These coordinates are RELATIVE TO THE MAP.
#
# x_frac:
#       0 = left
#       0.5 = center
#       1 = right
#
# y_frac:
#       0 = bottom
#       0.5 = center
#       1 = top
#
# width_frac:
#       fraction of the map width
#
# Example:
#       0.20 = 20% of map width
#
# For your ~1 degree map:
#       0.20 -> ~0.20 degrees
#

cluster = {
    "x_frac": 0.5,
    "y_frac": 0.5,
    "width_frac": 0.20,
}


# ----------------------------------------------------------------
# BACKGROUND REGIONS
# ----------------------------------------------------------------

background_boxes = [

    # upper-left
    {
        "x_frac": 0.20,
        "y_frac": 0.80,
        "width_frac": 0.20,
    },

    # upper-right
    {
        "x_frac": 0.80,
        "y_frac": 0.80,
        "width_frac": 0.20,
    },

    # lower-left
    {
        "x_frac": 0.20,
        "y_frac": 0.20,
        "width_frac": 0.20,
    },

    # lower-right
    {
        "x_frac": 0.80,
        "y_frac": 0.20,
        "width_frac": 0.20,
    },
]


# ================================================================
# RUN
# ================================================================

make_covariance_yaml(
    map_file=map_file,
    template_yaml=template_yaml,
    output_yaml=output_yaml,
    cluster=cluster,
    background_boxes=background_boxes,
)
