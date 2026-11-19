"""Crop ACT FITS maps and inverse-variance maps on their native sky grid."""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from pixell import enmap

from src.map_geometry import require_sky_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--box", nargs=4, type=float, required=True,
        metavar=("RA_MIN", "DEC_MIN", "RA_MAX", "DEC_MAX"),
        help="Sky bounds in degrees, in increasing order.",
    )
    parser.add_argument(
        "--product", action="append", nargs=2, type=Path, required=True,
        metavar=("SOURCE_FITS", "OUTPUT_FITS"),
        help="A sky-referenced input and its cutout filename. Repeat for more files.",
    )
    parser.add_argument(
        "--temperature-only", action="store_true",
        help="Keep only Stokes I from I,Q,U map products.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cutouts.")
    args = parser.parse_args()

    west, south, east, north = args.box
    if not np.isfinite(args.box).all() or not (
        west < east < west + 360 and -90 <= south < north <= 90
    ):
        parser.error("--box needs increasing finite RA and Dec bounds in degrees")
    box = np.deg2rad([[south, west], [north, east]])

    sources = {source.resolve() for source, _ in args.product}
    outputs = {output.resolve() for _, output in args.product}
    if sources & outputs or len(outputs) != len(args.product):
        parser.error("Output paths must be distinct and must differ from the source paths")

    for source, output in args.product:
        if output.exists() and not args.overwrite:
            parser.error(f"{output} exists. Use --overwrite to replace it.")
        require_sky_coverage(source, [box])

    for source, output in args.product:
        cutout = enmap.read_map(str(source), box=box)
        if args.temperature_only and cutout.ndim == 3 and cutout.shape[0] == 3:
            cutout = cutout[0]
        header = cutout.wcs.to_header(relax=True)
        source_header = fits.getheader(source)
        for key in ("BUNIT", "POLCCONV"):
            if key in source_header:
                header[key] = source_header.cards[key].value, source_header.comments[key]
        header["SRCFILE"] = source.name
        output.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(output, np.asarray(cutout), header, overwrite=args.overwrite)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
