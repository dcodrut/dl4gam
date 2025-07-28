import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

# local imports
from dl4gam.utils import prep_glacier_dataset, extract_date_from_fn, run_in_parallel

log = logging.getLogger(__name__)


def assign_image(glacier_id: str, raw_images_dir: str | Path, date: Optional[str] = None) -> tuple[str, Path] | None:
    """
    Assigns a single raw image to a glacier, either based on a provided date or by selecting the best image
    using metadata.

    If a date is not provided and multiple images exist for the glacier, the function tries to read a metadata file
    (metadata_filtered.csv) from the glacier directory (considered to be right under raw_images_dir, e.g.
    raw_images_dir/glacier_id/metadata_filtered.csv). This metadata file is expected to contain at least two columns,
    i.e. 'date' and 'score', and selects the image with the highest score.
    See `rank_images.py` for producing such a metadata file if not already present after downloading the images.

    Finally, if neither dates_csv is provided nor metadata file exists, we expect a single image per glacier directory.

    :param glacier_id: Glacier ID to find images for
    :param raw_images_dir: Directory containing raw satellite (tif) images, organized by glacier ID subdirectories
    :param date: Optional date string in 'YYYY-MM-DD' format to filter images by date.
    :return: Path to the selected image file or None if no image is found.
    """

    # Read all the raw images for the current glacier
    fp_img_list_all = sorted(list(Path(raw_images_dir).rglob('*.tif')))

    if len(fp_img_list_all) == 0:
        log.warning(f"No images found for glacier {glacier_id} in {raw_images_dir}")
        return None

    # Map each date to the corresponding image file path
    dates = [extract_date_from_fn(fp.stem) for fp in fp_img_list_all]
    date_to_fp = dict(zip(dates, fp_img_list_all))

    # If the date is provided, check if it exists and return the corresponding image
    if date is not None:
        if date in date_to_fp:
            return date, date_to_fp[date]
        else:
            log.warning(f"No image found for glacier {glacier_id} on the provided date ({date})")
            return None

    # If no date is provided, check if we have a single image and return it
    if len(fp_img_list_all) == 1:
        return dates[0], fp_img_list_all[0]

    # Otherwise, we need to select the best image based on metadata
    df_meta_fp = Path(raw_images_dir) / 'metadata_filtered.csv'
    if not df_meta_fp.exists():
        raise FileNotFoundError(f"Metadata file not found for glacier {glacier_id} at {df_meta_fp}")
    df_meta = pd.read_csv(df_meta_fp)

    # Keep the best image based on the 'score' column
    r = df_meta.sort_values(by='score', ascending=False).iloc[0]
    if r.date not in date_to_fp:
        raise ValueError(
            f"The best image date {r.date} according to the metadata from {df_meta_fp} was not found. "
            f"This should not happen, please check the metadata file."
        )

    return r.date, date_to_fp[r.date]


def filter_and_assign_images(
        gdf: gpd.GeoDataFrame,
        raw_data_base_dir: str | Path,
        year: str | int,
        dates_csv: Optional[str | Path] = None,
) -> tuple[list[str], list[str], list[Path]]:
    """
    Assign the raw images to the glaciers.
    Calls `assign_image` for each glacier ID in the GeoDataFrame.

    :param gdf: GeoDataFrame with the glacier outlines (we need at least the columns 'entry_id' and 'date_inv')
    :param dates_csv: Path to CSV with allowed dates (can be None)
    :param raw_data_base_dir: Base directory for raw images
    :param year: Year or 'inv' for inventory year
    :return: Tuple containing: the list of glacier IDs, the list of dates, and the list of file paths to the images.
    """

    if dates_csv is not None:
        allowed_dates = pd.read_csv(dates_csv)
        log.info(
            f"Loaded date restrictions from {dates_csv}; "
            f"we will use only the glaciers present in this file (n = {len(allowed_dates.entry_id.unique())})"
        )
        allowed_date_set = dict(zip(allowed_dates.entry_id, allowed_dates.date))
        gdf = gdf[gdf.entry_id.isin(allowed_date_set.keys())]
    else:  # get the inventory dates
        allowed_date_set = dict(zip(gdf.entry_id, gdf.date_inv))

    # Get the inventory years if needed, otherwise use the provided year (needed for the raw images directory structure)
    glacier_ids = list(gdf.entry_id)
    years = [pd.to_datetime(allowed_date_set[x]).year for x in glacier_ids] if year == 'inv' else [str(year)] * len(gdf)
    raw_images_dirs = [Path(raw_data_base_dir) / str(y) / str(gid) for gid, y in zip(glacier_ids, years)]

    # Assign the raw images to the glaciers
    res = run_in_parallel(
        fun=assign_image,
        glacier_id=glacier_ids,
        date=[allowed_date_set.get(gid) for gid in glacier_ids] if dates_csv else None,
        raw_images_dir=raw_images_dirs,
    )

    # Collect all the dates and file paths from the results, filtering out None values
    dates, fp_images = map(list, zip(*[x for x in res if x is not None]))
    glacier_ids = [gid for gid, r in zip(glacier_ids, res) if r is not None]

    return glacier_ids, dates, fp_images


def main(
        geoms_fp: str | Path,
        base_dir: str | Path,
        raw_data_base_dir: str | Path,
        year: str | int,
        dates_csv: Optional[str | Path] = None,
        extra_vectors: Optional[dict[str, str | Path]] = None,
        **kwargs
):
    # Read the outlines of the selected glaciers + all of them (needed for building the segmentation masks)
    log.info(f"Reading the glacier outlines from {geoms_fp}")
    gdf_sel = gpd.read_file(geoms_fp, layer='glacier_sel')
    gdf_all = gpd.read_file(geoms_fp, layer='glacier_all')

    # Read the FP, infer and patch sampling buffers and save them as extra vectors to be converted into binary masks
    # We will save them as list of subsets of GeoDataFrames, one for each glacier in gdf_sel (for run_in_parallel)
    extra_gdf_per_glacier = [{} for _ in range(len(gdf_sel))]  # one dict per glacier
    for buffer_name in ['infer', 'fp', 'patch_sampling']:
        log.info(f"Reading the buffer geometries for {buffer_name} from {geoms_fp}")
        _gdf = gpd.read_file(geoms_fp, layer=f"buffer_{buffer_name}")

        # Covert into a list of GeoDataFrames, one for each glacier in gdf_sel with a single geometry
        for i, gid in enumerate(gdf_sel.entry_id):
            extra_gdf_per_glacier[i][buffer_name] = _gdf[_gdf.entry_id == gid]

    # Include the extra vectors if provided
    if extra_vectors is not None:
        for k, fp in extra_vectors.items():
            log.info(f"Reading the extra vectors {k} from {fp}")
            _gdf = gpd.read_file(fp)

            # We will duplicate all the geometries for each glacier in the gdf_sel (for run_in_parallel)
            # Here we don't filter by entry_id as we don't know how they are structured
            for i, gid in enumerate(gdf_sel.entry_id):
                extra_gdf_per_glacier[i][k] = _gdf

    # Assign the images to the glaciers
    glacier_ids, dates, fp_images = filter_and_assign_images(
        gdf=gdf_sel,
        raw_data_base_dir=raw_data_base_dir,
        year=year,
        dates_csv=dates_csv,
    )

    # Build the output paths for the glacier cubes which will be created
    fp_output_cubes = [
        Path(base_dir) / gid / f"{pd.to_datetime(d).strftime('%Y%m%d')}.nc" for gid, d in zip(glacier_ids, dates)
    ]

    # Finally, build a xarray cube for each glacier, with the images & binary masks (+ additional features if provided)
    run_in_parallel(
        prep_glacier_dataset,
        fp_img=fp_images,
        entry_id=glacier_ids,
        gl_df=gdf_all,  # we use all the glaciers for the segmentation masks
        fp_out=fp_output_cubes,
        extra_geodataframes=extra_gdf_per_glacier,
        **kwargs,
    )
