import logging
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import shapely.ops

from dl4gam.utils import download_best_images, run_in_parallel

log = logging.getLogger(__name__)


def main(
        root_dir: str | Path,
        gee_project_name: str,
        geoms_fp: str | Path,
        buffer_roi: int,
        buffer_qc_metrics: int,
        year: str | int,
        download_window: tuple[str, str] = None,
        automated_selection: bool = False,
        dates_csv: str | Path = None,
        **kwargs,
):
    """
    Main function to download the best images from Google Earth Engine for all the entries in an inventory.
    The downloading will be done in parallel, per entry in the inventory (see the main config for the number of processes).
    If multiple images per entry are requested, then these will be downloaded sequentially for each entry.

    :param root_dir: output directory (root) where the images will be downloaded; a subdirectory will be then created as `out_dir/year/entry_id`
    :param gee_project_name: name of the Google Earth Engine project to use for the download
    :param geoms_fp: file path to the processed glacier outlines
    :param buffer_roi: buffer size in meters for the region of interest around each glacier
    :param buffer_qc_metrics: buffer size in meters for the quality control metrics (e.g. NDSI, cloud coverage)
    :param year: year for which to download the images
    :param download_window: MM:DD tuple with the start and end dates for the download window (e.g. '08-01', '10-15')
    :param automated_selection: whether to use the automated selection of the best images (if not, dates_csv must be provided)
    :param dates_csv: optional CSV file with the start and end dates for each glacier
    :param kwargs: all the parameters for the `download_best_images` function
    :return: None
    """

    log.info(f"Initializing Earth Engine API for project: {gee_project_name}")
    ee.Initialize(project=gee_project_name)

    log.info(f"Reading the glacier outlines from {geoms_fp}")
    gdf = gpd.read_file(geoms_fp, layer='glacier_sel').iloc[10:]

    # Set the start-end dates (use the csv if provided, otherwise the year + download_window)
    if not automated_selection:
        assert dates_csv is not None, "If automated_selection is False, dates_csv must be provided."
        log.info(f"Reading the start and end dates from {dates_csv}")
        dates_df = pd.read_csv(dates_csv, index_col='entry_id')
        start_dates = list(dates_df.loc[gdf.entry_id, 'date'])
        end_dates = [(pd.to_datetime(s) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') for s in start_dates]
        years = [pd.to_datetime(s).year for s in start_dates]
    else:
        # Get the inventory year for each glacier if year is 'inv', otherwise use the provided year
        years = list(gdf.date_inv.apply(lambda x: pd.to_datetime(x).year)) if year == 'inv' else [year] * len(gdf)
        start_dates = [f"{y}-{download_window[0]}" for y in years]
        end_dates = [f"{y}-{download_window[1]}" for y in years]

    # Prepare the geometries for each glacier in the GeoDataFrame (to run in parallel)
    entries = [gdf.iloc[i:i + 1] for i in range(len(gdf))]
    geoms_glacier = [r.geometry for r in entries]
    geoms_roi = [r.geometry.to_crs(r.iloc[0].crs_epsg).buffer(buffer_roi).to_crs('epsg:4326') for r in entries]
    geoms_glacier_buffered = [
        r.geometry.to_crs(r.iloc[0].crs_epsg).buffer(buffer_qc_metrics).to_crs('epsg:4326') for r in entries
    ]

    # For cloud coverage and albedo, we use the simple buffered glacier geometries
    geoms_clouds = geoms_glacier_buffered
    geoms_albedo = geoms_glacier_buffered

    # Prepare the NDSI geometries (non-glacier pixels within the buffered glacier geometries)
    geom_all_glaciers = shapely.ops.unary_union(geoms_glacier)
    geom_all_glaciers_buffered = shapely.ops.unary_union(geoms_glacier_buffered)
    geom_non_glacier = geom_all_glaciers_buffered.difference(geom_all_glaciers)
    geoms_ndsi = [r.geometry.intersection(geom_non_glacier) for r in geoms_glacier_buffered]

    # Prepare the output directory structure
    out_dirs = [Path(root_dir) / str(y) / entry_id for entry_id, y in zip(list(gdf.entry_id), years)]

    run_in_parallel(
        download_best_images,
        out_dir=out_dirs,
        geom=geoms_roi,
        geom_clouds=geoms_clouds,
        geom_ndsi=geoms_ndsi,
        geom_albedo=geoms_albedo,
        start_date=start_dates,
        end_date=end_dates,
        **kwargs,
    )
