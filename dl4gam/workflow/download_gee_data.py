import logging
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

from dl4gam.utils import download_best_images, run_in_parallel

log = logging.getLogger(__name__)


def main(
        base_dir: str | Path,
        gee_project_name: str,
        geoms_fp: str | Path,
        buffer_roi: int,
        year: str | int,
        download_time_window: tuple[str, str] = None,
        automated_selection: bool = False,
        dates_csv: str | Path = None,
        **kwargs,
):
    """
    Main function to download the best images from Google Earth Engine for all the entries in an inventory.
    The downloading will be done in parallel, per entry in the inventory (see the main config for the no. of processes).
    If multiple images per entry are requested, then these will be downloaded sequentially for each entry.

    :param base_dir: output directory (root) where the images will be downloaded;
        a subdirectory will be then created as `out_dir/year/entry_id`
    :param gee_project_name: name of the Google Earth Engine project to use for the download
    :param geoms_fp: file path to the processed glacier outlines
    :param buffer_roi: buffer size in meters for the region of interest around each glacier
    :param year: year for which to download the images
    :param download_time_window: MM:DD tuple with the start & end dates for the download window (e.g. '08-01', '10-15')
    :param automated_selection: whether to automatically select the best images;
        if False, dates must be provided either as a column in the GeoDataFrame (`date_acq`) or as a csv file
        with two columns: `entry_id` and `date` (in 'YYYY-MM-DD' format).
    :param dates_csv: optional csv file with the acquisition dates for each glacier (has priority over `date_acq`)
    :param kwargs: all the parameters for the `download_best_images` function
    :return: None
    """

    log.info(f"Initializing Earth Engine API for project: {gee_project_name}")
    ee.Initialize(project=gee_project_name)

    log.info(f"Reading the glacier outlines from {geoms_fp} (layer 'glacier_sel')")
    gdf = gpd.read_file(geoms_fp, layer='glacier_sel')

    # Set the start-end dates (use the date_acq column if needed, otherwise the year + download_time_window)
    if not automated_selection:
        # Read the csv file with the acquisition dates if provided
        if dates_csv is not None:
            log.info(f"Reading the acquisition dates from {dates_csv}")
            dates_df = pd.read_csv(dates_csv)
            if 'entry_id' not in dates_df.columns or 'date' not in dates_df.columns:
                raise ValueError("The dates CSV file must contain 'entry_id' and 'date' columns.")

            # Save the dates as a new column in the GeoDataFrame
            gdf['date_acq'] = gdf['entry_id'].map(dates_df.set_index('entry_id')['date'])
        else:
            if 'date_acq' not in gdf.columns:
                raise ValueError(
                    "The GeoDataFrame must contain a 'date_acq' column with the acquisition dates "
                    "if automated_selection is set to False."
                )
            log.info("Using the 'date_acq' column from the GeoDataFrame for acquisition dates.")

        # Make sure the dates are not missing for any glacier
        if gdf.date_acq.isnull().any():
            raise ValueError(
                "The GeoDataFrame contains missing values in the 'date_acq' column. "
                "Please ensure all glaciers have valid acquisition dates."
            )

        start_dates = gdf.date_acq.apply(lambda x: pd.to_datetime(x)).tolist()
        end_dates = [(s + pd.Timedelta(days=1)).strftime('%Y-%m-%d') for s in start_dates]
        years = [pd.to_datetime(s).year for s in start_dates]
    else:
        log.info(f"Automated selection of the best images with year: {year} & time window: {download_time_window}")
        # Get the inventory year for each glacier if year is 'inv', otherwise use the provided year
        years = list(gdf.date_inv.apply(lambda x: pd.to_datetime(x).year)) if year == 'inv' else [year] * len(gdf)
        start_dates = [f"{y}-{download_time_window[0]}" for y in years]
        end_dates = [f"{y}-{download_time_window[1]}" for y in years]

    # Project each geometry to the local CRS (EPSG code) provided in the GeoDataFrame
    # And build a list of GeoDataFrames for each glacier to run in parallel
    gdf = gdf.set_index('entry_id')  # we will use the index in the logs
    geoms_glacier = [gdf.iloc[i:i + 1].to_crs(gdf.iloc[i].crs_epsg).geometry for i in range(len(gdf))]
    geoms_roi = [r.buffer(buffer_roi).envelope for r in geoms_glacier]

    # Read the additional geometries needed for the QC metrics and turn them into lists
    log.info(f"Reading the QC geometries from {geoms_fp} ('buffer_clouds' and 'buffer_ndsi' layers)")
    gdf_clouds = gpd.read_file(geoms_fp, layer='buffer_clouds')
    gdf_ndsi = gpd.read_file(geoms_fp, layer='buffer_ndsi')

    # Align them with the glacier geometries and turn them into lists
    gdf_clouds = gdf_clouds.set_index('entry_id').reindex(gdf.index)
    gdf_ndsi = gdf_ndsi.set_index('entry_id').reindex(gdf.index)
    geoms_clouds = [gdf_clouds.iloc[i:i + 1].geometry for i in range(len(gdf_clouds))]
    geoms_ndsi = [gdf_ndsi.iloc[i:i + 1].geometry for i in range(len(gdf_ndsi))]

    # Prepare the output directory structure
    out_dirs = [Path(base_dir) / str(y) / entry_id for entry_id, y in zip(list(gdf.index), years)]

    run_in_parallel(
        download_best_images,
        out_dir=out_dirs,
        geom=geoms_roi,
        geom_clouds=geoms_clouds,
        geom_ndsi=geoms_ndsi,
        geom_albedo=geoms_clouds,  # we use the same buffered geometries for albedo as for clouds
        start_date=start_dates,
        end_date=end_dates,
        **kwargs,
    )
