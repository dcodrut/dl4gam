import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from dl4gam import utils
from dl4gam.configs.datasets import BaseDatasetCfg

log = logging.getLogger(__name__)


def add_auxiliary_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Adds the (missing) auxiliary columns to the GeoDataFrame:
    1. An integer ID for each glacier
    2. Area in km2
    3. The local UTM zone (for each glacier)
    4. (lat, lon) of the centroid of the glacier (forced to be inside the glacier)
    5. Country name and code

    :param gdf: GeoDataFrame with the glacier polygons
    :return: GeoDataFrame with the auxiliary columns
    """

    # Add the integer ID for each glacier
    if 'entry_id_i' not in gdf.columns:
        log.info("Assigning an integer ID to each glacier (1:n), which will be used for the mask construction.")
        gdf['entry_id_i'] = np.arange(len(gdf)) + 1

    # Add the area in km2
    if 'area_km2' not in gdf.columns:
        log.info("Computing the glacier areas")
        gdf['area_km2'] = gdf.geometry.area / 1e6

    if 'cenlat' not in gdf.columns or 'cenlon' not in gdf.columns:
        log.info("Adding centroid coordinates (forced within the glacier polys)")
        centroids = gdf.geometry.apply(lambda geom: utils.get_interior_centroid(geom))
        centroids = gpd.GeoSeries(centroids, crs=gdf.crs).to_crs('epsg:4326')
        gdf['cenlat'] = [centroid.y for centroid in centroids]
        gdf['cenlon'] = [centroid.x for centroid in centroids]

    # Add the local UTM zone (separately for each glacier)
    if 'utm_zone' not in gdf.columns or 'utm_epsg' not in gdf.columns:
        log.info("Adding the local UTM zone (separately for each glacier)")
        epsg_codes = [utils.latlon_to_utm_epsg(lat=y, lon=x) for (x, y) in zip(gdf['cenlon'], gdf['cenlat'])]
        utm_zones = [int(code[-2:]) for code in epsg_codes]
        gdf['utm_zone'] = utm_zones
        gdf['utm_epsg'] = epsg_codes
        log.info(f"Found {len(set(utm_zones))} UTM zones: {sorted(set(utm_zones))} in the dataset.")

    # Add the country name and code
    if 'country_name' not in gdf.columns or 'country_code' not in gdf.columns:
        log.info("Assigning country names and codes to glaciers")
        gdf = utils.add_country(gdf)

    return gdf


def compute_buffers(
        gdf: gpd.GeoDataFrame,
        buffers: BaseDatasetCfg.Buffers,
        tol: Optional[float] = None
) -> dict[str, gpd.GeoDataFrame]:
    """
    Computes the following geometries for each glacier:
    1. Geometry for computing the cloud coverage (and albedo), i.e. glacier + simple buffer
    2. Geometry for computing the NDSI, i.e. non-glacier pixels within the glacier buffer
    3. Geometry from which the patch centres will be sampled (simple buffer)
    4. Geometry within which the inference will be performed (non-overlapping buffer)
    5. Geometry within which false positives will be calculated (non-overlapping buffer)

    These geometries will be turned into binary masks later on.

    :param gdf: GeoDataFrame with all the glacier polygons
    :param buffers: Buffers object with the buffer sizes
    :param tol: tolerance for simplifying the geometries (in meters)
    :return: a dictionary with the GeoDataFrames for each geometry
    """

    # Before computing the buffers, let's simplify the geometries to reduce the processing time & storage size
    _gdf = gdf.copy()
    if tol is not None:
        log.info(f"Simplifying the geometries with tolerance {tol:.2f} m")
        _gdf['geometry'] = _gdf.geometry.simplify(tolerance=tol, preserve_topology=True).buffer(0)

    gdfs_out = {}

    # 1. The geometry for computing the cloud coverage (and albedo), i.e. glacier + simple buffer
    # We will also set the resolution to 2 (to avoid too many vertices in the buffer)
    log.info(f"Computing the glacier buffers for cloud coverage (and albedo) with size {buffers.qc_metrics} m")
    geoms_buffered = _gdf.buffer(buffers.qc_metrics, resolution=2)
    gdfs_out['buffer_clouds'] = geoms_buffered

    # 2. The geometry for computing the NDSI, i.e. non-glacier pixels within the glacier buffer
    log.info("Computing the glacier buffers for NDSI (non-glacier pixels within the glacier buffer)")
    gdfs_out['buffer_ndsi'] = gpd.overlay(
        gpd.GeoDataFrame(geometry=geoms_buffered),
        _gdf,
        how='difference',
        keep_geom_type=False
    ).geometry

    # 3. The geometry from which the patch centres will be sampled (simple buffer)
    gdfs_out['buffer_patch_sampling'] = _gdf.buffer(buffers.patch_sampling, resolution=2)

    # The next geometries require non-overlapping buffers:
    # 4. The geometry within which the inference will be performed
    log.info(f"Computing the non-overlapping buffers for inference with size {buffers.infer} m")
    if buffers.infer > 0:
        geoms_infer = utils.buffer_non_overlapping(_gdf, buffer_distance=buffers.infer, grid_size=tol)
    else:
        geoms_infer = gdf.geometry.buffer(0)  # just a copy of the original geometries

    # 5. The geometry within which false positives will be calculated
    log.info(f"Computing the non-overlapping buffers for false positives with sizes {buffers.fp}")

    # If the FP buffer is the same as the infer buffer, we can use the same geometry
    if buffers.fp[0] == buffers.infer:
        geoms_fp_min = geoms_infer.copy()
    elif buffers.fp[0] == 0:
        geoms_fp_min = gdf.geometry.buffer(0)  # just a copy of the original geometries
    else:
        geoms_fp_min = utils.buffer_non_overlapping(_gdf, buffer_distance=buffers.fp[0], grid_size=tol)
    geoms_fp_max = utils.buffer_non_overlapping(_gdf, buffer_distance=buffers.fp[1], grid_size=tol)

    # Take the difference between the two geometries to get the FP buffer
    geoms_fp = geoms_fp_max.difference(geoms_fp_min)
    gdfs_out['buffer_infer'] = geoms_infer
    gdfs_out['buffer_fp'] = geoms_fp

    # Covert the geometries to GeoDataFrames (we will export them as layers) and reproject them to WGS84
    for label, geom in gdfs_out.items():
        gdfs_out[label] = gpd.GeoDataFrame(
            {'entry_id': gdf.entry_id}, geometry=geom.buffer(0), crs=gdf.crs  # make sure the geometries are valid
        )

    return gdfs_out


def main(
        fp_in: str | Path,
        fp_out: str | Path,
        min_glacier_area: float,
        buffers: BaseDatasetCfg.Buffers,
        crs: str,
        gsd: float,
        dates_csv: Optional[str | Path] = None,
):
    """
    Adds the required columns to the GeoDataFrame and initializes the glacier-directories.

    It also prepares a series of buffers for each glacier, which will be used later in the patch sampling, inference,
    and evaluation.

    See :class:`dl4gam.configs.datasets.BaseDatasetCfg.Buffers`
    """

    fp_out = Path(fp_out)
    if fp_out.exists():
        log.warning(f"Output file {fp_out} already exists. Delete it if you want to reprocess the outlines.")
        return

    log.info(f"Reading from {fp_in}")
    gdf = gpd.read_file(fp_in)

    # Add a zero buffer to the geometries to fix potential issues with the geometries
    gdf['geometry'] = gdf.geometry.buffer(0)

    # Get rid of the z-coordinates if present
    gdf['geometry'] = gdf.geometry.force_2d()

    # Use (temporarily) a projected CRS for the processing; in case crs is 'UTM' we use a global equal-area projection
    # because we don't know the UTM zone yet and there might be multiple of them in the dataset
    crs_projected = crs if crs != 'UTM' else gdf.crs.srs if gdf.crs.is_projected else 'ESRI:54034'
    log.info(f"Reprojecting (temporarily) to {crs_projected}")
    gdf = gdf.to_crs(crs_projected)

    # Check if we have the minimum required columns
    required_columns = ['geometry', 'entry_id', 'date_inv']
    missing_columns = [col for col in required_columns if col not in gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Add the auxiliary columns
    gdf = add_auxiliary_columns(gdf)

    # Reorder the columns; keep date_inv as string
    _c = ['entry_id', 'entry_id_i', 'date_inv', 'area_km2']
    gdf = gdf[_c + [col for col in gdf.columns if col not in _c]]
    gdf['date_inv'] = gdf['date_inv'].apply(
        lambda d: d.strftime('%Y-%m-%d') if isinstance(d, (np.datetime64, pd.Timestamp)) else d
    )

    # Set the final CRS for the processing of the outlines
    gdf['crs_epsg'] = crs if crs != 'UTM' else gdf['utm_epsg']

    # Select the glaciers with the area larger than the minimum required
    glaciers_to_process = list(gdf[gdf['area_km2'] >= min_glacier_area].entry_id)
    log.info(
        f"Selected glaciers with area larger than {min_glacier_area} km^2: "
        f"{(n_crt := len(glaciers_to_process))} / {(n := len(gdf))} glaciers ({n_crt / n:.2%})"
    )

    # Load the csv with the externally provided acquisition dates if available
    if dates_csv is not None:
        log.info(f"Reading the acquisition dates from {dates_csv}")
        dates_df = pd.read_csv(dates_csv, index_col='entry_id')

        # Check if all the selected glaciers are present in the dates dataframe
        missing_entries = set(glaciers_to_process) - set(dates_df.index)
        if missing_entries:
            log.warning(
                f"No dates provided for {len(missing_entries)} glaciers: {missing_entries}. "
                f"We will skip these glaciers in the processing."
            )

        # Filter the GeoDataFrame to keep only the selected glaciers with provided dates
        glaciers_to_process = sorted(set(glaciers_to_process) & set(dates_df.index))

        # Save the dates as a new column in the GeoDataFrame
        gdf['date_acq'] = gdf['entry_id'].map(dates_df['date'])
    else:
        if 'date_acq' in gdf.columns:
            log.info("Acquisition dates already present in the GeoDataFrame, using them.")
        else:
            # Assume we will use exactly the same dates as the inventory date
            log.info("No acquisition dates provided, using the inventory date as the acquisition date.")
            gdf['date_acq'] = gdf['date_inv']

    # Filter the GeoDataFrame to keep only the selected glaciers
    gdf_sel = gdf[gdf['entry_id'].isin(glaciers_to_process)]

    # Compute the upper bound FP buffer if needed (provide both all glaciers and selected glaciers)
    if buffers.fp[1] == 'auto':
        log.info(f"Computing the upper bound for the FP buffer (this might take a while)")
        gdf_rest = gdf[~gdf['entry_id'].isin(glaciers_to_process)]
        lim_max = utils.calculate_equal_area_buffer(gdf_sel, gdf_rest, start_distance=buffers.fp[0], step=gsd)
        log.info(f"Upper bound for the FP buffer: {lim_max} m")
        buffers.fp = (buffers.fp[0], lim_max)

    log.info("Preparing the buffered outlines (using all the glaciers)")
    gdfs_buffers = compute_buffers(gdf=gdf, buffers=buffers, tol=gsd / 2.5)

    # Store the selected glacier outlines separately
    gdfs_out = {
        'glacier_sel': gdf_sel,
        'glacier_all': gdf
    }
    gdfs_out.update(gdfs_buffers)

    # Subset also the buffered outlines to the selected glaciers; reproject all to WGS84
    for _label, _gdf in gdfs_out.items():
        if 'buffer' in _label:
            gdfs_out[_label] = _gdf[_gdf['entry_id'].isin(glaciers_to_process)]
        gdfs_out[_label] = gdfs_out[_label].to_crs('epsg:4326')

    # Export the processed outlines with the auxiliary columns (after reprojecting to WGS84)
    fp_out = Path(fp_out)
    log.info(f"Exporting the processed outlines to {fp_out}")
    if fp_out.exists():
        log.warning(f"Output file {fp_out} already exists. Deleting it.")
        fp_out.unlink()
    else:
        fp_out.parent.mkdir(parents=True, exist_ok=True)
    for _label, _gdf in gdfs_out.items():
        _gdf.to_file(fp_out, layer=_label, driver='GPKG')
