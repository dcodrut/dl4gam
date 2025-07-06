import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from dl4gam import utils
from dl4gam.configs.datasets import BaseDatasetConfig

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


def add_buffers(
        entry_id: str,
        gdf: gpd.GeoDataFrame,
        buffers: BaseDatasetConfig.Buffers
):
    """
    Computes the following geometries for a glacier:
    1. Box for the final processed glacier cube
    2. Geometry from which the patch centres will be sampled
    3. Geometry within which the inference will be performed
    4. Geometry within which false positives will be calculated

    :param entry_id: the ID of the current glacier entry
    :param gdf: GeoDataFrame with all the glacier polygons
    :param buffers: Buffers object with the buffer sizes
    :return: a dictionary with the GeoDataFrames for each geometry
    """

    # reproject to the target CRS of the current glacier
    idx_crt_entry = gdf[gdf['entry_id'] == entry_id].index[0]
    target_crs = gdf.loc[idx_crt_entry].crs_epsg
    gdf = gdf.to_crs(target_crs)
    gdf_entry = gdf.loc[idx_crt_entry]

    gdfs_out = {}

    # The next geometries are simple buffers of the current glacier:
    # 1. The box for the final processed glacier cube
    # (should be large enough to cover the sampled patches but small enough to have enough raw data)
    # 2. The geometry from which the patch centres will be sampled
    for k in ['cube', 'patch_sampling']:
        buffer_geom = gdf_entry.geometry.buffer(getattr(buffers, k))

        # For the cube boxe, we use the box of the buffer geometry
        if k  == 'cube':
            buffer_geom = buffer_geom.envelope

        gdfs_out[f'buffer_{k}'] = gpd.GeoDataFrame(
            {'entry_id': [entry_id]}, geometry=[buffer_geom], crs=target_crs
        )

    # The next geometries require non-overlapping buffers (=> we need to use all the neighbours):
    # 3. The geometry within which the inference will be performed
    # 4. The geometry within which false positives will be calculated
    gdf_entry_with_neighbours = gdf[gdf.geometry.intersects(gdfs_out['buffer_cube'].geometry.iloc[0])]
    gdf_infer = utils.buffer_non_overlapping(gdf=gdf_entry_with_neighbours, buffer_distance=buffers.infer)
    gdf_fp_min = utils.buffer_non_overlapping(gdf=gdf_entry_with_neighbours, buffer_distance=buffers.fp[0])
    gdf_fp_max = utils.buffer_non_overlapping(gdf=gdf_entry_with_neighbours, buffer_distance=buffers.fp[1])
    gdf_fp = gdf_fp_max.difference(gdf_fp_min)
    gdfs_out['buffer_infer'] = gpd.GeoDataFrame(
        {'entry_id': [entry_id]}, geometry=[gdf_infer.geometry.loc[idx_crt_entry]], crs=target_crs
    )
    gdfs_out['buffer_fp'] = gpd.GeoDataFrame(
        {'entry_id': [entry_id]}, geometry=[gdf_fp.geometry.loc[idx_crt_entry]], crs=target_crs
    )

    # Covert the GeoDataFrames to WGS84 for the output
    for _label, _gdf in gdfs_out.items():
        gdfs_out[_label] = _gdf.to_crs('epsg:4326')

    return gdfs_out


def main(
        fp_in: str | Path,
        fp_out: str | Path,
        min_glacier_area: float,
        buffers: BaseDatasetConfig.Buffers,
        crs: str,
        gsd: float,
):
    """
    Adds the required columns to the GeoDataFrame and initializes the glacier-directories.

      prepares the following outlines (for each glacier):

    Prepare the outlines within which the inference will be performed.

    See :class:`dl4gam.configs.datasets.BaseDatasetConfig.Buffers`
    """
    log.info(f"Reading from {fp_in}")
    gdf = gpd.read_file(fp_in)

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

    # Compute the upper bound FP buffer if needed (using all the glaciers)
    if buffers.fp[1] == 'auto':
        log.info(f"Computing the upper bound for the FP buffer (this might take a while)")
        lim_max = utils.calculate_equal_area_buffer(
            gdf,
            start_distance=buffers.fp[0] + 5 * gsd,  # start with at least 5 pixels
            step=gsd,
        )
        log.info(f"Upper bound for the FP buffer: {lim_max} m")
        buffers.fp = (buffers.fp[0], lim_max)

    # Select the glaciers with the area larger than the minimum required
    glaciers_to_process = list(gdf[gdf['area_km2'] >= min_glacier_area].entry_id)
    glaciers_to_process = glaciers_to_process[:50]  # TODO: remove this line to process all glaciers
    log.info(
        f"Selected glaciers with area larger than {min_glacier_area} km^2: "
        f"{(n_crt:= len(glaciers_to_process))} / {(n := len(gdf))} glaciers ({n_crt / n:.2%})"
    )
    log.info(f"Preparing the buffered outlines")
    res = utils.run_in_parallel(
        fun=add_buffers,
        pbar_desc="Adding buffers to glacier outlines",
        entry_id=glaciers_to_process,
        gdf=gdf,
        buffers=buffers,
    )
    gdfs_out = {}
    for k in res[0]:
        gdfs_out[k] = gpd.GeoDataFrame(
            pd.concat([r[k] for r in res], ignore_index=True),
            crs='epsg:4326',  # all buffers are in WGS84
        )

    # Add the selected glacier outlines and also all of them
    gdfs_out['glacier_sel'] = gdf[gdf['entry_id'].isin(glaciers_to_process)].to_crs('epsg:4326')
    gdfs_out['glacier_all'] = gdf.to_crs('epsg:4326')

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
