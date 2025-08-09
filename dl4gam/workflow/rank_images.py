import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from dl4gam.configs.datasets import QCMetric
from dl4gam.utils import rank_images, run_in_parallel

log = logging.getLogger(__name__)


def main(
        raw_data_base_dir: str | Path,
        year: str | int,
        geoms_fp: str | Path,
        sort_by: tuple[QCMetric, ...],
        max_cloud_p: float = 1.0,
        min_coverage: float = 0.9,
        score_weights: Optional[tuple] = None,
        buffer: int = 0,
        bands_name_map: Optional[dict[str, str]] = None,
        bands_nok_mask: Optional[tuple[str]] = None,
        overwrite: bool = False,
):
    """
    In case the images were not automatically downloaded from Google Earth Engine using
    `dl4gam.workflow.download_gee_data` (which already computes and exports the QC statistics), this script can be used
    to compute the statistics offline and rank the images.

    We load all the images for each glacier, compute the QC statistics (e.g. cloud coverage, NDSI) and rank them.
    The stats of all images are saved in a csv file ('metadata.csv') in the corresponding glacier directory.
    Then, after filtering & ranking, we export `metadata_filtered.csv` with the ranked images for each glacier.
    """

    log.info(f"Reading the glacier outlines from {geoms_fp}")
    gdf = gpd.read_file(geoms_fp, layer='glacier_sel')
    gdf_all = gpd.read_file(geoms_fp, layer='glacier_all')

    # Load the geometries that will be used to build the binary masks for the QC metrics
    log.info(f"Reading the QC geometries from {geoms_fp} ('buffer_clouds' and 'buffer_ndsi' layers)")
    gdf_clouds = gpd.read_file(geoms_fp, layer='buffer_clouds')
    gdf_albedo = gdf_clouds  # we use the same buffered geometries for albedo as for clouds
    gdf_ndsi = gpd.read_file(geoms_fp, layer='buffer_ndsi')
    gdf_qc = {
        'qc_roi_cloud_p': gdf_clouds.set_index('entry_id').reindex(gdf.entry_id),
        'qc_roi_ndsi': gdf_ndsi.set_index('entry_id').reindex(gdf.entry_id),
        'qc_roi_albedo': gdf_albedo.set_index('entry_id').reindex(gdf.entry_id),
    }

    # Save all the QC geoms as list of subsets of GeoDataFrames, one for each glacier (for run_in_parallel)
    extra_gdf_per_glacier = [{k: _gdf.iloc[i:i + 1] for k, _gdf in gdf_qc.items()} for i in range(len(gdf))]

    # Get all the glacier directories
    years = list(gdf.date_inv.apply(lambda x: pd.to_datetime(x).year)) if year == 'inv' else [year] * len(gdf)
    glacier_dirs = [Path(raw_data_base_dir) / str(y) / str(gid) for gid, y in zip(gdf.entry_id, years)]

    run_in_parallel(
        fun=rank_images,
        min_coverage=min_coverage,
        max_cloud_p=max_cloud_p,
        sort_by=sort_by,
        score_weights=score_weights,
        raw_images_dir=glacier_dirs,
        gl_df=gdf_all,
        extra_geodataframes=extra_gdf_per_glacier,
        buffer=buffer,
        bands_name_map=bands_name_map,
        bands_nok_mask=bands_nok_mask,
        overwrite=overwrite
    )
