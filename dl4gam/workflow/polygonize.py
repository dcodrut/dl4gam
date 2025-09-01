"""
    Reads the predictions of a model and transforms the binary masks into multi-polygons exported to a shapefile.
    The results are collected from all the splits.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pyproj
import xarray as xr

log = logging.getLogger(__name__)

from dl4gam.utils import run_in_parallel, polygonize, extract_date_from_fn


def load_and_polygonize(fp: str | Path, preds_version: str = 'pred_b', min_segment_area_km2: float = 0.0):
    """
    Load the predictions from a NetCDF file and polygonize them.
    :param fp: path to the NetCDF dataset with the predictions
    :param preds_version: variable name to use for the predictions (e.g. 'pred_b', 'pred_low_b', 'pred_high_b')
    :param min_segment_area_km2: minimum area of the polygons to keep; polygons smaller than this will be discarded
    :return: the geometries extracted from the predictions
    """

    with xr.open_dataset(fp, decode_coords='all') as ds:
        # Get the predictions
        da_preds = (ds[preds_version] == 1)

        # Set the predictions to 0 on the pixels outside the infer + FP buffers as we won't use them
        da_preds = da_preds.where((ds.mask_infer == 1) | (ds.mask_fp == 1), False)

        # Apply the polygonization
        geoms = polygonize(da_preds, min_segment_area_km2=min_segment_area_km2)

        return geoms


def main(
        checkpoint_dir: str | Path,
        dataset_name: str,
        year: str | int,
        fold: str,
        min_segment_area_km2: float = 0.0,
):
    # Get the predictions directory
    model_dir = checkpoint_dir.parent
    preds_dir = model_dir / 'preds' / dataset_name / str(year) / fold
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")

    # Load the predictions
    fp_list = sorted(list(preds_dir.rglob('*.nc')))

    # Polygonize the predictions
    geoms_pred = run_in_parallel(
        fun=load_and_polygonize,
        fp=fp_list,
        preds_version='pred_b',
        min_segment_area_km2=min_segment_area_km2,
    )

    gdf = gpd.GeoDataFrame({
        'geometry': geoms_pred,
        'entry_id': [fp.parent.name for fp in fp_list],
        'image_id': [fp.stem for fp in fp_list],
        'image_date': [extract_date_from_fn(fp.stem) for fp in fp_list],
    }, crs='EPSG:4326')

    # Compute the area of the polygons in km2
    geod = pyproj.Geod(ellps="WGS84")
    gdf['area_km2'] = gdf.geometry.apply(lambda geom: abs(geod.geometry_area_perimeter(geom)[0]) / 1e6)

    fp_out = preds_dir / 'preds_polygons.gpkg'
    fp_out.parent.mkdir(exist_ok=True, parents=True)
    gdf.to_file(str(fp_out))
    log.info(f"Polygonized predictions saved to {fp_out}")
