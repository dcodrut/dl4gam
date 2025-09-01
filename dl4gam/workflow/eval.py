import logging
from pathlib import Path

import geopandas as gpd
import pyproj

from dl4gam.configs.config import BaseDatasetCfg
from dl4gam.utils import compute_stats

log = logging.getLogger(__name__)


def main(
        dataset_cfg: BaseDatasetCfg,
        fold: str,
        checkpoint_dir: str | Path,
):
    # Get the predicted geometries
    ds = dataset_cfg  # TODO: allow different dataset than the one used for training
    fp_geoms_pred = checkpoint_dir.parent / 'preds' / ds.name / str(ds.year) / fold / 'preds_polygons.gpkg'
    if not fp_geoms_pred.parent.exists():
        raise FileNotFoundError(f"Predictions geometries directory not found: {fp_geoms_pred.parent}")
    gdf_pred = gpd.read_file(fp_geoms_pred).set_index('entry_id')
    gdf_pred.geometry = gdf_pred.buffer(0)  # buffer(0) to fix potential invalid geoms
    log.info(f"Loaded {len(gdf_pred)} predicted geometries from {fp_geoms_pred}")

    # Read the following outlines from the processed geometries file:
    # 1, the selected glaciers (i.e. the ground truth)
    # 2, the buffer used for computing the FP (i.e. the area where we allow FP predictions)
    # 3. the buffer used at deployment time to crop the predictions (i.e. the area where we expect to have predictions)
    gdfs_ref = {}
    for name, layer in [('pos', 'glacier_sel'), ('neg', 'buffer_fp'), ('infer_allowed', 'buffer_infer')]:
        gdf = gpd.read_file(dataset_cfg.geoms_fp, layer=layer).set_index('entry_id')
        gdf.geometry = gdf.geometry.buffer(0)
        log.info(f"Loaded {len(gdf)} geometries for '{name}' from layer '{layer}'")

        # Keep only those for which we have predictions
        gdf = gdf.loc[gdf_pred.index]
        log.info(f"Kept {len(gdf)} geometries for '{name}' after filtering to those with predictions")

        gdfs_ref[name] = gdf

    # Save all the geometries in a single file to make it easier to visualize them together
    gdfs_out = gdfs_ref
    gdfs_out['pred'] = gdf_pred.reset_index()

    # Compute the geometries for TP, FP, FN (no TN as they are not relevant for this task)
    gdfs_out.update({
        'tp': gdfs_ref['pos'].intersection(gdf_pred).buffer(0).to_frame('geometry').reset_index(),
        'fn': gdfs_ref['pos'].difference(gdf_pred).buffer(0).to_frame('geometry').reset_index(),
        'fp': gdf_pred.intersection(gdfs_ref['neg']).buffer(0).to_frame('geometry').reset_index(),
        'infer': gdf_pred.intersection(gdfs_ref['infer_allowed']).buffer(0).to_frame('geometry').reset_index()
    }
    )

    # Compute the areas for the polygons in km2 and export the geometries to a GeoPackage
    geod = pyproj.Geod(ellps='WGS84')
    fp_out = checkpoint_dir.parent / 'eval' / f"geoms_dataset={ds.name}_year={ds.year}_fold={fold}.gpkg"
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    for k in gdfs_out:
        gdfs_out[k]['area_km2'] = gdfs_out[k].geometry.apply(
            lambda geom: abs(geod.geometry_area_perimeter(geom)[0]) / 1e6 if not geom.is_empty else 0.0
        )
        gdfs_out[k].to_file(fp_out, layer=k, driver='GPKG')
    log.info(f"Exported the evaluation geometries to {fp_out} (layers: {list(gdfs_out.keys())})")

    # Compute the total areas for TP, FP, FN
    area_totals = {k: float(round(gdfs_out[k].area_km2.sum(), 2)) for k in gdfs_out}
    log.info(f"Total areas (km^2): {area_totals}")

    # Get the debris geometries if available (to compute debris-specific metrics)
    fp_debris = ds.extra_vectors['debris'] if ds.extra_vectors and 'debris' in ds.extra_vectors else None

    log.info("Computing evaluation statistics")
    df_stats = compute_stats(fp_geoms=fp_out, fp_debris=fp_debris)
    log.info(f"Summary statistics:\n{df_stats.describe(percentiles=[]).T}")
    fp_out_stats = fp_out.parent / f"stats_dataset={ds.name}_year={ds.year}_fold={fold}.csv"
    df_stats.to_csv(fp_out_stats, index=True)
    log.info(f"Exported the evaluation statistics to {fp_out_stats}")
