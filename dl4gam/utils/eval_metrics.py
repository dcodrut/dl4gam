from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


def compute_stats(fp_geoms: str | Path, fp_debris: Optional[str | Path] = None) -> pd.DataFrame:
    gdf_pos = gpd.read_file(fp_geoms, layer='pos')
    gdf_neg = gpd.read_file(fp_geoms, layer='neg')
    gdf_tp = gpd.read_file(fp_geoms, layer='tp')
    gdf_fp = gpd.read_file(fp_geoms, layer='fp')
    gdf_fn = gpd.read_file(fp_geoms, layer='fn')

    # Keep only the TP predictions and add some stats
    df_stats = pd.DataFrame(index=gdf_tp.entry_id)
    df_stats['P'] = gdf_pos.area_km2.values
    df_stats['TP'] = gdf_tp.area_km2.values
    df_stats['FN'] = gdf_fn.area_km2.values
    df_stats['N'] = gdf_neg.area_km2.values
    df_stats['FP'] = gdf_fp.area_km2.values

    # Compute the total predicted area (TP + FP)
    df_stats['pred'] = df_stats.TP + df_stats.FP

    # Compute the rates
    df_stats['TPR'] = df_stats.TP / df_stats.P
    df_stats['FNR'] = df_stats.FN / df_stats.P
    df_stats['FPR'] = df_stats.FP / df_stats.N

    # If debris geometries are provided, compute the debris metrics
    if fp_debris is not None and Path(fp_debris).exists():
        gdf_debris = gpd.read_file(fp_debris)
        df_stats_debris = compute_debris_metrics(gdf_pos, gdf_tp, gdf_debris)
        df_stats = df_stats.join(df_stats_debris, how='left')

    return df_stats


def _areas_by_id_km2(gdf):
    """Compute per-row areas in km^2 using per-feature projected CRS from the column 'crs_epsg',
    (strings like 'epsg:32633'), then sum by `entry_id`."""
    if gdf.empty:
        return pd.DataFrame({'entry_id': [], 'area_km2': []})

    parts = []
    for code, sub in gdf.dropna(subset=['crs_epsg']).groupby('crs_epsg'):
        code_use = str(code).upper()  # 'epsg:32633' -> 'EPSG:32633'
        sub_proj = sub.to_crs(code_use)
        areas = sub_proj.geometry.area / 1e6  # m^2 -> km^2
        parts.append(pd.DataFrame({'entry_id': sub['entry_id'].values, 'area_km2': areas.values}))

    if not parts:
        return pd.DataFrame({'entry_id': gdf['entry_id'].unique(), 'area_km2': 0.0})

    out = pd.concat(parts, ignore_index=True)
    out = out.groupby('entry_id', as_index=False)['area_km2'].sum()
    return out


def compute_debris_metrics(gdf_gt, gdf_pred, gdf_debris):
    """
    Return a Pandas DataFrame with columns:
    area_km2, area_debris, debris_coverage_f, TP_debris (indexed like gdf_gt).
    """

    # Make sure all GeoDataFrames use the same CRS
    if gdf_pred.crs != gdf_gt.crs:
        gdf_pred = gdf_pred.to_crs(gdf_gt.crs)

    if gdf_debris.crs != gdf_gt.crs:
        gdf_debris = gdf_debris.to_crs(gdf_gt.crs)

    # Make sure the two GeoDataFrames area aligned (i.e. same entry_id order)
    if not gdf_pred.entry_id.equals(gdf_gt.entry_id):
        raise ValueError("gdf_pred and gdf_gt must have the same entry_id values in the same order.")

    # Select the relevant columns
    gt = gdf_gt[['entry_id', 'area_km2', 'crs_epsg', 'geometry']].copy()
    pred = gdf_pred[['entry_id', 'geometry']].copy()
    debris = gdf_debris[['geometry']].copy()

    # Make geometries valid just in case
    gt['geometry'] = gt.geometry.buffer(0)
    pred['geometry'] = pred.geometry.buffer(0)
    debris['geometry'] = debris.geometry.buffer(0)

    # 1. Compute the debris coverage within the GT outlines and its relative fraction to the total glacier area
    gt_debris = gpd.overlay(gt, debris, how='intersection', keep_geom_type=False)
    if gt_debris.empty:
        area_debris_df = pd.DataFrame({'entry_id': gt['entry_id'], 'area_debris': 0.0})
    else:
        areas1 = _areas_by_id_km2(gt_debris[['entry_id', 'crs_epsg', 'geometry']])
        area_debris_df = areas1.rename(columns={'area_km2': 'area_debris'})

    out = gt[['entry_id', 'area_km2']].merge(area_debris_df, on='entry_id', how='left')
    out['area_debris'] = out['area_debris'].fillna(0.0)
    out['debris_coverage_f'] = out['area_debris'] / out['area_km2']

    # 2. Compute the TP debris area (i.e. debris within both GT and predictions)
    debris_in_gt = gpd.clip(debris, gt)  # keep debris only within GT in case debris comes from a different dataset
    debris_in_gt['geometry'] = debris_in_gt.geometry.buffer(0)
    if debris_in_gt.empty:
        tp_df = pd.DataFrame({'entry_id': pred['entry_id'], 'TP_debris': 0.0})
    else:
        pred_with_epsg = pred.merge(gt[['entry_id', 'crs_epsg']], on='entry_id', how='left')
        pred_debris = gpd.overlay(pred_with_epsg, debris_in_gt, how='intersection', keep_geom_type=False)
        if pred_debris.empty:
            tp_df = pd.DataFrame({'entry_id': pred['entry_id'], 'TP_debris': 0.0})
        else:
            areas2 = _areas_by_id_km2(pred_debris[['entry_id', 'crs_epsg', 'geometry']])
            tp_df = areas2.rename(columns={'area_km2': 'TP_debris'})

    # Build a dataframe indexed by entry_id with all the metrics
    out = (
        out.merge(tp_df, on='entry_id', how='left')
        .fillna({'TP_debris': 0.0})
        .set_index('entry_id')[['area_km2', 'area_debris', 'debris_coverage_f', 'TP_debris']]
    )
    out['TPR_debris'] = np.where(out['area_debris'] > 1e-5, out['TP_debris'] / out['area_debris'], np.nan)
    return out
