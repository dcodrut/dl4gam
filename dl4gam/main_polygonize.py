"""
    Reads the predictions of a model and transforms the binary masks into multi-polygons exported to a shapefile.
    The results are collected from all the splits.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely.ops
import xarray as xr
from tqdm import tqdm

# local imports
from config import C


def parse_args():
    parser = argparse.ArgumentParser(description="Polygonize model predictions")
    parser.add_argument(
        '--model_dir', type=str,
        help='The root directory of the model outputs (e.g. /path/to/experiments/dataset/model_name)'
    )
    parser.add_argument(
        '--model_version', type=str, required=True,
        help='Version of the model'
    )
    parser.add_argument(
        '--split_list', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='The list of cross-validation iterations (e.g. [1, 2, 3, 4, 5])'
    )
    parser.add_argument(
        '--seed', type=str, required=True,
        help='Model training seed (set it to "all" for using the ensemble average results)'
    )
    parser.add_argument(
        '--use_calib', action='store_true',
        help='Flag for using the calibrated predictions'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_version = args.model_version
    split_list = list(args.split_list)
    seed = args.seed
    preds_version = 'preds_calib' if args.use_calib else 'preds'

    print(f"Model directory: {model_dir}")

    # build a dataframe with the paths to the predictions
    ds_name = Path(C.WD).name
    subdir = C.SUBDIR
    df_all = []
    for i_split in split_list:
        infer_dir = (
                model_dir / f'split_{i_split}' / f'seed_{seed}' / model_version /
                'output' / preds_version / ds_name / subdir / 's_test'
        )
        assert infer_dir.exists(), f"Inferences directory not found: {infer_dir}"
        print(f"Reading predictions from {infer_dir}")
        fp_list = sorted(list(infer_dir.glob('**/*.nc')))
        df_fp_crt = pd.DataFrame({
            'split': i_split,
            'seed': seed,
            'fp': fp_list,
        })
        df_all.append(df_fp_crt)
    df_all = pd.concat(df_all)
    print(df_all)

    # polygonize the predictions
    geom_all = {k: [] for k in ['entry_id', 'geometry', 'nc_crs', 'image_id', 'image_date', 'label', 'split']}
    for i in tqdm(range(len(df_all)), desc=f"Polygonizing predictions"):
        fp = df_all.fp.iloc[i]
        nc = xr.open_dataset(fp, decode_coords='all')

        # get the predictions (check if an interpolation was done)
        preds = nc.pred_b.values

        # set to 0 the pixels outside the 20m buffer or those that belong to other glaciers
        mask_crt_g = (nc.mask_crt_g.values == 1)
        mask_crt_g_b20 = (nc.mask_crt_g_b20.values == 1)
        mask_other_g = (~np.isnan(nc.mask_all_g_id.values)) & (~mask_crt_g)
        preds[mask_other_g | ~mask_crt_g_b20] = 0
        to_export = {preds_version: preds}

        # if there is a stddev among the variables, then the predictions come from an ensemble
        # in that case, compute the lower and higher bounds of the glacier extent
        if 'pred_low_b' in nc:
            preds_low = nc.pred_low_b.values
            preds_high = nc.pred_high_b.values

            # set to 0 the pixels that belong to other glaciers
            preds_low[mask_other_g | ~mask_crt_g_b20] = 0
            preds_high[mask_other_g | ~mask_crt_g_b20] = 0
            to_export[f'{preds_version}_low'] = preds_low
            to_export[f'{preds_version}_high'] = preds_high

        # polygonize the predictions
        for k, v in to_export.items():
            # apply sieve filter to remove connected components smaller than 10 pixels
            min_size_threshold = 10
            sieved_data = rasterio.features.sieve(to_export[k].astype(np.uint8), min_size_threshold)

            # get the shapes and convert them to polygons
            shapes = list(rasterio.features.shapes(sieved_data, transform=nc.rio.transform()))
            geometries = [shapely.geometry.shape(shape) for shape, value in shapes if value == 1]

            # check if nothing was found
            multipoly = shapely.ops.unary_union(geometries) if len(geometries) > 0 else shapely.geometry.Polygon()

            # combine the polygons into a single one and create a geodataframe
            geom_all['entry_id'].append(fp.parent.name)
            geom_all['geometry'].append(multipoly)
            geom_all['nc_crs'].append(nc.rio.crs)
            geom_all['image_id'].append(fp.stem)
            geom_all['image_date'].append(pd.to_datetime(fp.stem[:8], format='%Y%m%d').strftime('%Y-%m-%d'))
            geom_all['label'].append(k)
            geom_all['split'].append(f"split_{df_all.split.iloc[i]}")

    # export the geodataframe(s), separated by the label
    gdf = gpd.GeoDataFrame(geom_all)

    # sort by entry_id
    gdf = gdf.sort_values(by='entry_id')

    # reproject to WGS84 (from the raster crs)
    gdf = gdf.groupby('nc_crs', group_keys=False, sort=False)[[c for c in gdf.columns]].apply(
        lambda sdf: sdf.set_crs(crs=sdf.nc_crs.iloc[0]).to_crs(epsg=4326)
    )
    gdf = gdf.drop(columns=['nc_crs'])
    for k in gdf.label.unique():
        gdf_crt = gdf[gdf.label == k]
        fp_out = model_dir / 'gdf_all_splits' / f'seed_{seed}' / model_version / ds_name / subdir / f'{subdir}_{k}.shp'
        fp_out.parent.mkdir(exist_ok=True, parents=True)
        gdf_crt.to_file(fp_out)
        print(f"Vectors exported to {fp_out}")
