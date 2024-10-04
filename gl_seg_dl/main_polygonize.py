"""
    Reads the predictions of a model and transforms the binary masks into multi-polygons exported in a shapefile.
    The results from all the splits are collected.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely.ops
import xarray as xr
from tqdm import tqdm

# local imports

if __name__ == "__main__":
    results_root_dir = Path('../data/external/experiments_server/s2_alps_plus/unet')

    seed = 'all'
    split_list = [1, 2, 3, 4, 5]
    dir_infer = 's2_alps_plus'
    subdirs_infer = ['inv', '2023']
    version = '0'

    for subdir_infer in subdirs_infer:
        print(f"Preparing the paths to the predictions for subdir {subdir_infer}")
        # build a dataframe with the paths to the predictions
        df_all = []
        for split in split_list:
            model_output_dir = (
                    results_root_dir / f'split_{split}' / f'seed_{seed}' / f'version_{version}' /
                    'output' / 'preds' / dir_infer / subdir_infer / 's_test'
            )
            assert model_output_dir.exists(), f"Model output directory not found: {model_output_dir}"
            fp_list = sorted(list(model_output_dir.glob('**/*.nc')))
            df_fp_crt = pd.DataFrame({
                'split': split,
                'seed': seed,
                'fp': fp_list,
            })
            df_all.append(df_fp_crt)
        df_all = pd.concat(df_all)
        print(df_all)

        # polygonize the predictions
        geom_all = {'entry_id': [], 'geometry': [], 'image_id': [], 'image_date': [], 'label': [], 'split': []}
        for i in tqdm(range(len(df_all)), desc=f"Polygonizing predictions"):
            fp = df_all.fp.iloc[i]
            nc = xr.open_dataset(fp, decode_coords='all')

            # get the predictions (check if an interpolation was done)
            preds = nc.pred_i_nn_b.values if 'pred_i_nn' in nc else nc.pred_b.values

            # set to 0 the pixels outside the 20m buffer or those that belong to other glaciers
            mask_crt_g = (nc.mask_crt_g.values == 1)
            mask_crt_g_b20 = (nc.mask_crt_g_b20.values == 1)
            mask_other_g = (~np.isnan(nc.mask_all_g_id.values)) & (~mask_crt_g)
            preds[mask_other_g | ~mask_crt_g_b20] = 0
            to_export = {'pred': preds}

            # if there is a stddev among the variables, then the predictions come from an ensemble
            # in that case, compute the lower and higher bounds of the glacier extent
            if 'pred_low_b' in nc:
                preds_low = nc.pred_low_b_i_nn.values if 'pred_low_b_i_nn' in nc else nc.pred_low_b.values
                preds_high = nc.pred_high_b_i_nn.values if 'pred_high_b_i_nn' in nc else nc.pred_high_b.values

                # set to 0 the pixels that belong to other glaciers
                preds_low[mask_other_g | ~mask_crt_g_b20] = 0
                preds_high[mask_other_g | ~mask_crt_g_b20] = 0
                to_export['pred_low'] = preds_low
                to_export['pred_high'] = preds_high

            # polygonize the predictions
            for k, v in to_export.items():
                shapes = list(rasterio.features.shapes(to_export[k].astype(np.uint8), transform=nc.rio.transform()))
                geometries = [shapely.geometry.shape(shape) for shape, value in shapes if value == 1]

                # combine the polygons into a single one and create a geodataframe
                multipoly = shapely.ops.unary_union(geometries)
                geom_all['entry_id'].append(fp.parent.name)
                geom_all['geometry'].append(multipoly)
                geom_all['image_id'].append(fp.stem)
                geom_all['image_date'].append(pd.to_datetime(fp.stem[:8], format='%Y%m%d').strftime('%Y-%m-%d'))
                geom_all['label'].append(k)
                geom_all['split'].append(f"split_{df_all.split.iloc[i]}")

        # export the geodataframe(s), separated by the label
        gdf = gpd.GeoDataFrame(geom_all)
        gdf.set_crs(crs=nc.rio.crs, inplace=True)
        for k in gdf.label.unique():
            gdf_crt = gdf[gdf.label == k]
            fp_out = (
                    results_root_dir / 'gdf_all_splits' / f'seed_{seed}' / f'version_{version}' /
                    dir_infer / subdir_infer / f'{subdir_infer}_{k}.shp'
            )
            fp_out.parent.mkdir(exist_ok=True, parents=True)
            gdf_crt.to_file(fp_out)
