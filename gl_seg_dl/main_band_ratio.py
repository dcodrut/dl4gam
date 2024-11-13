"""
    Computes the best thresholds for the R/SWIR ratio either regionally, for each split (using the combined training
    and validation folds) or independently for each glacier using a single split.
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# local imports
from config import C
from task.data import extract_inputs
from utils.general import run_in_parallel
from utils.postprocessing import nn_interp

input_settings = {
    'bands_input': ['B4', 'B3', 'B2', 'B8', 'B11'],
    'bands_mask': ['~FILL_MASK', '~CLOUDLESS_MASK'],
    'dem': False,
    'dhdt': False,
    'optical_indices': False,
    'dem_features': False,
    'velocity': False
}


def band_ratio(filepath, r_swir_thr_step=0.1):
    nc = xr.open_dataset(filepath)
    stats = {'entry_id': [], 'glacier_area': [], 'thr': [], 'iou': []}

    # compute the band ratio (red / SWIR)
    r = nc.band_data.isel(band=nc.band_data.long_name.index('B4')).values
    swir = nc.band_data.isel(band=nc.band_data.long_name.index('B11')).values

    # keep only the clean pixels
    mask_no_data = extract_inputs(fp=filepath, ds=nc, input_settings=input_settings)['mask_no_data']
    r = r[~mask_no_data]
    swir = swir[~mask_no_data]

    # R/SWIR ratio
    ratio = r / np.maximum(swir, 1)

    # ground truth
    y_true = (~np.isnan(nc.mask_all_g_id.values))[~mask_no_data]

    for thr in np.arange(0.5, 5.5 + r_swir_thr_step, r_swir_thr_step):
        y_pred = (ratio >= thr)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
        iou = tp / (tp + fp + fn)
        stats['entry_id'].append(filepath.parent.name)
        stats['glacier_area'].append(nc.attrs['glacier_area'])
        stats['thr'].append(thr)
        stats['iou'].append(iou)

    return pd.DataFrame(stats)


def compute_regional_thresholds(root_dir_patches, num_cv_folds, r_swir_thr_step=0.1, num_procs=1):
    """
        Compute the best threshold for the R/SWIR ratio regionally, using the combined training and validation patches,
        independently for each validation split.
    """

    best_thrs = {'split': [], 'thr': [], 'iou': [], 'w_iou': []}
    for i_split in range(1, num_cv_folds + 1):
        dir_patches_crt_split = Path(root_dir_patches) / f'split_{i_split}'

        # get all the training and validation patches for the current split
        fp_list_train = list(Path(dir_patches_crt_split / 'fold_train').rglob('*.nc'))
        fp_list_valid = list(Path(dir_patches_crt_split / 'fold_valid').rglob('*.nc'))
        fp_list = sorted(fp_list_train + fp_list_valid)

        # compute the best threshold for each patch in parallel
        all_df_stats = run_in_parallel(
            fun=functools.partial(band_ratio, r_swir_thr_step=r_swir_thr_step),
            filepath=fp_list,
            num_procs=num_procs,
            pbar=True
        )
        df_stats = pd.concat(all_df_stats)

        # compute the average per glacier
        df_stats = df_stats.groupby(['entry_id', 'thr']).mean().reset_index()

        # weight the IOU based on the glaciers' areas
        total_area = df_stats.groupby('entry_id').first().glacier_area.sum()
        df_stats['w'] = df_stats.glacier_area / total_area * len(set(df_stats.entry_id))
        df_stats['w_iou'] = df_stats.iou * df_stats.w

        # compute the best threshold using the weighted IOU
        thr_best = df_stats.groupby('thr').w_iou.mean().idxmax()

        # save the best threshold and the corresponding IOU
        best_thrs['split'].append(f'split_{i_split}')
        best_thrs['thr'].append(thr_best)
        best_thrs['iou'].append(df_stats[df_stats.thr == thr_best].iou.mean())
        best_thrs['w_iou'].append(df_stats[df_stats.thr == thr_best].w_iou.mean())

    return pd.DataFrame(best_thrs)


def compute_glacier_wide_thresholds(root_dir_patches, num_cv_folds, r_swir_thr_step=0.005, num_procs=1):
    """
        Compute the best threshold for the R/SWIR ratio independently for each glacier, using the testing fold patches.
    """
    # we will compute a threshold independently for each glacier in the test fold
    best_thrs = {'entry_id': [], 'thr': [], 'iou': []}

    for i_split in range(1, num_cv_folds + 1):
        dir_patches_crt_split = Path(root_dir_patches) / f'split_{i_split}'

        # get all the testing patches for the current split
        fp_list = sorted(list(Path(dir_patches_crt_split / 'fold_test').rglob('*.nc')))

        # compute the best threshold for each patch in parallel
        all_df_stats = run_in_parallel(
            fun=functools.partial(band_ratio, r_swir_thr_step=r_swir_thr_step),
            filepath=fp_list,
            num_procs=num_procs,
            pbar=True
        )
        df_stats = pd.concat(all_df_stats)
        df_stats_best_iou = df_stats.sort_values('iou', ascending=False).groupby('entry_id').first()

        best_thrs['entry_id'].extend(df_stats_best_iou.index)
        best_thrs['thr'].extend(df_stats_best_iou.thr)
        best_thrs['iou'].extend(df_stats_best_iou.iou)

    return pd.DataFrame(best_thrs)


if __name__ == "__main__":
    results_dir_root = Path('../data/external/_experiments')
    out_dir_root = Path(C.DIR_GL_PATCHES).parent.parent / 'aux_data' / 'band_ratio_stats' / Path(C.DIR_GL_PATCHES).name
    print(f"out_dir_root = {out_dir_root}")
    for thr_type in ('regional', 'glacier-wide'):
        fp_out = out_dir_root / f'best_band_ratio_thrs_{thr_type}.csv'

        if not fp_out.exists():
            if thr_type == 'regional':
                df_best_thrs = compute_regional_thresholds(
                    root_dir_patches=C.DIR_GL_PATCHES,
                    num_cv_folds=C.NUM_CV_FOLDS,
                    r_swir_thr_step=0.1,
                    num_procs=C.NUM_PROCS
                )
            else:
                df_best_thrs = compute_glacier_wide_thresholds(
                    root_dir_patches=C.DIR_GL_PATCHES,
                    num_cv_folds=C.NUM_CV_FOLDS,
                    r_swir_thr_step=0.1,
                    num_procs=C.NUM_PROCS
                )
                df_best_thrs = df_best_thrs.sort_values('entry_id')
            fp_out.parent.mkdir(parents=True, exist_ok=True)
            df_best_thrs.to_csv(fp_out, index=False)
            print(f"Best thresholds saved to {fp_out}")
        else:
            df_best_thrs = pd.read_csv(fp_out, dtype={'entry_id': str})
            print(f"Best thresholds loaded from {fp_out}")

        print(f"Best thresholds for the R/SWIR ratio ({thr_type}):")
        print(df_best_thrs)
        # now apply the thresholds to the test set; save the "predictions" in the same format as for the trained models

        # first, prepare the testing glaciers using the split dataframe
        split_fp = Path(C.DIR_GL_INVENTORY).parent.parent / 'cv_split_outlines' / 'map_all_splits_all_folds.csv'
        split_df = pd.read_csv(split_fp, dtype={'entry_id': str})

        model_name = f"band_ratio_{thr_type}"
        for subdir in ('inv', '2023'):  # run on both inference subdirectories
            for i_split in range(1, C.NUM_CV_FOLDS + 1):
                glacier_ids = sorted(list(split_df[split_df[f'split_{i_split}'] == 'fold_test'].entry_id))
                ds_name = Path(C.WD).name

                # keep the same directory structure as for the trained models
                model_dir = results_dir_root / ds_name / model_name / f'split_{i_split}' / 'seed_0' / 'version_0'

                # export the input settings to YAML, similar to the train models (will be needed in the evaluation)
                input_settings_fp = model_dir / 'settings.yaml'
                input_settings_fp.parent.mkdir(parents=True, exist_ok=True)
                with open(input_settings_fp, 'w') as fp:
                    yaml.dump({
                        'data': {'rasters_dir': str(C.DIR_GL_INVENTORY)},
                        'model': {'inputs': input_settings}
                    }, fp, sort_keys=False)

                preds_dir = model_dir / 'output' / 'preds' / ds_name / subdir / 's_test'
                print(f"preds_dir = {preds_dir}")
                for entry_id in tqdm(glacier_ids, desc=f"subdir = {subdir}; split_{i_split}"):
                    # get the nc for the current glacier
                    dir_crt_g = Path(C.WD) / subdir / 'glacier_wide' / entry_id
                    fp_list = list(dir_crt_g.glob("*.nc"))
                    assert len(fp_list) == 1, f"Expected 1 file in {dir_crt_g}, got {len(fp_list)}"
                    fp = fp_list[0]
                    nc = xr.open_dataset(fp, decode_coords='all')

                    # get the best threshold for the current split
                    if thr_type == 'regional':
                        idx = (df_best_thrs.split == f'split_{i_split}')
                    else:
                        idx = (df_best_thrs.entry_id == entry_id)
                    assert sum(idx) == 1, "There should be a single entry here."
                    thr = df_best_thrs[idx].thr.values[0]

                    # get the predictions
                    r = nc.band_data.isel(band=nc.band_data.long_name.index('B4')).values
                    swir = nc.band_data.isel(band=nc.band_data.long_name.index('B11')).values
                    ratio = r / np.maximum(swir, 1)
                    preds_b = ratio > thr

                    # store the predictions as xarray based on the original nc
                    nc_pred = nc.copy()
                    nc_pred['r_swir_ratio'] = (('y', 'x'), ratio.astype(np.float32))
                    nc_pred['pred'] = (('y', 'x'), preds_b)  # no probabilities, just the binary prediction
                    nc_pred['pred_b'] = (('y', 'x'), preds_b)

                    # fill-in the masked pixels (only within 50m buffer) using the NN interpolation
                    mask_to_fill = extract_inputs(ds=nc, fp=fp, input_settings=input_settings)['mask_no_data']
                    mask_crt_g_b50 = (nc.mask_crt_g_b50.values == 1)
                    mask_to_fill &= mask_crt_g_b50
                    mask_ok = (~mask_to_fill) & mask_crt_g_b50

                    n_px = 30  # how many pixels to use as source for interpolation value
                    if mask_to_fill.sum() > 0 and mask_ok.sum() >= n_px:
                        # use the nearest neighbours
                        data_interp = nn_interp(data=preds_b, mask_to_fill=mask_to_fill, mask_ok=mask_ok, num_nn=n_px)
                        nc_pred['pred_i_nn'] = (('y', 'x'), data_interp.astype(np.float32))
                        nc_pred['pred_i_nn_b'] = (('y', 'x'), data_interp >= 0.5)

                    # add the CRS to the new data arrays
                    for c in [_c for _c in nc_pred.data_vars if 'pred' in _c] + ['r_swir_ratio']:
                        nc_pred[c].rio.write_crs(nc_pred.rio.crs, inplace=True)

                    # drop the data variables to save space
                    nc_pred = nc_pred[[c for c in nc_pred.data_vars if 'pred' in c or 'mask' in c] + ['r_swir_ratio']]

                    out_fp = preds_dir / entry_id / fp.name
                    out_fp.parent.mkdir(parents=True, exist_ok=True)
                    out_fp.unlink(missing_ok=True)
                    nc_pred.to_netcdf(out_fp)
