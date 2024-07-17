import itertools
import multiprocessing
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

# local imports
from config import C
from task.data import extract_inputs


def compute_stats(fp, rasters_dir, input_settings, band_target='mask_crt_g', exclude_bad_pixels=True,
                  return_rasters=False):
    stats = {'fp': fp}

    # read the predictions
    nc = xr.open_dataset(fp)

    # read the raw data and add it to the predictions dataset
    ds_name = fp.parent.parent.parent.name
    entry_id = fp.parent.name
    fp_orig = Path(rasters_dir).parent.parent / ds_name / 'glacier_wide' / entry_id / fp.name
    nc_orig = xr.open_dataset(fp_orig)
    for c in nc_orig.data_vars:
        if 'pred' not in c or 'mask' not in c:
            nc[c] = nc_orig[c]
    nc_data = extract_inputs(fp=fp_orig, ds=nc_orig, input_settings=input_settings)

    # hack: assume perfect predictions when plotting only the images and not the results
    if 'pred' not in nc.data_vars:
        nc['pred'] = (nc.mask_crt_g == 1)
        nc['pred_b'] = nc['pred']

    # get the ground truth based on the given target band
    mask = (nc[band_target].values == 1)

    # prepare the scaling constant for area computation in km2
    dx = nc.rio.resolution()[0]
    f_area = (dx ** 2) / 1e6

    # extract the mask for no-data pixels (which depends on the training yaml settings)
    mask_exclude = nc_data['mask_no_data'] if exclude_bad_pixels else np.zeros_like(mask)
    area_ok = np.sum(mask & (~mask_exclude)) * f_area
    stats['area_inv'] = np.sum(nc.mask_crt_g.values) * f_area
    stats['area_ok'] = area_ok
    area_excluded = np.sum(mask & mask_exclude) * f_area
    stats['area_excluded'] = area_excluded

    # get the debris mask and its area
    mask_debris = nc_data['mask_debris_crt_g']
    area_debris = np.sum(mask_debris) * f_area
    stats['area_debris'] = area_debris

    # loop over the original predictions and its interpolated versions, if any
    # interpolation should be found when pixels are missing and there are more than 30 non-masked pixels
    pred_bands = ['pred_b']
    pred_bands += [c for c in ['pred_i_nn_b', 'pred_i_hypso_b'] if c in nc.data_vars]

    for pred_band in pred_bands:
        suffix = pred_band.split('pred')[1].split('_b')[0]

        preds = nc[pred_band].values.copy()

        if pred_band == 'pred_b':  # apply the mask only on the raw predictions
            preds[mask_exclude] = False
            area = area_ok
        else:
            area = area_ok + area_excluded

        area_recalled = np.sum(mask & preds) * f_area
        recall = area_recalled / area if area > 0 else np.nan
        stats[f"area_recalled{suffix}"] = area_recalled
        stats[f"recall{suffix}"] = recall

        # add debris-specific stats
        area_debris_recalled = np.sum(mask_debris & preds) * f_area
        recall_debris = area_debris_recalled / area_debris if area_debris > 0 else np.nan
        stats[f"area_debris_recalled{suffix}"] = area_debris_recalled
        stats[f"recall_debris{suffix}"] = recall_debris

    # compute the FPs for the non-glacierized area (where predictions are made); use the default predictions
    preds = nc['pred_b'].values
    mask_preds_exist = ~np.isnan(nc.pred.values)
    mask_non_g = np.isnan(nc.mask_all_g_id.values) & mask_preds_exist & (~mask_exclude)
    area_non_g = np.sum(mask_non_g) * f_area
    mask_fp = preds & mask_non_g
    area_fp = np.sum(mask_fp) * f_area
    stats['area_non_g'] = area_non_g
    stats['area_fp'] = area_fp

    # compute the FPs for the non-glacierized area but only within a certain buffer
    nc['mask_crt_g_b0'] = nc['mask_crt_g']
    for b1, b2 in list(itertools.combinations(['b-20', 'b-10', 'b0', 'b10', 'b20', 'b50'], 2)):
        mask_crt_b_interval = (nc[f'mask_crt_g_{b1}'].values == 0) & (nc[f'mask_crt_g_{b2}'].values == 1)
        mask_non_g_crt_b = mask_non_g & mask_crt_b_interval
        mask_fp_crt_b = preds & mask_non_g_crt_b

        # compute the total non-glacier area in the current buffer and the corresponding FP area
        stats[f"area_{b1}_{b2}"] = np.sum(mask_crt_b_interval) * f_area
        stats[f"area_non_g_{b1}_{b2}"] = np.sum(mask_non_g_crt_b) * f_area
        stats[f"area_fp_{b1}_{b2}"] = np.sum(mask_fp_crt_b) * f_area

    # estimate the altitude & location of the terminus
    # first by the lower ice-predicted pixel (if it's not masked), then by the median of the lowest 30 pixels
    # if there are multiple pixels with the same minimum altitude, use the average
    nc_g = nc.where(nc.mask_crt_g == 1)
    dem_pred_on = nc_g.dem.values.copy()
    dem_pred_on[~preds] = np.nan
    h_pred_on_sorted = np.sort(np.unique(dem_pred_on.flatten()))
    h_pred_on_sorted = h_pred_on_sorted[~np.isnan(h_pred_on_sorted)]

    for num_px_thr in [1, 30]:
        if len(h_pred_on_sorted) > 0:  # it can happen that all the pixels are masked
            i = 0
            h_thr = h_pred_on_sorted[i]
            while i < len(h_pred_on_sorted) - 1 and np.sum(dem_pred_on <= h_thr) < num_px_thr:
                i += 1
                h_thr = h_pred_on_sorted[i]
        else:
            h_thr = -1

        # exclude the masked pixels; if all of them are masked, the result will be NaN
        mask_lowest = (dem_pred_on <= h_thr)
        mask_lowest[mask_exclude] = False

        all_masked = (np.sum(mask_lowest) == 0)
        idx = np.where(mask_lowest)
        stats[f'term_h_{num_px_thr}_px'] = np.nan if all_masked else float(h_thr)
        stats[f'term_x_i_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(idx[1]))
        stats[f'term_y_i_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(idx[0]))
        stats[f'term_x_m_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(nc.x.values[idx[1]]))
        stats[f'term_y_m_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(nc.y.values[idx[0]]))

    # save the filename of the original S2 data
    stats['fn'] = nc.attrs['fn']

    if not return_rasters:
        return stats

    rasters = {
        'mask': mask,
        'mask_exclude': mask_exclude,
        'preds': preds,
        'mask_debris': mask_debris,
    }

    return stats, rasters


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--inference_dir', type=str, metavar='path/to/inference_dir', required=True,
        help='directory where the model predictions are stored',
    )
    parser.add_argument('--fold', type=str, metavar='s_train|s_valid|s_test', required=True,
                        help='which subset to evaluate on: either s_train, s_valid or s_test')
    parser.add_argument('--rasters_dir', type=str, required=False,
                        help='directory where the original images are stored; '
                             'if not provided, the one from the config file is used'
                        )

    args = parser.parse_args()
    inference_dir_root = Path(args.inference_dir)
    print(f'inference_dir_root = {inference_dir_root}')
    assert inference_dir_root.exists()

    # set the raster directory to the command line argument if given, otherwise use the inference dirs from the config
    if args.rasters_dir is not None:
        rasters_dir = args.rasters_dir
    else:
        rasters_dir = C.DIR_GL_INVENTORY

    # replace the 'preds' subdirectory with 'stats'
    p = list(inference_dir_root.parts)
    stats_dir_root = Path(*p[:p.index('preds')]) / 'stats' / Path(*p[p.index('preds') + 1:])

    # get the training settings (needed for building the data masks)
    settings_fp = Path(*p[:p.index('output')]) / 'settings.yaml'

    with open(settings_fp, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)

    fold = args.fold
    preds_dir = inference_dir_root / fold
    fp_list = list(preds_dir.glob('**/*.nc'))
    print(f'fold = {fold}; #glaciers = {len(fp_list)}')
    if len(fp_list) == 0:
        print(f'No predictions found for fold = {fold}. Skipping.')
        exit(0)

    for band_target in ('mask_crt_g', 'mask_crt_g_b20'):
        for exclude_bad_pixels in (True, False):

            _compute_stats = partial(
                compute_stats,
                rasters_dir=rasters_dir,
                band_target=band_target,
                exclude_bad_pixels=exclude_bad_pixels,
                input_settings=all_settings['model']['inputs'],
            )

            with multiprocessing.Pool(C.NUM_CORES_EVAL) as pool:
                all_metrics = []
                for metrics in tqdm(
                        pool.imap_unordered(_compute_stats, fp_list, chunksize=1), total=len(fp_list),
                        desc=f'Computing evaluation metrics '
                             f'(exclude_bad_pixels = {exclude_bad_pixels}; mask_name = {band_target})'):
                    all_metrics.append(metrics)
                metrics_df = pd.DataFrame.from_records(all_metrics)

                stats_fp = stats_dir_root / fold / f'stats_excl_{exclude_bad_pixels}_{band_target}.csv'
                stats_fp.parent.mkdir(parents=True, exist_ok=True)
                metrics_df = metrics_df.sort_values('fp')
                metrics_df.to_csv(stats_fp, index=False)
                print(f'Evaluation metrics exported to {stats_fp}')
