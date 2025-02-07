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


def compute_stats(fp, rasters_dir, input_settings, exclude_bad_pixels=True, return_rasters=False,
                  estimate_terminus_loc=True):
    stats = {'fp': fp}

    # read the predictions
    nc = xr.open_dataset(fp)

    # read the raw data and add it to the predictions dataset
    ds_name = fp.parent.parent.parent.name
    entry_id = fp.parent.name
    fp_orig = Path(rasters_dir).parent.parent / ds_name / 'glacier_wide' / entry_id / fp.name
    assert fp_orig.exists(), f'Original raster not found: {fp_orig}'

    nc_orig = xr.open_dataset(fp_orig)
    for c in nc_orig.data_vars:
        if 'pred' not in c or 'mask' not in c:
            nc[c] = nc_orig[c]
    nc_data = extract_inputs(fp=fp_orig, ds=nc_orig, input_settings=input_settings)

    # hack: assume perfect predictions when plotting only the images and not the results
    if 'pred' not in nc.data_vars:
        nc['pred'] = (nc.mask_crt_g == 1)
        nc['pred_b'] = nc['pred']

    # get the ground truth for the current glacier
    mask_gt = (nc.mask_crt_g.values == 1)

    # get the mask for the non-glacierized area
    mask_non_g = np.isnan(nc.mask_all_g_id.values)

    # get the mask of all the other glaciers except the current one
    mask_other_g = (~mask_non_g) & (~mask_gt)

    # prepare the scaling constant for area computation in km2
    dx = nc.rio.resolution()[0]
    f_area = (dx ** 2) / 1e6

    # extract the mask for no-data pixels (which depends on the training yaml settings)
    mask_exclude = nc_data['mask_no_data'] if exclude_bad_pixels else np.zeros_like(mask_gt)
    area_ok = np.sum(mask_gt & (~mask_exclude)) * f_area
    area_excluded = np.sum(mask_gt & mask_exclude) * f_area
    stats['area_ok'] = area_ok
    stats['area_nok'] = area_excluded
    stats['area_inv'] = area_ok + area_excluded

    # get the debris mask and its area
    mask_debris = nc_data['mask_debris_crt_g']
    area_debris = np.sum(mask_debris) * f_area
    stats['area_debris'] = area_debris

    # loop over the original predictions and its interpolated versions, if any
    pred_bands_suffixes = ['', '_i_nn', '_i_hypso']

    nc['mask_crt_g_b0'] = nc['mask_crt_g']
    for s in pred_bands_suffixes:
        for k in ['', '_low', '_high']:
            pred_band = f'pred{k}{s}_b'

            # interpolation should be found when pixels are missing and there are more than 30 non-masked pixels
            if s != '' and pred_band not in nc.data_vars:
                continue

            # low and high bounds are present only for the ensemble predictions
            if k != '' and pred_band not in nc.data_vars:
                continue

            preds = nc[pred_band].values.copy()

            if pred_band == 'pred_b':  # apply the mask only on the raw predictions
                preds[mask_exclude] = False

            # compute the predicted area using multiple buffers which will be later used for various metrics
            for b in ['b-20', 'b-10', 'b0', 'b10', 'b20', 'b50']:
                mask_crt_b = (nc[f'mask_crt_g_{b}'].values == 1)

                # exclude the other glacier pixels (i.e., keeping the same ice divides)
                mask_crt_b &= (~mask_other_g)

                # compute the total area in the current buffer and the corresponding predicted area
                stats[f"area_{b}"] = np.sum(mask_crt_b) * f_area
                stats[f"area_pred{k}_{b}{s}"] = np.sum(preds & mask_crt_b) * f_area

                # add debris-specific stats
                area_debris_pred = np.sum(preds & mask_crt_b & mask_debris) * f_area
                stats[f"area_debris_pred{k}_{b}{s}"] = area_debris_pred

                # compute the total non-glacier area in the current buffer, and the corresponding FP area
                # skip if the buffer is completely within the glacier
                if b in ['b-20', 'b-10', 'b0']:
                    continue
                mask_non_g_crt_b = mask_non_g & mask_crt_b
                stats[f"area_non_g_{b}"] = np.sum(mask_non_g_crt_b) * f_area
                stats[f"area_non_g_pred{k}_{b}{s}"] = np.sum(preds & mask_non_g_crt_b) * f_area

    # estimate the altitude & location of the terminus if needed
    if estimate_terminus_loc:
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
        'full_raster': nc,
        'mask': mask_gt,
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
    assert inference_dir_root.exists(), f'Inference directory not found: {inference_dir_root}'

    # set the raster directory to the command line argument if given, otherwise use the inference dirs from the config
    if args.rasters_dir is not None:
        rasters_dir = args.rasters_dir
    else:
        rasters_dir = C.DIR_GL_RASTERS

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

    exclude_bad_pixels = True
    _compute_stats = partial(
        compute_stats,
        rasters_dir=rasters_dir,
        exclude_bad_pixels=exclude_bad_pixels,
        input_settings=all_settings['model']['inputs'],
    )

    with multiprocessing.Pool(C.NUM_PROCS_EVAL) as pool:
        all_metrics = []
        for metrics in tqdm(
                pool.imap_unordered(_compute_stats, fp_list, chunksize=1), total=len(fp_list),
                desc=f'Computing evaluation metrics (exclude_bad_pixels = {exclude_bad_pixels})'):
            all_metrics.append(metrics)
        metrics_df = pd.DataFrame.from_records(all_metrics)

        stats_fp = stats_dir_root / fold / f'stats_excl_{exclude_bad_pixels}.csv'
        stats_fp.parent.mkdir(parents=True, exist_ok=True)
        metrics_df = metrics_df.sort_values('fp')
        metrics_df.to_csv(stats_fp, index=False)
        print(f'Evaluation metrics exported to {stats_fp}')
