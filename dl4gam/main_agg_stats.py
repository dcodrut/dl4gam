"""
    Aggregate the statistics for the glacier-wide predictions from all the folds and seeds (if multiple are available).
    The statistics are produced by the main_eval.py script and are stored in the 'stats' subdirectory of the 'output'.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# local imports
from config import C


def print_regional_stats(
        df_stats: pd.DataFrame,
        buffer_size_pred: int = 20,
        buffer_size_fp: int = 50,
        compute_uncertainties: bool = False
):
    """
    Compute and print the regional statistics based on the glacier-wide predictions.
    :param df_stats: the dataframe with the glacier-wide statistics
    :param buffer_size_pred: the buffer size in meters until where the predictions are considered as glacierized
    :param buffer_size_fp: the buffer size in meters until where the predictions are considered as false positives
                           (starting from buffer_size_pred)
    :param compute_uncertainties: whether to add the uncertainties to the statistics

    :return: None
    """

    # show the stats when combining all the glaciers in the dataframe
    t_a_ok = df_stats.area_ok.sum()
    t_a_nok = df_stats.area_nok.sum()
    t_a_inv = df_stats.area_inv.sum()

    t_a_recalled = df_stats.area_recalled.sum()

    t_a_pred = df_stats.area_pred.sum()

    t_a_fp = df_stats[f'area_non_g_b{buffer_size_pred}_{buffer_size_fp}'].sum()
    t_a_fp_pred = df_stats[f'area_non_g_pred_b{buffer_size_pred}_{buffer_size_fp}'].sum()

    t_a_debris = df_stats.area_debris.sum()
    t_a_debris_recalled = df_stats.area_debris_recalled.sum()

    if compute_uncertainties:
        # assuming no spatial correlation
        # t_a_recalled_std = (df_stats.area_recalled_std ** 2).sum() ** 0.5  # no spatial correlation
        # t_a_pred_std = (df_stats.area_pred_std ** 2).sum() ** 0.5
        # t_a_fp_pred_std = (df_stats.area_non_g_pred_b20_50_std ** 2).sum() ** 0.5  # no spatial correlation
        # t_a_debris_recalled_std = (df_stats.area_debris_recalled_std ** 2).sum() ** 0.5  # no spatial correlation

        # assuming perfect correlation
        t_a_recalled_std = df_stats.area_recalled_std.sum()
        t_a_pred_std = df_stats.area_pred_std.sum()
        t_a_fp_pred_std = df_stats.area_non_g_pred_b20_50_std.sum()
        t_a_debris_recalled_std = df_stats.area_debris_recalled_std.sum()
    else:
        t_a_recalled_std = np.nan
        t_a_pred_std = np.nan
        t_a_fp_pred_std = np.nan
        t_a_debris_recalled_std = np.nan

    print(
        f'Regional statistics:'
        f'\n\t#glaciers = {len(df_stats)}'
        f'\n\ttotal area = {t_a_inv:.1f}; '
        f'total area NOK = {t_a_nok:.1f}; total area OK = {t_a_ok:.1f} ({t_a_ok / t_a_inv * 100:.2f}%)'
        f'\n\tno  buffer: recall = {t_a_recalled:.1f} ± {t_a_recalled_std:.2f} of {t_a_inv:.1f} km²'
        f' = {t_a_recalled / t_a_inv * 100:.1f} ± {t_a_recalled_std / t_a_inv * 100:.2f} %'
        f'\n\t{buffer_size_pred}m buffer: predicted area = {t_a_pred:.1f} ± {t_a_pred_std:.2f} km²'
        f'\n\t{buffer_size_pred}-{buffer_size_fp}m buffer: FP rate = {t_a_fp_pred:.1f} ± {t_a_fp_pred_std:.2f} '
        f'of {t_a_fp:.1f} km² = {t_a_fp_pred / t_a_fp * 100:.1f} ± {t_a_fp_pred_std / t_a_fp * 100:.2f} %'
        f'\n\tdebris percentage = {t_a_debris / t_a_inv * 100:.2f} %'
        f'\n\trecall debris = {t_a_debris_recalled:.1f} ± {t_a_debris_recalled_std:.2f} of {t_a_debris:.1f} km² '
        f'= {t_a_debris_recalled / t_a_debris * 100:.1f} ± {t_a_debris_recalled_std / t_a_debris * 100:.2f} %'
    )


def aggregate_all_stats(
        gl_df: gpd.GeoDataFrame,
        model_dir: str | Path,
        model_version: str,
        seed_list: list,
        split_list: list,
        ds_name: str,
        subdir: str,
        buffer_pred_m: int = 20,
        buffer_fp_m: int = 50,
        excl_masked_pixels: bool = False,
        use_calib: bool = False,
):
    """
    Aggregate the glacier-wide statistics from the testing folds of all the cross-validation iterations and seeds.
    Then compute the uncertainties based on the lower and upper bounds of the predictions, if available (i.e.
    predictions are already ensemble-aggregated) or based on the standard deviation of the predictions over the
    available seeds (if multiple seeds are available).

    The seed-level statistics amd the aggregated ones are exported to CSV files.

    :param gl_df: the dataframe with the glacier inventory
    :param model_dir: the root directory of the model outputs
    :param model_version: the version of the model
    :param seed_list: the list of seeds (or 'all' to use the ensemble-aggregated predictions)
    :param split_list: the list of cross-validation iterations (e.g. [1, 2, 3, 4, 5])
    :param ds_name: the name of the dataset used for inference
    :param subdir: the subdirectory of the dataset used for inference (e.g. 'inv', '2023')
    :param buffer_pred_m: the buffer size in meters until where the predictions are considered as glacierized
    :param buffer_fp_m: the buffer size in meters until where the predictions are considered as false positives
                        (starting from buffer_pred_m)
    :param excl_masked_pixels: whether to exclude the masked-out pixels from the statistics
    :param use_calib: whether to use the calibrated predictions

    :return: the dataframe with the aggregated statistics
    """
    df_stats_list = []
    stats_version = 'stats_calib' if use_calib else 'stats'
    if excl_masked_pixels:
        stats_version += '_excl_bad_pixels'
    for seed in seed_list:
        for i_split in split_list:
            stats_dir = model_dir / f"split_{i_split}/seed_{seed}/{model_version}/output/{stats_version}/{ds_name}"
            assert stats_dir.exists(), f"{stats_dir} doesn't exist"

            fp = stats_dir / str(subdir) / 's_test' / f"{stats_version}.csv"
            print(f'seed = {seed}; split = {i_split}; model_version = {model_version}; subdir = {subdir}')
            print(f'\tfp = {fp}')
            assert fp.exists(), f"{fp} doesn't exist"
            df_glacier = pd.read_csv(fp)

            df_glacier['seed'] = seed
            df_glacier['i_split'] = i_split
            df_glacier['subdir'] = subdir
            df_glacier['fold'] = 'test'
            df_glacier['fn'] = df_glacier.fp.apply(lambda s: Path(s).name)
            df_glacier['entry_id'] = df_glacier.fp.apply(lambda x: Path(x).parent.name)
            df_glacier = df_glacier[['entry_id'] + [c for c in df_glacier if c != 'entry_id']]  # entry_id first col
            df_glacier['date_actual'] = df_glacier.fn.apply(lambda x: pd.to_datetime(x[:8]))
            df_glacier['year'] = df_glacier.date_actual.apply(lambda x: x.year)
            del df_glacier['fp']

            # replace the inventory area with the one from the inventory
            # (the one we calculated is slightly different due to the resampling and rasterization)
            del df_glacier['area_inv']
            df_glacier = df_glacier.merge(gl_df[['entry_id', 'area_inv']], on='entry_id')
            df_glacier = df_glacier[['area_inv'] + [c for c in df_glacier if c != 'area_inv']]  # area_inv first col
            df_glacier['area_nok_p'] = df_glacier.area_nok / df_glacier.area_inv

            # dummy columns in case buffer_size_pred is zero
            df_glacier[f'area_non_g_b0'] = 0.0
            df_glacier[f'area_non_g_pred_b0'] = 0.0
            df_glacier[f'area_non_g_pred_low_b0'] = 0.0
            df_glacier[f'area_non_g_pred_high_b0'] = 0.0

            # percentage of debris-covered area
            df_glacier['area_debris_p'] = df_glacier.area_debris / df_glacier.area_inv

            # non-glacierized area between the limit of OK predictions & the false positive limit
            df_glacier[f'area_non_g_b{buffer_pred_m}_{buffer_fp_m}'] = (
                    df_glacier[f'area_non_g_b{buffer_fp_m}'] -
                    df_glacier[f'area_non_g_b{buffer_pred_m}']
            )

            # compute the recall and the false positive rates
            for s in ['', '_low', '_high']:
                if s != '' and seed_list != ['all']:
                    continue  # low and high bounds are available only for the aggregated ensemble predictions

                # recall
                df_glacier[f'area_recalled{s}'] = df_glacier[f'area_pred{s}_b0']
                df_glacier[f'recall{s}'] = df_glacier[f'area_recalled{s}'] / df_glacier.area_inv

                # predicted debris area & recall (ignore if glaciers have less than 1% debris)
                df_glacier[f'area_debris_recalled{s}'] = df_glacier[f'area_debris_pred{s}_b0']
                df_glacier.loc[df_glacier.area_debris_p <= 0.01, f'area_debris_recalled{s}'] = np.nan
                df_glacier[f'recall_debris{s}'] = df_glacier[f'area_debris_recalled{s}'] / df_glacier.area_debris

                # false-positive area & rate
                df_glacier[f'area_non_g_pred{s}_b{buffer_pred_m}_{buffer_fp_m}'] = (
                        df_glacier[f'area_non_g_pred{s}_b{buffer_fp_m}'] -
                        df_glacier[f'area_non_g_pred{s}_b{buffer_pred_m}']
                )
                df_glacier[f'fp_rate{s}'] = (
                        df_glacier[f'area_non_g_pred{s}_b{buffer_pred_m}_{buffer_fp_m}'] /
                        df_glacier[f'area_non_g_b{buffer_pred_m}_{buffer_fp_m}']
                )

                # include the buffer area in the predicted area
                df_glacier[f'area_pred{s}'] = (
                        df_glacier[f'area_recalled{s}'] +
                        df_glacier[f'area_non_g_pred{s}_b{buffer_pred_m}']
                )

            df_stats_list.append(df_glacier)

    # merge with the glacier dataframe
    df_stats_all = pd.concat(df_stats_list)
    assert set(df_stats_all.entry_id).issubset(set(gl_df.entry_id)), \
        "Some glaciers are missing from the inventory or the wrong inventory was used"
    df_stats_all = df_stats_all.merge(
        gl_df.drop(columns=[
            'area_inv',
            'SECTOR_NR',
            'SEC_NAMES',
            'Tile_Name',
            'S2_footpri',
            'Date',
            'Funding',
            'Analyst',
            'O1_REGION',
            'O2_REGION', 'Footpr_ID', 'geometry'
        ], errors='ignore'), on='entry_id'
    )

    # mark which images used for training are exactly the same from the inventory
    df_stats_all['is_inv_year'] = (df_stats_all.year == df_stats_all.year_inv)
    df_stats_all = df_stats_all.sort_values(['entry_id', 'seed', 'date_actual'])

    # export the processed statistics
    label = 'ensemble' if seed_list == ['all'] else 'individual'
    fp_out = model_dir / 'stats_all_splits' / f"df_{stats_version}_all_{ds_name}_{subdir}_{model_version}_{label}.csv"
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    df_stats_all.to_csv(fp_out, index=False)
    print(f"\nSaved the processed & combined statistics to {fp_out}\n")

    # show the regional statistics for each seed
    if len(seed_list) > 1:
        print("\nRegional statistics for each seed:")
        for seed in seed_list:
            print(f"seed = {seed}")
            df_stats_seed = df_stats_all[df_stats_all.seed == seed]
            print_regional_stats(df_stats_seed, buffer_size_pred=buffer_pred_m, buffer_size_fp=buffer_fp_m)

    # compute the uncertainties
    # is the seed is 'all', then the standard deviation is derived from the lower and upper bounds of the predictions
    # these are based on the stats computed on the ensemble-aggregated rasters (see main_agg_ensemble.py)
    # otherwise we compute the standard deviation of the predicted areas over the available seeds
    # (the latter assumes that the pixel-wise predictions within a glacier are independent)

    # stop here if the model is a band ratio model
    if 'band_ratio' in model_dir.name:
        print("No uncertainties available for the band ratio models!")
        return

    if label == 'ensemble':
        print("Computing the standard deviations based on the lower and upper bounds of the predictions...")
        df_stats_agg = df_stats_all.copy()  # the aggregated average is already there
        for c in df_stats_all.select_dtypes(include=np.number):
            if (
                    ('pred' in c and 'low' not in c and 'high' not in c) or
                    c in ['recall', 'area_recalled', 'area_debris_recalled']
            ):
                if 'pred' in c:
                    c_l = c.replace('pred', 'pred_low')
                    c_u = c.replace('pred', 'pred_high')
                else:
                    c_l = c + '_low'
                    c_u = c + '_high'
                assert c_l in df_stats_all.columns and c_u in df_stats_all.columns, f"{c_l} or {c_u} not in columns"
                stddevs = (df_stats_all[c_u] - df_stats_all[c_l]) * 0.5
                df_stats_agg.insert(df_stats_agg.columns.get_loc(c) + 1, f'{c}_std', stddevs)

    else:
        print("Computing the standard deviations based on the predictions over the available seeds...")
        df_stats_agg = df_stats_all.groupby(['entry_id', 'is_inv_year']).first().reset_index()
        for c in df_stats_all.select_dtypes(include=np.number):
            if 'pred' in c or c in ['recall', 'area_recalled', 'area_debris_recalled']:
                df_stats_agg[c] = df_stats_all.groupby(['entry_id', 'is_inv_year'])[c].mean().values
                stddevs = df_stats_all.groupby(['entry_id', 'is_inv_year'])[c].std().values

                # add the standard deviation right after the average
                df_stats_agg.insert(df_stats_agg.columns.get_loc(c) + 1, f'{c}_std', stddevs)

    # print the regional statistics for the aggregated predictions
    print_regional_stats(
        df_stats_agg, buffer_size_pred=buffer_pred_m, buffer_size_fp=buffer_fp_m, compute_uncertainties=True
    )

    # export
    fp_out = model_dir / 'stats_all_splits' / fp_out.name.replace(f'df_{stats_version}_all', f'df_{stats_version}_agg')
    df_stats_agg.to_csv(fp_out, index=False)
    print(f"\nSaved the dataframe with the aggregated predictions & uncertainties to {fp_out}")

    return df_stats_agg


def compute_change_rates(df_t0: pd.DataFrame, df_t1: pd.DataFrame):
    """
    Compute the glacier-wide area change rates between two time steps.
    THe rates are exported to a CSV file.

    :param df_t0: dataframe with the glacier-wide statistics at time t0
    :param df_t1: dataframe with the glacier-wide statistics at time t1

    :return: None
    """

    # make sure we have the same glaciers in both dataframes
    assert (df_t0.entry_id == df_t1.entry_id).all()

    # compute the annual area change rates per glacier
    df_rates = pd.DataFrame({
        'entry_id': df_t0.entry_id.values,
        'area_inv': df_t0.area_inv.values,
        'recall': df_t0.recall.values,
        'area_t0': df_t0.area_pred.values,
        'area_t0_std': df_t0.area_pred_std.values,
        'area_t1': df_t1.area_pred.values,
        'area_t1_std': df_t1.area_pred_std.values,
        'year_t0': df_t0.year.values,
        'year_t1': df_t1.year.values,
    })

    # compute the rate of change per year
    df_rates['num_years'] = df_rates.year_t1 - df_rates.year_t0
    df_rates['area_rate'] = (df_rates.area_t1 - df_rates.area_t0) / df_rates.num_years

    # compute the rate of change per year in percentage
    df_rates['area_rate_prc'] = df_rates.area_rate / df_rates.area_t0

    # compute the standard deviation of the glacier-wide change rate (assuming the errors are independent)
    df_rates['area_rate_std'] = (df_rates.area_t0_std ** 2 + df_rates.area_t1_std ** 2) ** 0.5 / df_rates.num_years

    t_a_t0 = df_rates.area_t0.sum()
    t_a_t0_std = df_rates.area_t0_std.sum()  # the errors are dependent

    t_a_t1 = df_rates.area_t1.sum()
    t_a_t1_std = df_rates.area_t1_std.sum()  # the errors are dependent

    annual_area_change = df_rates.area_rate.sum()
    annual_area_change_std = df_rates.area_rate_std.sum()  # the errors are dependent

    t_a_diff = t_a_t1 - t_a_t0
    t_a_diff_std = (t_a_t0_std ** 2 + t_a_t1_std ** 2) ** 0.5  # the errors are independent

    # sanity check: the total area change should be equal to the sum of the rates of change
    assert np.isclose((df_rates.area_rate * df_rates.num_years).sum(), t_a_diff)

    print(
        "\nRegional area changes statistics:"
        f"\n\t#glaciers = {len(df_rates)}"
        f"\n\ttotal area t0  = {t_a_t0:.1f} ± {t_a_t0_std:.2f} km²"
        f"\n\ttotal area t1 = {t_a_t1:.1f} ± {t_a_t1_std:.2f} km²"
        f"\n\tarea change = {t_a_diff:.2f} ± {t_a_diff_std:.2f} km²"
        f"\n\tannual area change rate = {annual_area_change:.2f} ± {annual_area_change_std:.2f} km² / year"
        f"\n\tannual area change rate (%) = {annual_area_change / t_a_t0 * 100:.2f} ± "
        f"{annual_area_change_std / t_a_t0 * 100:.2f} % / year"
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate the glacier-wide statistics from all the folds and seeds')
    parser.add_argument(
        '--model_dir', type=str,
        help='The root directory of the model outputs (e.g. /path/to/experiments/dataset/model_name)'
    )
    parser.add_argument('--model_version', type=str, required=True, help='Version of the model')
    parser.add_argument(
        '--seed_list', type=str, nargs='+', required=True,
        help='The list of seeds (or "all" to use the ensemble-aggregated predictions)'
    )
    parser.add_argument(
        '--split_list', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='The list of cross-validation iterations (e.g. [1, 2, 3, 4, 5])'
    )
    parser.add_argument(
        '--subdir_list', type=str, nargs='+', required=True,
        help='The subdirectories of the model outputs (e.g. inv, 2023). If two are provided, area change rates are '
             'computed between them, assuming they are provided in chronological order.'
    )
    parser.add_argument(
        '--use_calib', action='store_true',
        help='Flag for using the calibrated predictions'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _model_dir = Path(args.model_dir)
    _model_version = args.model_version
    _seed_list = ["all"] if args.seed_list == "all" else list(args.seed_list)
    _split_list = list(args.split_list)
    _subdir_list = list(args.subdir_list)

    # read the inventory
    print(f"Loading the glacier outlines from {C.GLACIER_OUTLINES_FP}")
    _gl_df = gpd.read_file(C.GLACIER_OUTLINES_FP)

    # add the year for each outline
    if 'year_acq' in _gl_df.columns:
        _gl_df['year_inv'] = _gl_df.year_acq
    elif 'date_inv' in _gl_df.columns:
        _gl_df['year_inv'] = _gl_df.date_inv.apply(lambda d: pd.to_datetime(d).year)
    else:
        raise ValueError('Neither of acquisition year (`year_acq`) or date (`date_inv`) was found in the inventory')

    # mark the outlines as inventory
    _gl_df['area_inv'] = _gl_df.area_km2

    print(f"Model directory: {_model_dir}")

    _ds_name = Path(C.WD).name
    df_stats_agg_t0 = aggregate_all_stats(
        gl_df=_gl_df,
        model_dir=_model_dir,
        model_version=_model_version,
        seed_list=_seed_list,
        split_list=_split_list,
        ds_name=_ds_name,
        subdir=_subdir_list[0],
        use_calib=args.use_calib,
    )

    if len(_subdir_list) == 2:
        df_stats_agg_t1 = aggregate_all_stats(
            gl_df=_gl_df,
            model_dir=_model_dir,
            model_version=_model_version,
            seed_list=_seed_list,
            split_list=_split_list,
            ds_name=_ds_name,
            subdir=_subdir_list[1],
            use_calib=args.use_calib,
        )

        # compute the change rates
        compute_change_rates(df_stats_agg_t0, df_stats_agg_t1)
