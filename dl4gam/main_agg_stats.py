"""
    Aggregate the statistics for the glacier-wide predictions from all the folds and seeds (if multiple are available).
    The statistics are produced by the main_eval.py script and are stored in the 'stats' subdirectory of the 'output'.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

# local imports
from config import C


def print_regional_area_stats(
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
        f'Regional area statistics:'
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
        f'= {t_a_debris_recalled / t_a_debris * 100:.1f} ± {t_a_debris_recalled_std / t_a_debris * 100:.2f} %\n'
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
            print_regional_area_stats(df_stats_seed, buffer_size_pred=buffer_pred_m, buffer_size_fp=buffer_fp_m)

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
    print_regional_area_stats(
        df_stats_agg, buffer_size_pred=buffer_pred_m, buffer_size_fp=buffer_fp_m, compute_uncertainties=True
    )

    # export
    fp_out = model_dir / 'stats_all_splits' / fp_out.name.replace(f'df_{stats_version}_all', f'df_{stats_version}_agg')
    df_stats_agg.to_csv(fp_out, index=False)
    print(f"\nSaved the dataframe with the aggregated predictions & uncertainties to {fp_out}")

    return df_stats_agg


def print_regional_area_change_stats(df_rates: pd.DataFrame):
    t_a_t0 = df_rates.area_t0.sum()
    t_a_t0_std = df_rates.area_t0_std.sum()  # errors are dependent # TODO: revisit this assumption and the ones below

    t_a_t1 = df_rates.area_t1.sum()
    t_a_t1_std = df_rates.area_t1_std.sum()  # errors are dependent

    annual_area_change = df_rates.area_rate.sum()
    annual_area_change_std = df_rates.area_rate_std.sum()  # errors are dependent

    t_a_diff = t_a_t1 - t_a_t0
    t_a_diff_std = (t_a_t0_std ** 2 + t_a_t1_std ** 2) ** 0.5  # errors are independent

    # sanity check: the total area change should be equal to the sum of the rates of change
    assert np.isclose((df_rates.area_rate * df_rates.num_years).sum(), t_a_diff)

    print(
        "Regional area change statistics:"
        f"\n\t#glaciers = {len(df_rates)}"
        f"\n\ttotal area inv = {df_rates.area_inv.sum():.1f} km²"
        f"\n\ttotal area t0 = {t_a_t0:.1f} ± {t_a_t0_std:.2f} km²"
        f"\n\ttotal area t1 = {t_a_t1:.1f} ± {t_a_t1_std:.2f} km²"
        f"\n\tarea change = {t_a_diff:.2f} ± {t_a_diff_std:.2f} km²"
        f"\n\tannual area change rate = {annual_area_change:.2f} ± {annual_area_change_std:.2f} km² / year"
        f"\n\tannual area change rate (%) = {annual_area_change / t_a_t0 * 100:.2f} ± "
        f"{annual_area_change_std / t_a_t0 * 100:.2f} % / year\n"
    )


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

    print("\nGlacier-wide area change rates statistics (before filtering & interpolation):")
    print_regional_area_change_stats(df_rates)

    return df_rates


def mark_noisy_change_rates(gl_df: gpd.GeoDataFrame, df_rates: pd.DataFrame, min_snr=1, recall_thr=0.8):
    """
    Mark the change rates based on:
    1) SNR: the estimated changes over their uncertainties
        and
    2) the recall at inventory time (since the SNR filter is not perfect, due to issues with the estimated unc)

    :param gl_df: the dataframe with the glacier inventory
    :param df_rates: dataframe with the glacier-wide area change statistics at time t0 and t1
    :param min_snr: the threshold for the signal-to-noise ratio (SNR)
    :param recall_thr: the threshold for the recall at inventory time

    :return: None
    """

    ta_inv = gl_df.area_inv.sum()
    print(
        f"Before filtering: "
        f"n = {len(df_rates)}; "
        f"total area = {df_rates.area_t0.sum():.1f} km²; "
        f"area coverage = {df_rates.area_inv.sum() / ta_inv * 100:.2f}%"
    )

    # threshold based on the SNR
    df_rates['snr'] = df_rates.area_rate.abs() / df_rates.area_rate_std.clip(1e-9)
    idx_high_snr = df_rates.snr >= min_snr
    print(
        f"SNR >= {min_snr}: "
        f"\n\tn = {sum(idx_high_snr)}; "
        f"total area = {df_rates[idx_high_snr].area_t0.sum():.1f} km²; "
        f"area coverage = {df_rates[idx_high_snr].area_inv.sum() / ta_inv * 100:.2f}%"
    )

    # threshold based on the recall at inventory time
    idx_high_recall = (df_rates.recall >= recall_thr)
    print(
        f"recall >= {recall_thr}: "
        f"\n\tn = {sum(idx_high_recall)}; "
        f"total area = {df_rates[idx_high_recall].area_t0.sum():.1f} km²; "
        f"area coverage = {df_rates[idx_high_recall].area_inv.sum() / ta_inv * 100:.2f}%"
    )

    # both thresholds
    idx_ok = idx_high_snr & idx_high_recall
    print(
        f"SNR >= {min_snr} AND recall >= {recall_thr}: "
        f"\n\tn = {sum(idx_ok)}; "
        f"total area = {df_rates[idx_ok].area_t0.sum():.1f} km²; "
        f"area coverage = {df_rates[idx_ok].area_inv.sum() / ta_inv * 100:.2f}%\n"
    )

    # save the information for each filter and their combination
    df_rates['filtered_by_unc'] = ~idx_high_snr
    df_rates['filtered_by_recall'] = ~idx_high_recall
    df_rates['filtered'] = ~idx_ok

    print_regional_area_change_stats(df_rates[idx_ok])

    return df_rates


def extrapolate_area_change_rates(
        gl_df: gpd.GeoDataFrame,
        df_rates: pd.DataFrame,
        y_stop: int = None,
        model_type: str = 'piecewise-linear'
):
    """
    Extrapolate the area change rates to the entire glacier inventory.

    The extrapolation is done using a piecewise linear interpolation based on the area of the glaciers.
    (It's actually interpolation for glaciers larger than 0.1 km² and extrapolation for smaller ones.)

    :param gl_df: the dataframe with the glacier inventory
    :param df_rates: dataframe with the glacier-wide area change statistics at time t0 and t1
    :param y_stop: the year of the second time step (if None, the second year is taken from the dataframe)
    :param model_type: the type of model to use for the extrapolation ('piecewise-linear' or '2d-polynomial')
    :return: the dataframe with the extrapolated area change rates
    """

    assert model_type in ['piecewise-linear', '2d-polynomial'], \
        f"model_type must be 'piecewise-linear' or '2d-polynomial', not {model_type}"

    # create a dataframe with all the glaciers for interpolating the area change rates
    df_rates_all_g = df_rates.drop(columns='area_inv').merge(
        gl_df[['entry_id', 'area_inv']], on='entry_id', how='right'
    )

    # add the years
    df_rates_all_g['year_t0'] = gl_df.sort_values('entry_id').year_inv.values
    if y_stop is None:
        # we expect the second year to be the same for all glaciers
        assert len(set(df_rates.year_t1)) == 1, \
            f"More than one second year in the dataframe: {set(df_rates.year_t1)}"
        y_stop = df_rates.year_t1.values[0]

    df_rates_all_g['year_t1'] = y_stop
    df_rates_all_g['num_years'] = df_rates_all_g.year_t1 - df_rates_all_g.year_t0

    # needed for the interpolation
    df_rates_all_g = df_rates_all_g.sort_values('area_inv')

    # set the classes for the area which will be used for the piecewise linear interpolation or uncertainties
    area_thrs = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, np.inf]  # based on Paul et al. 2020; # TODO: parameterize this
    df_rates_all_g['area_class'] = None
    area_classes = []
    for i in range(len(area_thrs) - 1):
        min_area = area_thrs[i]
        max_area = area_thrs[i + 1]
        idx_all = (min_area <= df_rates_all_g.area_inv) & (df_rates_all_g.area_inv < max_area)

        area_class = f"[{min_area}, {max_area})" if max_area != np.inf else f">= {min_area}"
        area_classes.append(area_class)
        df_rates_all_g.loc[idx_all, 'area_class'] = area_class
        if i == 0:
            area_class = f"[0.01, {min_area})"
            area_classes.insert(0, area_class)
            idx_extrapolate = (df_rates_all_g.area_inv < min_area)
            df_rates_all_g.loc[idx_extrapolate, 'area_class'] = area_class

    print(f"model_type = {model_type}")
    df_rates_all_g['area_rate_pred'] = np.nan
    df_rates_all_g['area_rate_pred_std'] = np.nan
    for i in range(len(area_thrs) - 1):
        min_area = area_thrs[i]
        max_area = area_thrs[i + 1]
        idx_all = (min_area <= df_rates_all_g.area_inv) & (df_rates_all_g.area_inv < max_area)
        idx_ok = idx_all & (df_rates_all_g.filtered == False)

        print(
            f"{min_area} <= area < {max_area}: "
            f"n_all_g = {sum(idx_all)}; n_ok = {sum(idx_ok)} ({sum(idx_ok) / sum(idx_all) * 100:.2f} %)"
        )

        if sum(idx_all) == sum(idx_ok):
            print(f"\tAll glaciers are covered, skipping the interpolation.")
            continue

        # interpolate the area change rate
        x = df_rates_all_g[idx_ok].area_inv.values
        y = df_rates_all_g[idx_ok].area_rate.values
        x_all = df_rates_all_g[idx_all].area_inv.values

        if model_type == 'piecewise-linear':
            model = sm.OLS(y, sm.add_constant(x)).fit()
            # display(model.summary())
            y_pred = model.predict(sm.add_constant(x_all))
            df_rates_all_g.loc[idx_all, 'area_rate_pred'] = y_pred

        # for the standard deviations, use the average of the entire class
        area_rate_pred_std = np.sqrt(np.mean(df_rates_all_g[idx_ok].area_rate_std.values ** 2))
        area_t0_pred_std = np.sqrt(np.mean(df_rates_all_g[idx_ok].area_t0_std.values ** 2))
        area_t1_pred_std = np.sqrt(np.mean(df_rates_all_g[idx_ok].area_t1_std.values ** 2))
        df_rates_all_g.loc[idx_all, 'area_rate_pred_std'] = area_rate_pred_std
        df_rates_all_g.loc[idx_all, 'area_t0_pred_std'] = area_t0_pred_std
        df_rates_all_g.loc[idx_all, 'area_t1_pred_std'] = area_t1_pred_std

        # for the uncovered class (i.e. < 0.1 km²), extrapolate using the following class
        if i == 0:
            idx_extrapolate = (df_rates_all_g.area_inv < min_area)
            if sum(idx_extrapolate) == 0:
                continue

            if model_type == 'piecewise-linear':
                x_extrapolate = df_rates_all_g[idx_extrapolate].area_inv.values
                y_extrapolate = model.predict(sm.add_constant(x_extrapolate))
                df_rates_all_g.loc[idx_extrapolate, 'area_rate_pred'] = y_extrapolate

            # to compensate for the uncertainties introduced by the extrapolation,
            # use the average standard deviation of the first class
            df_rates_all_g.loc[idx_extrapolate, 'area_rate_pred_std'] = area_rate_pred_std
            df_rates_all_g.loc[idx_extrapolate, 'area_t0_pred_std'] = area_t0_pred_std
            df_rates_all_g.loc[idx_extrapolate, 'area_t1_pred_std'] = area_t1_pred_std

        df_rates_all_g['area_rate_prc_pred'] = df_rates_all_g.area_rate_pred / df_rates_all_g.area_inv

    if model_type == '2d-polynomial':
        df = df_rates_all_g[df_rates_all_g.filtered == False]

        log_area = np.log10(df.area_inv)
        rel_change = df.area_rate_prc * 100
        X = pd.DataFrame({
            "log_area": log_area,
            "log_area_sq": log_area ** 2
        })
        X = sm.add_constant(X)
        model = sm.OLS(rel_change, X).fit()
        # print(model.summary())

        a = model.params["const"]
        b = model.params["log_area"]
        c = model.params["log_area_sq"]
        print(
            f"\nΔA/A = {a:.3f} "
            f"{'+' if b >= 0 else '-'} {abs(b):.3f}·log10(A) "
            f"{'+' if c >= 0 else '-'} {abs(c):.3f}·log10(A)^2"
        )

        log_area_range = np.log10(df_rates_all_g.area_inv.values)
        X_pred = pd.DataFrame({
            "const": 1,
            "log_area": log_area_range,
            "log_area_sq": log_area_range ** 2
        })
        rel_change_pred = model.predict(X_pred).values
        df_rates_all_g['area_rate_prc_pred'] = rel_change_pred / 100
        df_rates_all_g['area_rate_pred'] = rel_change_pred / 100 * df_rates_all_g.area_inv

    # save the original area change rates and then replace them with the interpolated ones
    df_rates_all_g['area_t0_orig'] = df_rates_all_g.area_t0
    df_rates_all_g['area_t0_std_orig'] = df_rates_all_g.area_t0_std
    df_rates_all_g['area_t1_orig'] = df_rates_all_g.area_t1
    df_rates_all_g['area_t1_std_orig'] = df_rates_all_g.area_t1_std
    df_rates_all_g['area_rate_orig'] = df_rates_all_g.area_rate
    df_rates_all_g['area_rate_std_orig'] = df_rates_all_g.area_rate_std
    idx_to_fill_in = ~(df_rates_all_g.filtered == False)
    df_rates_all_g.loc[idx_to_fill_in, 'area_rate'] = df_rates_all_g[idx_to_fill_in].area_rate_pred
    df_rates_all_g.loc[idx_to_fill_in, 'area_rate_std'] = df_rates_all_g[idx_to_fill_in].area_rate_pred_std
    df_rates_all_g.loc[idx_to_fill_in, 'area_t0'] = df_rates_all_g[idx_to_fill_in].area_inv
    df_rates_all_g.loc[idx_to_fill_in, 'area_t0_std'] = df_rates_all_g[idx_to_fill_in].area_t0_pred_std
    df_rates_all_g.loc[idx_to_fill_in, 'area_t1'] = df_rates_all_g[idx_to_fill_in].area_t0 + df_rates_all_g[
        idx_to_fill_in].area_rate * df_rates_all_g[idx_to_fill_in].num_years
    df_rates_all_g.loc[idx_to_fill_in, 'area_t1_std'] = df_rates_all_g[idx_to_fill_in].area_t1_pred_std

    # sort back
    df_rates_all_g = df_rates_all_g.sort_values('entry_id')

    df_rates_all_g.area_class = pd.Categorical(df_rates_all_g.area_class, categories=area_classes)
    print(df_rates_all_g.groupby('area_class', observed=False).area_rate_prc.describe().reset_index())

    # compute the R2 over all the predictions
    idx_training = (~df_rates_all_g.area_rate_pred.isna()) & (df_rates_all_g.filtered == False)
    y = df_rates_all_g.area_rate[idx_training].values
    y_pred = df_rates_all_g.area_rate_pred[idx_training].values
    r2 = r2_score(y, y_pred)
    print(f"R² = {r2 * 100:.1f}%")

    print_regional_area_change_stats(df_rates_all_g)

    return df_rates_all_g


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

    use_calib = args.use_calib
    _ds_name = Path(C.WD).name
    df_stats_agg_t0 = aggregate_all_stats(
        gl_df=_gl_df,
        model_dir=_model_dir,
        model_version=_model_version,
        seed_list=_seed_list,
        split_list=_split_list,
        ds_name=_ds_name,
        subdir=_subdir_list[0],
        use_calib=use_calib,
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
            use_calib=use_calib,
        )

        # compute the change rates
        df_stats_changes = compute_change_rates(df_stats_agg_t0, df_stats_agg_t1)

        #  mark the noisy change rates
        df_stats_changes = mark_noisy_change_rates(_gl_df, df_stats_changes, min_snr=1, recall_thr=0.9)

        stats_version = 'stats_calib' if use_calib else 'stats'
        fp_out = (
                _model_dir / 'stats_all_splits' /
                f"df_changes_{stats_version}_all_{_ds_name}_{_model_version}_ensemble.csv"
        )
        df_stats_changes.to_csv(fp_out, index=False)
        print(f"\nSaved the dataframe with the aggregated predicted changes & uncertainties to {fp_out}")

        df_stats_changes = extrapolate_area_change_rates(_gl_df, df_stats_changes, model_type='2d-polynomial')
        fp_out = (
                _model_dir / 'stats_all_splits' /
                f"df_changes_{stats_version}_all_{_ds_name}_{_model_version}_ensemble_extrapolated.csv"
        )
        df_stats_changes.to_csv(fp_out, index=False)
        print(f"\nSaved the dataframe with the extrapolated changes & uncertainties to {fp_out}")
