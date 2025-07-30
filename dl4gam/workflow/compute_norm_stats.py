import logging
from pathlib import Path

import pandas as pd

from dl4gam.utils import run_in_parallel, compute_normalization_stats, aggregate_normalization_stats

log = logging.getLogger(__name__)


def main(
        data_dir: str | Path,
        split_csv: str | Path,
        fp_out: str | Path,
):
    """
    Compute normalization statistics for the entire dataset and then aggregate them for the training folds of the given
    cross-validation iteration.

    :param data_dir: data directory containing the glacier-wide NetCDF files or patches.
    :param split_csv: csv file containing the train/val/test splits for the current cross-validation iteration.
    :param fp_out: where to save the aggregated normalization statistics for the training fold
    :return:
    """

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV file {split_csv} does not exist.")

    # Let's first check if we already computed the stats for other CV iterations
    fp_stats_all = Path(fp_out).parent / 'stats_all.csv'
    if fp_stats_all.exists():
        log.info(f"Stats for all files already computed. Loading them from {fp_stats_all}.")
        df_stats_all = pd.read_csv(fp_stats_all)
    else:
        # Get the list of all patches / glacier-wide cubes and group them by glacier
        fp_list = sorted(list(Path(data_dir).rglob('*.nc')))
        gl_to_files = {x.parent.name: [] for x in fp_list}
        for fp in fp_list:
            gl_to_files[fp.parent.name].append(fp)

        log.info(f"Found {len(fp_list)} files belonging to {len(gl_to_files)} glaciers in {data_dir}")

        all_stats = run_in_parallel(compute_normalization_stats, fp=fp_list)
        df_stats_all = pd.concat([pd.DataFrame(stats) for stats in all_stats])
        fp_stats_all.parent.mkdir(parents=True, exist_ok=True)
        df_stats_all.to_csv(fp_stats_all, index=False)
        log.info(f"Stats for all files saved to {fp_stats_all}")

    # Now extract the entry IDs from the train fold of the current CV split
    log.info(f"Extracting training fold entry ID for the current CV iteration from {split_csv}")
    df_split = pd.read_csv(split_csv)
    gl_entry_ids_train = set(df_split[df_split['fold'] == 'train'].entry_id)
    df_stats_train = df_stats_all[df_stats_all.entry_id.isin(gl_entry_ids_train)]
    print(len(df_stats_train))

    # aggregate the statistics
    df_stats_train_agg = aggregate_normalization_stats(df_stats_train)
    df_stats_train_agg.to_csv(fp_out, index=False)
    log.info(f"Aggregated stats for the training fold saved to {fp_out}")
