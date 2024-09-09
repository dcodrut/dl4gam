"""
    Script which computes the mean and standard deviation of the training patches from each cross-validation fold.
"""
import pandas as pd
from pathlib import Path

# local imports
from config import C
from utils.data_stats import compute_normalization_stats, aggregate_normalization_stats
from utils.general import run_in_parallel

if __name__ == '__main__':
    data_dir_root = Path(C.DIR_GL_PATCHES)
    out_dir_root = data_dir_root.parent.parent / 'aux_data' / 'norm_stats' / Path(C.DIR_GL_PATCHES).name
    for i_split in range(1, C.NUM_CV_FOLDS + 1):
        data_dir_crt_split = data_dir_root / f'split_{i_split}' / 'fold_train'
        fp_list = sorted(list(Path(data_dir_crt_split).glob('**/*.nc')))

        print(f'Computing normalization stats for split = {i_split}')
        all_stats = run_in_parallel(compute_normalization_stats, fp=fp_list, num_procs=C.NUM_PROCS, pbar=True)
        all_df = [pd.DataFrame(stats) for stats in all_stats]

        df = pd.concat(all_df)
        df = df.sort_values('fn')
        out_dir_crt_split = out_dir_root / f'split_{i_split}'
        out_dir_crt_split.mkdir(parents=True, exist_ok=True)
        fp_out = Path(out_dir_crt_split) / 'stats_train_patches.csv'
        df.to_csv(fp_out, index=False)
        print(f'Stats (per patch) saved to {fp_out}')

        # aggregate the statistics
        df_stats_agg = aggregate_normalization_stats(df)
        fp_out = Path(out_dir_crt_split) / 'stats_train_patches_agg.csv'
        df_stats_agg.to_csv(fp_out, index=False)
        print(f'Aggregated stats saved to {fp_out}')
