"""
    Script which computes the mean and standard deviation of the training patches from each cross-validation fold.
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# local imports
import config as C
from utils.data_stats import compute_normalization_stats, aggregate_normalization_stats

if __name__ == '__main__':
    data_dir_root = Path(C.S2.DIR_GL_PATCHES)
    out_dir_root = Path(C.S2.DIR_AUX_DATA) / Path(C.S2.DIR_GL_PATCHES).name
    for i_split in range(1, C.S2.NUM_CV_FOLDS + 1):
        data_dir_crt_split = data_dir_root / f'split_{i_split}' / 'fold_train'
        fp_list = sorted(list(Path(data_dir_crt_split).glob('**/*.nc')))

        all_df = []
        for fp_crt_patch in tqdm(fp_list, desc=f'Compute stats for train patches from split = {i_split}'):
            stats_crt_patch = compute_normalization_stats(fp_crt_patch)
            df = pd.DataFrame(stats_crt_patch)
            all_df.append(df)

        df = pd.concat(all_df)
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
