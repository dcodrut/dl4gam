"""
    Script for preparing the training data:

    1. Script which extracts patches from the glacier-wide NetCDF files.
    2. Builds the splits for the cross-validation and the corresponding training, validation and test folds.
    3. Computes the normalization statistics for the training patches of each cross-validation split.

"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# local imports
from config import C
from utils.sampling_utils import patchify_data, data_cv_split
from utils.general import run_in_parallel
from utils.data_stats import compute_normalization_stats, aggregate_normalization_stats

if __name__ == "__main__":
    # 1. patchify the data
    dir_patches = Path(C.DIR_GL_PATCHES)
    patchify_data(
        rasters_dir=C.DIR_GL_INVENTORY,
        patches_dir=dir_patches,
        patch_radius=C.PATCH_RADIUS,
        sampling_step=C.SAMPLING_STEP_TRAIN,
    )

    # 2. build the splits for the cross-validation

    # import the constants corresponding to the desired dataset
    outlines_fp = C.GLACIER_OUTLINES_FP
    print(f'Reading S2-based glacier outlines from {outlines_fp}')
    gl_df = gpd.read_file(outlines_fp)
    initial_area = gl_df.area_km2.sum()
    print(f'#glaciers = {len(gl_df)}; area = {initial_area:.2f} km2')

    print(f'Keeping only the glaciers above {C.MIN_GLACIER_AREA}')
    gl_df = gl_df[gl_df.area_km2 >= C.MIN_GLACIER_AREA].copy()
    area = gl_df.area_km2.sum()
    print(f'#glaciers = {len(gl_df)}; area = {area:.2f} km2 ({area / initial_area * 100:.2f}%)')

    # remove the glaciers for which there is no data
    assert Path(C.DIR_GL_INVENTORY).exists(), f"{C.DIR_GL_INVENTORY} does not exist"
    print(f'Keeping only the glaciers that have data in {C.DIR_GL_INVENTORY}')
    gl_entry_ids_ok = [x.parent.name for x in Path(C.DIR_GL_INVENTORY).glob('**/*.nc')]
    gl_df = gl_df[gl_df.entry_id.isin(gl_entry_ids_ok)]
    area = gl_df.area_km2.sum()
    print(f'#glaciers = {len(gl_df)}; area = {area:.2f} km2 ({area / initial_area * 100:.2f}%)')

    # split the data regionally
    data_cv_split(
        gl_df=gl_df,
        num_folds=C.NUM_CV_FOLDS,
        valid_fraction=C.VALID_FRACTION,
        outlines_split_dir=C.DIR_OUTLINES_SPLIT
    )

    # save one dataframe which maps each glacier to the corresponding fold of each split (will be used at inference)
    df_list = []
    for i_split in range(1, C.NUM_CV_FOLDS + 1):
        crt_split_dir = Path(C.DIR_OUTLINES_SPLIT) / f"split_{i_split}"
        for fold in ['train', 'valid', 'test']:
            fp = crt_split_dir / f"fold_{fold}.shp"
            df = gpd.read_file(fp)
            df['split'] = f"split_{i_split}"
            df['fold'] = f"fold_{fold}"
            df_list.append(df)

    df_all = pd.concat(df_list)
    df_all = df_all.pivot(index='entry_id', columns='split', values='fold').reset_index()
    fp = Path(C.DIR_OUTLINES_SPLIT) / 'map_all_splits_all_folds.csv'
    df_all.to_csv(fp, index=False)
    print(f"Dataframe with the split-fold mapping saved to {fp}")

    # 3. compute the normalization statistics for the training patches of each cross-validation split
    # get the list of all patches and group them by glacier
    fp_all_patches = sorted(list(Path(dir_patches).glob('**/*.nc')))
    print(f'Found {len(fp_all_patches)} patches')
    gl_to_patches = {x.parent.name: [] for x in fp_all_patches}
    for fp in fp_all_patches:
        gl_to_patches[fp.parent.name].append(fp)
    out_dir_root = dir_patches.parent.parent / 'aux_data' / 'norm_stats' / dir_patches.name
    for i_split in range(1, C.NUM_CV_FOLDS + 1):
        # get the list of patches for the training fold of the current cross-validation split
        gl_entry_ids_train = set(df_all[df_all[f"split_{i_split}"] == 'fold_train'].entry_id)
        fp_list = [x for gl, lst in gl_to_patches.items() if gl in gl_entry_ids_train for x in lst]

        print(
            f'Computing normalization stats for the training fold of the split = {i_split} '
            f'(#glaciers = {len(gl_entry_ids_train)}, '
            f'#patches = {len(fp_list)})'
        )
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
