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
    ################################################# STEP 1 - PATCHIFY ################################################
    # patchify the data if needed (otherwise patches will be sampled on the fly while training)
    if C.EXPORT_PATCHES:
        dir_patches = Path(C.DIR_GL_PATCHES)
        patchify_data(
            rasters_dir=C.DIR_GL_RASTERS,
            patches_dir=dir_patches,
            patch_radius=C.PATCH_RADIUS,
            sampling_step=C.SAMPLING_STEP_TRAIN,
        )

    ########################################## STEP 2 - CROSS-VALIDATION SPLIT #########################################
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
    assert Path(C.DIR_GL_RASTERS).exists(), f"{C.DIR_GL_RASTERS} does not exist"
    print(f'Keeping only the glaciers that have data in {C.DIR_GL_RASTERS}')
    fp_rasters = sorted(list(Path(C.DIR_GL_RASTERS).rglob('*.nc')))
    gl_entry_ids_ok = [x.parent.name for x in fp_rasters]
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

    ####################################### STEP 3 - COMPUTE NORMALIZATION STATS #######################################
    # prepare the files (of patches or rasters) for all the glaciers
    if C.EXPORT_PATCHES:
        # get the list of all patches and group them by glacier
        fp_list = sorted(list(Path(dir_patches).rglob('*.nc')))
        print(f'Found {len(fp_list)} patches')
        gl_to_files = {x.parent.name: [] for x in fp_list}
        for fp in fp_list:
            gl_to_files[fp.parent.name].append(fp)
        out_dir_root = Path(C.WD) / Path(C.SUBDIR) / 'aux_data' / 'norm_stats' / dir_patches.name
    else:
        # get the raster path for each glacier
        fp_list = fp_rasters
        gl_to_files = {x.parent.name: [x] for x in fp_list}
        out_dir_root = Path(C.WD) / Path(C.SUBDIR) / 'aux_data' / 'norm_stats' / 'rasters'

    print(f"Computing normalization stats for all {'patches' if C.EXPORT_PATCHES else 'rasters'} "
          f"(#files = {len(fp_list)}; #glaciers = {len(gl_to_files)})")
    all_stats = run_in_parallel(compute_normalization_stats, fp=fp_list, num_procs=C.NUM_PROCS, pbar=True)
    all_df = [pd.DataFrame(stats) for stats in all_stats]
    df = pd.concat(all_df)
    fp_out = Path(out_dir_root) / 'stats_all.csv'
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp_out, index=False)
    print(f'Stats (per file) saved to {fp_out}')

    # extract & aggregate the statistics for the training folds of each cross-validation split
    for i_split in range(1, C.NUM_CV_FOLDS + 1):
        gl_entry_ids_train = set(df_all[df_all[f"split_{i_split}"] == 'fold_train'].entry_id)
        df_crt = df[df.entry_id.isin(gl_entry_ids_train)]

        # aggregate the statistics
        df_crt_agg = aggregate_normalization_stats(df_crt)
        fp_out = Path(out_dir_root) / f'stats_agg_split_{i_split}.csv'
        df_crt_agg.to_csv(fp_out, index=False)
        print(f'Aggregated stats for split {i_split} saved to {fp_out}')
