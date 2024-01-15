import geopandas as gpd
import pandas as pd
from pathlib import Path

# local imports
import config
from utils.sampling_utils import patchify_s2_data, data_cv_split

if __name__ == "__main__":
    # import the constants corresponding to the desired dataset
    C = getattr(config, config.DATASET_NAME)

    s2_outlines_fp = Path(C.DIR_OUTLINES_ROOT) / 'raw' / 'c3s_gi_rgi11_s2_2015_v2/c3s_gi_rgi11_s2_2015_v2.shp'
    print(f'Reading S2-based glacier outlines from {s2_outlines_fp}')
    s2_df = gpd.read_file(s2_outlines_fp)
    initial_area = s2_df.AREA_KM2.sum()
    print(f'#glaciers = {len(s2_df)}; area = {initial_area} km2')

    print(f'Keeping only the glaciers above {C.MIN_GLACIER_AREA}')
    s2_df = s2_df[s2_df.AREA_KM2 >= C.MIN_GLACIER_AREA].copy()
    s2_df['entry_id'] = s2_df.GLACIER_NR.apply(lambda k: f"{k:04d}")
    area = s2_df.AREA_KM2.sum()
    print(f'#glaciers = {len(s2_df)}; area = {area} km2 ({area / initial_area * 100:.2f}%)')

    # split the data regionally
    data_cv_split(
        sdf=s2_df,
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
            df = gpd.read_file(fp)[['entry_id', 'LAT', 'LON']].rename(columns={'LAT': 'cen_lat', 'LON': 'cen_lon'})
            df['split'] = f"split_{i_split}"
            df['fold'] = f"fold_{fold}"
            df_list.append(df)

    df_all = pd.concat(df_list)
    df_all = df_all.pivot(index='entry_id', columns='split', values='fold').reset_index()
    fp = Path(C.DIR_OUTLINES_SPLIT) / 'map_all_splits_all_folds.csv'
    df_all.to_csv(fp, index=False)
    print(f"Dataframe with the split-fold mapping saved to {fp}")

    # extract patches
    patchify_s2_data(
        rasters_dir=C.DIR_GL_RASTERS_INV,
        outlines_split_dir=C.DIR_OUTLINES_SPLIT,
        num_folds=C.NUM_CV_FOLDS,
        patches_dir=C.DIR_GL_PATCHES,
        patch_radius=C.PATCH_RADIUS,
        sampling_step=C.SAMPLING_STEP,
    )
