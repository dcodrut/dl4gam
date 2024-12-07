from pathlib import Path

import geopandas as gpd
import pandas as pd

# local imports
from config import C
from utils.sampling_utils import patchify_data, data_cv_split

if __name__ == "__main__":
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

    # extract patches
    patchify_data(
        rasters_dir=C.DIR_GL_INVENTORY,
        outlines_split_dir=C.DIR_OUTLINES_SPLIT,
        num_folds=C.NUM_CV_FOLDS,
        patches_dir=C.DIR_GL_PATCHES,
        patch_radius=C.PATCH_RADIUS,
        sampling_step=C.SAMPLING_STEP_TRAIN,
        folds=('train', 'valid', 'test')
    )
