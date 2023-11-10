import geopandas as gpd
from pathlib import Path

# local imports
from utils.sampling_utils import patchify_s2_data, data_cv_split
import config as C

if __name__ == "__main__":
    s2_outlines_fp = Path(C.S2.DIR_OUTLINES_ROOT) / 'raw' / 'c3s_gi_rgi11_s2_2015_v2/c3s_gi_rgi11_s2_2015_v2.shp'

    print(f'Reading S2-based glacier outlines from {s2_outlines_fp}')
    s2_df = gpd.read_file(s2_outlines_fp)
    initial_area = s2_df.AREA_KM2.sum()
    print(f'#glaciers = {len(s2_df)}; area = {initial_area} km2')

    print(f'Keeping only the glaciers above {C.S2.MIN_GLACIER_AREA}')
    s2_df = s2_df[s2_df.AREA_KM2 >= C.S2.MIN_GLACIER_AREA]
    area = s2_df.AREA_KM2.sum()
    print(f'#glaciers = {len(s2_df)}; area = {area} km2 ({area / initial_area * 100:.2f}%)')

    # split the data regionally
    data_cv_split(
        sdf=s2_df.copy(),
        num_folds=C.S2.NUM_CV_FOLDS,
        valid_fraction=C.S2.VALID_FRACTION,
        outlines_split_dir=C.S2.DIR_OUTLINES_SPLIT
    )

    # extract patches
    patchify_s2_data(
        rasters_dir=C.S2.DIR_GL_RASTERS_INV,
        outlines_split_dir=C.S2.DIR_OUTLINES_SPLIT,
        num_folds=C.S2.NUM_CV_FOLDS,
        patches_dir=C.S2.DIR_GL_PATCHES,
        patch_radius=C.S2.PATCH_RADIUS,
        sampling_step=C.S2.SAMPLING_STEP,
    )
