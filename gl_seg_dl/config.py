NUM_CORES = 16


# ######################################################################################################################
# ######################################################## S2 ##########################################################
# ######################################################################################################################
class S2:
    DIR = f'../data/external/rasters/s2'
    DIR_AUX_DATA = f'{DIR}/aux_data'
    DIR_OUTLINES_ROOT = '../data/outlines/s2'
    DIR_OUTLINES_SPLIT = f'{DIR_OUTLINES_ROOT}/cv_split'
    NUM_CV_FOLDS = 5
    VALID_FRACTION = 0.15
    DIR_GL_RASTERS_INV = f'{DIR}/inv'
    MIN_GLACIER_AREA = 0.1  # km2
    DIR_GL_RASTERS_2023 = f'{DIR}/2023'
    DIRS_INFER = [DIR_GL_RASTERS_INV, DIR_GL_RASTERS_2023]  # directories on which to make glacier-wide inferences
    NUM_CORES_EVAL = 16

    # raw -> rasters settings
    BANDS = [
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
        'CLOUDLESS_MASK',
        'FILL_MASK'
    ]
    GSD = 10
    NODATA = -9999

    # patch sampling settings
    PATCH_RADIUS = 128
    SAMPLING_STEP = 128
    DIR_GL_PATCHES = f'{DIR}/patches_inv_r_{PATCH_RADIUS}_s_{SAMPLING_STEP}'


class GLAMOS(S2):
    DIR = '../data/external/rasters/glamos'
    DIR_GL_RASTERS_INV = f'{DIR}/inv'
    DIR_GL_RASTERS_2023 = f'{DIR}/2023'
    DIRS_INFER = [DIR_GL_RASTERS_INV, DIR_GL_RASTERS_2023]  # directories on which to make glacier-wide inferences


class INFERENCE:
    DIRS_INFER = S2.DIRS_INFER  # directories on which to make glacier-wide inferences
