from pathlib import Path


class BaseConfig:
    """ Class which defines the constants applied for all configs and specifies the dataset-specific ones """

    # the next properties are the same for all the datasets
    NUM_CV_FOLDS = 5
    VALID_FRACTION = 0.15
    MIN_GLACIER_AREA = 0.1  # km2
    DEMS_DIR = Path('../data/external/copdem_eea10m')
    DHDT_DIR = Path('../data/external/dhdt_hugonnet/11_rgi60_2000-01-01_2020-01-01/dhdt')
    GLACIER_OUTLINES_FP = Path('../data/outlines/s2/rgi_format/c3s_gi_rgi11_s2_2015_v2/c3s_gi_rgi11_s2_2015_v2.shp')
    DEBRIS_OUTLINES_FP = Path('../data/data_gmb/debris/combined_debris_s2_inv/combined_debris_s2_inv.shp')
    NODATA = -9999

    # how many cores to use when building the rasters
    NUM_CORES = 16

    # how many cores to use when evaluating the models per glacier
    NUM_CORES_EVAL = 16
    PRELOAD_DATA_INFER = True  # whether to load the netcdf files in memory before patchifying them

    # the next properties have to be specified for each dataset
    @classmethod
    @property
    def RAW_DATA_DIR(cls):
        # where the original raw (tif) images are stored
        raise NotImplementedError

    @classmethod
    @property
    def WD(cls):
        # working directory (will store the glacier-wide rasters, patches, stats etc.)
        raise NotImplementedError

    @classmethod
    @property
    def BANDS(cls):
        # which bands to keep from the original raw files (used when building the glacier-wide rasters)
        raise NotImplementedError

    @classmethod
    @property
    def PATCH_RADIUS(cls):
        """ patch radius in pixels """
        raise NotImplementedError

    @classmethod
    @property
    def SAMPLING_STEP_TRAIN(cls):
        """
            The step (in pixels) between two consecutive patches for training.
            These patches will be exported to disk.
        """
        raise NotImplementedError

    @classmethod
    @property
    def SAMPLING_STEP_INFER(cls):
        """
            The step (in pixels) between two consecutive patches for inference.
            By default, the step is the half of the one used in training (so double the overlap).
            These patches will be built in memory.
        """
        return cls.SAMPLING_STEP_TRAIN // 2

    # the next properties are derived based on the above
    @classmethod
    @property
    def DIR_OUTLINES_SPLIT(cls):
        return Path(cls.WD) / 'cv_split_outlines'

    @classmethod
    @property
    def DIR_GL_INVENTORY(cls):
        return Path(cls.WD) / Path(cls.RAW_DATA_DIR).name / 'glacier_wide'

    @classmethod
    @property
    def DIR_GL_PATCHES(cls):
        return (
                Path(cls.WD) / Path(cls.RAW_DATA_DIR).name / 'patches' /
                f"r_{cls.PATCH_RADIUS}_s_{cls.SAMPLING_STEP_TRAIN}"
        )


class S2(BaseConfig):
    """ Settings for Sentinel-2 data using the Paul. et al. 2020 outlines """

    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2/inv'
    # RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2/2023'
    WD = f'../data/external/wd/s2'

    # raw -> rasters settings
    BANDS = (
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
        'CLOUDLESS_MASK',
        'FILL_MASK'
    )

    GSD = 10  # ground sampling distance in meters

    # patch sampling settings
    PATCH_RADIUS = 128
    SAMPLING_STEP_TRAIN = 128


class S2_PLUS(S2):
    """ Settings for Sentinel-2 data using the Paul. et al. 2020 outlines with manually checked images
    (same images were replaced because they had too many clouds/shadows or had too much seasonal snow)"""

    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2_plus/inv'
    # RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2_plus/2023'
    WD = f'../data/external/wd/s2_plus'

    # csv file with the finals dates
    CSV_DATES_ALLOWED = Path(WD) / Path(RAW_DATA_DIR).name / 'aux_data' / 'final_dates.csv'
    # CSV_DATES_ALLOWED = None

    # raw -> rasters settings
    BANDS = (
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
        'CLOUDLESS_MASK',
        'FILL_MASK'
    )

    # patch sampling settings
    PATCH_RADIUS = 128
    SAMPLING_STEP_TRAIN = 128


class S2_GLAMOS(S2):
    """ Settings for Sentinel-2 data using the SGI2016 outlines. Most of the settings are the same as in S2. """

    # RAW_DATA_DIR = '../data/sat_data_downloader/external/download/glamos/inv'
    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/glamos/2023'

    WD = '../data/external/wd/s2_glamos'

    # some dates are skipped when building the rasters because of seasonal snow
    CSV_DATES_ALLOWED = Path(WD) / Path(RAW_DATA_DIR).name / 'aux_data' / 'dates_allowed.csv'

    # SGI2016 outlines in the RGI format
    GLACIER_OUTLINES_FP = Path(
        '../data/data_gmb/glamos/inventory_sgi2016_r2020_rgi_format/SGI_2016_glaciers_rgi_format.shp'
    )

    # we need specific DEMs given that the outlines are different
    DEMS_DIR = Path('../data/external/oggm/s2_glamos')


class PS(BaseConfig):
    """ Settings for manually downloaded Planet data """

    WD = f'../data/external/wd/ps'
    RAW_DATA_DIR = '../data/external/planet/raw_processed/inv'
    # RAW_DATA_DIR = '../data/external/planet/raw_processed/2023'

    # raw -> rasters settings
    BANDS = ('B', 'G', 'R', 'NIR', 'cloud', 'shadow')

    GSD = 3  # ground sampling distance in meters

    # patch sampling settings
    PATCH_RADIUS = 256
    SAMPLING_STEP_TRAIN = 256
    DIR_GL_PATCHES = f'{WD}/inv/patches/r_{PATCH_RADIUS}_s_{SAMPLING_STEP_TRAIN}'


class S2_PS(S2):
    """
        Settings for Sentinel-2 data that matches the manually downloaded Planet data.
        Most of the settings are the same as in S2.
    """
    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2_ps/inv'
    WD = '../data/external/wd/s2_ps'
    CSV_FINAL_DATES = '../data/sat_data_downloader/external/aux/s2_dates/dates_ps_s2.csv'

# specify which dataset to use
# C = S2
C = S2_PLUS
# C = S2_GLAMOS
# C = S2_PS
# C = PS
