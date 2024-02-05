from pathlib import Path


class BaseConfig:
    """ Class which defines the constants applied for all configs and specifies the dataset specific ones """

    # the next properties are the same for all the datasets
    NUM_CV_FOLDS = 5
    VALID_FRACTION = 0.15
    MIN_GLACIER_AREA = 0.1  # km2
    DEMS_DIR = Path('../data/external/oggm/s2')
    GLACIER_OUTLINES_FP = Path('../data/outlines/s2/rgi_format/c3s_gi_rgi11_s2_2015_v2/c3s_gi_rgi11_s2_2015_v2.shp')
    DEBRIS_OUTLINES_FP = dict(
        debris_scherler_2018='../data/data_gmb/debris/scherler_2018/11_rgi60_CentralEurope_S2_DC_2015_2017_mode.shp',
        debris_sgi_2016='../data/data_gmb/glamos/inventory_sgi2016_r2020/SGI_2016_debriscover.shp',
    )
    NODATA = -9999

    # how many cores to use when building the rasters
    NUM_CORES = 32

    # how many cores to use when evaluating the models per glacier
    NUM_CORES_EVAL = 16

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
    def SAMPLING_STEP(cls):
        """ The step (in pixels) between two consecutive patches """
        raise NotImplementedError

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
        return Path(cls.WD) / Path(cls.RAW_DATA_DIR).name / 'patches' / f"r_{cls.PATCH_RADIUS}_s_{cls.SAMPLING_STEP}"


class S2(BaseConfig):
    """ Settings for Sentinel-2 data using the Paul. et al. 2020 outlines """

    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2/inv'
    WD = f'../data/external/wd/s2'

    # raw -> rasters settings
    BANDS = (
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
        'CLOUDLESS_MASK',
        'FILL_MASK'
    )

    # patch sampling settings
    PATCH_RADIUS = 128
    SAMPLING_STEP = 128


class S2_GLAMOS(S2):
    """ Settings for Sentinel-2 data using the SGI2016 outlines. Most of the settings are the same as in S2. """

    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2_glamos/inv'
    WD = '../data/external/wd/s2_glamos'


class PS(BaseConfig):
    """ Settings for manually downloaded Planet data """

    WD = f'../data/external/wd/ps'
    RAW_DATA_DIR = '../data/external/planet/raw_processed/inv'

    # raw -> rasters settings
    BANDS = ('B', 'G', 'R', 'NIR', 'cloud', 'shadow')

    # patch sampling settings
    PATCH_RADIUS = 256
    SAMPLING_STEP = 256
    DIR_GL_PATCHES = f'{WD}/inv/patches/r_{PATCH_RADIUS}_s_{SAMPLING_STEP}'


class S2_PS(S2):
    """
        Settings for Sentinel-2 data that matches the manually downloaded Planet data.
        Most of the settings are the same as in S2.
    """
    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2_ps/inv'
    WD = '../data/external/wd/s2_ps'


# specify which dataset to use
C = S2
