from pathlib import Path


class BaseConfig:
    """ Class which defines the constants applied for all configs and specifies the dataset-specific ones """

    # the next properties are the same for all the datasets
    NUM_CV_FOLDS = 5
    VALID_FRACTION = 0.1
    MIN_GLACIER_AREA = 0.1  # km2
    NODATA = -9999

    # how many processes to use when building the rasters
    NUM_PROCS = 8

    # how many processes to use when evaluating the models per glacier
    NUM_PROCS_EVAL = 16
    PRELOAD_DATA_INFER = True  # whether to load the netcdf files in memory before patchifying them (at inference time)

    # the next properties have to be specified for each dataset
    # we raise a NotImplementedError for the properties that need to be implemented
    @classmethod
    @property
    def GLACIER_OUTLINES_FP(cls):
        # path to the glacier outlines (a shapefile); two columns (except 'geometry') are expected 'Area' and 'entry_id'
        raise NotImplementedError

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
            By default, the step is the half of the one used in training
                (so double the overlap & ~four time more patches).
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

    @classmethod
    @property
    def EXTRA_RASTERS(cls):
        """
            A dictionary {name -> path} with the paths to various directories that contain additional raster data to be
             added to the optical data. The data is expected to be in a raster format (e.g. tif) and it will be
             automatically matched to the glacier directories. This means that we automatically find the files that
             intersect the current glacier, merge them if needed, resample them to the same resolution as the optical
             and finally add them to the glacier optical dataset.
        """
        return {}

    @classmethod
    @property
    def EXTRA_GEOMETRIES(cls):
        """
            A dictionary {name -> path} with the paths to various shape files to be rasterized as binary masks and added
             to the optical data. The data is expected to be in a vector format (e.g. shp), with one file per feature.
        """
        return {}

    @classmethod
    @property
    def CSV_DATES_ALLOWED(cls):
        """
            Path to a csv file containing which dates are allowed for each glacier.
            If None, all the images from the raw image directory will be accessed and the best will be automatically
             selected based on cloud coverage and the snow index.
        """

        return None


class S2_ALPS(BaseConfig):
    """ Settings for Sentinel-2 data using the Paul. et al. 2020 outlines """

    # glacier outlines
    GLACIER_OUTLINES_FP = Path('../data/outlines/paul_et_al_2020/c3s_gi_rgi11_s2_2015_v2.shp')

    # get the year from the environment variables if exists, otherwise assume it's the inventory year (i.e. mainly 2015)
    import os
    _year = os.environ.get('S2_ALPS_YEAR', 'inv')
    assert _year in ('inv', '2023')

    if _year == 'inv':
        # extra rasters to be added to the optical data
        EXTRA_RASTERS = {
            'dem': Path('../data/external/copdem_30m'),
            'dhdt': Path('../data/external/dhdt_hugonnet/11_rgi60_2010-01-01_2015-01-01/dhdt'),
            'v': Path('../data/external/velocity/its_live/2015'),
        }
    elif _year == '2023':
        # extra rasters to be added to the optical data
        EXTRA_RASTERS = {
            'dem': Path('../data/external/copdem_30m'),
            'dhdt': Path('../data/external/dhdt_hugonnet/11_rgi60_2015-01-01_2020-01-01/dhdt'),
            'v': Path('../data/external/velocity/its_live/2022'),
        }

    # extra vector data
    EXTRA_GEOMETRIES = {
        'debris': Path('../data/outlines/debris_multisource/debris_multisource.shp'),
    }

    RAW_DATA_DIR = f'../data/sat_data_downloader/external/download/s2_alps/{_year}'
    WD = f'../data/external/wd/s2_alps'

    # raw -> rasters settings
    BANDS = (
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
        'CLOUDLESS_MASK',
        'FILL_MASK'
    )

    GSD = 10  # ground sampling distance in meters

    # patch sampling settings
    PATCH_RADIUS = 128
    SAMPLING_STEP_TRAIN = 64


class S2_ALPS_PLUS(S2_ALPS):
    """
        Settings for Sentinel-2 data using the Paul. et al. 2020 outlines with manually checked images.
        Same inventory images were removed or replaced (when possible) because they had too many clouds/shadows or
        had too much seasonal snow. The final dates are read from a csv file (CSV_DATES_ALLOWED).
    """

    WD = f'../data/external/wd/s2_alps_plus'

    # csv file with the finals inventory dates
    CSV_DATES_ALLOWED = '../data/inv_images_qc/final_dates.csv' if S2_ALPS._year == 'inv' else None


class S2_SGI(S2_ALPS):
    """ Settings for Sentinel-2 data using the SGI2016 outlines. Most of the settings are the same as in S2. """

    RAW_DATA_DIR = f'../data/sat_data_downloader/external/download/s2_sgi/{S2_ALPS._year}'

    WD = '../data/external/wd/s2_sgi'

    # some dates are skipped when building the rasters because of seasonal snow
    # CSV_DATES_ALLOWED = Path(WD) / Path(RAW_DATA_DIR).name / 'aux_data' / 'dates_allowed.csv'
    CSV_DATES_ALLOWED = None

    # SGI2016 outlines in the RGI format
    GLACIER_OUTLINES_FP = Path('../data/outlines/sgi/inventory_sgi2016_r2020_processed/SGI_2016_glaciers_processed.shp')


class PS(S2_ALPS):
    """ Settings for manually downloaded Planet data """

    # extra rasters to be added to the optical data
    EXTRA_RASTERS = {
        'dem': Path('../data/external/copdem_30m')
    }

    WD = f'../data/external/wd/ps_alps'
    RAW_DATA_DIR = '../data/external/planet/raw_processed/inv'
    # RAW_DATA_DIR = '../data/external/planet/raw_processed/2023'

    # raw -> rasters settings
    BANDS = ('B', 'G', 'R', 'NIR', 'cloud', 'shadow')

    GSD = 3  # ground sampling distance in meters

    # patch sampling settings
    PATCH_RADIUS = 256
    SAMPLING_STEP_TRAIN = 256
    DIR_GL_PATCHES = f'{WD}/inv/patches/r_{PATCH_RADIUS}_s_{SAMPLING_STEP_TRAIN}'


class S2_PS(S2_ALPS):
    """
        Settings for Sentinel-2 data that matches the manually downloaded Planet data.
        Most of the settings are the same as in S2.
    """
    RAW_DATA_DIR = '../data/sat_data_downloader/external/download/s2_ps/inv'
    WD = '../data/external/wd/s2_ps'
    CSV_FINAL_DATES = '../data/sat_data_downloader/external/aux/s2_dates/dates_ps_s2.csv'

# specify which dataset to use
# C = S2_ALPS
C = S2_ALPS_PLUS
# C = S2_SGI
# C = S2_PS
# C = PS
