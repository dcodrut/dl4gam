from abc import ABC, abstractmethod
from pathlib import Path


class BaseConfig(ABC):
    """ Class which defines the constants applied for all configs and specifies the dataset-specific ones """

    # root working directory, where the glacier-wide rasters, patches, stats etc. are stored
    # (this is the same for all the datasets)
    ROOT_WD = Path('../data/external/wd')

    # the next properties are the same for all the datasets
    NUM_CV_FOLDS = 5
    VALID_FRACTION = 0.1
    MIN_GLACIER_AREA = 0.1  # km2
    NODATA = -9999

    # how many processes to use in the parallel processing (when possible/implemented, e.g. building the rasters)
    NUM_PROCS = 8

    # whether to load the netcdf files in memory before patchifying them
    # (at inference time or at training time if EXPORT_PATCHES is False)
    PRELOAD_DATA = False

    # whether to export the patches to disk (if False, the patches will be built on the fly);
    # building the patches on the fly is a bit slower but not by much as we use xarray's lazy loading;
    # the advantage is that we save disk space, especially when using a small sampling stride;
    # additionally, it can increase the diversity of the patches as we can sample more and the epochs will be different
    EXPORT_PATCHES = False

    # the next properties have to be specified for each dataset
    @classmethod
    @property
    @abstractmethod
    def GLACIER_OUTLINES_FP(cls):
        # path to the glacier outlines (a shapefile); two columns (except 'geometry') are expected 'Area' and 'entry_id'
        ...

    @classmethod
    @property
    @abstractmethod
    def RAW_DATA_DIR(cls):
        # where the original raw (tif) images are stored
        ...

    @classmethod
    @property
    @abstractmethod
    def SUBDIR(cls):
        # subdirectory of the WD where the data is stored (for separating the data by years e.g. 'inv', '2023')
        ...

    @classmethod
    @property
    def BANDS_NAME_MAP(cls):
        # a dict with the optical bands to keep from the raw images when building the glacier-wide rasters (keys)
        # AND the new names for the bands if needed (values)
        # (in case we want to derive optical indices (i.e. NDVI, NDWI & NDSI) we expect the R, G, B, NIR, SWIR bands)
        # !Note that we will attempt to use the "long_name" attribute of the bands in the raw files
        # and if this is not available, the map keys should be B1, B2, B3, ... etc.

        # if it is None, no bands subset or renaming will be applied
        return None

    @classmethod
    @property
    def BANDS_QC_MASK(cls):
        # a list of the bands that are used to build the quality mask (e.g. FILL_MASK, CLOUD_MASK, SHADOW_MASK)
        # the mask is built using the bitwise OR of the bands
        # use ~ as prefix to invert a mask
        return None

    @classmethod
    @property
    @abstractmethod
    def PATCH_RADIUS(cls):
        """ patch radius in pixels """
        ...

    @classmethod
    @property
    @abstractmethod
    def SAMPLING_STEP_TRAIN(cls):
        """
            The step (in pixels) between two consecutive patches for training.
            These patches will be either be exported to disk or prepared on the fly, see EXPORT_PATCHES.
        """
        ...

    @classmethod
    @property
    def NUM_PATCHES_TRAIN(cls):
        """
            The number of patches to sample for training. Has to be smaller than the total number of patches available
            (which is controlled by SAMPLING_STEP_TRAIN).

            We will then sample (without replacement) the required number of patches, either at the beginning of each
            epoch (see SAMPLE_PATCHES_EACH_EPOCH) or once at the beginning of the training.

            This is used only when EXPORT_PATCHES is False.
        """
        if cls.EXPORT_PATCHES:
            return None

        raise NotImplementedError

    @classmethod
    @property
    def SAMPLE_PATCHES_EACH_EPOCH(cls):
        """
            Whether to take a new sample of the initially generated patches at the beginning of each epoch.
            This is used only when EXPORT_PATCHES is False.
            If False, the patches will be the same for each epoch and the order will be shuffled. This could be useful
            when training an ensemble of models (~bootstrapping).
            Note however that we keep at least one patch per glacier so it's not a bootstrapping in a classical sense.
        """
        if cls.EXPORT_PATCHES:
            return False

        return True

    @classmethod
    @property
    def SAMPLING_STEP_VALID(cls):
        """
            The step (in pixels) between two consecutive patches for validation.
            By default, the step is the same as the training one if patches are exported to disk
            or half of the training one otherwise.
        """
        return cls.SAMPLING_STEP_TRAIN if cls.EXPORT_PATCHES else cls.SAMPLING_STEP_TRAIN * 2

    @classmethod
    @property
    def SAMPLING_STEP_TEST(cls):
        """
            The step (in pixels) between two consecutive patches for (glacier-wide) inference.
            By default, the step is the half of the one used in training if patches are exported to disk.
                (so double the overlap & (potentially) x4 more patches) or the same as the training one otherwise.
            These patches will be built on the fly, one glacier at a time.
        """
        return cls.SAMPLING_STEP_TRAIN // 2 if cls.EXPORT_PATCHES else cls.SAMPLING_STEP_TRAIN

    # the next properties are derived based on the above
    @classmethod
    @property
    def WD(cls):
        # derived working directory from the class name containing the dataset-specific settings
        return cls.ROOT_WD / cls.__name__.lower()

    @classmethod
    @property
    def DIR_OUTLINES_SPLIT(cls):
        return Path(cls.WD) / 'cv_split_outlines'

    @classmethod
    @property
    def DIR_GL_RASTERS(cls):
        return Path(cls.WD) / cls.SUBDIR / 'glacier_wide'

    @classmethod
    @property
    def DIR_GL_PATCHES(cls):
        """ Directory where the patches are stored, in we don't train with on-the-fly patch generation """
        return Path(cls.WD) / cls.SUBDIR / 'patches' / f"r_{cls.PATCH_RADIUS}_s_{cls.SAMPLING_STEP_TRAIN}" \
            if cls.EXPORT_PATCHES else None

    @classmethod
    @property
    def EXTRA_RASTERS(cls):
        """
            A dictionary {name -> path} with the paths to various directories that contain additional raster data to be
             added to the optical data. The data is expected to be in a raster format (e.g. tif) and it will be
             automatically matched to the glacier directories. This means that we automatically find the files that
             intersect the current glacier, merge them if needed, resample them to the same resolution as the optical
             and finally add them to the glacier optical dataset. Note that we expect the data to be static (not sure
             what happens in the merging step now if multiple files for the same pixels are found).
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
             selected based on cloud coverage and the snow index (currently this works for Sentinel-2 data only).
        """

        return None


class S2_ALPS(BaseConfig):
    """ Settings for Sentinel-2 data using the Paul. et al. 2020 outlines """

    # glacier outlines
    GLACIER_OUTLINES_FP = Path('../data/outlines/paul_et_al_2020/c3s_gi_rgi11_s2_2015_v2.shp')

    # get the year from the environment variables if exists, otherwise assume it's the inventory year (i.e. mainly 2015)
    import os
    SUBDIR = os.environ.get('S2_ALPS_YEAR', 'inv')
    assert SUBDIR in ('inv', '2023')

    if SUBDIR == 'inv':
        # extra rasters to be added to the optical data
        EXTRA_RASTERS = {
            'dem': Path('../data/external/copdem_30m'),
            'dhdt': Path('../data/external/dhdt_hugonnet/11_rgi60_2010-01-01_2015-01-01/dhdt'),
            'v': Path('../data/external/velocity/its_live/2015'),
        }
        RAW_DATA_DIR = f'../data/sat_data_downloader/external/download/s2_alps/yearly/'  # multiple years are needed
    elif SUBDIR == '2023':
        # extra rasters to be added to the optical data
        EXTRA_RASTERS = {
            'dem': Path('../data/external/copdem_30m'),
            'dhdt': Path('../data/external/dhdt_hugonnet/11_rgi60_2015-01-01_2020-01-01/dhdt'),
            'v': Path('../data/external/velocity/its_live/2022'),
        }
        RAW_DATA_DIR = f'../data/sat_data_downloader/external/download/s2_alps/yearly/{SUBDIR}'

    # extra vector data
    EXTRA_GEOMETRIES = {
        'debris': Path('../data/outlines/debris_multisource/debris_multisource.shp'),
    }

    # raw -> rasters settings
    BANDS_NAME_MAP = {k: k for k in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']}
    BANDS_NAME_MAP.update({
        'B2': 'B',
        'B3': 'G',
        'B4': 'R',
        'B8': 'NIR',
        'B11': 'SWIR'  # this is SWIR1, but we will call it SWIR
    })
    # BANDS_QC_MASK = ['FILL_MASK', 'SHADOW_MASK', 'CLOUD_MASK']
    BANDS_QC_MASK = ['~CLOUDLESS_MASK']  # see https://geedim.readthedocs.io/en/latest/cli.html#geedim-download

    GSD = 10  # ground sampling distance in meters

    # patch sampling settings
    PATCH_RADIUS = 128
    SAMPLING_STEP_TRAIN = 32
    NUM_PATCHES_TRAIN = 10000
    SAMPLE_PATCHES_EACH_EPOCH = False
    SAMPLING_STEP_VALID = 64
    SAMPLING_STEP_TEST = 32


class S2_ALPS_PLUS(S2_ALPS):
    """
        Settings for Sentinel-2 data using the Paul. et al. 2020 outlines with manually checked images.
        Same inventory images were removed or replaced (when possible) because they had too many clouds/shadows or
        had too much seasonal snow. The final dates are read from a csv file (CSV_DATES_ALLOWED).
    """

    # csv file with the finals inventory dates
    CSV_DATES_ALLOWED = '../data/inv_images_qc/final_dates.csv' if S2_ALPS.SUBDIR == 'inv' else None


class S2_SGI(S2_ALPS):
    """ Settings for Sentinel-2 data using the SGI2016 outlines. Most of the settings are the same as in S2. """

    RAW_DATA_DIR = f'../data/sat_data_downloader/external/download/s2_sgi/{S2_ALPS.SUBDIR}'

    # some dates are skipped when building the rasters because of seasonal snow
    # CSV_DATES_ALLOWED = Path(WD) / Path(RAW_DATA_DIR).name / 'aux_data' / 'dates_allowed.csv'
    CSV_DATES_ALLOWED = None

    # SGI2016 outlines in the RGI format
    GLACIER_OUTLINES_FP = Path('../data/outlines/sgi/inventory_sgi2016_r2020_processed/SGI_2016_glaciers_processed.shp')


# specify which dataset to use
# C = S2_ALPS
C = S2_ALPS_PLUS
# C = S2_SGI
