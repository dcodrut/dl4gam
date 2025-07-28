from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Union, Any, Optional

from omegaconf import MISSING


# ======================================================================================================================
# Base dataset configuration
# ======================================================================================================================
@dataclass
class BaseDatasetConfig:
    # Dataset identifier
    name: str = MISSING

    # A label used to create a subdir where the data is stored
    # We will use 'inv' to denote the data that match the glacier outlines, which can originate from multiple years
    # If the data is from a single year, we will use the year as a label (e.g. '2023')
    year: str = MISSING

    # Path to the glacier outlines (vector-file format)
    outlines_fp: str = MISSING

    # CRS and GSD which will be used for the rasterization of the glacier outlines and the raw data
    crs: str = "UTM"  # we use local UTM projection by default; use "EPSG:XXXXX" for a specific projection
    gsd: Union[int, float] = MISSING

    # Whether to download the raw data automatically from Google Earth Engine (see raw_data settings below)
    gee_download: bool = False

    @dataclass
    class RawDataConfig:
        """
        Settings for the raw data, which can be either downloaded automatically from Google Earth Engine or provided
        as pre-downloaded images.
        """
        # Where the original raw (tif) images are stored needed to produce the glacier cubes
        # If the data is manually downloaded, we expect to find it stored as root_dir/year/glacier_id/date.tif
        base_dir: str = MISSING

        # Whether to automatically select the best images for each glacier based on cloud coverage, NDSI and albedo
        # This is useful if we have multiple images per glacier and want to select the best one.
        # If True, for each glacier, we will read all the images from the corresponding subdirectory of root_dir and
        # automatically select the best image based on cloud coverage, NDSI and albedo.
        # If False, we expect the dates_csv to be provided (see below) with the dates to be used for each glacier
        # or to have the images already selected and stored in the root_dir.
        # Note that it can also be set to True if a single image per glacier is available (e.g. already pre-processed);
        # in this case no dates_csv is expected and no QC metrics will be computed (we will just use the image as is).
        automated_selection: bool = False

        # Alternatively, we can use a csv file with the dates (e.g. from an inventory or manually chosen)
        # Note that the dates should be in the format 'YYYY-MM-DD' and the csv file should have a column 'date_acq'.
        # If automatic image selection is enabled, all the images from the raw image directory will be loaded
        # and the best will be automatically selected based on cloud coverage, NDSI and albedo.
        # The selected dates will be saved to this csv file (the default will be set in __post_init__).
        dates_csv: Optional[str] = None

        ################################################################################################################
        # The following settings can be set for automatic data downloading using Google Earth Engine
        # Note that currently this was tested for Sentinel-2 data only.  # TODO: add Landsat-8

        # (rectangular) buffer around the glaciers (in meters) used for downloading
        buffer_roi: Optional[Union[int, float]] = None

        # GEE project to use for downloading
        # (see https://developers.google.com/earth-engine/guides/transition_to_cloud_projects)
        gee_project_name: Optional[str] = None

        # GEE image collections e.g. COPERNICUS/S2_HARMONIZED
        img_collection_name: Optional[str] = None

        # date interval (within the given year) for downloading if strict dates are not imposed (see dates_csv)
        # format: DD:MM e.g. ('08-01', '10-15')
        download_window: Optional[Tuple[str, str]] = None

        # Parameters for the cloud masks
        cloud_collection_name: Optional[str] = None  # e.g. 'COPERNICUS/S2_CLOUD_PROBABILITY'
        cloud_band: Optional[str] = None  # band name for the cloud probability mask, e.g. 'probability'
        cloud_mask_thresh_p: Optional[float] = None  # threshold for binarizing the cloud probability mask, e.g. 0.4

        # Once we computed the cloud mask, we can set a maximum cloud coverage to accept for the images
        # Note that the cloud coverage (incl. other stats) is computed on the glacier + a buffer
        # (see the class `Buffers`).
        min_coverage: float = 0.9  # minimum coverage percentage to accept the scene
        max_cloud_p: float = 0.3  # max cloud coverage (glacier + buffer); if fixed dates are used, this will be ignored

        # number of images (i.e. days) to download per glacier
        # (we choose the best automatically if strict dates are not imposed)
        num_days_to_keep: int = 1

        # Specify how to sort the images for each glacier when selecting the best one.
        # After filtering the scenes using min_coverage, max_cloud_p we sort the rest by a (weighted) score using:
        sort_by: Tuple[str, str] = ('cloud_p', 'ndsi')  # 'albedo' can also be used
        # If we want to weight the scores, we can set score_weights to a tuple of floats.
        score_weights: Optional[Tuple[float, float]] = None  # if None, equal weights will be used

        skip_existing: bool = True  # whether to skip the existing images in the root_dir
        try_reading: bool = True  # whether to try reading the images before skipping the existing ones

    raw_data: RawDataConfig = field(default_factory=RawDataConfig)

    @dataclass
    class OGGMDataConfig:
        """
        Settings for the OGGM data, which will be downloaded automatically and later added to the training data after
        reprojection and clipping.
        """

        # Root data directory
        base_dir: str = "${working_dir}/raw_data/oggm"

        # Where to get the DEMs from (see https://tutorials.oggm.org/stable/notebooks/tutorials/dem_sources.html)
        dem_source = 'NASADEM'  # or None, for using the default OGGM DEM source

        # GSD of the data (most of the variables in OGGM have maximum 30m so we set it to this by default)
        gsd: int = 30

    oggm_data: OGGMDataConfig = field(default_factory=OGGMDataConfig)

    # Patch radius in pixels
    patch_radius: int = MISSING

    # The step (in pixels) between two consecutive patches for training, validation and inference.
    @dataclass
    class Strides:
        train: int = MISSING
        valid: Optional[int] = None  # we will set it automatically in __post_init__
        infer: Optional[int] = None  # we will set it automatically in __post_init__

    strides: Strides = field(default_factory=Strides)

    # Whether to export the patches to disk or build them on the fly
    export_patches: bool = MISSING

    # Number of patches to sample for training
    # It has to be smaller than the total number of patches available (which is controlled by strides.train).
    # We will then sample (without replacement) the required number of patches, either at the beginning of each
    # epoch (see sample_patches_each_epoch) or only once at the beginning of the training.
    num_patches_train: int = MISSING

    # Whether to take a different sample of the initially generated patches at the beginning of each epoch.
    # It can be enabled only when export_patches is False.
    # This can be useful in ensemble training (leading to a version  of bootstrapping). If False, the patches will be
    # the same for each epoch and only the order will be shuffled. Note however that we keep at least one patch per
    # glacier so it's not a bootstrapping in a classical sense.
    sample_patches_each_epoch: bool = MISSING

    # Buffer sizes (in meters) around the glacier outlines for the different steps of the pipeline
    @dataclass
    class Buffers:
        # Buffer for computing the Quality Control (QC) metrics, e.g. cloud coverage, NDSI, albedo.
        # For NDSI, only non-ice & non-cloud pixels will be used. See `gee_download.py` or 'rank_images.py` for details.
        buffer_qc_metrics: Union[int, float] = 100  # in meters; if None, the entire ROI will be used for the metrics

        # Buffer for the final processed glacier cube (should be large enough to allow patch sampling)
        cube: Optional[Union[int, float]] = None  # will be set automatically in __post_init__

        # Buffer from within which patch centres are sampled
        patch_sampling: Union[int, float] = MISSING

        # Buffer size used at inference time within which the positive pixels will be counted
        # Useful if we expect that outlines are not perfect, and we want to include some buffer around the glacier.
        # Or if we expect the glaciers to grow w.r.t. the date of the outlines.
        infer: Union[int, float] = 0

        # Buffer interval on which False Positive pixels will be counted
        # (if 'auto', we will automatically derive it s.t. the two classes - glacier/non-glacier - are balanced)
        fp: Tuple[Any, Any] = (0, 'auto')

    buffers: Buffers = field(default_factory=Buffers)

    # Whether to check if the raw data covers 100% of the glacier outlines (incl. the buffer) when building the glacier
    # cubes. If False, we will keep the desired buffer but fill the missing pixels with NaNs.
    check_data_coverage = True

    # Bands re-naming when processing the raw data. Note that only the bands used as keys will be kept, the rest will be
    # dropped. We also need this information to compute the NDSI and albedo.
    bands_rename: Dict[str, str] = MISSING

    # A list of bands to be merged into a QC mask (e.g. ['~CLOUDLESS_MASK', 'SOME_OTHER_MASK']), where ~ means negation
    bands_nok_mask: List[str] = field(default_factory=list)

    # A dictionary {name -> path} with the paths to various directories that contain additional raster data to be
    # added to the glacier cubes. The data is expected to be in a raster format (e.g. tif) and it will be
    # automatically matched (i.e. all the scenes intersecting the glacier cube will be merged and reprojected).
    extra_rasters: Dict[str, str] = field(default_factory=dict)

    # A dictionary {name -> path} with the paths to various vector files to be rasterized as binary masks and added
    # to the glacier cubes. The data is expected to be in a vector format (e.g. shp).
    # We will automatically reproject and clip the data to the glacier cube.
    extra_vectors: Dict[str, str] = field(default_factory=dict)

    # A list of features to compute using xDEM (e.g. slope, aspect, etc.)
    # See https://xdem.readthedocs.io/en/stable/gen_modules/xdem.DEM.get_terrain_attribute.html
    xdem_features: Optional[List[str]] = None

    # ==================================================================================================================
    # Local paths; # TODO: see later if these should be set in __post_init__ such that they are frozen
    # ==================================================================================================================

    # Root directory for all the data which will be processed
    base_dir: str = "${working_dir}/datasets/${dataset.name}/${dataset.year}"

    # Processed inventory outlines with all the additional derived geometries
    geoms_fp: str = "${dataset.base_dir}/geoms.gpkg"

    # A csv file with the glacier IDs and their corresponding folds under each cross-validation iteration.
    split_csv: str = "${dataset.base_dir}/cv_split_outlines/map_cv_iter_${pl.cv_iter}.csv"

    # Path to the processed glacier cubes (netcdf) after processing
    cubes_dir: str = "${dataset.base_dir}/glacier_cubes"

    # Path to a csv with the normalization stats of the current cross-validation iteration
    norm_stats_csv: str = "${dataset.base_dir}/norm_stats/stats_cv_iter_${pl.cv_iter}.csv"

    # Directory for the patches (if they are exported, otherwise will be later set to None)
    patches_dir: Optional[str] = "${dataset.base_dir}/patches/r_${dataset.patch_radius}_s_${dataset.strides.train}"

    def __post_init__(self):
        """
        Generate intermediate paths, set dynamic default values in case they are not provided and a few other checks.
        """

        # If we don't export patches, we will probably use a small stride for training;
        # => set the default valid & infer depending on the training stride
        if self.export_patches:
            if self.strides.valid is None:
                self.strides.valid = self.strides.train
            if self.strides.infer is None:
                self.strides.infer = self.strides.train // 2
            self.sample_patches_each_epoch = False
        else:
            if self.strides.valid is None:
                self.strides.valid = self.strides.train * 2
            if self.strides.infer is None:
                self.strides.infer = self.strides.train

            # set the patch directory to None
            self.patches_dir = None

        # Check if the infer stride is small enough to have some overlap
        # (we discard some border pixels, i.e. 5%, to avoid edge effects)
        if self.strides.infer >= int(self.patch_radius * 0.9):
            raise ValueError(
                f"Inference stride ({self.strides.infer}) is too large for the patch radius ({self.patch_radius}). "
                f"It should be smaller than {self.patch_radius * 0.9} to have a minimum of 10% overlap."
            )

        # Set the default cube buffer only enough to cover sampled patches
        if self.buffers.cube is None:
            self.buffers.cube = self.patch_radius * self.gsd + self.buffers.patch_sampling

        # Generate the path to the dates csv file if needed.
        if self.raw_data.dates_csv is None:
            if self.raw_data.automated_selection:
                self.raw_data.dates_csv = f"{self.base_dir}/dates.csv"
            else:
                raise ValueError(
                    "raw_data.dates_csv must be set when automatic_selection is False. "
                    "Please provide the path to the csv file with the dates."
                )

        # Check if the required raw data settings are set
        if self.gee_download:
            if self.raw_data.buffer_roi is None:
                raise ValueError("raw_data.buffer_roi must be set for automatic download.")
            if self.raw_data.gee_project_name is None or self.raw_data.img_collection_name is None:
                raise ValueError("raw_data.gee_project and raw_data.gee_collection must be set for GEE download.")

        # If automatic selection is not enabled, we disable the coverage and cloud filtering
        if not self.raw_data.automated_selection:
            self.raw_data.min_coverage = 0.0
            self.raw_data.max_cloud_p = 1.0
            self.raw_data.num_days_to_keep = 1


# ======================================================================================================================
# Sentinel-2 Alps dataset config
# ======================================================================================================================
@dataclass
class S2AlpsConfig(BaseDatasetConfig):
    """ Configuration for the Sentinel-2 Alps dataset based on the inventory from Paul et al. (2020).

    The images are either downloaded for the same dates as the inventory outlines (year='inv') or for a specific year
    (e.g. '2023'). For the latter, the dates will be automatically selected based on e.g. the cloud coverage and NDSI
    (depending on the `raw_data` settings).
    """

    name: str = 's2_alps'
    gsd: int = 10
    outlines_fp: str = "../data/outlines/paul_et_al_2020/c3s_gi_rgi11_s2_2015_v2.shp"
    year: str = 'inv'  # or a certain year (e.g. '2023')
    gee_download: bool = True  # whether to download the raw data automatically from GEE

    export_patches: bool = False
    sample_patches_each_epoch: bool = False
    patch_radius: int = 128  # in pixels
    num_patches_train: int = 7500 // 16 * 16  # multiple of 16 (batch size)
    strides: BaseDatasetConfig.Strides = field(default_factory=lambda: BaseDatasetConfig.Strides(train=32))
    buffers: BaseDatasetConfig.Buffers = field(
        default_factory=lambda: BaseDatasetConfig.Buffers(
            patch_sampling=50,
            infer=20,
            fp=(20, 'auto')
        )
    )

    raw_data: BaseDatasetConfig.RawDataConfig = field(
        default_factory=lambda: BaseDatasetConfig.RawDataConfig(
            base_dir="../data/external/dl4gam/raw_data/images/s2_alps/yearly",
            automated_selection=False,
            dates_csv="../data/inv_images_qc/final_dates.csv",
            # we need a buffer >= patch radius,
            # but we use a larger buffer in case we later want to increase the patch size and avoid redownloading
            buffer_roi=(S2AlpsConfig.patch_radius * 2 + 5) * S2AlpsConfig.gsd,
            gee_project_name='your-project-id',  # set it to your GEE project
            img_collection_name='COPERNICUS/S2_HARMONIZED',  # L2 doesn't include 2015 data
            cloud_collection_name='GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED',
            cloud_band='cs',  # band name for the cloud probability mask
            cloud_mask_thresh_p=0.6,  # threshold for binarizing the cloud probability mask
        )
    )

    # Always add the COPDEM30 no matter the year
    # TODO: take it from OGGM dirs
    extra_rasters: Dict[str, str] = field(default_factory=lambda: {
        'dem': '../data/external/copdem_30m',
    })
    # whether to include the dhdt raster from Hugonnet et al. (2021) (see also `__post_init__`)
    include_dhdt_raster: bool = True

    bands_rename: Dict[str, str] = field(default_factory=lambda: {
        'B1': 'B1',
        'B2': 'B',
        'B3': 'G',
        'B4': 'R',
        'B5': 'B5',
        'B6': 'B6',
        'B7': 'B7',
        'B8': 'NIR',
        'B8A': 'B8A',
        'B9': 'B9',
        'B10': 'B10',
        'B11': 'SWIR',
        'B12': 'B12'
    })

    bands_nok_mask: List[str] = field(default_factory=lambda: ['cloud_mask'])

    # Use the (mixed) debris product processed in the notebook
    extra_vectors: Dict[str, str] = field(default_factory=lambda: {
        'debris': '../data/outlines/debris_multisource/debris_multisource.shp'
    })

    # Which features to compute using xDEM
    xdem_features: List[str] = field(
        default_factory=lambda: [
            'slope',
            'aspect',
            'planform_curvature',
            'profile_curvature',
            'terrain_ruggedness_index'
        ]
    )

    def __post_init__(self):
        super().__post_init__()

        # Automatically add the 'dhdt' raster from Hugonnet et al. (2021)
        if self.include_dhdt_raster:
            if self.year == 'inv':
                self.extra_rasters['dhdt'] = '../data/external/dhdt_hugonnet/11_rgi60_2010-01-01_2015-01-01/dhdt'
            else:
                # For the year, we will use the closest pentad to the given year (e.g. 2023)
                # Note that the data is available for four 5-year periods: 2000-2005, 2005-2010, 2010-2015, 2015-2020
                pentad_start = (int(self.year) // 5) * 5
                pentad_start = max(min(pentad_start, 2015), 2000)  # limit to the available pentads
                pentad_end = pentad_start + 5
                dirname = f"11_rgi60_{pentad_start}-01-01_{pentad_end}-01-01"
                self.extra_rasters['dhdt'] = f"../data/external/dhdt_hugonnet/{dirname}/dhdt"

        # Set the dates to the inventory ones (exported in the notebook)
        if self.year == 'inv':
            self.csv_dates = "../data/outlines/paul_et_al_2020/dates.csv"
        else:
            # Automatically select the images in the given year and the following window
            self.raw_data.automated_selection = True
            self.raw_data.download_window = ('08-01', '10-15')

            # Thresholds applied to the images before ranking them
            self.raw_data.min_coverage = 0.9
            self.raw_data.max_cloud_p = 0.3


# ======================================================================================================================
# Sentinel-2 Alps+ dataset config
# ======================================================================================================================
@dataclass
class S2AlpsPlusConfig(S2AlpsConfig):
    """
    Configuration for the Sentinel-2 Alps dataset with manually curated imagery dates (for the inventory year).
    Everything else is the same as in S2AlpsConfig.
    """

    name: str = 's2_alps_plus'
    raw_data: BaseDatasetConfig.RawDataConfig = field(
        default_factory=lambda: replace(
            S2AlpsConfig().raw_data,
            dates_csv='../data/inv_images_qc/final_dates.csv',  # manually curated dates
        )
    )
