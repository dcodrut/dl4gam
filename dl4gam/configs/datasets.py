from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Any, Optional

from omegaconf import MISSING


class QCMetric(str, Enum):
    """
    Quality Control (QC) metrics that can be used for selecting the best images.
    If provided, each of these metrics will be computed on separate geometries.
    Otherwise, we will use the entire ROI (i.e. glacier + rectangular buffer).
    
    A (weighted) score will be then computed based on the specified metrics (a tuple of QCMetric values)
    and the images will be sorted by this score (see `dl4gam.utils.gee.sort_images` for details).
    """

    CLOUD_P = 'cloud_p'  # average cloud coverage percentage
    NDSI = 'ndsi'  # average Normalized Difference Snow Index
    ALBEDO = 'albedo'  # average Albedo (average reflectance in visible bands)

    def __str__(self) -> str:
        return self.value


# ======================================================================================================================
# Raw data configurations: 1) local (i.e. assumed to be already downloaded) and 2) automatically downloaded from GEE
# ======================================================================================================================
@dataclass
class RawImagesCfg:
    """
    Settings for locally stored (i.e. pre-downloaded) raw images.

    We expect the images to be stored in a directory structure like:
    root_dir/year/glacier_id/date*.tif
    where:
    - root_dir is the base directory where the raw images are stored (see `base_dir` below)
    - year is the year of the images (e.g. '2023' or 'inv' for a multi-year inventory; see `year` in BaseDatasetCfg)
    - glacier_id is the ID of the glacier (e.g. 'RGI60-11.00100')
    - date* is the filename containing the date of the image, e.g. '2023-08-15_ABC.tif'
    """

    # Where the original raw (tif) images are stored
    base_dir: str = MISSING

    # Which bands to keep from the raw images.
    bands: Tuple[str, ...] = MISSING

    # In case multiple images are available for a glacier, we can specify the dates to use for each glacier in a CSV.
    # The CSV should contain two columns: 'entry_id' and 'date' as "YYYY-MM-DD".
    # Alternatively, we can use the `automated_selection` (see below) to automatically select the best images.
    dates_csv: Optional[str] = None

    # Whether to automatically select the best images for each glacier, in case multiple are available
    # If True, for each glacier, we will read all the images from the corresponding subdirectory of root_dir and
    # automatically select the best image based on e.g. cloud coverage, NDSI and albedo (see below).
    # If False, we expect either:
    #  - the glacier has a single image available in the root_dir/year/glacier_id/ subdirectory
    #  - the dates_csv to be provided (see above)
    automated_selection: bool = False

    # Whether to expect that in the end all glaciers have at least one image available.
    # If True, we will raise an error if any glacier does not have an image after assigning the images.
    ensure_all_glaciers_have_images: bool = True

    # We assume that the images have the required bands to compute these metrics:
    # - for using average cloud coverage (`cloud_p`), we expect the band 'cloud_mask' to be present
    # - for using NDSI ('ndsi'), we expect the bands 'G' and 'NIR' to be present
    # - for using albedo ('albedo'), we expect the bands 'R', 'G', 'B' to be present
    # In case the required optical bands exist but with different names, we can set the following map:
    bands_rename: Optional[Dict[str, str]] = None

    # Buffer for computing the Quality Control (QC) metrics
    # For NDSI, only non-ice & non-cloud pixels will be used. See `gee_download.py` or 'rank_images.py` for details.
    buffer_qc_metrics: float = 100  # in meters; if None, the entire ROI will be used for the metrics

    # Assuming we have a cloud mask, we can set a maximum cloud coverage to accept for the images
    # (Only used if `automated_selection` is True)
    min_coverage: float = 0.9  # minimum coverage percentage to accept the scene
    max_cloud_p: float = 0.3  # max cloud coverage (glacier + buffer)

    # Specify how to sort the images for each glacier when selecting the best one.
    # After filtering the scenes using min_coverage & max_cloud_p, we sort the rest by a (weighted) score using:
    sort_by: Tuple[QCMetric, ...] = (QCMetric.CLOUD_P, QCMetric.NDSI)
    # If we want to weight the scores, we can set score_weights to a tuple of floats.
    score_weights: Optional[Tuple[float, ...]] = None  # if None, equal weights will be used


@dataclass
class LocalRawImagesCfg(RawImagesCfg):
    """
    No new settings are added here, this class is just a placeholder to distinguish between local and GEE raw images.
    """


@dataclass
class GEERawImagesCfg(RawImagesCfg):
    """
    On top of the local raw data settings, this class contains the additional settings for automatically downloading
    the raw data using Google Earth Engine. See the `download_best_images` function for details.

    Note that currently this was tested for Sentinel-2 data only.  # TODO: add Landsat-8
    """

    # (rectangular) buffer around the glaciers (in meters) used for downloading
    # (could be set to a larger value to prevent redownloading if later we want to increase the patch size)
    buffer_roi: float = MISSING

    # GEE project to use for downloading
    # (see https://developers.google.com/earth-engine/guides/transition_to_cloud_projects)
    gee_project_name: str = MISSING

    # GEE image collections e.g. COPERNICUS/S2_HARMONIZED
    img_collection_name: str = MISSING

    # Date interval (within the given year) for downloading if strict dates are not imposed (see `dates_csv` above).
    # If both `download_time_window` and `dates_csv` are set, we expect a column `date_inv` in the inventory outlines
    # format: DD:MM e.g. ('08-01', '10-15')
    download_time_window: Optional[Tuple[str, str]] = None

    # Number of images (i.e. days) to download per glacier
    # (we choose the best automatically if exact dates are not used; see `automated_selection` in RawImagesCfg)
    num_days_to_keep: int = 1

    # Parameters for the cloud masks
    # (required if we want to filter out the images based on cloud coverage or mask out the cloudy pixels in the loss)
    # The mask will be added as an additional band called 'cloud_mask' to the downloaded images.
    cloud_collection_name: Optional[str] = None  # e.g. 'COPERNICUS/S2_CLOUD_PROBABILITY'
    cloud_band: Optional[str] = None  # band name for the cloud probability mask, e.g. 'probability'
    cloud_mask_thresh_p: Optional[float] = None  # threshold for binarizing the cloud probability mask, e.g. 0.4

    skip_existing: bool = True  # whether to skip the existing images in the root_dir
    try_reading: bool = True  # whether to try reading the images before skipping the existing ones

    # Whether to consider only the latest processed tile per acquisition day (in case of multiple reprocessed versions)
    latest_tile_only: bool = True

    def __post_init__(self):
        """ Validate the configuration and check if the required settings are set. """
        # Check if we need the cloud mask
        if (self.max_cloud_p < 1.0 or QCMetric.CLOUD_P in self.sort_by) and (
                self.cloud_collection_name is None or
                self.cloud_band is None or
                self.cloud_mask_thresh_p is None
        ):
            raise ValueError(
                "If cloud coverage filtering/sorting is enabled, `cloud_collection_name`, `cloud_band` and "
                "`cloud_mask_thresh_p` must be set."
            )


# ======================================================================================================================
# Base dataset configuration
# ======================================================================================================================
@dataclass
class BaseDatasetCfg:
    # Dataset identifier
    # (should be dependent on the glacier outlines & glaciers covered, kept constant across the years)
    name: str = MISSING

    # Source name, e.g. 'Copernicus Sentinel-2', 'PlanetScope', etc., for plotting purposes
    source_name: str = MISSING

    # A label used to create a subdir where the data is stored
    # We will use 'inv' to denote the data that match the glacier outlines, which can originate from multiple years
    # If the data is from a single year, we will use the year as a label (e.g. '2023')
    year: str = MISSING

    # Path to the glacier outlines (vector-file format)
    outlines_fp: Optional[str] = MISSING

    # Minimum glacier area to consider a glacier in the dataset (in km^2)
    min_glacier_area: float = MISSING

    # CRS and GSD which will be used for the rasterization of the glacier outlines and the raw data
    crs: str = "UTM"  # we use local UTM projection by default; use "EPSG:XXXXX" for a specific projection
    gsd: float = MISSING

    # Settings for the raw data, which can be either downloaded automatically from GEE or assumed to be already
    # downloaded and stored locally; see RawImagesCfg and GEERawImagesCfg for details.
    raw_data: RawImagesCfg = MISSING

    @dataclass
    class OGGMDataCfg:
        """
        Settings for the OGGM data, which will be downloaded automatically and later added to the training data after
        reprojection and clipping.
        """

        # Root data directory
        base_dir: str = "${working_dir}/raw_data/oggm"

        # Where to get the DEMs from (see https://tutorials.oggm.org/stable/notebooks/tutorials/dem_sources.html)
        dem_source = 'COPDEM30'

        # GSD of the data (most of the variables in OGGM have maximum 30m so we set it to this by default)
        gsd: int = 30

    oggm_data: OGGMDataCfg = field(default_factory=OGGMDataCfg)

    # Patch radius in pixels
    patch_radius: int = MISSING

    # The step (in pixels) between two consecutive patches for training, validation and inference.
    @dataclass
    class Strides:
        train: int = MISSING
        val: Optional[int] = None  # we will set it automatically in __post_init__
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
        # Pointer to the raw data settings
        qc_metrics: float = "${..raw_data.buffer_qc_metrics}"

        # Buffer for the final processed glacier cube (should be large enough to allow patch sampling)
        cube: Optional[float] = None  # if None, will be set automatically in __post_init__

        # Buffer from within which patch centres are sampled
        patch_sampling: float = MISSING

        # Buffer size used at inference time within which the positive pixels will be counted
        # Useful if we expect that outlines are not perfect, and we want to include some buffer around the glacier.
        # Or if we expect the glaciers to grow w.r.t. the date of the outlines.
        infer: float = 0

        # Buffer interval on which False Positive pixels will be counted
        # (if 'auto', we will automatically derive it s.t. the two classes - glacier/non-glacier - are balanced)
        fp: Tuple[Any, Any] = (0, 'auto')

    buffers: Buffers = field(default_factory=Buffers)

    # Whether to check if the raw data covers 100% of the glacier outlines (incl. the buffer) when building the glacier
    # cubes. If False, we will keep the desired buffer but fill the missing pixels with NaNs.
    check_data_coverage: bool = True

    # A list of bands to be merged into a QC mask (e.g. ('~CLOUDLESS_MASK', 'SOME_OTHER_MASK')), where ~ means negation
    # (we will use the union of the specified bands).
    bands_nok_mask: Optional[Tuple[str, ...]] = None

    # A dictionary {name -> path} with the paths to various directories that contain additional raster data to be
    # added to the glacier cubes. The data is expected to be in a raster format (e.g. tif) and it will be
    # automatically matched (i.e. all the scenes intersecting the glacier cube will be merged and reprojected).
    extra_rasters: Optional[Dict[str, str]] = None

    # A dictionary {name -> path} with the paths to various vector files to be rasterized as binary masks and added
    # to the glacier cubes. The data is expected to be in a vector format (e.g. shp).
    # We will automatically reproject and clip the data to the glacier cube.
    extra_vectors: Optional[Dict[str, str]] = None

    # A list of features to compute using xDEM (e.g. slope, aspect, etc.)
    # See https://xdem.readthedocs.io/en/stable/gen_modules/xdem.DEM.get_terrain_attribute.html
    xdem_features: Optional[Tuple[str, ...]] = None

    # ==================================================================================================================
    # Paths to the processed data, dynamically generated based on the relevant parameters.
    # ==================================================================================================================

    # Root directory for all the data which will be processed
    base_dir: str = "${working_dir}/datasets/${.name}"

    # Processed inventory outlines with all the additional derived geometries
    geoms_fp: str = "${.base_dir}/inventory/geoms.gpkg"

    # A csv file with the glacier IDs and their corresponding folds under each cross-validation iteration.
    split_csv: str = "${.base_dir}/cv_splits/map_cv_iter_${cv_iter}.csv"

    # A csv with the normalization stats of the current cross-validation iteration
    norm_stats_csv: str = "${.base_dir}/norm_stats/stats_cv_iter_${cv_iter}.csv"

    # Path to the root of the processed glacier cubes (netcdf) after processing; this will be separated by year
    cubes_dir: str = "${.base_dir}/glacier_cubes/year=${.year}"

    # Directory for the patches (in case they are exported to disk)
    patches_dir: Optional[str] = "${.base_dir}/patches/year=${.year}/r_${.patch_radius}_s_${.strides.train}"

    def __post_init__(self):
        """
        Generate intermediate paths, set dynamic default values in case they are not provided and a few other checks.
        """

        # If we don't export patches, we will probably use a small stride for training;
        # => set the default val & infer depending on the training stride
        if self.export_patches:
            if self.strides.val is None:
                self.strides.val = self.strides.train  # we sample all the patches with the same stride and then split
            if self.strides.infer is None:
                self.strides.infer = self.strides.train // 2  # smaller because we sample them on the fly
            self.sample_patches_each_epoch = False
        else:
            if self.strides.val is None:
                self.strides.val = self.strides.train * 2  # if we sample on the fly, we have too many train patches
            if self.strides.infer is None:
                self.strides.infer = self.strides.train

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

        # If automatic selection is not enabled, we disable the coverage and cloud filtering
        if not self.raw_data.automated_selection:
            self.raw_data.min_coverage = 0.0
            self.raw_data.max_cloud_p = 1.0
            self.raw_data.num_days_to_keep = 1


@dataclass
class S2GEERawImagesCfg(GEERawImagesCfg):
    """Default configuration for Sentinel-2 raw images downloaded from GEE."""
    automated_selection: bool = False
    dates_csv: Optional[str] = None

    bands: Tuple[str, ...] = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12')
    bands_rename: Dict[str, str] = field(default_factory=lambda: {
        'B2': 'B',
        'B3': 'G',
        'B4': 'R',
        'B8': 'NIR',
        'B11': 'SWIR',
    })

    # GEE collections (the project name will have to be set by the user in the YAML config or CLI)
    img_collection_name: str = 'COPERNICUS/S2_HARMONIZED'
    cloud_collection_name: str = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
    cloud_band: str = 'cs'
    cloud_mask_thresh_p: float = 0.6


# ======================================================================================================================
# Sentinel-2 dataset config
# ======================================================================================================================
@dataclass
class S2DatasetCfg(BaseDatasetCfg):
    """Base configuration for any Sentinel-2 dataset.

    The regional specifics will be set via YAML or CLI arguments, e.g. the outlines_fp, year, etc.

    By default, the images are automatically downloaded using GEE, either for the same dates as the inventory outlines
    (year='inv') or for a specific year (e.g. '2023'). For the latter, the dates will be automatically selected based
    on e.g. the cloud coverage and NDSI (depending on the `raw_data` settings).
    """

    source_name: str = "Copernicus Sentinel-2"

    # Sentinel-2 typical GSD
    gsd: float = 10.0

    # Patch / sampling defaults
    export_patches: bool = False
    sample_patches_each_epoch: bool = False
    patch_radius: int = 128

    # Strides and buffers can be overridden in YAML
    strides: BaseDatasetCfg.Strides = field(default_factory=lambda: BaseDatasetCfg.Strides(train=32))
    buffers: BaseDatasetCfg.Buffers = field(
        default_factory=lambda: BaseDatasetCfg.Buffers(
            patch_sampling=50,
            infer=20,
            fp=(0, 'auto')
        )
    )

    # Will be set in the YAML config or CLI arguments to LocalRawImagesCfg or S2GEERawImagesCfg
    raw_data: RawImagesCfg = MISSING

    # The cloud mask will be downloaded automatically so we include it by default in the final QC mask.
    bands_nok_mask: Tuple[str, ...] = field(default_factory=lambda: ('cloud_mask',))
