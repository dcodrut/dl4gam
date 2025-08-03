from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from dl4gam.configs.datasets import (
    LocalRawImagesConfig,
    GEERawImagesConfig,
    S2GEERawImagesConfig,
    BaseDatasetConfig,
    S2DatasetConfig,
)
from dl4gam.configs.models import UnetModelConfig
from dl4gam.configs.training import PLConfig


@dataclass
class DL4GAMConfig:
    """ Full application config for the DL4GAM workflow.

    First, we define here the composition scheme (i.e. the dataset, the model & training settings)
    and some global settings.

    Second, we do some post-processing in the `__post_init__` method to set some parameters that depend on the dataset
    and model configs.

    Last, we use @property to define the parameters for each workflow step to avoid passing the whole config and thus
    make clear which parameters are used for each step. This also allows us to easily refactor and also to make sure
    we use the parameters set in the post_init methods.
    """

    # The following are pointers to the dataset and model configs that will be populated by Hydra using the yaml files.
    dataset: Any = MISSING
    model: Any = MISSING

    # For the training setup we have a single config that is used for all models
    pl: PLConfig = field(default_factory=PLConfig)

    # Root working directory (everything is relative to this)
    working_dir: str = './data/external/dl4gam'

    # Global settings
    min_glacier_area: float = 0.1  # km^2
    num_procs: int = 24  # for the steps that run in parallel
    pbar: bool = True  # whether to show progress bars when processing time-consuming steps

    def __post_init__(self):
        # In case we sample patches on the fly & different sample every epoch, we implement this by altering the
        # `limit_train_batches` parameter.
        # With shuffle=True and a large enough number of patches (which will be checked), the epochs will be different.
        if self.dataset.sample_patches_each_epoch:
            self.pl.trainer.limit_train_batches = self.pl.data.num_patches_train // self.pl.data.train_batch_size

        # Set the strides for validation and test sets in the data config
        self.pl.data.stride_val = self.dataset.strides.val
        self.pl.data.stride_test = self.dataset.strides.infer

    # Workflow step configs: all the parameters are derived from the existing settings
    @property
    def step_process_inventory(self) -> dict:
        return {
            '_target_': 'dl4gam.workflow.process_inventory.main',
            'fp_in': self.dataset.outlines_fp,
            'fp_out': self.dataset.geoms_fp,
            'min_glacier_area': self.min_glacier_area,
            'buffers': self.dataset.buffers,
            'crs': self.dataset.crs,
            'gsd': self.dataset.gsd,
            'dates_csv': self.dataset.raw_data.dates_csv,
        }

    @property
    def step_raw_data_download(self) -> dict:
        # We take a few parameters from the dataset config plus (most of) the raw data settings
        return {
            '_target_': 'dl4gam.workflow.download_gee_data.main',
            'base_dir': self.dataset.raw_data.base_dir,
            'geoms_fp': self.dataset.geoms_fp,
            'buffer_roi': self.dataset.raw_data.buffer_roi,
            'dates_csv': self.dataset.raw_data.dates_csv,
            'year': self.dataset.year,
            'gsd': self.dataset.gsd,
            'automated_selection': self.dataset.raw_data.automated_selection,
            'download_time_window': self.dataset.raw_data.download_time_window,
            'gee_project_name': self.dataset.raw_data.gee_project_name,
            'img_collection_name': self.dataset.raw_data.img_collection_name,
            'cloud_collection_name': self.dataset.raw_data.cloud_collection_name,
            'cloud_band': self.dataset.raw_data.cloud_band,
            'cloud_mask_thresh_p': self.dataset.raw_data.cloud_mask_thresh_p,
            'max_cloud_p': self.dataset.raw_data.max_cloud_p,
            'min_coverage': self.dataset.raw_data.min_coverage,
            'sort_by': self.dataset.raw_data.sort_by,
            'score_weights': self.dataset.raw_data.score_weights,
            'num_days_to_keep': self.dataset.raw_data.num_days_to_keep,
            'bands_to_keep': self.dataset.raw_data.bands,
            'bands_name_map': self.dataset.raw_data.bands_rename,
            'latest_tile_only': self.dataset.raw_data.latest_tile_only,
            'skip_existing': self.dataset.raw_data.skip_existing,
            'try_reading': self.dataset.raw_data.try_reading,
        }

    @property
    def step_oggm_data_download(self) -> dict:
        return {
            '_target_': 'dl4gam.workflow.download_oggm_data.main',
            'geoms_fp': self.dataset.geoms_fp,
            'working_dir': self.dataset.oggm_data.base_dir,
            'gsd': self.dataset.oggm_data.gsd,
            'border_m': self.dataset.buffers.cube,
            'dem_source': self.dataset.oggm_data.dem_source,
            'num_procs_download': self.num_procs,
        }

    @property
    def step_build_dataset(self) -> dict:
        return {
            '_target_': 'dl4gam.workflow.build_dataset.main',
            'geoms_fp': self.dataset.geoms_fp,
            'buffer': self.dataset.buffers.cube,
            'check_data_coverage': self.dataset.check_data_coverage,
            'base_dir': self.dataset.cubes_dir,
            'raw_data_base_dir': self.dataset.raw_data.base_dir,
            'year': self.dataset.year,
            'automated_selection': self.dataset.raw_data.automated_selection,
            'bands_name_map': self.dataset.raw_data.bands_rename,
            'bands_nok_mask': self.dataset.bands_nok_mask,
            'extra_vectors': self.dataset.extra_vectors,
            'extra_rasters': self.dataset.extra_rasters,
            'xdem_features': self.dataset.xdem_features,
            'overwrite': True,  # whether to overwrite existing netCDF files
        }

    @property
    def step_patchify_dataset(self) -> dict:
        return {
            '_target_': 'dl4gam.utils.patchify_data',
            'cubes_dir': self.dataset.cubes_dir,
            'patch_radius': self.dataset.patch_radius,
            'patches_dir': self.dataset.patches_dir,
            'stride': self.dataset.strides.train,
        }

    @property
    def step_data_split(self) -> dict:
        return {
            '_target_': 'dl4gam.utils.data_cv_split',
            'geoms_fp': self.dataset.geoms_fp,
            'num_folds': self.pl.num_cv_folds,
            'cv_iter': self.pl.cv_iter,
            'val_fraction': self.pl.val_fraction,
            'fp_out': self.dataset.split_csv,
        }

    @property
    def step_compute_norm_stats(self) -> dict:
        return {
            '_target_': 'dl4gam.workflow.compute_norm_stats.main',
            # Use the patches directory if exporting patches, otherwise use the cubes directory
            'data_dir': self.dataset.patches_dir if self.dataset.export_patches else self.dataset.cubes_dir,
            'split_csv': self.dataset.split_csv,
            'fp_out': self.dataset.norm_stats_csv,
        }

    @property
    def step_train_model(self) -> dict:
        return {
            '_target_': 'dl4gam.workflow.train_model.main',
            'seed': self.pl.seed,
            'logger': self.pl.logger,
            'data': self.pl.data,
            'model': self.model,
            'task': self.pl.task,
            'checkpoint_callback': self.pl.checkpoint_callback,
            'trainer': self.pl.trainer,
        }

    # Which step to execute
    current_step: str = MISSING


# Register configs with Hydra's ConfigStore
def register_configs():
    cs = ConfigStore.instance()
    cs.store(group='dataset', name='base', node=BaseDatasetConfig)
    cs.store(group='dataset', name='s2_base', node=S2DatasetConfig)
    cs.store(group='dataset/raw_data', name='local', node=LocalRawImagesConfig)
    cs.store(group="dataset/raw_data", name='gee_base', node=GEERawImagesConfig)
    cs.store(group="dataset/raw_data", name='gee_s2', node=S2GEERawImagesConfig)
    cs.store(group='model', name='unet', node=UnetModelConfig)
    cs.store(name='dl4gam_config', node=DL4GAMConfig)
