from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, cast, Optional, Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING

from dl4gam.configs.datasets import (
    LocalRawImagesCfg,
    GEERawImagesCfg,
    S2GEERawImagesCfg,
    BaseDatasetCfg,
    S2DatasetCfg,
    PSDatasetCfg
)
from dl4gam.configs.models import SMPModelCfg, InputCfg
from dl4gam.configs.training import RunCfg


@dataclass
class DL4GAMCfg:
    """ Full application config for the DL4GAM workflow.

    First, we define here the composition scheme (i.e. the dataset, the model & training settings)
    and some global settings.

    Second, we do some post-processing in the `__post_init__` method to set some parameters that depend on the dataset
    and model configs.

    Last, we use  to define the parameters for each workflow step to avoid passing the whole config and thus
    make clear which parameters are used for each step. This also allows us to easily refactor and also to make sure
    we use the parameters set in the post_init methods.
    """

    # The following are pointers to the dataset and model configs that will be populated by Hydra using the yaml files.
    dataset: BaseDatasetCfg = MISSING
    model: SMPModelCfg = MISSING
    input: InputCfg = MISSING

    # For the training setup we have a single config that is used for all models
    # (we will overwrite only the seed and the cross-validation iteration)
    run: RunCfg = field(default_factory=RunCfg)

    # Root working directory (everything is relative to this); will also be the root of the Hydra config
    working_dir: str = MISSING

    # Global settings
    num_procs: int = MISSING  # for the steps that run in parallel
    pbar: bool = True  # whether to show progress bars when processing time-consuming steps
    seed: int = 1  # only for model training, the data split is controlled only by the `cv_iter` and `num_cv_folds`

    # Cross-validation settings
    num_cv_folds: int = 5  # number of cross-validation folds
    cv_iter: int = 1  # current cross-validation iteration
    val_fraction: float = 0.1  # fraction of the training set to use for validation

    # Settings used at inference time (see `stage_inference` method)
    # Fold (e.g. 'test' or 'val')
    fold_inference: str = 'test'

    # Timestamp of the model run to use for inference; by default, we use the latest one but can be set by CLI
    run_timestamp: Optional[str] = MISSING  # we will then look for subdirs: run=run_timestamp

    # Set this > 0 to discard small segments in the polygonization step
    # (after extracting the segments from the binarized predictions)
    min_segment_area_km2: float = 0.0  # km^2

    # Custom logging settings, if we want to control (e.g. mute) existing loggers (i.e. coming from external libraries)
    external_modules_log_level: Dict[str, str] = field(default_factory=dict)

    internal: Optional[Dict[str, Any]] = field(default_factory=dict)  # for any internal use in config.yaml

    def __post_init__(self):
        # In case we sample patches on the fly & different sample every epoch, we implement this by altering the
        # `limit_train_batches` parameter.
        # With shuffle=True and a large enough number of patches (which will be checked), the epochs will be different.
        if self.dataset.sample_patches_each_epoch:
            self.run.trainer.limit_train_batches = self.run.data.num_patches_train // self.run.data.train_batch_size

        # Set the strides for validation and test sets in the data config
        self.run.data.stride_val = self.dataset.strides.val
        self.run.data.stride_test = self.dataset.strides.infer

    # Workflow stages configs: all the parameters are derived from the existing settings
    def stage_process_inventory(self):
        from dl4gam.workflow.process_inventory import main
        main(
            fp_in=self.dataset.outlines_fp,
            fp_out=self.dataset.geoms_fp,
            min_glacier_area=self.dataset.min_glacier_area,
            buffers=self.dataset.buffers,
            crs=self.dataset.crs,
            gsd=self.dataset.gsd,
            dates_csv=self.dataset.raw_data.dates_csv,
        )

    def stage_download_raw_data(self):
        from dl4gam.workflow.download_gee_data import main

        # For this stage, we expect the raw_data to be from Google Earth Engine
        if not isinstance(self.dataset.raw_data, GEERawImagesCfg):
            raise ValueError(
                f"The raw_data configuration must be of type GEERawImagesCfg for this stage "
                f"but got {type(self.dataset.raw_data)}. Check the dataset configuration."
            )

        self.dataset.raw_data = cast(GEERawImagesCfg, self.dataset.raw_data)
        main(
            base_dir=self.dataset.raw_data.base_dir,
            geoms_fp=self.dataset.geoms_fp,
            buffer_roi=self.dataset.raw_data.buffer_roi,
            dates_csv=self.dataset.raw_data.dates_csv,
            year=self.dataset.year,
            gsd=self.dataset.gsd,
            automated_selection=self.dataset.raw_data.automated_selection,
            download_time_window=self.dataset.raw_data.download_time_window,
            gee_project_name=self.dataset.raw_data.gee_project_name,
            img_collection_name=self.dataset.raw_data.img_collection_name,
            cloud_collection_name=self.dataset.raw_data.cloud_collection_name,
            cloud_band=self.dataset.raw_data.cloud_band,
            cloud_mask_thresh_p=self.dataset.raw_data.cloud_mask_thresh_p,
            max_cloud_p=self.dataset.raw_data.max_cloud_p,
            min_coverage=self.dataset.raw_data.min_coverage,
            sort_by=self.dataset.raw_data.sort_by,
            score_weights=self.dataset.raw_data.score_weights,
            num_days_to_keep=self.dataset.raw_data.num_days_to_keep,
            bands_to_keep=self.dataset.raw_data.bands,
            bands_name_map=self.dataset.raw_data.bands_rename,
            latest_tile_only=self.dataset.raw_data.latest_tile_only,
            skip_existing=self.dataset.raw_data.skip_existing,
            try_reading=self.dataset.raw_data.try_reading,
        )

    def stage_rank_images(self):
        from dl4gam.workflow.rank_images import main
        main(
            raw_data_base_dir=self.dataset.raw_data.base_dir,
            year=self.dataset.year,
            geoms_fp=self.dataset.geoms_fp,
            min_coverage=self.dataset.raw_data.min_coverage,
            max_cloud_p=self.dataset.raw_data.max_cloud_p,
            sort_by=self.dataset.raw_data.sort_by,
            score_weights=self.dataset.raw_data.score_weights,
            buffer=self.dataset.buffers.cube,
            bands_name_map=self.dataset.raw_data.bands_rename,
            bands_nok_mask=self.dataset.bands_nok_mask,
        )

    def stage_download_oggm_data(self):
        from dl4gam.workflow.download_oggm_data import main
        main(
            geoms_fp=self.dataset.geoms_fp,
            working_dir=self.dataset.oggm_data.base_dir,
            gsd=self.dataset.oggm_data.gsd,
            border_m=int(self.dataset.buffers.cube),
            dem_source=self.dataset.oggm_data.dem_source,
            num_procs=self.num_procs,
        )

    def stage_build_dataset(self):
        from dl4gam.workflow.build_dataset import main
        main(
            geoms_fp=self.dataset.geoms_fp,
            base_dir=self.dataset.cubes_dir,
            raw_data_base_dir=self.dataset.raw_data.base_dir,
            year=self.dataset.year,
            extra_vectors=self.dataset.extra_vectors,
            automated_selection=self.dataset.raw_data.automated_selection,
            ensure_all_glaciers_have_images=self.dataset.raw_data.ensure_all_glaciers_have_images,
            buffer=self.dataset.buffers.cube,
            check_data_coverage=self.dataset.check_data_coverage,
            bands_name_map=self.dataset.raw_data.bands_rename,
            bands_nok_mask=tuple(self.dataset.bands_nok_mask) if self.dataset.bands_nok_mask else None,
            extra_rasters=self.dataset.extra_rasters,
            xdem_features=tuple(self.dataset.xdem_features) if self.dataset.xdem_features else None,
            overwrite=True,
        )

    def stage_patchify_dataset(self):
        from dl4gam.utils.sampling_utils import patchify_data
        patchify_data(
            cubes_dir=self.dataset.cubes_dir,
            patch_radius=self.dataset.patch_radius,
            patches_dir=self.dataset.patches_dir,
            stride=self.dataset.strides.train,
        )

    def stage_split_data(self):
        from dl4gam.utils.sampling_utils import data_cv_split
        data_cv_split(
            geoms_fp=self.dataset.geoms_fp,
            num_folds=self.num_cv_folds,
            cv_iter=self.cv_iter,
            val_fraction=self.val_fraction,
            fp_out=self.dataset.split_csv,
        )

    def stage_compute_norm_stats(self):
        from dl4gam.workflow.compute_norm_stats import main
        main(
            data_dir=self.dataset.patches_dir if self.dataset.export_patches else self.dataset.cubes_dir,
            split_csv=self.dataset.split_csv,
            fp_out=self.dataset.norm_stats_csv,
        )

    def stage_train(self):
        from dl4gam.workflow.train_model import main
        main(
            seed=self.seed,
            run_cfg=self.run,
            model_cfg=self.model,
        )

    def stage_infer(self):
        from dl4gam.workflow.test_model import main
        main(
            run_cfg=self.run,
            model_cfg=self.model,
            dataset_cfg=self.dataset,
            fold=self.fold_inference,
            checkpoint_dir=Path(self.run.logger.save_dir) / 'checkpoints'
        )

    def stage_polygonize(self):
        from dl4gam.workflow.polygonize import main
        main(
            checkpoint_dir=Path(self.run.logger.save_dir) / 'checkpoints',
            dataset_name=self.dataset.name,
            year=self.dataset.year,
            fold=self.fold_inference,
            min_segment_area_km2=self.min_segment_area_km2,
        )

    def stage_eval(self):
        from dl4gam.workflow.eval import main
        main(
            dataset_cfg=self.dataset,
            fold=self.fold_inference,
            checkpoint_dir=Path(self.run.logger.save_dir) / 'checkpoints',
        )

    def stage_plot_dataset(self):
        from dl4gam.workflow.plot import main
        main(
            dataset_cfg=self.dataset,
        )

    def stage_plot_dataset_with_preds(self):
        from dl4gam.workflow.plot import main
        main(
            dataset_cfg=self.dataset,
            fold=self.fold_inference,
            checkpoint_dir=Path(self.run.logger.save_dir) / 'checkpoints',
            bands_train_input=self.input.bands_input,
        )

    # Which stage of the pipeline to execute currently (to be set by the user)
    stage: str = MISSING


# Register configs with Hydra's ConfigStore
def register_configs():
    cs = ConfigStore.instance()
    cs.store(group='dataset', name='base', node=BaseDatasetCfg)
    cs.store(group='dataset', name='s2_base', node=S2DatasetCfg)
    cs.store(group='dataset', name='ps_base', node=PSDatasetCfg)
    cs.store(group='dataset/raw_data', name='local', node=LocalRawImagesCfg)
    cs.store(group="dataset/raw_data", name='gee_base', node=GEERawImagesCfg)
    cs.store(group="dataset/raw_data", name='gee_s2', node=S2GEERawImagesCfg)
    cs.store(group='model', name='smp_base', node=SMPModelCfg)
    cs.store(group='input', name='input_base', node=InputCfg)
    cs.store(name='dl4gam_config', node=DL4GAMCfg)


# Register resolvers with OmegaConf
def register_resolvers():
    # Get the parent directory of a given path
    OmegaConf.register_new_resolver("parentdir", lambda p: str(Path(p).parent))

    # Strip whitespace from a given path
    OmegaConf.register_new_resolver("fpstrip", lambda p: str(Path(*[x.strip() for x in Path(p).parts])))

    # Get the latest run timestamp from a given directory
    def get_latest_run_timestamp(base_dir: str) -> Optional[str]:
        runs = sorted(Path(base_dir).glob('run=*jobnum=*'))
        if not runs:
            raise FileNotFoundError(
                f"No runs found in {base_dir}. Please check the directory. "
                f"You can also set the `run_timestamp` parameter manually in the CLI or config file."
            )
        return str(runs[-1].name).replace('run=', '')

    OmegaConf.register_new_resolver("latest_run_timestamp", lambda base_dir: get_latest_run_timestamp(base_dir))
