from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class PLConfig:
    # Directory under the main working directory for the current experiment
    # Note that we include the common sweeping parameters in the subdirectory but this could be overwritten in the cl
    base_dir: str = (
        "${working_dir}/"
        "experiments/"
        "${dataset.name}/"
        "${model.name}/"
        "cv_iter_${pl.cv_iter}/"
        "seed_${pl.seed}/"
    )

    num_cv_folds: int = 5  # number of cross-validation folds
    cv_iter: int = 1  # current cross-validation iteration
    val_fraction: float = 0.1  # fraction of the training set to use for validation
    seed: int = 1

    @dataclass
    class Data:
        _target_: str = 'dl4gam.pl_modules.data.GlSegDataModule'
        input_settings: Any = "${model.input_settings}"
        split_csv: str = "${dataset.split_csv}"
        patches_on_disk: bool = "${dataset.export_patches}"
        patches_dir: str = "${dataset.patches_dir}"
        num_patches_train: int = "${dataset.num_patches_train}"
        seed: int = "${pl.seed}"
        cubes_dir: str = "${dataset.cubes_dir}"
        patch_radius: int = "${dataset.patch_radius}"
        stride_train: int = "${dataset.strides.train}"
        norm_stats_csv: str = "${dataset.norm_stats_csv}"  # path to the precomputed normalization stats csv file
        standardize_data: bool = False  # whether to standardize the data (mean=0, std=1)
        minmax_scale_data: bool = True  # whether to apply min-max scaling to the data
        scale_each_band: bool = False  # whether to scale each optical band separately or all together
        train_batch_size: int = 16
        val_batch_size: int = 32
        test_batch_size: int = 32
        train_shuffle: bool = True  # whether to (re)shuffle the training data before each epoch
        use_augmentation: bool = False  # only D4 transforms are applied but makes the training slower
        num_workers: int = 16
        preload_data: bool = False  # whether to load ALL! the netcdf files in memory before sampling patches
        pin_memory: bool = False  # see https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

        # These stride parameters will be set in the main config
        # (they are not static but dynamically computed in the dataset config)
        stride_val: Optional[int] = field(init=False, default=None)
        stride_test: Optional[int] = field(init=False, default=None)

    @dataclass
    class Task:
        _target_: str = 'dl4gam.pl_modules.seg_task.GlSegTask'

        @dataclass
        class Loss:
            _target_: str = 'dl4gam.pl_modules.loss.MaskedLoss'
            metric: str = 'focal'

        @dataclass
        class Optimizer:
            _target_: str = 'torch.optim.Adam'
            lr: float = 0.0001

        @dataclass
        class LRScheduler:  # Note that we're not using it by default
            _target_: str = 'torch.optim.lr_scheduler_params.CosineAnnealingLR'
            T_max: int = 50
            eta_min: float = 0.0

        loss: Loss = field(default_factory=Loss)
        optimizer: Optimizer = field(default_factory=Optimizer)
        lr_scheduler: Optional[LRScheduler] = None  # we're not using it by default

        # interpolation method for the predictions, can be 'nn' or 'hypso'; see `dl4gam.utils.postprocessing`
        interp: str = 'nn'

    @dataclass
    class Logger:
        _target_: str = 'pytorch_lightning.loggers.TensorBoardLogger'
        save_dir: str = "${pl.base_dir}"
        name: str = ''  # keep empty so it doesn't create a subfolder
        default_hp_metric: bool = False  # disable the default hyperparameter metric logging
        version: str = "${now:%Y-%m-%d_%H-%M-%S}"

    @dataclass
    class CheckpointCallback:
        _target_: str = 'pytorch_lightning.callbacks.ModelCheckpoint'
        monitor: str = 'w_JaccardIndex_val_epoch_avg_per_g'
        filename: str = 'ckpt-{epoch:02d}-{w_JaccardIndex_val_epoch_avg_per_g:.4f}'
        save_top_k: int = 1
        save_last: bool = False
        mode: str = 'max'
        every_n_epochs: int = 1

    @dataclass
    class Trainer:
        _target_: str = 'pytorch_lightning.Trainer'
        devices: List[int] = field(default_factory=lambda: [0])
        accelerator: str = 'gpu'
        log_every_n_steps: int = 10
        max_epochs: int = 30
        deterministic: bool = True
        limit_train_batches: Optional[int] = None

    data: Data = field(default_factory=Data)
    task: Task = field(default_factory=Task)
    logger: Logger = field(default_factory=Logger)
    checkpoint_callback: CheckpointCallback = field(default_factory=CheckpointCallback)
    trainer: Trainer = field(default_factory=Trainer)
