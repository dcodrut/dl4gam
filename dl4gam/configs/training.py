from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class PLConfig:
    # Directory under the main working directory for the current experiment
    # Note that we include the common sweeping parameters in the subdirectory but this could be overwritten in the cl
    base_dir: str = (
        "${working_dir}/"
        "experiments/"
        "${dataset.name}/"
        "${model.name}/"
        "cv_iter=${pl.cv_iter}/"
        "seed=${pl.task.seed}/"
    )

    num_cv_folds: int = 5  # number of cross-validation folds
    cv_iter: int = 1  # current cross-validation iteration
    valid_fraction: float = 0.1  # fraction of the training set to use for validation

    @dataclass
    class Data:
        minmax_scale_data: bool = False
        standardize_data: bool = True
        scale_each_band: bool = True
        train_batch_size: int = 16
        val_batch_size: int = 32
        test_batch_size: int = 32
        num_workers: int = 16
        use_augmentation: bool = False

    @dataclass
    class Task:
        seed: int = 1
        loss: str = 'focal'
        optimizer: Dict[str, Any] = field(default_factory=lambda: {'lr': 0.0001})
        lr_schedule: Optional[Any] = None

    @dataclass
    class Logger:
        save_dir: str = "${pl.base_dir}"
        name: str = ''  # keep empty so it doesn't create a subfolder  # TODO: check this later

    @dataclass
    class Trainer:
        devices: List[int] = field(default_factory=lambda: [0])
        accelerator: str = 'gpu'
        log_every_n_steps: int = 10
        max_epochs: int = 30
        deterministic: bool = True

    data: Data = field(default_factory=Data)
    task: Task = field(default_factory=Task)
    logger: Logger = field(default_factory=Logger)
    trainer: Trainer = field(default_factory=Trainer)
