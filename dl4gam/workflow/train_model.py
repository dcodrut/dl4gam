import logging
import os

from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelSummary

from dl4gam.configs.config import SMPModelCfg, RunCfg

# https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
if 'SLURM_NTASKS' in os.environ:
    del os.environ['SLURM_NTASKS']
if 'SLURM_JOB_NAME' in os.environ:
    del os.environ['SLURM_JOB_NAME']

log = logging.getLogger(__name__)


def main(run_cfg: RunCfg, model_cfg: SMPModelCfg, seed: int):
    """
    Script to train a model using the provided configuration.
    The best checkpoint will be saved in the output directory specified in the run configuration.

    :param run_cfg: run configuration containing data, task, trainer, logger, and checkpoint callback settings;
        see `dl4gam.configs.training.RunCfg`
    :param model_cfg: model configuration containing the model architecture and parameters;
        see `dl4gam.configs.model.SMPModelCfg`
    :param seed: seed for reproducibility
    :return: None
    """
    # Fix the seed
    log.info(f"Seeding everything to {seed}")
    seed_everything(seed, workers=True)

    # Set up the tensorboard logger
    log.info(f"Setting up the TensorBoard logger: {run_cfg.logger}")
    tb_logger = instantiate(run_cfg.logger)

    # Data
    log.info(f"Instantiating the data module: {run_cfg.data}")
    dm = instantiate(run_cfg.data)

    # Model
    log.info(f"Instantiating the model: {model_cfg}")
    model = instantiate(model_cfg)

    # Task
    log.info(f"Instantiating the task: {run_cfg.task}")
    task = instantiate(run_cfg.task, model=model, _recursive_=False)

    # Callbacks
    log.info(f"Instantiating the model checkpoint callback: {run_cfg.checkpoint_callback}")
    checkpoint_callback = instantiate(run_cfg.checkpoint_callback)
    summary = ModelSummary(max_depth=-1)

    # Trainer
    log.info(f"Instantiating the trainer: {run_cfg.trainer}")
    trainer = instantiate(run_cfg.trainer, logger=tb_logger, callbacks=[checkpoint_callback, summary])

    # Finally, fit the model
    trainer.fit(task, dm)
    log.info(f'Best model {checkpoint_callback.best_model_path} score {checkpoint_callback.best_model_score}')
