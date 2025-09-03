import logging
import logging.config

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig, SCMode

from dl4gam.configs.config import register_configs, register_resolvers, DL4GAMCfg
from dl4gam.utils import parallel_utils

register_configs()
register_resolvers()


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg_dict: DictConfig):
    # First check if the provided stage is valid
    valid_stages = {
        attr.removeprefix('stage_') for attr in dir(DL4GAMCfg)
        if attr.startswith('stage_') and callable(getattr(DL4GAMCfg, attr))
    }
    if cfg_dict.stage not in valid_stages:
        raise RuntimeError(f"Invalid stage='{cfg_dict.stage}'; expected one of {sorted(valid_stages)}")

    # Resolve and show the hydra directory to trigger the search for the run_timestamp if the stage requires it
    # (as for certain stages, e.g. 'infer', the output directory assumes a checkpoint directory)
    output_dir = HydraConfig.get().runtime.output_dir
    log = logging.getLogger(__name__)
    log.info(f"Current output directory: {output_dir}")

    # Once hydra is initialized, we can delete the run_timestamp and run_dir_per_stage field from the config
    # (to avoid failed instantiation as a consequence of missing the run_timestamp field for stages that don't use it)
    del cfg_dict.run_timestamp
    del cfg_dict.internal

    # Covert the DictConfig to our internal dataclass structure; check if there are any missing fields
    cfg: DL4GAMCfg = OmegaConf.to_container(
        cfg_dict,
        resolve=True,
        throw_on_missing=True,
        structured_config_mode=SCMode.INSTANTIATE,
    )

    # Set the logging levels for external modules
    for module_name, log_level in cfg.external_modules_log_level.items():
        log.info(f"Setting logging level for module '{module_name}' to '{log_level}'")
        logging.getLogger(module_name).setLevel(log_level)

    # Set the number of processes and progress bar for parallel processing
    parallel_utils.set_default_num_procs(cfg.num_procs)
    parallel_utils.set_default_pbar(cfg.pbar)

    try:
        log.info(f"Stage to be executed: {cfg_dict.stage}")

        # Run the stage function
        stage_func = getattr(cfg, f"stage_{cfg.stage}")
        stage_func()

    except KeyboardInterrupt:
        log.exception("Keyboard interrupt received, stopping the process.")
    except Exception:
        log.exception("An error occurred")


if __name__ == '__main__':
    main()
