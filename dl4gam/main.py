import logging
import logging.config

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig, SCMode

from dl4gam.configs.config import register_configs
from dl4gam.utils import parallel_utils

register_configs()


@hydra.main(config_path=None, config_name='config', version_base=None)
def main(cfg_dict: DictConfig):
    # Covert the DictConfig to our internal dataclass structure
    cfg = OmegaConf.to_container(
        cfg_dict,
        resolve=True,
        throw_on_missing=True,
        structured_config_mode=SCMode.INSTANTIATE,
    )

    # Check if the config has all the required parameters
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    # Print all the parameters after all evaluations and substitutions
    log = logging.getLogger(__name__)
    log.info(f"Final config: \n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Get the hydra config and set the logging level for external modules
    hydra_cfg = HydraConfig.get()
    log.info(
        f"Setting logging level for external modules "
        f"({hydra_cfg.job_logging.external_modules}) to {hydra_cfg.job_logging.external_modules_log_level}"
    )
    for _ in hydra_cfg.job_logging.external_modules:
        logging.getLogger(_).setLevel(hydra_cfg.job_logging.external_modules_log_level)

    # Set the number of processes and progress bar for parallel processing
    parallel_utils.set_default_num_procs(cfg.num_procs)
    parallel_utils.set_default_pbar(cfg.pbar)

    try:
        # Get the settings for the current step to execute
        settings_crt_step = getattr(cfg, cfg.current_step)
        log.info(f"Executing step: {cfg.current_step} with settings: {settings_crt_step}")
        hydra.utils.instantiate(settings_crt_step, _recursive_=False)
    except KeyboardInterrupt:
        log.exception("Keyboard interrupt received, stopping the process.")
    except Exception:
        log.exception("An error occurred")


if __name__ == '__main__':
    main()
