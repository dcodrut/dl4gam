from dataclasses import dataclass, field

from hydra.conf import HydraConf


@dataclass
class DL4GAMJobLogging:
    formatters: dict = field(default_factory=lambda: {
        'simple': {
            'format': "[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)"
        }
    })

    root: dict = field(default_factory=lambda: {
        'handlers': ['console', 'file'],
        'level': 'INFO',  # this is also the default
        'propagate': False,
    })

    disable_existing_loggers: bool = False

    # to avoid logging from external modules, manually set them to WARN or ERROR
    external_modules: list = field(default_factory=lambda: ['pyogrio'])
    external_modules_log_level: str = 'WARN'

@dataclass
class DL4GAMHydraConfig(HydraConf):
    job_logging: DL4GAMJobLogging = field(default_factory=DL4GAMJobLogging)

