import os
import logging.config


config_dict = {
    "version": 1,
    "handlers": {
        "train_file_handler": {
            "class": "logging.FileHandler",
            "formatter": "my_formatter",
            "filename": "train.log",
        },
        "test_file_handler": {
            "class": "logging.FileHandler",
            "formatter": "my_formatter",
            "filename": "test.log",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "my_formatter",
        },
    },
    "loggers": {
        "train_logger": {
            "handlers": ["train_file_handler", "console_handler"],
            "level": "INFO",
        },
        "test_logger": {
            "handlers": ["test_file_handler", "console_handler"],
            "level": "INFO",
        },
    },
    "formatters": {
        "my_formatter": {
            "format": "%(asctime)s : [%(name)s] : [%(levelname)s] - %(message)s"
        }
    }
}


def init_loggers():
    if os.path.exists("./train.log"):
        os.remove("./train.log")

    if os.path.exists("./test.log"):
        os.remove("./test.log")

    logging.config.dictConfig(config_dict)
