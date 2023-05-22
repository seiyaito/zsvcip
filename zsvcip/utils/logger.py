import logging


def setup_logger(name, level=logging.INFO, output=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    if output is not None:
        fh = logging.FileHandler(output)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        fh.setFormatter(fh_formatter)

        logger.addHandler(fh)

    return logger
