import logging
import colorlog

def get_logger(name: str) -> logging.Logger:
    log_format = "%(log_color)s%(filename)s:%(lineno)d%(reset)s  %(message)s"
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        log_format,
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))

    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger