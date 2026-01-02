from loguru import logger as log_config
import sys



def log_func():
    log_config.remove()
    log_config.add(sys.stderr,
                   format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
                   level="INFO")
    log_config.add("logs/mirai_aiko_debug.log",
                   level="DEBUG",
                   rotation="10 MB")
    return log_config

logger = log_func()

logger.info("LOGGER STARTING NOW")