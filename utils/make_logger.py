import logging.config
import os
import sys

import dotenv

dotenv.load_dotenv(".env")
PATH_TO_LOGGER_CONFIG_FILE = os.getenv("PATH_TO_LOGGER_CONFIG_FILE")

try:
    logging.config.fileConfig(PATH_TO_LOGGER_CONFIG_FILE)
except KeyError as err:
    print(f"Error. Configuration file `{PATH_TO_LOGGER_CONFIG_FILE}` not found ...")
    sys.exit(-1)
else:
    logger = logging.getLogger("sLogger")
