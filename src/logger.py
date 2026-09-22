import logging
from src.config import BASE_DIR

LOG_PATH= BASE_DIR/ "logs"/ "pipeline.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler=logging.StreamHandler()
file_handler=logging.FileHandler(LOG_PATH)

formatter=logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)