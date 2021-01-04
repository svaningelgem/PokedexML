import logging
from pathlib import Path
from typing import Union

from cv2 import cv2

root = Path(__file__).parent
DATASET_DIR = root / 'submodules/PogoAssets/pokemon_icons'
MODEL_OUTPUT = root / 'trained_model'
LABELBIN_OUTPUT = root / 'labelbin.dat'

__first_image = str(next(DATASET_DIR.glob('*.png')))
IMAGE_DIMENSIONS = cv2.imread(__first_image, cv2.IMREAD_UNCHANGED).shape


def get_logger(
        name: str,
        level: Union[str, int] = logging.INFO,
        fmt: str = '[%(asctime)-15s] [%(levelname)s] %(module)s > %(message)s'
) -> logging.Logger:
    logging.basicConfig(level=level, format=fmt)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(root / f'{name}.log')
    file_handler.setFormatter(logging.Formatter(fmt=fmt))
    logger.addHandler(file_handler)

    return logger
