"""Contains useful paths within the repository"""
import pathlib

_PATH = pathlib.Path(__file__)
UTIL_PATH = _PATH.parent
SRC_PATH = _PATH.parent.parent
DATA_PATH = SRC_PATH / "data"

TRAIN_DATA_FILENAME = "bs140513_032310.csv"
TRAIN_DATA_PATH = DATA_PATH / TRAIN_DATA_FILENAME

METRICS_FILENAME = "metrics.txt"
METRICS_PATH = DATA_PATH / METRICS_FILENAME
