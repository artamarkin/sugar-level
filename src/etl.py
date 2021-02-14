import logging
import pathlib
from dataclasses import InitVar, dataclass
from functools import wraps
from typing import Callable

import pandas as pd


def decorate_df(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.log(f"[{func.__name__}]: Input Data Shape={args[0].shape}")
        logger.log(level, f"[DONE], Resultind Data Shape={func(*args, **kwargs).shape}")


@dataclass
class SugarData:

    data_path: InitVar[pathlib.Path]
    data: pd.DataFrame = field(init=False)
    def __post_init__(self, data_path: pathlib.Path):
        self.data_path = data_path
        self.data = __read_raw_data(self)
        self.data_cleaned = __clean_data(self)
