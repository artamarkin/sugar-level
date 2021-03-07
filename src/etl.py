import logging
import pathlib
from dataclasses import InitVar, dataclass, field
from functools import wraps
from typing import Callable, Dict, List
import gin

import numpy as np
import pandas as pd

local_path = pathlib.Path().resolve()
GIN_PATH = local_path / "src" / "config.gin"


def _decorate_df(func: Callable):
    """
    Function decorator to log processed data shape
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        func_result = func(*args, **kwargs)
        logging.info(
            f" {func.__name__} [DONE], Resulting Data Shape={func_result.shape}"
        )
        return func_result

    return wrapper


def _norm_text(text: str) -> str:
    return text.lower().replace("  ", " ")


def _search_by_category(
    text: str,
    products: str,
    product_groups: Dict[str, List[str]],
) -> int:
    for p in product_groups[products]:
        if p in text:
            return 1
    return 0


@gin.configurable
def _assign_high_level_product_group(
    df: pd.DataFrame, product_groups: Dict[str, List[str]] = gin.REQUIRED
) -> pd.DataFrame:
    """
    Instead of using original product notes we group all products into
    high level categories (see config.gin).
    TODO: use quantity of products as well.
    """
    for k in product_groups.keys():
        df[k] = df["notes_norm"].apply(
            lambda d: _search_by_category(d, k, product_groups)
        )
    return df


@dataclass
class SugarData:
    """
    Data class which preprocess and store measurement of the glucose in blood.
    """

    data_path: InitVar[pathlib.Path]
    raw: pd.DataFrame = field(init=False)
    cleaned: pd.DataFrame = field(init=False)
    glucose: pd.DataFrame = field(init=False)
    glucose_resampled: pd.DataFrame = field(init=False)
    notes: pd.DataFrame = field(init=False)

    def __post_init__(self, data_path: pathlib.Path):
        gin.parse_config_file(GIN_PATH)
        self.raw = self.__read_data(data_path)
        self.cleaned = self.__clean_data()
        self.glucose = self.__glucose()
        self.notes = self.__notes()
        self.glucose_resampled = self.__glucose_resampled()

    @_decorate_df
    def __read_data(self, data_path: pathlib.Path) -> pd.DataFrame:
        return pd.read_csv(data_path, skiprows=1)

    @_decorate_df
    def __clean_data(self) -> pd.DataFrame:
        data = (
            self.raw.rename(columns={"Notes": "notes"})
            .assign(
                # get glucose level from manual and automatic scans
                glucose_level=lambda d: np.where(
                    d["Historic Glucose mmol/L"].isnull(),
                    d["Scan Glucose mmol/L"],
                    d["Historic Glucose mmol/L"],
                ),
                datetime=lambda d: pd.to_datetime(
                    d["Device Timestamp"], format="%d-%m-%Y %H:%M"
                ),
            )
            .loc[
                lambda d: (d["glucose_level"].notnull()) | (d["notes"].notnull()),
                [
                    "datetime",
                    "glucose_level",
                    "notes",
                ],
            ]
            .sort_values(by="datetime")
            .reset_index(drop=True)
        )

        return data

    @_decorate_df
    def __glucose(self) -> pd.DataFrame:
        return self.cleaned.loc[
            lambda d: d["glucose_level"].notnull(), ["datetime", "glucose_level"]
        ]

    @_decorate_df
    def __notes(self) -> pd.DataFrame:
        return (
            self.cleaned.loc[lambda d: d["notes"].notnull(), ["datetime", "notes"]]
            .assign(notes_norm=lambda d: d["notes"].apply(_norm_text))
            # fix naming. TODO: make dictionary with new names
            .assign(
                notes=lambda d: d["notes"]
                .replace("orange polenta cake", "polenta cake")
                .replace("orange wine", "wine")
            )
            .pipe(_assign_high_level_product_group)
        )

    @_decorate_df
    def __glucose_resampled(self) -> pd.DataFrame:
        return (
            self.glucose.set_index("datetime")
            .resample("1min")
            .mean()
            .interpolate()
            .reset_index()
        )
