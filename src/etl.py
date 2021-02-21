import logging
import pathlib
from dataclasses import InitVar, dataclass, field
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd


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


@dataclass
class SugarData:
    """
    Data class which preprocess and store measurement of the glucose in blood.
    """

    data_path: InitVar[pathlib.Path]
    raw: pd.DataFrame = field(init=False)
    cleaned: pd.DataFrame = field(init=False)
    glucose: pd.DataFrame = field(init=False)
    notes: pd.DataFrame = field(init=False)

    def __post_init__(self, data_path: pathlib.Path):
        self.raw = self.__read_data(data_path)
        self.cleaned = self.__clean_data()
        self.glucose = self.__glucose()
        self.notes = self.__notes()

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
        return self.cleaned.loc[
            lambda d: d["notes"].notnull(), ["datetime", "notes"]
        ].assign(notes_norm=lambda d: d["notes"].apply(_norm_text))
