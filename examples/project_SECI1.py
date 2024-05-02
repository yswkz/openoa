from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import openoa.utils.unit_conversion as un
import openoa.utils.met_data_processing as met
from openoa.plant import PlantData
from openoa.utils import filters, timeseries
from openoa.logging import logging


logger = logging.getLogger()

def extract_data(path="data/SECI1_data"):
    """
    Extract zip file containing project engie data.
    """
    path = Path(path).resolve()
    if not path.exists():
        logger.info("Extracting compressed data files")
        with ZipFile(path.with_suffix(".zip")) as zipfile:
            zipfile.extractall(path)


def clean_scada(seci1_file: str | Path) -> pd.DataFrame:
    """Reads in and cleans up the SCADA data

    Args:
        seci1_file (:obj: `str` | `Path`): The file object corresponding to the SCADA data.
    Returns:
        pd.DataFrame: The cleaned up SCADA data that is ready for loading into a `PlantData` object.
    """
    scada_freq = "10min"

    logger.info("Loading SECI1 data")
    seci1_df = pd.read_csv(seci1_file)
    logger.info("SECI data loaded")

    # We know that the timestamps are in local time, so we want to convert them to UTC
    logger.info("Timestamp conversion to datetime and UTC")
    seci1_df["Timestamp_ist"] = pd.to_datetime(seci1_df["Timestamp_ist"], utc=True).dt.tz_localize(None)

    # There are duplicated timestamps, so let's ensure we drop the duplicates for each turbine
    seci1_df = seci1_df.drop_duplicates(subset=["Timestamp_ist","loc_id"], keep="first")

    # Remove extreme values from the temperature field
    logger.info("Removing out of range of temperature readings")
    seci1_df = seci1_df[(seci1_df["Ot_avg"] >= -15.0) & (seci1_df["Ot_avg"] <= 45.0)]

    # Filter out the unresponsive sensors
    # Due to data discretization, there appear to be a large number of repeating values
    logger.info("Flagging unresponsive sensors")
    turbine_id_list = seci1_df.loc_id.unique()
    sensor_cols = ["Ba_avg", "P_avg", "Ws_avg", "Va_avg", "Ot_avg", "Ya_avg", "Wa_avg"]
    for t_id in turbine_id_list:
        ix_turbine = seci1_df["loc_id"] == t_id

        # Cancel out readings where the wind vane direction repeats more than 3 times in a row
        ix_flag = filters.unresponsive_flag(seci1_df.loc[ix_turbine], 3, col=["Va_avg"])
        seci1_df.loc[ix_flag.loc[ix_flag["Va_avg"]].index, sensor_cols] = np.nan

        # Cancel out the temperature readings where the value repeats more than 20 times in a row
        ix_flag = filters.unresponsive_flag(seci1_df.loc[ix_turbine], 20, col=["Ot_avg"])
        seci1_df.loc[ix_flag.loc[ix_flag["Ot_avg"]].index, "Ot_avg"] = np.nan

    logger.info("Converting pitch to the range [-180, 180]")
    seci1_df.loc[:, "Ba_avg"] = seci1_df["Ba_avg"] % 360
    ix_gt_180 = seci1_df["Ba_avg"] > 180.0
    seci1_df.loc[ix_gt_180, "Ba_avg"] = seci1_df.loc[ix_gt_180, "Ba_avg"] - 360.0

    logger.info("Calculating energy production")
    seci1_df["energy_kwh"] = un.convert_power_to_energy(seci1_df.P_avg * 1000, scada_freq) / 1000

    return seci1_df