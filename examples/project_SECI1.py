#%%
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
#%%


logger = logging.getLogger()

def extract_data(path="data/SECI1_data"):
    """
    Extract zip file containing SECI1 data.
    """
    path = Path(path).resolve()
    if not path.exists():
        logger.info("Extracting compressed data files")
        with ZipFile(path.with_suffix(".zip")) as zipfile:
            zipfile.extractall(path)
#%%

def clean_scada(seci1_file: str | Path) -> pd.DataFrame:
    """Reads in and cleans up the SCADA data

    Args:
        seci1_file (:obj: `str` | `Path`): The file object corresponding to the seci1 data.
    Returns:
        pd.DataFrame: The cleaned up seci1 data that is ready for loading into a `PlantData` object.
    """
    seci1_freq = "10min"

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
    seci1_df = seci1_df[(seci1_df["TempOutdoor_Avg"] >= -15.0) & (seci1_df["TempOutdoor_Avg"] <= 45.0)]

    # Filter out the unresponsive sensors
    # Due to data discretization, there appear to be a large number of repeating values
    logger.info("Flagging unresponsive sensors")
    turbine_id_list = seci1_df.loc_id.unique()
    sensor_cols = ["PitchAngle_Avg", "ActivePowerkW_Avg", "WindSpeedms_Avg", "WindDirection_Avg", "TempOutdoor_Avg", "NacellePos_Avg", "absolute_wind_direction_avg"]
    for t_id in turbine_id_list:
        ix_turbine = seci1_df["loc_id"] == t_id

        # Cancel out readings where the wind vane direction repeats more than 3 times in a row
        ix_flag = filters.unresponsive_flag(seci1_df.loc[ix_turbine], 3, col=["WindDirection_Avg"])
        seci1_df.loc[ix_flag.loc[ix_flag["WindDirection_Avg"]].index, sensor_cols] = np.nan

        # Cancel out the temperature readings where the value repeats more than 20 times in a row
        ix_flag = filters.unresponsive_flag(seci1_df.loc[ix_turbine], 20, col=["TempOutdoor_Avg"])
        seci1_df.loc[ix_flag.loc[ix_flag["TempOutdoor_Avg"]].index, "TempOutdoor_Avg"] = np.nan

    logger.info("Converting pitch to the range [-180, 180]")
    seci1_df.loc[:, "PitchAngle_Avg"] = seci1_df["PitchAngle_Avg"] % 360
    ix_gt_180 = seci1_df["PitchAngle_Avg"] > 180.0
    seci1_df.loc[ix_gt_180, "PitchAngle_Avg"] = seci1_df.loc[ix_gt_180, "PitchAngle_Avg"] - 360.0

    logger.info("Calculating energy production")
    seci1_df["energy_kwh"] = un.convert_power_to_energy(seci1_df.ActivePowerkW_Avg * 1000, seci1_freq) / 1000

    return seci1_df
#%%

def load_cleansed_data(path: str | Path, return_value="plantdata") -> PlantData:
    """Loads the already created data in `path`/cleansed, if previously parsed.

    Args:
        path (str | Path, optional):The file path to the SECI1 data. Defaults to
            "data/SECI1_data".
        return_value (str, optional): "plantdata" will return a fully constructed PlantData object.
            "dataframes" will return a list of dataframes instead. Defaults to "plantdata".

    Returns:
        PlantData | tuple[pandas.DataFrame, ...]
    """
    logger.info("Reading in the previously cleansed data")

    path = path / "cleansed"
    scada_df = pd.read_csv(path / "scada.csv")
    meter_df = pd.read_csv(path / "meter.csv")
    curtail_df = pd.read_csv(path / "curtail.csv")
    asset_df = pd.read_csv(path / "asset.csv")
    reanalysis = dict(
        merra2=pd.read_csv(path / "reanalysis_merra2.csv"),
    )

    # Return the appropriate data format
    if return_value == "dataframes":
        return scada_df, meter_df, curtail_df, asset_df, reanalysis
    elif return_value == "plantdata":
        # Build and return PlantData
        engie_plantdata = PlantData(
            analysis_type="MonteCarloAEP",  # Choosing a random type that doesn't fail validation
            metadata=path / "metadata.yml",
            scada=scada_df,
            meter=meter_df,
            curtail=curtail_df,
            asset=asset_df,
            reanalysis=reanalysis,
        )
        return engie_plantdata
    else:
        raise ValueError("`return_value` must be one of 'plantdata' or 'dataframes'.")
#%%    


def prepare(
    path: str | Path = "data/SECI1_data/SECI1_data", return_value="plantdata", use_cleansed: bool = False
):
    """
    Do all loading and preparation of the data for this plant.
    args:
    - path (str): Path to SECI1 data folder. If it doesn't exist, we will try to extract a zip file of the same name.
    - scada_df (pandas.DataFrame): Override the scada dataframe with one provided by the user.
    - return_value (str): "plantdata" will return a fully constructed PlantData object. "dataframes" will return a list of dataframes instead.
    - use_cleansed (bool): Use previously prepared data if the the "cleansed" folder exists above the main `path`. Defaults to False.
    """

    if type(path) == str:
        path = Path(path).resolve()
    # Load the pre-cleaned data, if available
    if use_cleansed and (path.parent / "cleansed").is_dir():
        return load_cleansed_data(path=path.parent, return_value=return_value)

    # Extract data if necessary
    extract_data(path)

    ###################
    # Plant Metadata - not used
    ###################

    # lat_lon = (48.452, 5.588)
    # plant_capacity = 8.2  # MW
    # num_turbines = 4
    # turbine_capacity = 2.05  # MW

    ###################
    # SECI1 DATA #
    ###################
    scada_df = clean_scada(path / "cleaned_seci1-data-jan2023.csv")

    ##############
    # METER DATA #
    ##############
    logger.info("Reading in the meter data")
    meter_curtail_df = pd.read_csv(path / "cleaned_seci1-plantdata-3turbines.csv")
    meter_df = meter_curtail_df.copy()

    # Create datetime field
    meter_df["time"] = pd.to_datetime(meter_df.timestamp).dt.tz_localize(None)

    # Drop the fields we don't need
    meter_df.drop(["timestamp", "availability_kWh", "curtailment_kWh"], axis=1, inplace=True)

    #####################################
    # Availability and Curtailment Data #
    #####################################
    logger.info("Reading in the curtailment data")
    curtail_df = meter_curtail_df.copy()  # Make another copy for modifying the curtailment data

    # Create datetime field with a UTC base
    curtail_df["time"] = pd.to_datetime(curtail_df.timestamp).dt.tz_localize(None)

    # Drop the fields we don't need
    curtail_df.drop(["timestamp"], axis=1, inplace=True)

    ###################
    # REANALYSIS DATA #
    ###################
    logger.info("Reading in the reanalysis data and calculating the extra fields")

    # MERRA2
    reanalysis_merra2_df = pd.read_csv(path / "merra2_seci1_new.csv")

    # Create datetime field with a UTC base
    reanalysis_merra2_df["datetime"] = pd.to_datetime(
        reanalysis_merra2_df["timestamp"], utc=True
    ).dt.tz_localize(None)

    # calculate wind direction from u, v
    reanalysis_merra2_df["winddirection_deg"] = met.compute_wind_direction(
        reanalysis_merra2_df["U50M"],
        reanalysis_merra2_df["V50M"],
    )

    # Drop the fields we don't need
    #reanalysis_merra2_df.drop(["Unnamed: 0"], axis=1, inplace=True)

    ##############
    # ASSET DATA #
    ##############

    logger.info("Reading in the asset data")
    asset_df = pd.read_csv(path / "Seci1_Asset_Table.csv")

    # Assign type to turbine for all assets
    asset_df["type"] = "turbine"

    # Return the appropriate data format
    if return_value == "dataframes":
        return (
            scada_df,
            meter_df,
            curtail_df,
            asset_df,
            dict(merra2=reanalysis_merra2_df),
        )
    elif return_value == "plantdata":
        # Build and return PlantData
        engie_plantdata = PlantData(
            analysis_type="MonteCarloAEP",  # Choosing a random type that doesn't fail validation
            metadata=path.parent.parent / "SECI1_plant_meta.yml",
            scada=scada_df,
            meter=meter_df,
            curtail=curtail_df,
            asset=asset_df,
            reanalysis={"merra2": reanalysis_merra2_df},
        )
        return engie_plantdata
    else:
        raise ValueError("`return_value` must be one of 'plantdata' or 'dataframes'.")


if __name__ == "__main__":
    prepare()

# %%
