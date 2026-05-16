"""
Pytest validation checks for Integrated Project P3 outputs.

Expected input files in the same folder:
- sampled_weather_df.csv
- sampled_field_df.csv
"""

import pandas as pd


WEATHER_CSV = "sampled_weather_df.csv"
FIELD_CSV = "sampled_field_df.csv"


def _load_weather_df():
    return pd.read_csv(WEATHER_CSV)


def _load_field_df():
    return pd.read_csv(FIELD_CSV)


def test_read_weather_DataFrame_shape():
    weather_df = _load_weather_df()
    assert weather_df.shape == (1843, 4)


def test_read_field_DataFrame_shape():
    field_df = _load_field_df()
    assert field_df.shape == (5654, 20)


def test_weather_DataFrame_columns():
    weather_df = _load_weather_df()
    expected = {"Weather_station_ID", "Message", "Measurement", "Value"}
    assert set(weather_df.columns) == expected


def test_field_DataFrame_columns():
    field_df = _load_field_df()
    cols = set(field_df.columns)

    # Required core columns used later in the notebook pipeline.
    # Note: some notebook versions keep `Ave_temps` until just before the
    # hypothesis section, where it is renamed to `Temperature`.
    expected_subset = {
        "Field_ID",
        "Elevation",
        "Rainfall",
        "Pollution_level",
        "Crop_type",
        "Annual_yield",
        "Weather_station",
    }

    assert expected_subset.issubset(cols)
    assert ("Temperature" in cols) or ("Ave_temps" in cols)


def test_field_DataFrame_non_negative_elevation():
    field_df = _load_field_df()
    assert (field_df["Elevation"] >= 0).all()


def test_crop_types_are_valid():
    field_df = _load_field_df()
    # These are the known bad labels corrected in the pipeline.
    invalid_labels = {"cassaval", "wheatn", "teaa"}
    assert not field_df["Crop_type"].isin(invalid_labels).any()


def test_positive_rainfall_values():
    field_df = _load_field_df()
    # Rainfall should not be negative.
    assert (field_df["Rainfall"] >= 0).all()
