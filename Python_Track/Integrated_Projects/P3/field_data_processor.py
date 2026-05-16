"""
Field data processing module for the Maji Ndogo project.
"""

import logging

from week7.data_ingestion import create_db_engine, query_data, read_from_web_CSV


class FieldDataProcessor:
    """
    Ingest, clean, and enrich field-level agricultural data.
    """

    def __init__(self, config_params, logging_level="INFO"):
        """
        Initialize processor configuration and runtime state.

        Args:
            config_params (dict): Pipeline configuration parameters.
            logging_level (str): Logging level for this processor.
        """
        self.db_path = config_params["db_path"]
        self.sql_query = config_params["sql_query"]
        self.columns_to_rename = config_params["columns_to_rename"]
        self.values_to_rename = config_params["values_to_rename"]
        self.weather_map_data = config_params["weather_map_data"]

        self.df = None
        self.engine = None
        self.initialize_logging(logging_level)

    def initialize_logging(self, logging_level):
        """
        Configure and attach a class-scoped logger.

        Args:
            logging_level (str): Requested logger level.
        """
        logger_name = __name__ + ".FieldDataProcessor"
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(getattr(logging, logging_level.upper(), logging.INFO))

    def ingest_sql_data(self):
        """
        Load raw field data from SQL into self.df.

        Returns:
            pandas.DataFrame: Raw field dataset.
        """
        self.df = query_data(create_db_engine(self.db_path), self.sql_query)
        self.logger.info("Successfully loaded data.")
        return self.df

    def rename_columns(self):
        """
        Swap mislabeled crop and annual yield columns safely.

        The source dataset has these two columns interchanged. This method
        performs a three-step rename via a temporary column to avoid name
        collision and preserve values.
        """
        self.df.rename(columns={"Annual_yield": "Crop_type_Temp"}, inplace=True)
        self.df.rename(columns={"Crop_type": "Annual_yield"}, inplace=True)
        self.df.rename(columns={"Crop_type_Temp": "Crop_type"}, inplace=True)
        self.logger.info("Swapped columns: 'Annual_yield' and 'Crop_type'.")

    def apply_corrections(self, column_name="Crop_type", abs_column="Elevation"):
        """
        Apply value corrections to key columns.

        Args:
            column_name (str): Crop type column name.
            abs_column (str): Column to force non-negative values.

        Notes:
            Crop labels are stripped before mapping so variants with trailing
            spaces are normalized (for example "wheat " -> "wheat").
        """
        self.df[abs_column] = self.df[abs_column].abs()
        self.df[column_name] = self.df[column_name].apply(
            lambda crop: self.values_to_rename.get(str(crop).strip(), str(crop).strip())
        )
        self.logger.info("Applied corrections to crop labels and elevation.")

    def weather_station_mapping(self):
        """
        Load Field_ID-to-weather station mapping table.

        Returns:
            pandas.DataFrame: Weather mapping DataFrame.
        """
        return read_from_web_CSV(self.weather_map_data)

    def process(self):
        """
        Run the full field-data processing pipeline.

        Steps:
            1. Ingest SQL data.
            2. Swap mislabeled columns.
            3. Correct crop labels and elevation values.
            4. Merge weather-station mapping by Field_ID.
            5. Remove helper/index columns not needed downstream.
        """
        self.ingest_sql_data()
        self.rename_columns()
        self.apply_corrections()
        weather_station_df = self.weather_station_mapping()
        self.df = self.df.merge(weather_station_df, on="Field_ID", how="left")
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)
        self.logger.info("Field data processing completed.")
