"""
Weather data processing module for the Maji Ndogo project.
"""

import logging
import re

from week7.data_ingestion import read_from_web_CSV


class WeatherDataProcessor:
    """
    Process weather station messages into structured weather measurements.
    """

    def __init__(self, config_params, logging_level="INFO"):
        """
        Initialize weather data source parameters and logger state.

        Args:
            config_params (dict): Config containing weather CSV path and regex patterns.
            logging_level (str): Logging level (e.g., INFO, DEBUG, NONE).
        """
        self.weather_station_data = config_params["weather_csv_path"]
        self.patterns = config_params["regex_patterns"]
        self.weather_df = None
        self.initialize_logging(logging_level)

    def initialize_logging(self, logging_level):
        """
        Configure and attach a class-scoped logger.

        Args:
            logging_level (str): Requested logger level.
        """
        logger_name = __name__ + ".WeatherDataProcessor"
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False

        if logging_level.upper() == "DEBUG":
            log_level = logging.DEBUG
        elif logging_level.upper() == "INFO":
            log_level = logging.INFO
        elif logging_level.upper() == "NONE":
            self.logger.disabled = True
            return
        else:
            log_level = logging.INFO

        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def weather_station_mapping(self):
        """
        Load weather station message data into self.weather_df.
        """
        self.weather_df = read_from_web_CSV(self.weather_station_data)
        self.logger.info("Successfully loaded weather station data from the web.")

    def extract_measurement(self, message):
        """
        Extract measurement label and numeric value from a weather message.

        Args:
            message (str): Raw weather message.

        Returns:
            tuple: (measurement_name, numeric_value) or (None, None).
        """
        for key, pattern in self.patterns.items():
            match = re.search(pattern, str(message))
            if match:
                self.logger.debug("Measurement extracted: %s", key)
                value = next((x for x in match.groups() if x is not None), None)
                return key, float(value)
        self.logger.debug("No measurement match found.")
        return None, None

    def process_messages(self):
        """
        Parse all messages and append Measurement and Value columns.

        Returns:
            pandas.DataFrame: Processed weather DataFrame.
        """
        if self.weather_df is not None:
            result = self.weather_df["Message"].apply(self.extract_measurement)
            self.weather_df["Measurement"], self.weather_df["Value"] = zip(*result)
            self.logger.info("Messages processed and measurements extracted.")
        else:
            self.logger.warning(
                "weather_df is not initialized, skipping message processing."
            )
        return self.weather_df

    def calculate_means(self):
        """
        Compute mean values by Weather_station_ID and Measurement.

        Returns:
            pandas.DataFrame | None: Unstacked means table if available.
        """
        if self.weather_df is not None:
            means = self.weather_df.groupby(
                by=["Weather_station_ID", "Measurement"]
            )["Value"].mean()
            self.logger.info("Mean values calculated.")
            return means.unstack()
        self.logger.warning("weather_df is not initialized, cannot calculate means.")
        return None

    def process(self):
        """
        Run the full weather-data processing pipeline.
        """
        self.weather_station_mapping()
        self.process_messages()
        self.logger.info("Data processing completed.")
