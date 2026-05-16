"""
Data ingestion utilities for the Maji Ndogo project.

This module provides helper functions to:
1. Create a database engine for SQL queries.
2. Run SQL queries and return DataFrames.
3. Read CSV data from web URLs.
"""

import logging

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def create_db_engine(db_path):
    """
    Create and validate a SQLAlchemy database engine.

    Args:
        db_path (str): SQLAlchemy database URI.

    Returns:
        sqlalchemy.engine.Engine: A valid SQLAlchemy engine.

    Raises:
        ImportError: If SQLAlchemy is unavailable.
        Exception: If engine creation or connection test fails.
    """
    try:
        engine = create_engine(db_path)

        # Test DB connection
        with engine.connect():
            pass

        logger.info("Database engine created successfully.")
        return engine
    except ImportError as e:
        logger.error("SQLAlchemy is required to use this function. Please install it first.")
        raise e
    except Exception as e:
        logger.error(f"Failed to create database engine. Error: {e}")
        raise e


def query_data(engine, sql_query):
    """
    Execute a SQL query and return results as a DataFrame.

    Args:
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine.
        sql_query (str): SQL query string.

    Returns:
        pandas.DataFrame: Query results.

    Raises:
        ValueError: If the query returns an empty DataFrame.
        Exception: If query execution fails.
    """
    try:
        df = pd.read_sql_query(text(sql_query), engine)

        if df.empty:
            raise ValueError("Query executed successfully but returned an empty DataFrame.")

        logger.info("SQL query executed successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to query data. Error: {e}")
        raise e


def read_from_web_CSV(csv_url):
    """
    Read a CSV file from a web URL.

    Args:
        csv_url (str): URL to CSV file.

    Returns:
        pandas.DataFrame: Loaded CSV data.

    Raises:
        ValueError: If the CSV is empty.
        Exception: If CSV download/parsing fails.
    """
    try:
        df = pd.read_csv(csv_url)

        if df.empty:
            raise ValueError("CSV loaded but DataFrame is empty.")

        logger.info(f"CSV loaded successfully from: {csv_url}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV from URL. Error: {e}")
        raise e


# Backward-compatible alias for modules that import snake_case.
read_from_web_csv = read_from_web_CSV
