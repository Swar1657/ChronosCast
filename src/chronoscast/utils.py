import pandas as pd
import logging

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a simple console logger for the given module name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes all DataFrame column names by converting to lowercase and stripping whitespace.
    """
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df
