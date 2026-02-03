import pandas as pd
from typing import Dict
from .config import ProjectConfig
from .utils import setup_logger

logger = setup_logger(__name__)

class DataIngestor:
    """
    Handles the reliable loading of raw data sources into pandas DataFrames.
    """
    
    def __init__(self, config=ProjectConfig):
        self.config = config
        self.data_sources = self.config.RAW_DATA_SOURCES

    def load_source(self, source_key: str) -> pd.DataFrame:
        """
        Loads a single specified data source by its key.
        """
        if source_key not in self.data_sources:
            logger.error(f"Source key '{source_key}' not found in configuration.")
            raise KeyError(f"Source key '{source_key}' not found in configuration.")
            
        file_path = self.data_sources[source_key]
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded '{source_key}' from {file_path}. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load '{source_key}' from {file_path}. Error: {e}")
            raise

    def load_all_data_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Iterates through the configuration and loads all required data sources.
        """
        logger.info("Starting ingestion of all data sources...")
        all_data = {}
        for key in self.data_sources.keys():
            all_data[key] = self.load_source(key)
        logger.info("Finished loading all data sources.")
        return all_data
