import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.chronoscast.data_ingestion import DataIngestor
from src.chronoscast.config import ProjectConfig

def test_load_source_success():
    class DummyConfig(ProjectConfig):
        RAW_DATA_SOURCES = {'test_source': 'fake_path.csv'}
    
    ingestor = DataIngestor(config=DummyConfig)
    
    with patch('src.chronoscast.data_ingestion.pd.read_csv') as mock_read:
        mock_read.return_value = pd.DataFrame({'a': [1, 2]})
        df = ingestor.load_source('test_source')
        assert len(df) == 2
        assert 'a' in df.columns
        mock_read.assert_called_once_with('fake_path.csv')

def test_load_source_missing_key():
    ingestor = DataIngestor()
    with pytest.raises(KeyError):
        ingestor.load_source('nonexistent_key')

def test_load_source_file_not_found():
    class DummyConfig(ProjectConfig):
        RAW_DATA_SOURCES = {'test_source': 'does_not_exist.csv'}
    
    ingestor = DataIngestor(config=DummyConfig)
    with pytest.raises(Exception):
        ingestor.load_source('test_source')
