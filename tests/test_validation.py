import pytest
import pandas as pd
from src.chronoscast.validation import TimeSeriesValidator
from src.chronoscast.config import ProjectConfig

def test_generate_splits_insufficient_history():
    df = pd.DataFrame({
        'month_start': pd.date_range('2022-01-01', periods=10, freq='MS'),
        'billqty': range(10)
    })
    
    class DummyConfig(ProjectConfig):
        MIN_OBSERVATIONS = 24
        FORECAST_HORIZON_MONTHS = 4
        
    validator = TimeSeriesValidator(config=DummyConfig)
    splits = list(validator.generate_splits(df))
    assert len(splits) == 0

def test_generate_splits_valid_history():
    # 30 months of data
    df = pd.DataFrame({
        'month_start': pd.date_range('2022-01-01', periods=30, freq='MS'),
        'billqty': range(30)
    })
    
    class DummyConfig(ProjectConfig):
        MIN_OBSERVATIONS = 24
        FORECAST_HORIZON_MONTHS = 4
        
    validator = TimeSeriesValidator(config=DummyConfig)
    splits = list(validator.generate_splits(df))
    
    # We should have folds starting at train=24, ending at train=29 (so 6 folds)
    assert len(splits) == 6
    
    # Check first fold
    train_0, val_0 = splits[0]
    assert len(train_0) == 24
    assert len(val_0) == 4
    
    # Check last fold
    train_last, val_last = splits[-1]
    assert len(train_last) == 29
    assert len(val_last) == 1 # Since total is 30, only 1 month left for validation
