import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chronoscast.validation import TimeSeriesValidator
from src.chronoscast.config import ProjectConfig

def test_chronological_split_no_leakage():
    validator = TimeSeriesValidator()
    # Create mock series: 30 months
    dates = pd.date_range(start='2020-01-01', periods=30, freq='MS')
    df = pd.DataFrame({
        'month_start': dates,
        ProjectConfig.TARGET_VARIABLE: np.arange(30)
    })
    
    splits = list(validator.generate_splits(df))
    assert len(splits) > 0
    
    for train_df, val_df in splits:
        # Check no leakage: Max train date must be strictly before min val date
        assert train_df['month_start'].max() < val_df['month_start'].min()
        
def test_validation_length():
    validator = TimeSeriesValidator()
    dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
    df = pd.DataFrame({
        'month_start': dates,
        ProjectConfig.TARGET_VARIABLE: np.arange(36)
    })
    
    splits = list(validator.generate_splits(df))
    # Check that at least some splits have the exact validation horizon
    valid_splits = [s for s in splits if len(s[1]) == ProjectConfig.FORECAST_HORIZON_MONTHS]
    assert len(valid_splits) > 0
    
    train_df, val_df = valid_splits[-1]
    assert len(val_df) == ProjectConfig.FORECAST_HORIZON_MONTHS
