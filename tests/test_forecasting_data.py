import pytest
import pandas as pd
from src.chronoscast.forecasting_data import ForecastingDataBuilder
from src.chronoscast.config import ProjectConfig

@pytest.fixture
def mock_master_data():
    return pd.DataFrame({
        'month_start': [pd.Timestamp('2022-01-01'), pd.Timestamp('2022-03-01')],
        'sales_category': ['A', 'A'],
        'site': ['S1', 'S1'],
        'billqty': [10, 30],
        'temperature': [25.0, 28.0],
        'humidity': [60, 65],
        'cpi': [100.0, 102.0],
        'gdp_growth': [5.0, 5.0]
    })

def test_build_forecasting_data(mock_master_data):
    class DummyConfig(ProjectConfig):
        FORECASTING_GRAIN = ['sales_category', 'site']
        MIN_OBSERVATIONS = 2
        
    builder = ForecastingDataBuilder(mock_master_data, ['sales_category', 'site'], DummyConfig)
    df = builder.build()
    
    # Range is Jan to March, so 3 months total
    assert len(df) == 3
    
    # Missing Feb should be zero-filled for target
    feb_row = df[df['month_start'] == pd.Timestamp('2022-02-01')].iloc[0]
    assert feb_row['billqty'] == 0
    
    # Check if exogenous variables were forward filled for Feb
    assert feb_row['temperature'] == 25.0 # Ffilled from Jan
    
    # Total valid months since first non-zero is 3 (Jan, Feb, Mar)
    # Min obs is 2, so it should be sufficient history
    assert df['sufficient_history'].all() == True

def test_outlier_capping():
    # Create mock data with a massive outlier
    # Baseline demand is ~10-20. One month has 99999.
    df = pd.DataFrame({
        'month_start': [pd.Timestamp(f'2022-{m:02d}-01') for m in range(1, 6)],
        'sales_category': ['A'] * 5,
        'site': ['S1'] * 5,
        'billqty': [10, 15, 12, 99999, 14], # Month 4 is outlier
        'temperature': [25.0] * 5,
        'humidity': [60] * 5,
        'cpi': [100.0] * 5,
        'gdp_growth': [5.0] * 5
    })
    
    class DummyConfig(ProjectConfig):
        FORECASTING_GRAIN = ['sales_category', 'site']
        MIN_OBSERVATIONS = 2
        
    builder = ForecastingDataBuilder(df, ['sales_category', 'site'], DummyConfig)
    res_df = builder.build()
    
    # Q1 of [10, 12, 14, 15, 99999] is 12. Q3 is 15. IQR = 3.
    # Upper bound = 15 + 1.5 * 3 = 19.5
    # The 99999 should be capped at 19.5.
    outlier_val = res_df[res_df['month_start'] == pd.Timestamp('2022-04-01')]['billqty'].iloc[0]
    
    assert outlier_val <= 20.0
    assert outlier_val > 15.0
