import pytest
import pandas as pd
from src.chronoscast.eda import EDADiagnostics
from src.chronoscast.config import ProjectConfig

@pytest.fixture
def mock_master_data():
    return pd.DataFrame({
        'month_start': [pd.Timestamp('2022-01-01'), pd.Timestamp('2022-02-01'), pd.Timestamp('2022-01-01')],
        'sales_category': ['A', 'A', 'B'],
        'site': ['S1', 'S1', 'S1'],
        'billqty': [10, 20, 30]
    })

def test_analyze_sparsity(mock_master_data):
    eda = EDADiagnostics(mock_master_data, ProjectConfig)
    
    # We pass a simple grain
    grains = [['sales_category', 'site']]
    results = eda.analyze_sparsity(grains)
    
    assert len(results) == 1
    assert results['Grain'].iloc[0] == 'sales_category x site'
    assert results['Series_Count'].iloc[0] == 2 # (A, S1) and (B, S1)
    assert results['Observations'].iloc[0] == 3
    # Total months = 2 (Jan, Feb). Total possible = 2 series * 2 months = 4. 3/4 = 75%
    assert results['Fill_Rate_Pct'].iloc[0] == 75.0

def test_get_overall_demand_trend(mock_master_data):
    eda = EDADiagnostics(mock_master_data, ProjectConfig)
    trend = eda.get_overall_demand_trend()
    
    assert len(trend) == 2
    assert trend['billqty'].iloc[0] == 40 # Jan: 10 + 30
    assert trend['billqty'].iloc[1] == 20 # Feb: 20
