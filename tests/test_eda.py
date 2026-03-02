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

def test_select_forecasting_grain():
    # Mock analysis results to test the selection logic independently
    class MockConfig:
        MIN_OBSERVATIONS = 24
        TARGET_VARIABLE = 'target'
        
    class MockEDA(EDADiagnostics):
        def analyze_sparsity(self, grains):
            return pd.DataFrame([
                {'Grain': 'cat x site', 'Series_Count': 27, 'Fill_Rate_Pct': 98.0, 'Viability_Pct': 100.0},
                {'Grain': 'cat x brand x site', 'Series_Count': 72, 'Fill_Rate_Pct': 93.0, 'Viability_Pct': 100.0},
                {'Grain': 'cat x pc', 'Series_Count': 8, 'Fill_Rate_Pct': 100.0, 'Viability_Pct': 100.0},
                {'Grain': 'cat x brand x pc', 'Series_Count': 8, 'Fill_Rate_Pct': 99.0, 'Viability_Pct': 90.0} # not 100% viable
            ])
            
    mock_df = pd.DataFrame() # dummy
    eda = MockEDA(mock_df, MockConfig)
    
    grains = [
        ['cat', 'site'],
        ['cat', 'brand', 'site'],
        ['cat', 'pc'],
        ['cat', 'brand', 'pc']
    ]
    
    result = eda.select_forecasting_grain(grains)
    
    # Expect cat x brand x site because:
    # 1. cat x brand x pc is dropped (Viability < 100)
    # 2. remaining: cat x brand x site has highest depth (3) vs others (2)
    assert result['selected_grain'] == ['cat', 'brand', 'site']
    assert 'Depth=3' in result['selection_reason']

def test_select_forecasting_grain_tie_breaker():
    class MockConfig:
        MIN_OBSERVATIONS = 24
        TARGET_VARIABLE = 'target'
        
    class MockEDA(EDADiagnostics):
        def analyze_sparsity(self, grains):
            return pd.DataFrame([
                {'Grain': 'cat x pc', 'Series_Count': 8, 'Fill_Rate_Pct': 100.0, 'Viability_Pct': 100.0},
                {'Grain': 'cat x site', 'Series_Count': 27, 'Fill_Rate_Pct': 98.0, 'Viability_Pct': 100.0},
            ])
            
    eda = MockEDA(pd.DataFrame(), MockConfig)
    
    result = eda.select_forecasting_grain([['cat', 'pc'], ['cat', 'site']])
    
    # Both have depth 2. Tie break goes to Series_Count (27 > 8)
    assert result['selected_grain'] == ['cat', 'site']
