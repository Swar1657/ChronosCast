import pytest
import pandas as pd
import numpy as np
from src.chronoscast.config import ProjectConfig
from src.chronoscast.decomposition_engine import DecompositionEngine

@pytest.fixture
def dummy_master_df():
    # 2 categories, 1 site, 2 articles per category, over 3 months
    # Cat A, Site S1: Article 1 (75%), Article 2 (25%)
    # Cat B, Site S1: Article 3 (100%), Article 4 (0%)
    
    dates = pd.date_range('2024-01-01', periods=3, freq='MS')
    
    data = []
    for month in dates:
        # Cat A
        data.append({'month_start': month, 'sales_category': 'A', 'site': 'S1', 'articleno': '1', 'billqty': 75.0})
        data.append({'month_start': month, 'sales_category': 'A', 'site': 'S1', 'articleno': '2', 'billqty': 25.0})
        # Cat B
        data.append({'month_start': month, 'sales_category': 'B', 'site': 'S1', 'articleno': '3', 'billqty': 10.0})
        data.append({'month_start': month, 'sales_category': 'B', 'site': 'S1', 'articleno': '4', 'billqty': 0.0})
        
    return pd.DataFrame(data)

def test_decomposition_calculate_ratios(dummy_master_df):
    class DummyConfig(ProjectConfig):
        DECOMPOSITION_WINDOW_MONTHS = 6
        
    engine = DecompositionEngine(dummy_master_df, config=DummyConfig)
    ratios = engine.calculate_ratios()
    
    # Check shape (4 unique children)
    assert len(ratios) == 4
    
    # Check ratios for Cat A
    art1_ratio = ratios[ratios['articleno'] == '1']['ratio'].iloc[0]
    art2_ratio = ratios[ratios['articleno'] == '2']['ratio'].iloc[0]
    
    assert np.isclose(art1_ratio, 0.75)
    assert np.isclose(art2_ratio, 0.25)
    
    # Check ratios for Cat B
    art3_ratio = ratios[ratios['articleno'] == '3']['ratio'].iloc[0]
    art4_ratio = ratios[ratios['articleno'] == '4']['ratio'].iloc[0]
    
    assert np.isclose(art3_ratio, 1.0)
    assert np.isclose(art4_ratio, 0.0)

def test_decompose_and_reconcile(dummy_master_df):
    class DummyConfig(ProjectConfig):
        DECOMPOSITION_WINDOW_MONTHS = 6
        
    engine = DecompositionEngine(dummy_master_df, config=DummyConfig)
    
    # Parent forecasts
    parent_forecasts = pd.DataFrame({
        'sales_category': ['A', 'B'],
        'site': ['S1', 'S1'],
        'month_start': [pd.Timestamp('2024-04-01'), pd.Timestamp('2024-04-01')],
        'forecast': [200.0, 50.0]
    })
    
    # Decompose
    granular = engine.decompose(parent_forecasts)
    
    assert len(granular) == 4
    
    art1_fc = granular[granular['articleno'] == '1']['granular_forecast'].iloc[0]
    art2_fc = granular[granular['articleno'] == '2']['granular_forecast'].iloc[0]
    art3_fc = granular[granular['articleno'] == '3']['granular_forecast'].iloc[0]
    art4_fc = granular[granular['articleno'] == '4']['granular_forecast'].iloc[0]
    
    # 200 * 0.75 = 150
    assert np.isclose(art1_fc, 150.0)
    # 200 * 0.25 = 50
    assert np.isclose(art2_fc, 50.0)
    # 50 * 1.0 = 50
    assert np.isclose(art3_fc, 50.0)
    # 50 * 0.0 = 0
    assert np.isclose(art4_fc, 0.0)
    
    # Verify Reconciliation
    assert engine.reconcile(parent_forecasts, granular) == True
