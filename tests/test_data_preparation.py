import pytest
import pandas as pd
from src.chronoscast.data_preparation import DataPreparer
from src.chronoscast.config import ProjectConfig

@pytest.fixture
def sample_raw_data():
    sales = pd.DataFrame({
        'billingdate': [20220101, 20220101, 20220101], # One duplicate
        'articleno': [1, 1, 1],
        'articledescription': ['A', 'A', 'A'],
        'profitcenter': ['P1', 'P1', 'P1'],
        'brand': ['B1', 'B1', 'B1'],
        'product': ['Cat1', 'Cat1', 'Cat1'], # sales_category
        'site': ['S1', 'S1', 'S1'],
        'billqty': [10, 10, 10],
        'baseprice': [100.0, 100.0, 100.0]
    })
    
    product = pd.DataFrame({
        'articleno': [1],
        'category': ['Cat_A'],
        'product': ['BrandLine_A'] # brand_line
    })
    
    profit_center = pd.DataFrame({
        'profitcenter': ['P1'],
        'city': ['City_A'],
        'countrykey': ['IN'],
        'companyname': ['Comp']
    })
    
    showroom = pd.DataFrame({
        'refid': ['S1'],
        'showroom': ['Show_S1'],
        'plantdescription': ['Plant']
    })
    
    weather = pd.DataFrame({
        'month': [20220101],
        'city': ['City_A'],
        'temperature': [25.0],
        'humidity': [60]
    })
    
    economic = pd.DataFrame({
        'year': [2022],
        'country': ['India'],
        'cpi': [105.0],
        'gdp_growth': [5.0]
    })
    
    return {
        'sales': sales,
        'product': product,
        'profit_center': profit_center,
        'showroom': showroom,
        'weather': weather,
        'economic': economic
    }

def test_create_master_dataset(sample_raw_data):
    preparer = DataPreparer(raw_data=sample_raw_data, config=ProjectConfig)
    master = preparer.create_master_dataset()
    
    # 1 duplicate dropped, then aggregated (since the remaining 2 are identical except we just dropped the duplicate row entirely, wait, actually all 3 rows are identical in the sample, so drop_duplicates drops 2 rows leaving 1. Then aggregation just takes that 1 row).
    assert len(master) == 1
    
    # Check if target is correct (10 from the 1 deduplicated row)
    assert master['billqty'].iloc[0] == 10
    assert master['baseprice'].iloc[0] == 100.0
    
    # Check if dates were parsed
    assert master['month_start'].iloc[0] == pd.Timestamp('2022-01-01')
    assert master['year'].iloc[0] == 2022
    
    # Check semantic column renaming
    assert 'sales_category' in master.columns
    assert 'brand_line' in master.columns
    assert master['sales_category'].iloc[0] == 'Cat1'
    assert master['brand_line'].iloc[0] == 'BrandLine_A'
    
    # Check dimensional joins
    assert master['city'].iloc[0] == 'City_A'
    assert master['showroom'].iloc[0] == 'Show_S1'
    
    # Check exogenous joins
    assert master['temperature'].iloc[0] == 25.0
    assert master['cpi'].iloc[0] == 105.0
