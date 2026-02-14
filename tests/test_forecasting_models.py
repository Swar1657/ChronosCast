import pytest
import pandas as pd
import numpy as np

from src.chronoscast.config import ProjectConfig
from src.chronoscast.forecasting_models import (
    BaselineModel, ARIMAModel, SARIMAModel, ETSModel, 
    ARIMAXModel, ProphetModel, LSTMModel
)

@pytest.fixture
def dummy_train_data():
    # 30 months of data with a slight upward trend
    dates = pd.date_range('2022-01-01', periods=30, freq='MS')
    qty = np.linspace(10, 50, 30) + np.sin(np.arange(30)) * 5
    
    return pd.DataFrame({
        'month_start': dates,
        'billqty': qty,
        'temperature': np.random.uniform(20, 30, 30),
        'cpi': np.linspace(100, 110, 30)
    })

@pytest.fixture
def dummy_future_data():
    dates = pd.date_range('2024-07-01', periods=4, freq='MS')
    return pd.DataFrame({
        'month_start': dates,
        'temperature': np.random.uniform(20, 30, 4),
        'cpi': np.linspace(110, 112, 4)
    })

class DummyConfig(ProjectConfig):
    MIN_OBSERVATIONS = 24

def test_baseline_model(dummy_train_data):
    model = BaselineModel(config=DummyConfig, seasonal_period=12)
    assert model.fit(dummy_train_data) == True
    preds = model.predict(4)
    assert len(preds) == 4
    # It should copy values from exactly 12 periods ago (indices -12 to -9)
    np.testing.assert_array_almost_equal(preds, dummy_train_data['billqty'].iloc[-12:-8].values)

def test_arima_model(dummy_train_data):
    # Keep search space tiny for test speed
    model = ARIMAModel(config=DummyConfig, p_max=1, d_max=0, q_max=0)
    assert model.fit(dummy_train_data) == True
    preds = model.predict(4)
    assert len(preds) == 4

def test_sarima_model(dummy_train_data):
    model = SARIMAModel(config=DummyConfig, order=(1,0,0), seasonal_order=(0,0,0,12))
    assert model.fit(dummy_train_data) == True
    preds = model.predict(4)
    assert len(preds) == 4

def test_ets_model(dummy_train_data):
    model = ETSModel(config=DummyConfig, seasonal_period=12)
    assert model.fit(dummy_train_data) == True
    preds = model.predict(4)
    assert len(preds) == 4

def test_arimax_model(dummy_train_data, dummy_future_data):
    model = ARIMAXModel(config=DummyConfig, order=(1,0,0), exog_cols=['temperature', 'cpi'])
    assert model.fit(dummy_train_data) == True
    preds = model.predict(4, df_future=dummy_future_data)
    assert len(preds) == 4

def test_prophet_model(dummy_train_data):
    model = ProphetModel(config=DummyConfig)
    assert model.fit(dummy_train_data) == True
    preds = model.predict(4)
    assert len(preds) == 4

def test_lstm_model(dummy_train_data):
    model = LSTMModel(config=DummyConfig, seq_length=6, epochs=5, hidden_size=8)
    assert model.fit(dummy_train_data) == True
    preds = model.predict(4)
    assert len(preds) == 4

def test_insufficient_history():
    short_df = pd.DataFrame({'month_start': [1,2], 'billqty': [10, 20]})
    model = BaselineModel(config=DummyConfig)
    # Should fail validation cleanly
    assert model.fit(short_df) == False
    assert model.failure_reason == "Insufficient history"
    
def test_constant_series():
    const_df = pd.DataFrame({'month_start': range(30), 'billqty': [10]*30})
    model = ARIMAModel(config=DummyConfig)
    assert model.fit(const_df) == False
    assert model.failure_reason == "Constant series"
