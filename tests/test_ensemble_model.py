import pytest
import numpy as np
from src.chronoscast.ensemble_model import ForecastEnsemble

def test_ensemble_valid_models():
    ensemble = ForecastEnsemble()
    predictions = {
        'ModelA': np.array([10.0, 20.0, 30.0]),
        'ModelB': np.array([20.0, 30.0, 40.0])
    }
    
    final, contributors = ensemble.combine(predictions)
    
    # Should average to [15, 25, 35]
    np.testing.assert_array_equal(final, np.array([15.0, 25.0, 35.0]))
    assert len(contributors) == 2
    assert 'ModelA' in contributors
    assert 'ModelB' in contributors

def test_ensemble_discards_failed_models():
    ensemble = ForecastEnsemble()
    predictions = {
        'ModelA': np.array([10.0, 20.0, 30.0]),
        'ModelB': np.array([np.nan, np.nan, np.nan]),  # Should be dropped
        'ModelC': np.array([20.0, 30.0, 40.0]),
        'ModelD': np.array([]) # Should be dropped
    }
    
    final, contributors = ensemble.combine(predictions)
    
    # Should average only A and C to [15, 25, 35]
    np.testing.assert_array_equal(final, np.array([15.0, 25.0, 35.0]))
    assert len(contributors) == 2
    assert 'ModelA' in contributors
    assert 'ModelC' in contributors
    assert 'ModelB' not in contributors

def test_ensemble_all_failed():
    ensemble = ForecastEnsemble()
    predictions = {
        'ModelA': np.array([np.nan, np.nan]),
        'ModelB': np.array([np.inf, 20.0])
    }
    
    final, contributors = ensemble.combine(predictions)
    
    # Since all failed, it should return array of zeros length 2 (inferred from arrays)
    np.testing.assert_array_equal(final, np.array([0.0, 0.0]))
    assert len(contributors) == 0

def test_ensemble_length_mismatch():
    ensemble = ForecastEnsemble()
    predictions = {
        'ModelA': np.array([10.0, 20.0]),
        'ModelB': np.array([10.0, 20.0, 30.0]) # Dropped because A established length as 2
    }
    
    final, contributors = ensemble.combine(predictions)
    
    np.testing.assert_array_equal(final, np.array([10.0, 20.0]))
    assert len(contributors) == 1
    assert 'ModelB' not in contributors
