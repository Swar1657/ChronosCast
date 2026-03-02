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

def test_ensemble_weighted():
    ensemble = ForecastEnsemble()
    predictions = {
        'ModelA': np.array([10.0, 20.0]),
        'ModelB': np.array([20.0, 40.0]),
        'ModelC': np.array([30.0, 60.0])
    }
    # Weights should be normalized.
    # ModelC gets 0 weight. ModelA and ModelB share the weight (0.2 + 0.8 = 1.0).
    weights = {'ModelA': 0.2, 'ModelB': 0.8, 'ModelC': 0.0, 'ModelMissing': 1.0}
    
    final, contributors = ensemble.combine(predictions, weights=weights)
    
    # Expected: 0.2*10 + 0.8*20 = 2 + 16 = 18
    # 0.2*20 + 0.8*40 = 4 + 32 = 36
    np.testing.assert_array_almost_equal(final, np.array([18.0, 36.0]))
    assert len(contributors) == 3 # All 3 successfully ran, even if C had 0 weight

def test_ensemble_weights_sum_to_zero_fallback():
    ensemble = ForecastEnsemble()
    predictions = {
        'ModelA': np.array([10.0, 20.0]),
        'ModelB': np.array([20.0, 40.0])
    }
    # If all surviving models have 0 weight, it should fallback to simple mean.
    weights = {'ModelA': 0.0, 'ModelB': 0.0, 'ModelX': 1.0}
    
    final, contributors = ensemble.combine(predictions, weights=weights)
    
    # Expected: mean(10, 20) = 15, mean(20, 40) = 30
    np.testing.assert_array_almost_equal(final, np.array([15.0, 30.0]))

def test_ensemble_single_model_weight():
    ensemble = ForecastEnsemble()
    predictions = {
        'ModelA': np.array([10.0, 20.0]),
        'ModelB': np.array([np.nan, np.nan]) # Fails
    }
    weights = {'ModelA': 0.5, 'ModelB': 0.5}
    
    final, contributors = ensemble.combine(predictions, weights=weights)
    
    # ModelA is the only survivor, so its weight of 0.5 gets normalized to 1.0
    np.testing.assert_array_almost_equal(final, np.array([10.0, 20.0]))
    assert len(contributors) == 1
    assert 'ModelA' in contributors
