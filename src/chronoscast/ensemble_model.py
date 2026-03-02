import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from .utils import setup_logger

logger = setup_logger(__name__)

class ForecastEnsemble:
    """
    Implements a forecast ensemble that validates inputs, discards failed models,
    and combines successful predictions using optional validation-derived weights.
    """
    def __init__(self):
        pass
        
    def combine(
        self,
        model_predictions: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Combines predictions from multiple models.
        Args:
            model_predictions: Dict mapping model_name to its numpy array of predictions.
            weights: Optional dict mapping model_name to a validation-derived weight.
                     If provided, a weighted average is computed among the surviving
                     valid models (weights are renormalized to sum to 1.0).
                     If None, a simple equal-weight mean is used.
        Returns:
            Tuple containing:
            - Final ensembled prediction array.
            - List of model names that successfully contributed to the ensemble.
        """
        valid_predictions = []
        contributing_models = []
        
        expected_length = None
        
        for name, preds in model_predictions.items():
            # Verify basic structure
            if preds is None or len(preds) == 0:
                logger.debug(f"Ensemble dropped '{name}': Empty or None predictions.")
                continue
                
            if expected_length is None:
                expected_length = len(preds)
            elif len(preds) != expected_length:
                logger.debug(f"Ensemble dropped '{name}': Length mismatch ({len(preds)} vs {expected_length}).")
                continue
                
            # Verify validity (no NaNs or Infs)
            if np.isnan(preds).any() or np.isinf(preds).any():
                logger.debug(f"Ensemble dropped '{name}': Contains NaN or Inf values.")
                continue
                
            valid_predictions.append(preds)
            contributing_models.append(name)
            
        if not valid_predictions:
            logger.warning("Ensemble failed: No valid models available to combine. Falling back to zeros.")
            fallback_len = expected_length if expected_length is not None else 1
            return np.zeros(fallback_len), []

        stacked_preds = np.vstack(valid_predictions)

        # Determine whether to use weighted or equal-weight averaging
        if weights is not None:
            # Collect weights for the surviving contributing models only
            raw_weights = np.array([weights.get(name, 0.0) for name in contributing_models])
            weight_sum = raw_weights.sum()

            if weight_sum > 0:
                normalized_weights = raw_weights / weight_sum
                final_forecast = np.average(stacked_preds, axis=0, weights=normalized_weights)
            else:
                # All surviving models have zero or missing weight — fall back to equal average
                logger.warning("All surviving model weights are zero. Falling back to equal-weight average.")
                final_forecast = np.mean(stacked_preds, axis=0)
        else:
            # No weights provided — simple mean ensemble
            final_forecast = np.mean(stacked_preds, axis=0)
        
        return final_forecast, contributing_models
