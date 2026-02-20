import numpy as np
import logging
from typing import Dict, List, Tuple
from .utils import setup_logger

logger = setup_logger(__name__)

class ForecastEnsemble:
    """
    Implements a simple mean ensemble that validates inputs, discards failed models,
    and retains only successful predictions.
    """
    def __init__(self):
        pass
        
    def combine(self, model_predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Combines predictions from multiple models.
        Args:
            model_predictions: Dict mapping model_name to its numpy array of predictions.
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
            # If all models failed, we must return a safe fallback of 0s. 
            # If expected_length is None, we don't even know how long it should be.
            fallback_len = expected_length if expected_length is not None else 1
            return np.zeros(fallback_len), []
            
        # Simple mean ensemble as specified by PwC architecture
        stacked_preds = np.vstack(valid_predictions)
        final_forecast = np.mean(stacked_preds, axis=0)
        
        return final_forecast, contributing_models
