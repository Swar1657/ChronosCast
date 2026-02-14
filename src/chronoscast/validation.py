import pandas as pd
from typing import Iterator, Tuple
from .config import ProjectConfig

class TimeSeriesValidator:
    """
    Implements expanding-window / walk-forward validation.
    Ensures zero temporal leakage by strictly separating training and validation horizons chronologically.
    """
    def __init__(self, config=ProjectConfig):
        self.config = config
        self.val_horizon = self.config.FORECAST_HORIZON_MONTHS
        self.min_history = self.config.MIN_OBSERVATIONS
        
    def generate_splits(self, df: pd.DataFrame) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Yields (train_df, val_df) for each walk-forward fold.
        `df` must be a chronologically sorted DataFrame for a single series.
        """
        # Ensure chronological ordering
        df = df.sort_values('month_start').reset_index(drop=True)
        
        n_obs = len(df)
        
        # If total observations is less than the required minimum history + one validation step, 
        # we cannot do any validation folds.
        if n_obs < self.min_history + 1:
            return
            
        # We start our first fold such that training size is at least self.min_history
        # and we slide the window forward by the validation horizon (or by 1 month step).
        # A standard walk-forward evaluates on every step possible after min_history.
        
        # Let's slide by 1 month step at a time for maximum validation rigor, 
        # evaluating on the next `val_horizon` months.
        
        current_train_end = self.min_history
        
        while current_train_end < n_obs:
            train_df = df.iloc[:current_train_end].copy()
            
            # The validation window is up to val_horizon months ahead
            val_end = min(current_train_end + self.val_horizon, n_obs)
            val_df = df.iloc[current_train_end:val_end].copy()
            
            yield train_df, val_df
            
            # Slide forward
            # Moving forward by val_horizon is "non-overlapping" folds.
            # Moving forward by 1 is "expanding window" with overlapping validation horizons.
            # To avoid exploding compute, especially for expensive models, we step by the validation horizon
            # or a sensible fixed step. Let's step by 1 month to get a robust average, or if the user
            # wants less folds, we step by val_horizon.
            # For a 2-4 month horizon on ~31 months dataset (where min=24), we only have 7 months to test.
            # Stepping by 1 month gives ~7 folds, which is perfectly reasonable.
            current_train_end += 1
