import pandas as pd
import numpy as np
from typing import Tuple
from .config import ProjectConfig
from .utils import setup_logger

logger = setup_logger(__name__)

class DecompositionEngine:
    """
    Allocates aggregate parent forecasts (e.g., Category x Site) down to 
    granular child forecasts (e.g., Article x Site) based on recent historical contribution ratios.
    """
    def __init__(self, master_df: pd.DataFrame, config=ProjectConfig):
        self.master = master_df
        self.config = config
        self.target = self.config.TARGET_VARIABLE
        self.window = self.config.DECOMPOSITION_WINDOW_MONTHS
        
        # Parent: ['sales_category', 'site']
        self.parent_grain = self.config.FORECASTING_GRAIN
        # Child: ['articleno', 'site']
        self.child_grain = ['articleno', 'site']

    def calculate_ratios(self) -> pd.DataFrame:
        """
        Calculates the historical contribution ratio of each child to its parent
        over the configured recent history window.
        """
        logger.info(f"Calculating decomposition ratios over a {self.window}-month historical window.")
        
        # Get the most recent month in the dataset to calculate the window
        max_month = self.master['month_start'].max()
        cutoff_month = max_month - pd.DateOffset(months=self.window)
        
        # Filter strictly to the historical window
        recent_data = self.master[self.master['month_start'] >= cutoff_month]
        
        # Aggregate demand for children
        child_demand = recent_data.groupby(self.child_grain, as_index=False)[self.target].sum()
        child_demand.rename(columns={self.target: 'child_volume'}, inplace=True)
        
        # We need the parent mapping for each child.
        # Ensure we don't have duplicate columns (e.g., 'site' is in both parent and child grains)
        mapping_cols = list(dict.fromkeys(self.child_grain + self.parent_grain))
        mapping = self.master[mapping_cols].drop_duplicates()
        
        # Join mapping to child demand
        child_mapped = pd.merge(child_demand, mapping, on=self.child_grain, how='left')
        
        # Calculate total parent demand from the mapped children to ensure identical denominators
        parent_demand = child_mapped.groupby(self.parent_grain, as_index=False)['child_volume'].sum()
        parent_demand.rename(columns={'child_volume': 'parent_volume'}, inplace=True)
        
        # Calculate Ratios
        ratios = pd.merge(child_mapped, parent_demand, on=self.parent_grain, how='left')
        
        # Handle zero division safely (if a parent had 0 sales in the window)
        ratios['ratio'] = np.where(ratios['parent_volume'] > 0, ratios['child_volume'] / ratios['parent_volume'], 0.0)
        
        # Normalize ratios to strictly equal 1.0 per parent to prevent floating point leakage
        ratio_sums = ratios.groupby(self.parent_grain, as_index=False)['ratio'].sum()
        ratio_sums.rename(columns={'ratio': 'ratio_sum'}, inplace=True)
        ratios = pd.merge(ratios, ratio_sums, on=self.parent_grain, how='left')
        
        # Safely normalize
        ratios['ratio'] = np.where(ratios['ratio_sum'] > 0, ratios['ratio'] / ratios['ratio_sum'], 0.0)
        
        # Keep only the essential columns mapping child to parent and ratio
        final_cols = list(dict.fromkeys(self.child_grain + self.parent_grain + ['ratio']))
        self.ratios = ratios[final_cols].copy()
        return self.ratios

    def decompose(self, parent_forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Allocates parent forecast volume to children using calculated ratios.
        Expects parent_forecasts to have columns: parent_grain + ['month_start', 'forecast']
        """
        if not hasattr(self, 'ratios'):
            self.calculate_ratios()
            
        logger.info("Executing top-down decomposition to granular level.")
        
        # Merge parent forecasts with the ratio mapping
        decomposed = pd.merge(parent_forecasts, self.ratios, on=self.parent_grain, how='inner')
        
        # Allocate volume
        decomposed['granular_forecast'] = decomposed['forecast'] * decomposed['ratio']
        
        # Enforce non-negativity natively required by business logic
        decomposed['granular_forecast'] = np.maximum(decomposed['granular_forecast'], 0.0)
        
        return decomposed

    def reconcile(self, parent_forecasts: pd.DataFrame, child_forecasts: pd.DataFrame, tolerance: float = 0.001) -> bool:
        """
        Reconciles the aggregated granular forecasts back to the parent forecasts
        to prove that no volume was lost or hallucinated during decomposition.
        """
        logger.info("Reconciling decomposed forecasts to parent aggregates...")
        
        # Re-aggregate children back to parent level per month
        # Since child_forecasts has the parent_grain columns, we group by parent_grain + month_start
        agg_child = child_forecasts.groupby(self.parent_grain + ['month_start'], as_index=False)['granular_forecast'].sum()
        
        # Join with original parent forecasts
        comparison = pd.merge(parent_forecasts, agg_child, on=self.parent_grain + ['month_start'], how='inner')
        
        comparison['abs_diff'] = np.abs(comparison['forecast'] - comparison['granular_forecast'])
        max_error = comparison['abs_diff'].max()
        
        logger.info(f"Maximum reconciliation absolute error: {max_error:.6f}")
        
        if max_error > tolerance:
            logger.error(f"Reconciliation FAILED. Max error {max_error:.6f} exceeds tolerance {tolerance}.")
            return False
            
        logger.info("Reconciliation PASSED.")
        return True
