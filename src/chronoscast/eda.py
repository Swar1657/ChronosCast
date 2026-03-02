import pandas as pd
from typing import List, Dict
from .config import ProjectConfig
from .utils import setup_logger

logger = setup_logger(__name__)

class EDADiagnostics:
    """
    Provides reusable EDA and diagnostic functionality for ChronosCast.
    Analyzes sparsity and demand patterns.
    """
    
    def __init__(self, master_df: pd.DataFrame, config=ProjectConfig):
        self.master = master_df
        self.config = config
        
    def analyze_sparsity(self, candidate_grains: List[List[str]]) -> pd.DataFrame:
        """
        Analyzes the sparsity and viability of different forecasting grains.
        Returns a DataFrame of metrics.
        """
        logger.info("Starting sparsity analysis across candidate grains...")
        
        total_months = self.master['month_start'].nunique()
        results = []
        
        for grain in candidate_grains:
            # Group by the candidate grain and month
            grouped = self.master.groupby(grain + ['month_start'])[self.config.TARGET_VARIABLE].sum().reset_index()
            
            series_groups = grouped.groupby(grain)
            num_series = len(series_groups)
            
            total_observations = len(grouped)
            total_possible = num_series * total_months
            fill_rate = total_observations / total_possible if total_possible > 0 else 0
            
            viable_series = sum(series_groups.size() >= self.config.MIN_OBSERVATIONS)
            viability_pct = viable_series / num_series if num_series > 0 else 0
            
            results.append({
                'Grain': ' x '.join(grain),
                'Series_Count': num_series,
                'Observations': total_observations,
                'Fill_Rate_Pct': fill_rate * 100,
                'Viable_Series': viable_series,
                'Viability_Pct': viability_pct * 100
            })
            
        df_results = pd.DataFrame(results)
        logger.info("Sparsity analysis complete.")
        return df_results

    def get_overall_demand_trend(self) -> pd.DataFrame:
        """
        Calculates the aggregate monthly demand trend across the entire business.
        """
        trend = self.master.groupby('month_start')[self.config.TARGET_VARIABLE].sum().reset_index()
        return trend.sort_values('month_start')

    def select_forecasting_grain(self, candidate_grains: List[List[str]]) -> Dict:
        """
        Determines the optimal forecasting grain based on data viability and granularity.
        
        Selection Rule:
        1. A candidate is eligible ONLY if its Viability_Pct == 100% (all series meet min observations).
        2. Rank eligible candidates by Hierarchy Depth (len(Grain)) descending.
        3. Break depth ties by Series_Count descending (most granular possible).
        4. Break final ties by Fill_Rate_Pct descending.
        """
        df_results = self.analyze_sparsity(candidate_grains)
        
        # 1. Filter to strict viability
        eligible = df_results[df_results['Viability_Pct'] == 100.0].copy()
        if eligible.empty:
            raise ValueError("No candidate grain is 100% viable. Cannot securely determine forecasting grain.")
            
        # 2, 3, 4. Rank candidates
        # To rank by hierarchy depth, we calculate length of the grain
        eligible['Depth'] = eligible['Grain'].apply(lambda x: len(x.split(' x ')))
        
        ranked = eligible.sort_values(
            by=['Depth', 'Series_Count', 'Fill_Rate_Pct'],
            ascending=[False, False, False]
        )
        
        best_row = ranked.iloc[0]
        selected_grain_str = best_row['Grain']
        selected_grain = selected_grain_str.split(' x ')
        
        selection_reason = (
            f"Selected {selected_grain_str} (Depth={best_row['Depth']}, Series={best_row['Series_Count']}, "
            f"Fill={best_row['Fill_Rate_Pct']:.1f}%) because it is the most granular level that is 100% viable."
        )
        
        logger.info(selection_reason)
        
        return {
            'selected_grain': selected_grain,
            'candidate_results': df_results,
            'selection_reason': selection_reason
        }
