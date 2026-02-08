import pandas as pd
from .config import ProjectConfig
from .utils import setup_logger

logger = setup_logger(__name__)

class ForecastingDataBuilder:
    """
    Transforms the Monthly Master Dataset into a strict, zero-filled, 
    chronological forecasting dataset at the configured target grain.
    """
    
    def __init__(self, master_df: pd.DataFrame, config=ProjectConfig):
        self.master = master_df.copy()
        self.config = config
        self.grain = self.config.FORECASTING_GRAIN
        self.target = self.config.TARGET_VARIABLE

    def build(self) -> pd.DataFrame:
        """
        Executes the transformation to the forecasting dataset.
        """
        logger.info(f"Building forecasting dataset at grain: {self.grain}")
        
        # 1. Aggregate to the Target Grain
        agg_cols = self.grain + ['month_start']
        
        # We also want to preserve exogenous variables.
        # Since weather and economic indices are at the month level 
        # (and possibly city/year), we take the first/mean value since it's identical per month/city.
        
        # First, aggregate the target
        df_target = self.master.groupby(agg_cols, as_index=False)[self.target].sum()
        
        # Exogenous variables to preserve: temperature, humidity, cpi, gdp_growth
        # Because we're at Category x Site, we can group and take the mean for exogenous numeric features
        exog_cols = ['temperature', 'humidity', 'cpi', 'gdp_growth']
        df_exog = self.master.groupby(agg_cols, as_index=False)[exog_cols].mean()
        
        df_grain = pd.merge(df_target, df_exog, on=agg_cols, how='left')
        
        # 2. Create Continuous Timeline (Zero-filling)
        min_month = df_grain['month_start'].min()
        max_month = df_grain['month_start'].max()
        all_months = pd.date_range(start=min_month, end=max_month, freq='MS')
        
        # Get all unique series at this grain
        unique_series = df_grain[self.grain].drop_duplicates()
        
        # Cartesian product of series and months
        multi_idx = pd.MultiIndex.from_product(
            [unique_series[col] for col in self.grain] + [all_months],
            names=self.grain + ['month_start']
        )
        
        df_full = pd.DataFrame(index=multi_idx).reset_index()
        
        # Join actual data
        df_final = pd.merge(df_full, df_grain, on=agg_cols, how='left')
        
        # Fill missing demand with 0
        df_final[self.target] = df_final[self.target].fillna(0)
        
        # Forward/Backward fill exogenous variables securely within each site if needed
        # (Assuming weather is stable per site, and CPI/GDP is stable overall)
        df_final[exog_cols] = df_final.groupby('site')[exog_cols].transform(lambda x: x.ffill().bfill())
        
        # 3. Add Sufficient History Flag
        # Count non-zero or just total contiguous months? 
        # The requirement says "sufficient history flag". 
        # We count actual observed data points before zero-filling? 
        # Or we count from the first non-zero month.
        
        # Let's count months since the first non-zero demand
        def get_sufficient_history(group):
            non_zero_months = group[group[self.target] > 0]['month_start']
            if len(non_zero_months) == 0:
                return False
            first_obs = non_zero_months.min()
            valid_obs = group[group['month_start'] >= first_obs]
            return len(valid_obs) >= self.config.MIN_OBSERVATIONS
            
        history_map = df_final.groupby(self.grain).apply(get_sufficient_history)
        df_final = pd.merge(df_final, history_map.rename('sufficient_history'), left_on=self.grain, right_index=True)
        
        # 4. Sort chronologically
        df_final = df_final.sort_values(by=self.grain + ['month_start']).reset_index(drop=True)
        
        logger.info(f"Forecasting dataset built. Final shape: {df_final.shape}")
        return df_final
