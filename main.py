import os
import sys
import time
import sqlite3
import pandas as pd
import numpy as np
import warnings
import logging

from src.chronoscast.config import ProjectConfig
from src.chronoscast.utils import setup_logger
from src.chronoscast.data_ingestion import DataIngestor
from src.chronoscast.data_preparation import DataPreparer
from src.chronoscast.eda import EDADiagnostics
from src.chronoscast.forecasting_data import ForecastingDataBuilder
from src.chronoscast.validation import TimeSeriesValidator
from src.chronoscast.metrics import calculate_all_metrics
from src.chronoscast.forecasting_models import (
    BaselineModel, ARIMAModel, SARIMAModel, ETSModel, 
    ARIMAXModel, ProphetModel, LSTMModel
)
from src.chronoscast.ensemble_model import ForecastEnsemble
from src.chronoscast.decomposition_engine import DecompositionEngine

logger = setup_logger("chronoscast_pipeline")

def run():
    start_time = time.time()
    logger.info("Starting ChronosCast End-to-End Pipeline...")
    
    # 1. Ingestion
    ingestor = DataIngestor()
    raw_data = ingestor.load_all_data_sources()
    
    # 2. Preparation
    preparer = DataPreparer(raw_data)
    master_df = preparer.create_master_dataset()
    
    # 3. Sparsity Analysis & Grain Selection
    analyzer = EDADiagnostics(master_df)
    selection_result = analyzer.select_forecasting_grain(ProjectConfig.CANDIDATE_GRAINS)
    grain = selection_result['selected_grain']
    
    logger.info("--- SPARSITY ANALYSIS & GRAIN SELECTION ---")
    df_res = selection_result['candidate_results']
    for _, row in df_res.iterrows():
        logger.info(f"Candidate: {row['Grain']:<35} | Total: {row['Series_Count']:<3} | "
                    f"Viable: {row['Viable_Series']:<3} | Fill: {row['Fill_Rate_Pct']:>6.2f}%")
        
    logger.info(selection_result['selection_reason'])
    
    # 4. Forecasting Data 
    builder = ForecastingDataBuilder(master_df, grain=grain)
    forecasting_df = builder.build()
    
    horizon = ProjectConfig.FORECAST_HORIZON_MONTHS
    max_date = forecasting_df['month_start'].max()
    future_dates = pd.date_range(start=max_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    
    ensemble = ForecastEnsemble()
    grouped = list(forecasting_df.groupby(grain))
    
    logger.info(f"Total parent series identified: {len(grouped)}")
    
    def get_models():
        return [
            BaselineModel(),
            ARIMAModel(p_max=1, d_max=1, q_max=1),
            SARIMAModel(),
            ETSModel(),
            ARIMAXModel(exog_cols=['temperature', 'cpi', 'gdp_growth']),
            ProphetModel(),
            LSTMModel(epochs=20)
        ]
        
    validator = TimeSeriesValidator()
    model_metrics = {m.model_name: {'rmse': [], 'mae': [], 'success_series': 0, 'fail_series': 0} for m in get_models()}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        logger.info("--- PHASE 1: CHRONOLOGICAL VALIDATION ---")
        validation_start = time.time()
        
        for idx, (name, group_df) in enumerate(grouped):
            if not group_df['sufficient_history'].all():
                continue
                
            splits = list(validator.generate_splits(group_df))
            valid_splits = [s for s in splits if len(s[1]) == horizon]
            if not valid_splits:
                continue
                
            train_df, val_df = valid_splits[-1]
            actuals = val_df[ProjectConfig.TARGET_VARIABLE].values
            df_future = val_df.copy()
            
            models = get_models()
            for model in models:
                m_name = model.model_name
                try:
                    if model.fit(train_df):
                        preds = model.predict(steps=len(val_df), df_future=df_future)
                        mets = calculate_all_metrics(actuals, preds)
                        model_metrics[m_name]['rmse'].append(mets['rmse'])
                        model_metrics[m_name]['mae'].append(mets['mae'])
                        model_metrics[m_name]['success_series'] += 1
                    else:
                        model_metrics[m_name]['fail_series'] += 1
                except Exception as e:
                    logger.debug(f"{m_name} failed during validation on {name}: {e}")
                    model_metrics[m_name]['fail_series'] += 1
                    
        validation_elapsed = time.time() - validation_start
        logger.info(f"Validation completed in {validation_elapsed:.2f} seconds.")
        
        # Calculate ensemble weights
        logger.info("--- AGGREGATE VALIDATION METRICS & ENSEMBLE WEIGHTS ---")
        ensemble_weights = {}
        sum_inv_rmse = 0.0
        
        for m_name, mets in model_metrics.items():
            if mets['rmse']:
                agg_rmse = np.nanmean(mets['rmse'])
                agg_mae = np.nanmean(mets['mae'])
                if agg_rmse > 0:
                    inv_rmse = 1.0 / agg_rmse
                    sum_inv_rmse += inv_rmse
                    ensemble_weights[m_name] = inv_rmse
                elif agg_rmse == 0:
                    inv_rmse = 1e9 
                    sum_inv_rmse += inv_rmse
                    ensemble_weights[m_name] = inv_rmse
            else:
                agg_rmse = np.nan
                agg_mae = np.nan
                
            logger.info(f"Model: {m_name:<15} | RMSE: {agg_rmse:<8.4f} | MAE: {agg_mae:<8.4f} | Success: {mets['success_series']:<3} | Fail: {mets['fail_series']:<3}")
            
        if sum_inv_rmse > 0:
            for m_name in ensemble_weights:
                ensemble_weights[m_name] /= sum_inv_rmse
        
        logger.info("Calculated Ensemble Weights:")
        for m_name, w in ensemble_weights.items():
            logger.info(f"  {m_name}: {w:.4f}")
            
        # Final Forecasting
        logger.info("--- PHASE 2: FINAL FORECASTING (FULL HISTORY) ---")
        forecasting_start = time.time()
        all_parent_forecasts = []
        
        for idx, (name, group_df) in enumerate(grouped):
            name_tuple = name if isinstance(name, tuple) else (name,)
            logger.info(f"[{idx+1}/{len(grouped)}] Forecasting series: {' | '.join([f'{k}={v}' for k, v in zip(grain, name_tuple)])}")
            
            if not group_df['sufficient_history'].all():
                logger.warning(f"Skipping {name} due to insufficient history.")
                continue
                
            last_row = group_df.iloc[-1]
            df_future = pd.DataFrame({'month_start': future_dates})
            for exog in ['temperature', 'humidity', 'cpi', 'gdp_growth']:
                df_future[exog] = last_row[exog]
                
            models = get_models()
            predictions = {}
            for model in models:
                if model.fit(group_df):
                    preds = model.predict(steps=horizon, df_future=df_future)
                    predictions[model.model_name] = preds
                    
            final_preds, contributors = ensemble.combine(predictions, weights=ensemble_weights)
            logger.info(f"  Ensemble used {len(contributors)} models: {contributors}")
            
            for i, date in enumerate(future_dates):
                row_dict = {
                    'month_start': date,
                    'forecast': final_preds[i]
                }
                for g_col, g_val in zip(grain, name_tuple):
                    row_dict[g_col] = g_val
                all_parent_forecasts.append(row_dict)
                
        forecasting_elapsed = time.time() - forecasting_start
        logger.info(f"Final forecasting completed in {forecasting_elapsed:.2f} seconds.")
            
    parent_forecasts_df = pd.DataFrame(all_parent_forecasts)
    if parent_forecasts_df.empty:
        logger.error("No forecasts were generated. Pipeline halting.")
        sys.exit(1)
        
    # Decomposition & Reconciliation
    logger.info("--- PHASE 3: DECOMPOSITION & RECONCILIATION ---")
    decomp_engine = DecompositionEngine(master_df, parent_grain=grain)
    granular_forecasts_df = decomp_engine.decompose(parent_forecasts_df)
    
    reconciled = decomp_engine.reconcile(parent_forecasts_df, granular_forecasts_df)
    if not reconciled:
        logger.error("Pipeline HALTED due to reconciliation failure.")
        sys.exit(1)
        
    # Export to SQLite
    db_path = "chronoscast_forecasts.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    export_df = granular_forecasts_df[['articleno', 'site', 'month_start', 'granular_forecast']].rename(
        columns={
            'articleno': 'article',
            'month_start': 'forecast_month',
            'granular_forecast': 'expected_demand'
        }
    )
    export_df['forecast_month'] = export_df['forecast_month'].dt.strftime('%Y-%m-%d')
    export_df['expected_demand'] = export_df['expected_demand'].round(2)
    
    conn = sqlite3.connect(db_path)
    export_df.to_sql('granular_forecasts', conn, index=False, if_exists='replace')
    conn.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")
    logger.info(f"Generated {len(export_df)} granular forecast rows. Exported to {db_path}.")

if __name__ == "__main__":
    logging.getLogger('cmdstanpy').disabled = True
    run()
