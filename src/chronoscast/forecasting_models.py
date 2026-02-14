import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import warnings

from .config import ProjectConfig
from .utils import setup_logger

# Import modeling libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from prophet import Prophet
except ImportError:
    pass # Tests might mock this or we are waiting for dependencies to install

logger = setup_logger(__name__)

class ModelFailureException(Exception):
    pass

class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models in ChronosCast.
    Enforces a consistent interface and provides centralized failure handling.
    """
    def __init__(self, config=ProjectConfig, **kwargs):
        self.config = config
        self.model_name = self.__class__.__name__
        self.target = self.config.TARGET_VARIABLE
        self.is_fitted = False
        self.failure_reason = None
        
    @abstractmethod
    def _fit(self, df_train: pd.DataFrame):
        pass
        
    @abstractmethod
    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        pass

    def fit(self, df_train: pd.DataFrame) -> bool:
        self.is_fitted = False
        self.failure_reason = None
        
        try:
            if len(df_train) < self.config.MIN_OBSERVATIONS:
                raise ModelFailureException("Insufficient history")
                
            if df_train[self.target].nunique() <= 1:
                raise ModelFailureException("Constant series")
                
            self._fit(df_train)
            self.is_fitted = True
            return True
            
        except ModelFailureException as e:
            self.failure_reason = str(e)
            logger.debug(f"{self.model_name} skipped fitting: {e}")
            return False
        except Exception as e:
            self.failure_reason = f"Unhandled exception during fit: {e}"
            logger.warning(f"{self.model_name} failed during fit: {e}")
            return False

    def predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            logger.debug(f"{self.model_name} predict called but model is not fitted.")
            return np.full(steps, np.nan)
            
        try:
            predictions = self._predict(steps, df_future)
            if len(predictions) != steps:
                raise ModelFailureException(f"Expected {steps} predictions, got {len(predictions)}")
                
            predictions = np.maximum(predictions, 0.0)
            return predictions
            
        except Exception as e:
            self.failure_reason = f"Exception during predict: {e}"
            logger.warning(f"{self.model_name} failed during predict: {e}")
            return np.full(steps, np.nan)

class BaselineModel(BaseForecaster):
    """
    Seasonal Naive Baseline.
    Predicts the value from exactly one seasonal period ago (e.g., 12 months).
    If there isn't enough history for a full seasonal cycle, falls back to Naive.
    """
    def __init__(self, config=ProjectConfig, seasonal_period=12, **kwargs):
        super().__init__(config, **kwargs)
        self.seasonal_period = seasonal_period
        self.last_values = None
        self.fallback_value = 0.0

    def _fit(self, df_train: pd.DataFrame):
        self.fallback_value = df_train[self.target].iloc[-1]
        if len(df_train) >= self.seasonal_period:
            self.last_values = df_train[self.target].iloc[-self.seasonal_period:].values
        else:
            self.last_values = None

    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        preds = np.zeros(steps)
        for i in range(steps):
            if self.last_values is not None:
                preds[i] = self.last_values[i % self.seasonal_period]
            else:
                preds[i] = self.fallback_value
        return preds

class ARIMAModel(BaseForecaster):
    """ARIMA model with bounded parameter selection."""
    def __init__(self, config=ProjectConfig, p_max=2, d_max=1, q_max=2, **kwargs):
        super().__init__(config, **kwargs)
        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max
        self.model_fit = None

    def _fit(self, df_train: pd.DataFrame):
        y = df_train[self.target].values
        best_aic = np.inf
        best_order = (0, 0, 0)
        best_mdl = None
        
        # Suppress convergence warnings during search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(self.p_max + 1):
                for d in range(self.d_max + 1):
                    for q in range(self.q_max + 1):
                        try:
                            mdl = ARIMA(y, order=(p, d, q)).fit(method='innovations_mle')
                            if mdl.aic < best_aic:
                                best_aic = mdl.aic
                                best_order = (p, d, q)
                                best_mdl = mdl
                        except:
                            continue
                            
        if best_mdl is None:
            raise ModelFailureException("ARIMA failed to converge on any bounded parameters.")
            
        self.model_fit = best_mdl

    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.model_fit.forecast(steps=steps)

class SARIMAModel(BaseForecaster):
    """SARIMA model assuming a fixed seasonal structure."""
    def __init__(self, config=ProjectConfig, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), **kwargs):
        super().__init__(config, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def _fit(self, df_train: pd.DataFrame):
        y = df_train[self.target].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model_fit = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order).fit(method='innovations_mle')
            except Exception as e:
                raise ModelFailureException(f"SARIMA failed to fit: {e}")

    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.model_fit.forecast(steps=steps)

class ETSModel(BaseForecaster):
    """Exponential Smoothing (ETS) model."""
    def __init__(self, config=ProjectConfig, seasonal_period=12, **kwargs):
        super().__init__(config, **kwargs)
        self.seasonal_period = seasonal_period
        self.model_fit = None

    def _fit(self, df_train: pd.DataFrame):
        y = df_train[self.target].values
        # Add a tiny constant to avoid multiplicative errors on exact 0
        y_safe = np.where(y <= 0, 0.001, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # We use additive trend and seasonal since it's robust to zeros
                self.model_fit = ExponentialSmoothing(
                    y_safe, 
                    trend='add', 
                    seasonal='add', 
                    seasonal_periods=self.seasonal_period,
                    initialization_method="estimated"
                ).fit()
            except Exception as e:
                raise ModelFailureException(f"ETS failed to fit: {e}")

    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.model_fit.forecast(steps)

class ARIMAXModel(BaseForecaster):
    """ARIMAX using predefined exogenous features (e.g. CPI, GDP, Temperature)."""
    def __init__(self, config=ProjectConfig, order=(1, 0, 1), exog_cols=None, **kwargs):
        super().__init__(config, **kwargs)
        self.order = order
        self.exog_cols = exog_cols or ['temperature', 'cpi']
        self.model_fit = None

    def _fit(self, df_train: pd.DataFrame):
        y = df_train[self.target].values
        missing_cols = [c for c in self.exog_cols if c not in df_train.columns]
        if missing_cols:
            raise ModelFailureException(f"Missing exogenous columns: {missing_cols}")
            
        exog = df_train[self.exog_cols].values
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model_fit = ARIMA(y, exog=exog, order=self.order).fit(method='innovations_mle')
            except Exception as e:
                raise ModelFailureException(f"ARIMAX failed to fit: {e}")

    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if df_future is None or len(df_future) < steps:
            raise ModelFailureException("ARIMAX requires valid future exogenous data.")
            
        exog_future = df_future[self.exog_cols].iloc[:steps].values
        return self.model_fit.forecast(steps=steps, exog=exog_future)

class ProphetModel(BaseForecaster):
    """Prophet forecasting model."""
    def __init__(self, config=ProjectConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = None

    def _fit(self, df_train: pd.DataFrame):
        df_prophet = pd.DataFrame({
            'ds': df_train['month_start'],
            'y': df_train[self.target]
        })
        # Disable daily/weekly seasonality since data is monthly
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        # Suppress logging
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        try:
            self.model.fit(df_prophet)
        except Exception as e:
            raise ModelFailureException(f"Prophet failed to fit: {e}")

    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        # Create future dataframe matching the required steps
        future = self.model.make_future_dataframe(periods=steps, freq='MS')
        forecast = self.model.predict(future)
        return forecast['yhat'].iloc[-steps:].values

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    
    class SimpleLSTM(nn.Module):
        def __init__(self, hidden_size=16, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.linear(lstm_out[:, -1, :])
            return out
            
except ImportError:
    pass

class LSTMModel(BaseForecaster):
    """
    LSTM Forecasting Model.
    Uses strict chronological windowing and scales only using training data.
    """
    def __init__(self, config=ProjectConfig, seq_length=6, epochs=50, hidden_size=16, **kwargs):
        super().__init__(config, **kwargs)
        self.seq_length = seq_length
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.model = None
        self.scaler = None
        self.last_sequence = None

    def _create_sequences(self, data: np.ndarray):
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            xs.append(data[i:(i + self.seq_length)])
            ys.append(data[i + self.seq_length])
        return np.array(xs), np.array(ys)

    def _fit(self, df_train: pd.DataFrame):
        if 'torch' not in globals():
            raise ModelFailureException("PyTorch not available for LSTM.")
            
        if len(df_train) <= self.seq_length:
            raise ModelFailureException(f"History length ({len(df_train)}) must be > sequence length ({self.seq_length})")
            
        # Reproducibility
        torch.manual_seed(self.config.RANDOM_SEED)
        
        y = df_train[self.target].values.reshape(-1, 1)
        
        # Strictly scale on training data ONLY
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        y_scaled = self.scaler.fit_transform(y)
        
        # Save last sequence for prediction
        self.last_sequence = y_scaled[-self.seq_length:]
        
        X, Y = self._create_sequences(y_scaled)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        
        self.model = SimpleLSTM(hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Suppress training output
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(X_tensor)
            loss = criterion(out, Y_tensor)
            loss.backward()
            optimizer.step()
            
    def _predict(self, steps: int, df_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        self.model.eval()
        preds = []
        
        current_seq = self.last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(steps):
                # Shape for model: (batch_size=1, seq_length, features=1)
                seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
                pred_scaled = self.model(seq_tensor).item()
                preds.append(pred_scaled)
                
                # Update sequence for next prediction
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1] = [pred_scaled]
                
        # Inverse transform
        preds_array = np.array(preds).reshape(-1, 1)
        preds_unscaled = self.scaler.inverse_transform(preds_array).flatten()
        return preds_unscaled

