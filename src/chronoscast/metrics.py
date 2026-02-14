import numpy as np

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates zero-safe Mean Absolute Percentage Error.
    If actual demand is 0, the percentage error is undefined. We handle this
    by only computing MAPE on non-zero true values. If all true values are 0,
    we return a fallback (e.g., 0.0 or np.nan, typically np.nan).
    """
    non_zero_mask = (y_true != 0)
    if not np.any(non_zero_mask):
        return np.nan
        
    y_true_nz = y_true[non_zero_mask]
    y_pred_nz = y_pred[non_zero_mask]
    
    return float(np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100)

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Returns a dictionary of all standard forecasting metrics."""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}
        
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }
