import os

class ProjectConfig:
    """
    Centralized configuration for the ChronosCast project.
    """
    
    # Base directory calculation relative to this config file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Input file keys and filenames
    RAW_DATA_SOURCES = {
        "sales": os.path.join(DATA_RAW_DIR, "salesTran.csv"),
        "product": os.path.join(DATA_RAW_DIR, "dimProduct.csv"),
        "profit_center": os.path.join(DATA_RAW_DIR, "dimProfitCenter.csv"),
        "showroom": os.path.join(DATA_RAW_DIR, "dimShowRoom.csv"),
        "weather": os.path.join(DATA_RAW_DIR, "weatherIndex.csv"),
        "economic": os.path.join(DATA_RAW_DIR, "economicIndex.csv")
    }
    
    # Output targets
    MASTER_DATASET_PATH = os.path.join(RESULTS_DIR, "master_dataset.csv")
    
    # Important Schema Details
    DATE_COLUMN = "billingdate"
    TARGET_VARIABLE = "billqty"
    
    # Forecasting Bounds and Defaults
    FORECAST_HORIZON_MONTHS = 4
    DECOMPOSITION_WINDOW_MONTHS = 18
    MIN_OBSERVATIONS = 24
    
    # Seed for reproducibility
    RANDOM_SEED = 42
