import pandas as pd
from typing import Dict
from .config import ProjectConfig
from .utils import setup_logger, clean_column_names

logger = setup_logger(__name__)

class DataPreparer:
    """
    Cleans raw data and constructs the Monthly Master Dataset.
    """
    
    def __init__(self, raw_data: Dict[str, pd.DataFrame], config=ProjectConfig):
        # We copy to avoid modifying the raw ingestion data in place
        self.data = {k: v.copy() for k, v in raw_data.items()}
        self.config = config

    def _clean_and_format(self):
        """
        Cleans data types, drops known duplicates, and standardizes columns.
        """
        logger.info("Cleaning and formatting raw datasets...")
        
        # Standardize all column names first
        for key, df in self.data.items():
            self.data[key] = clean_column_names(df)
            
        sales = self.data['sales']
        weather = self.data['weather']
        
        # 1. Deduplicate sales (known 1 duplicate)
        initial_sales_rows = len(sales)
        sales.drop_duplicates(inplace=True)
        dropped = initial_sales_rows - len(sales)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate rows from sales.")
            
        # 2. Parse dates
        # Sales dates are integers like 20220101
        sales['billingdate_dt'] = pd.to_datetime(sales[self.config.DATE_COLUMN].astype(str), format='%Y%m%d')
        # Create a clean month-start timestamp for joining
        sales['month_start'] = sales['billingdate_dt'].dt.to_period('M').dt.to_timestamp()
        sales['year'] = sales['billingdate_dt'].dt.year
        
        # Weather dates are integers like 20220101 representing months
        weather['month_start'] = pd.to_datetime(weather['month'].astype(str), format='%Y%m%d').dt.to_period('M').dt.to_timestamp()
        
        self.data['sales'] = sales
        self.data['weather'] = weather

    def create_master_dataset(self) -> pd.DataFrame:
        """
        Aggregates daily sales to a monthly grain and joins all dimension and exogenous datasets.
        """
        self._clean_and_format()
        
        sales = self.data['sales']
        dim_prod = self.data['product']
        dim_pc = self.data['profit_center']
        dim_sr = self.data['showroom']
        weather = self.data['weather']
        econ = self.data['economic']
        
        # Resolve semantic naming conflicts
        # salesTran.product is a category group; dimProduct.product is a brand line.
        sales = sales.rename(columns={'product': 'sales_category'})
        dim_prod = dim_prod.rename(columns={'product': 'brand_line'})
        
        # Drop overlapping columns in dimensions to avoid _x and _y suffixing
        dim_prod_clean = dim_prod[['articleno', 'category', 'brand_line']]
        dim_pc_clean = dim_pc[['profitcenter', 'city', 'countrykey', 'companyname']]
        dim_sr_clean = dim_sr[['refid', 'showroom', 'plantdescription']]
        
        # 1. Aggregate Sales to Monthly Grain
        # We group by month_start, year, and all primary categorical dimensions.
        logger.info("Aggregating sales to monthly grain...")
        
        group_cols = [
            'month_start', 'year', 'articleno', 'articledescription', 
            'profitcenter', 'brand', 'sales_category', 'site'
        ]
        
        monthly_sales = sales.groupby(group_cols, as_index=False).agg({
            self.config.TARGET_VARIABLE: 'sum',
            'baseprice': 'mean' # Representative price for the month
        })
        
        # 2. Join Dimensions
        logger.info("Joining dimensions...")
        master = pd.merge(monthly_sales, dim_prod_clean, on='articleno', how='left')
        master = pd.merge(master, dim_pc_clean, on='profitcenter', how='left')
        master = pd.merge(master, dim_sr_clean, left_on='site', right_on='refid', how='left')
        master.drop(columns=['refid'], inplace=True)
        
        # 3. Join Exogenous (Weather)
        logger.info("Joining exogenous weather data...")
        master = pd.merge(master, weather[['month_start', 'city', 'temperature', 'humidity']], 
                          on=['month_start', 'city'], how='left')
                          
        # 4. Join Exogenous (Economic)
        # Note: dim_pc_clean provides countrykey='IN', but econ has country='India'. 
        # Since it's a single country, we can just join on 'year'.
        logger.info("Joining exogenous economic data...")
        master = pd.merge(master, econ[['year', 'cpi', 'gdp_growth']], on='year', how='left')
        
        logger.info(f"Master dataset created successfully. Final shape: {master.shape}")
        
        return master
