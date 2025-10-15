"""
Data Pipeline - Prepares and cleans data for demand forecasting
Part of the CONTEXT LAYER in the demand forecast architecture
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import kagglehub
from kagglehub import KaggleDatasetAdapter

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Handles data preparation, cleaning, and aggregation for forecasting.

    Responsibilities:
    1. Fetch data from multiple sources (external sources, Kaggle, PostgreSQL)
    2. Clean and validate time series data
    3. Handle missing values and outliers
    4. Aggregate data by time period
    5. Prepare data for Prophet forecasting engine
    """

    def __init__(self):
        self.raw_data = None
        self.cleaned_data = None
        self.kaggle_cache = None

    def load_kaggle_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load retail supply chain sales dataset from Kaggle.
        Used for testing and demo purposes.
        
        Note: Returns RAW data as-is from Kaggle. Column standardization
        is handled by Context Layer's _standardize_data_format() method.

        Args:
            force_refresh: If True, reload from Kaggle instead of using cache

        Returns:
            Raw DataFrame with Kaggle sales data (unstandardized columns)
        """
        try:
            if self.kaggle_cache is not None and not force_refresh:
                logger.info("Using cached Kaggle dataset")
                return self.kaggle_cache.copy()

            logger.info("Loading Kaggle retail supply chain dataset...")

            # First, download the dataset to get the 
            dataset_path = kagglehub.dataset_download("amangarg08/apple-retail-sales-dataset")
            logger.info(f"Kaggle dataset downloaded to: {dataset_path}")

            # Find CSV files in the dataset
            import os
            all_files = os.listdir(dataset_path)
            logger.info(f"Available files in dataset: {all_files}")
            
            csv_files = [f for f in all_files if f.endswith('.csv') or f.endswith('.xlsx')]

            if not csv_files:
                raise ValueError(f"No CSV files found in {dataset_path}")
            
            # Prioritize files with 'sales', 'transactions', 'orders' in the name
            sales_files = [f for f in csv_files if any(keyword in f.lower() for keyword in ['sales', 'transaction', 'order', 'invoice'])]
            
            if sales_files:
                csv_filename = sales_files[0]
                logger.info(f"Found sales data file: {csv_filename}")
            else:
                csv_filename = csv_files[0]
                logger.warning(f"No sales file found, using first available: {csv_filename}")

            # Now load using KaggleDatasetAdapter.PANDAS with the specific filename
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "amangarg08/apple-retail-sales-dataset",
                csv_filename  # Specify the actual CSV filename
            )

            logger.info(f"Kaggle dataset loaded: {len(df)} records, {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")

            # Note: Column standardization is now handled by Context Layer's _standardize_data_format()
            # This method just returns the raw data as-is from Kaggle
            
            # Cache the dataset
            self.kaggle_cache = df.copy()
            logger.info(f"✓ Successfully loaded and cached {len(df)} records from Kaggle dataset")

            return df

        except Exception as e:
            logger.error(f"Failed to load Kaggle dataset: {str(e)}")
            logger.error("Make sure 'kagglehub[pandas-datasets]' is installed: pip install 'kagglehub[pandas-datasets]'")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Order Date', 'Product Name', 'Sales', 'Quantity'])

    def detect_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Dynamically detect date and value columns from the DataFrame.

        Looks for columns that match common patterns:
        - Date columns: Contains 'date', 'time', 'dt', or is datetime type
        - Value columns: Contains 'sales', 'revenue', 'amount', 'total', 'value' or is numeric

        Args:
            df: Input DataFrame

        Returns:
            Dict with detected column names: {'date_col': str, 'value_col': str}
        """
        detected = {'date_col': None, 'value_col': None}

        # Detect date column
        for col in df.columns:
            col_lower = col.lower()
            # Check if it's already a datetime type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                detected['date_col'] = col
                break
            # Check if column name suggests it's a date
            elif any(keyword in col_lower for keyword in ['date', 'time', 'dt', 'day', 'timestamp']):
                # Try to parse as date
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    detected['date_col'] = col
                    break
                except:
                    continue

        # Detect value column (sales/revenue) - PRIORITY ORDER
        # 1. First, check for exact matches (case-insensitive)
        priority_names = ['sales', 'revenue', 'amount', 'total_sales']
        for priority_name in priority_names:
            for col in df.columns:
                if col.lower() == priority_name:
                    # Must be numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        detected['value_col'] = col
                        logger.info(f"Found exact match for value column: '{col}'")
                        break
            if detected['value_col']:
                break
        
        # 2. If no exact match, look for columns containing keywords (but exclude "people")
        if detected['value_col'] is None:
            for col in df.columns:
                col_lower = col.lower()
                # Skip if it's the date column
                if col == detected['date_col']:
                    continue
                # Exclude columns with "people" in the name (like "Retail Sales People")
                if 'people' in col_lower or 'person' in col_lower or 'employee' in col_lower:
                    continue
                # Check if column name suggests it's a sales/revenue column
                if any(keyword in col_lower for keyword in ['sales', 'sale', 'revenue', 'amount', 'total', 'value', 'price']):
                    # Must be numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        detected['value_col'] = col
                        logger.info(f"Found keyword match for value column: '{col}'")
                        break

        # 3. If still no value column found, use first numeric column
        if detected['value_col'] is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Exclude columns with "people" in the name
            numeric_cols = [col for col in numeric_cols if 'people' not in col.lower()]
            if len(numeric_cols) > 0:
                detected['value_col'] = numeric_cols[0]
                logger.info(f"Using first numeric column: '{detected['value_col']}'")

        logger.info(f"Detected columns - Date: '{detected['date_col']}', Value: '{detected['value_col']}'")
        return detected

    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, str]]:
        """
        Validate data quality for forecasting with dynamic column detection.

        Checks:
        - Minimum data points (30 days)
        - Required columns present (detects dynamically)
        - Date column validity
        - Numeric columns validity

        Args:
            df: Raw sales data

        Returns:
            Tuple of (is_valid, list_of_issues, detected_columns)
        """
        issues = []

        # Dynamically detect columns
        detected_cols = self.detect_columns(df)

        # Check if we found required columns
        if detected_cols['date_col'] is None:
            issues.append(f"No date column found. Available columns: {list(df.columns)}")
            return False, issues, detected_cols

        if detected_cols['value_col'] is None:
            issues.append(f"No numeric value column found. Available columns: {list(df.columns)}")
            return False, issues, detected_cols

        # Check minimum data points
        if len(df) < 30:
            issues.append(f"Insufficient data: {len(df)} records (minimum 30 required)")

        # Check date column
        try:
            pd.to_datetime(df[detected_cols['date_col']], errors='coerce')
        except Exception as e:
            issues.append(f"Invalid date column '{detected_cols['date_col']}': {str(e)}")

        # Check sales column is numeric
        if not pd.api.types.is_numeric_dtype(df[detected_cols['value_col']]):
            try:
                pd.to_numeric(df[detected_cols['value_col']], errors='coerce')
            except Exception as e:
                issues.append(f"Value column '{detected_cols['value_col']}' is not numeric: {str(e)}")

        # Check for all null sales
        if df[detected_cols['value_col']].isna().all():
            issues.append(f"All values in '{detected_cols['value_col']}' are null")

        is_valid = len(issues) == 0
        return is_valid, issues, detected_cols

    def clean_time_series_data(
        self,
        df: pd.DataFrame,
        date_col: str = 'Order Date',
        value_col: str = 'Sales'
    ) -> pd.DataFrame:
        """
        Clean and prepare time series data.

        Operations:
        - Convert dates to datetime
        - Remove invalid dates
        - Convert sales to numeric
        - Remove negative sales
        - Handle outliers
        - Sort by date

        Args:
            df: Raw data
            date_col: Name of date column
            value_col: Name of value column

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Convert date column
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')

        # Remove rows with invalid dates
        before_count = len(df_clean)
        df_clean = df_clean.dropna(subset=[date_col])
        removed = before_count - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid dates")

        # Convert sales to numeric
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')

        # Remove rows with null sales
        before_count = len(df_clean)
        df_clean = df_clean.dropna(subset=[value_col])
        removed = before_count - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} rows with null sales")

        # Remove negative sales (use .loc to avoid ambiguous boolean)
        before_count = len(df_clean)
        df_clean = df_clean.loc[df_clean[value_col] >= 0].copy()
        removed = before_count - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} rows with negative sales")

        # Sort by date
        df_clean = df_clean.sort_values(date_col)

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        logger.info(f"Cleaned data: {len(df_clean)} records remaining")

        return df_clean

    def handle_outliers(
        self,
        df: pd.DataFrame,
        value_col: str = 'Sales',
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in sales data.

        Args:
            df: Input data
            value_col: Column to check for outliers
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        df_result = df.copy()

        if method == 'iqr':
            Q1 = df_result[value_col].quantile(0.25)
            Q3 = df_result[value_col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Cap outliers instead of removing them
            df_result[value_col] = df_result[value_col].clip(lower=lower_bound, upper=upper_bound)

            logger.info(f"Capped outliers using IQR method (bounds: {lower_bound:.2f} - {upper_bound:.2f})")

        elif method == 'zscore':
            mean = df_result[value_col].mean()
            std = df_result[value_col].std()

            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

            df_result[value_col] = df_result[value_col].clip(lower=lower_bound, upper=upper_bound)

            logger.info(f"Capped outliers using Z-score method (bounds: {lower_bound:.2f} - {upper_bound:.2f})")

        return df_result

    def aggregate_by_period(
        self,
        df: pd.DataFrame,
        date_col: str = 'Order Date',
        value_col: str = 'Sales',
        freq: str = 'D',
        fill_method: str = 'zero'
    ) -> pd.DataFrame:
        """
        Aggregate sales data by time period.

        Args:
            df: Input data
            date_col: Date column name
            value_col: Value column name
            freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
            fill_method: Method to fill missing dates ('zero', 'forward', 'interpolate')

        Returns:
            Aggregated DataFrame with complete date range
        """
        # Group by date and sum sales
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()

        # Create complete date range
        date_range = pd.date_range(
            start=df_agg[date_col].min(),
            end=df_agg[date_col].max(),
            freq=freq
        )

        # Create complete DataFrame
        complete_df = pd.DataFrame({date_col: date_range})

        # Merge with aggregated data
        result = complete_df.merge(df_agg, on=date_col, how='left')

        # Fill missing values
        if fill_method == 'zero':
            result[value_col] = result[value_col].fillna(0)
        elif fill_method == 'forward':
            result[value_col] = result[value_col].fillna(method='ffill')
        elif fill_method == 'interpolate':
            result[value_col] = result[value_col].interpolate(method='linear')

        logger.info(f"Aggregated to {len(result)} {freq} periods")

        return result

    def filter_by_product(
        self,
        df: pd.DataFrame,
        product_name: str,
        product_col: str = 'Product Name'
    ) -> pd.DataFrame:
        """
        Filter data by product name (case-insensitive partial match).

        Args:
            df: Input data
            product_name: Product name to filter
            product_col: Product column name

        Returns:
            Filtered DataFrame
        """
        if product_col not in df.columns:
            logger.warning(f"Product column '{product_col}' not found, returning all data")
            return df

        filtered = df[df[product_col].str.contains(product_name, case=False, na=False)]
        logger.info(f"Filtered to {len(filtered)} records for product: {product_name}")

        return filtered

    def prepare_for_forecasting(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        value_col: Optional[str] = None,
        product_filter: Optional[str] = None,
        freq: str = 'D',
        remove_outliers: bool = False
    ) -> pd.DataFrame:
        """
        Complete data preparation pipeline for forecasting with dynamic column detection.

        This is the main method called by the Context Layer.

        Args:
            df: Raw sales data
            date_col: Date column name (auto-detected if None)
            value_col: Value column name (auto-detected if None)
            product_filter: Optional product name filter
            freq: Aggregation frequency
            remove_outliers: Whether to handle outliers (default: False for retail sales data)

        Returns:
            Clean, aggregated DataFrame ready for Prophet with standardized columns
        """
        logger.info("Starting data preparation pipeline...")

        # Step 1: Filter by product if specified
        if product_filter:
            df = self.filter_by_product(df, product_filter)

        # Step 2: Validate data quality and detect columns if not provided
        is_valid, issues, detected_cols = self.validate_data_quality(df)
        if not is_valid:
            logger.error(f"Data quality issues: {issues}")
            raise ValueError(f"Data quality validation failed: {', '.join(issues)}")

        # Use detected columns if not explicitly provided
        if date_col is None:
            date_col = detected_cols['date_col']
        if value_col is None:
            value_col = detected_cols['value_col']

        logger.info(f"Using columns - Date: '{date_col}', Value: '{value_col}'")

        # Step 3: Clean time series data
        df = self.clean_time_series_data(df, date_col, value_col)

        # Step 4: Handle outliers
        if remove_outliers:
            df = self.handle_outliers(df, value_col, method='iqr', threshold=3.0)

        # Step 5: Aggregate by period
        df = self.aggregate_by_period(df, date_col, value_col, freq, fill_method='zero')

        # Step 6: Standardize column names to 'Order Date' and 'Sales' for Prophet
        if date_col != 'Order Date':
            df = df.rename(columns={date_col: 'Order Date'})
            logger.info(f"Standardized date column '{date_col}' -> 'Order Date'")
        
        if value_col != 'Sales':
            df = df.rename(columns={value_col: 'Sales'})
            logger.info(f"Standardized value column '{value_col}' -> 'Sales'")

        # Store cleaned data
        self.cleaned_data = df.copy()

        logger.info("Data preparation pipeline completed successfully")

        return df

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset.

        Args:
            df: Input data

        Returns:
            Dict with summary statistics
        """
        if 'Order Date' in df.columns:
            date_col = 'Order Date'
        else:
            date_col = df.columns[0]

        if 'Sales' in df.columns:
            value_col = 'Sales'
        else:
            value_col = df.select_dtypes(include=[np.number]).columns[0]

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Safely get date strings
        start_date = None
        end_date = None
        days = 0
        
        if not df.empty and df[date_col].notna().any():
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            
            if pd.notna(min_date):
                start_date = min_date.isoformat() if hasattr(min_date, 'isoformat') else str(min_date)
            if pd.notna(max_date):
                end_date = max_date.isoformat() if hasattr(max_date, 'isoformat') else str(max_date)
            if pd.notna(min_date) and pd.notna(max_date):
                try:
                    # Try pandas Timedelta first
                    days = (max_date - min_date).days
                except (OverflowError, ValueError):
                    # Fallback: Convert to Python datetime for very large date ranges
                    try:
                        min_dt = min_date.to_pydatetime() if hasattr(min_date, 'to_pydatetime') else min_date
                        max_dt = max_date.to_pydatetime() if hasattr(max_date, 'to_pydatetime') else max_date
                        days = (max_dt - min_dt).days
                    except Exception:
                        # If all else fails, return 0
                        days = 0
                        logger.warning(f"Could not calculate date range between {min_date} and {max_date}")

        return {
            "total_records": len(df),
            "date_range": {
                "start": start_date,
                "end": end_date,
                "days": days
            },
            "sales_stats": {
                "total": float(df[value_col].sum()),
                "mean": float(df[value_col].mean()),
                "median": float(df[value_col].median()),
                "std": float(df[value_col].std()),
                "min": float(df[value_col].min()),
                "max": float(df[value_col].max())
            },
            "missing_values": int(df[value_col].isna().sum()),
            "zero_sales_days": int((df[value_col] == 0).sum())
        }
