"""
Context Layer - Orchestrates demand forecasting pipeline

The Context Layer sits between the AI Agent and the Forecasting Engine.
It handles:
1. Aggregates data from external sources/Kaggle/PostgreSQL
2. Retrieves user settings (economic events, supply chain)
3. Extracts forecast parameters from user prompt
4. Calls Prophet forecasting engine
5. Validates with Exponential Smoothing
6. Passes results to LLM and Advisor layers
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import re

from lib.config import DatabaseManager
from lib.data_pipeline import DataPipeline
from lib.forecast_settings import ForecastSettingsManager
from lib.forecasting_engine import ForecastingEngine
from lib.validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class ContextLayer:
    """
    Central coordinator for demand forecasting workflow.

    Implements your architecture's CONTEXT LAYER:
    USER PROMPT → SYSTEM → **CONTEXT LAYER** → LLM CALL → VALIDATION → ADVISOR
    """

    def __init__(self):
        self.data_pipeline = DataPipeline()
        self.forecasting_engine = ForecastingEngine()
        self.validation_engine = ValidationEngine()

    def prepare_forecast_context(
        self,
        session_id: str,
        user_prompt: str,
        product_filter: Optional[str] = None,
        forecast_mode: str = "aggregate",  # "aggregate", "single", "multi", "top_n"
        top_n_products: int = 10  # For "multi" or "top_n" mode
    ) -> Dict:
        """
        Main method: Aggregates all context for demand forecasting.

        This is called by the forecast endpoint and orchestrates the entire flow.

        Args:
            session_id: User session ID
            user_prompt: Natural language query (e.g., "forecast next 45 days weekly")
            product_filter: Optional product name/ID to filter (single product)
            forecast_mode: Forecasting strategy:
                - "aggregate": All products combined (default)
                - "single": One specific product (requires product_filter)
                - "multi": Multiple specific products (comma-separated in product_filter)
                - "top_n": Top N products by sales volume
            top_n_products: Number of products to forecast in "top_n" mode

        Returns:
            Complete forecast context ready for LLM and Advisor:
            {
                "forecast_data": {...},  # Prophet output
                "validation_result": {...},  # Validation check
                "settings": {...},  # User settings
                "data_summary": {...},  # Data quality info
                "user_query": {...}  # Extracted parameters
            }
        """
        try:
            logger.info(f"Preparing forecast context for session {session_id}")

            # Step 1: Extract forecast parameters from user prompt
            forecast_params = self._extract_forecast_parameters(user_prompt)
            logger.info(f"Extracted forecast parameters: {forecast_params}")

            # Step 2: Retrieve user settings
            settings = ForecastSettingsManager.get_settings(session_id)
            if not settings:
                logger.warning(f"No settings found for session {session_id}, using defaults")
                settings = ForecastSettingsManager.get_default_settings()

            economic_events = settings.get("economic_events", [])
            supply_chain_locations = settings.get("supply_chain_locations", [])

            # Step 3: Fetch historical data
            historical_data = self._fetch_historical_data(
                session_id,
                product_filter=None,  # Don't filter yet, get all data first
                lookback_days=forecast_params.get("lookback_days", 90)
            )

            if historical_data is None or len(historical_data) < 30:
                return {
                    "error": "Insufficient historical data",
                    "message": "Need at least 30 days of sales data for forecasting",
                    "available_data_points": len(historical_data) if historical_data is not None else 0
                }
            
            # Step 3b: Enrich with product names if needed (dynamic)
            historical_data = self._enrich_product_names(historical_data, session_id)
            
            # Step 3c: Determine which products to forecast based on mode
            products_to_forecast = self._determine_products_to_forecast(
                historical_data,
                forecast_mode,
                product_filter,
                top_n_products
            )
            
            logger.info(f"Forecast mode: {forecast_mode}, Products: {products_to_forecast}")

            # Step 4-6: Generate forecast (single or multi-product)
            if len(products_to_forecast) == 0:
                # AGGREGATE MODE: Forecast all products combined
                forecast_result = self._run_single_forecast(
                    historical_data,
                    forecast_params,
                    economic_events,
                    supply_chain_locations,
                    product_name="All Products"
                )
            elif len(products_to_forecast) == 1:
                # SINGLE PRODUCT MODE
                product_name = products_to_forecast[0]
                product_data = historical_data[historical_data['Product Name'] == product_name]
                
                if len(product_data) < 30:
                    return {
                        "error": "Insufficient product data",
                        "message": f"Product '{product_name}' has only {len(product_data)} records (minimum 30 required)",
                        "available_data_points": len(product_data)
                    }
                
                forecast_result = self._run_single_forecast(
                    product_data,
                    forecast_params,
                    economic_events,
                    supply_chain_locations,
                    product_name=product_name
                )
            else:
                # MULTI-PRODUCT MODE
                forecast_result = self._run_multi_product_forecast(
                    historical_data,
                    products_to_forecast,
                    forecast_params,
                    economic_events,
                    supply_chain_locations
                )

            # Check if forecast failed
            if "error" in forecast_result:
                return forecast_result

            # Note: Validation and confidence merging now happens inside _run_single_forecast()
            # or _run_multi_product_forecast() for multi-product scenarios

            # Step 7: Get data summary
            data_summary = self.data_pipeline.get_data_summary(historical_data)

            # Step 8: Build complete context
            # For single/aggregate mode, forecast_result contains validation_result
            # For multi mode, forecast_result contains per-product validations
            context = {
                "forecast_data": forecast_result,
                "settings": {
                    "economic_events": economic_events,
                    "supply_chain_locations": supply_chain_locations
                },
                "data_summary": data_summary,
                "user_query": {
                    "original_prompt": user_prompt,
                    "extracted_parameters": forecast_params,
                    "product_filter": product_filter,
                    "forecast_mode": forecast_mode,
                    "top_n_products": top_n_products
                },
                "context_generated_at": datetime.now().isoformat()
            }

            logger.info("Context preparation complete")
            return context

        except Exception as e:
            logger.error(f"Error preparing forecast context: {str(e)}")
            return {
                "error": "Context preparation failed",
                "message": str(e)
            }

    def _extract_forecast_parameters(self, user_prompt: str) -> Dict:
        """
        Extract forecast parameters from natural language prompt.

        Examples:
        - "forecast next 45 days" → periods=45, frequency='D'
        - "predict demand for 8 weeks" → periods=8, frequency='W'
        - "monthly forecast for 3 months" → periods=3, frequency='M'

        Args:
            user_prompt: Natural language query

        Returns:
            Dict with extracted parameters:
            {
                "periods": int,
                "frequency": str,  # 'D', 'W', or 'M'
                "lookback_days": int
            }
        """
        params = {
            "periods": 30,  # Default
            "frequency": "D",  # Default daily
            "lookback_days": 90  # Default 3 months of history
        }

        prompt_lower = user_prompt.lower()

        # Extract number of periods
        # Look for patterns like "45 days", "8 weeks", "3 months", "next 30 days"
        patterns = [
            r'(\d+)\s*(day|days)',
            r'(\d+)\s*(week|weeks)',
            r'(\d+)\s*(month|months)',
            r'next\s+(\d+)',
            r'for\s+(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                periods = int(match.group(1))

                # Determine frequency based on unit
                if 'week' in pattern:
                    params["periods"] = periods
                    params["frequency"] = "W"
                elif 'month' in pattern:
                    params["periods"] = periods
                    params["frequency"] = "M"
                else:
                    params["periods"] = periods
                    params["frequency"] = "D"

                break

        # Look for explicit frequency keywords
        if any(word in prompt_lower for word in ['weekly', 'week by week', 'per week']):
            params["frequency"] = "W"
        elif any(word in prompt_lower for word in ['monthly', 'month by month', 'per month']):
            params["frequency"] = "M"
        elif any(word in prompt_lower for word in ['daily', 'day by day', 'per day']):
            params["frequency"] = "D"

        # Adjust lookback based on forecast horizon (longer forecast = need more history)
        if params["periods"] > 90:
            params["lookback_days"] = 180  # 6 months
        elif params["periods"] > 30:
            params["lookback_days"] = 120  # 4 months
        else:
            params["lookback_days"] = 90  # 3 months (default)

        logger.debug(f"Extracted parameters from '{user_prompt}': {params}")
        return params

    def _fetch_historical_data(
        self,
        session_id: str,
        product_filter: Optional[str],
        lookback_days: int = 90
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical sales data from external sources, PostgreSQL, or Kaggle.
        
        This method acts as a FORMAT ADAPTER - it fetches data from any source
        and standardizes it into a consistent format before passing to Data Pipeline.

        Priority order:
        1. External data source (if connected and has data)
        2. PostgreSQL (if connected and has data)
        3. Kaggle dataset (always fallback for testing/demo)

        Args:
            session_id: User session ID
            product_filter: Optional product filter
            lookback_days: How many days of history to fetch

        Returns:
            Standardized DataFrame with consistent column names or None
        """
        try:
            start_date = datetime.now() - timedelta(days=lookback_days)
            end_date = datetime.now()

            raw_df = None
            data_source = None

            # Try PostgreSQL first
            if DatabaseManager.has_user_connection(session_id):
                logger.info("Fetching data from PostgreSQL...")
                sales_data = DatabaseManager.get_data(
                    session_id=session_id,
                    query_type="sales_data",
                    use_mock=False
                )

                if sales_data and len(sales_data) >= 50:
                    raw_df = pd.DataFrame(sales_data)
                    data_source = "postgresql"
                    logger.info(f"✓ Retrieved {len(raw_df)} records from PostgreSQL")
                else:
                    logger.warning(f"✗ PostgreSQL data insufficient ({len(sales_data) if sales_data else 0} records)")

            # Fallback to Kaggle dataset for testing/demo
            if raw_df is None:
                logger.info("⚠️ No sufficient data from user sources, falling back to Kaggle demo dataset...")
                raw_df = self.data_pipeline.load_kaggle_dataset()
                data_source = "kaggle"
                
                if raw_df is not None and not raw_df.empty:
                    logger.info(f"✓ Loaded {len(raw_df)} records from Kaggle dataset")

            if raw_df is None or raw_df.empty:
                logger.error("❌ All data sources failed")
                return None

            # STANDARDIZE: Format the data regardless of source
            standardized_df = self._standardize_data_format(raw_df, data_source)
            
            if standardized_df is not None and not standardized_df.empty:
                logger.info(f"✓ Standardized {len(standardized_df)} records from {data_source}")
                return standardized_df
            else:
                logger.error(f"❌ Failed to standardize data from {data_source}")
                return None

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _standardize_data_format(
        self,
        df: pd.DataFrame,
        data_source: str
    ) -> Optional[pd.DataFrame]:
        """
        Standardize data format from any source into consistent schema.
        
        This is the FORMAT ADAPTER - handles all column name variations and
        converts them to the standard format expected by the Data Pipeline.
        
        Standard Format:
        - Date column → 'Order Date' (datetime)
        - Value column → 'Sales' (numeric)
        - Product column → 'Product Name' (string, optional)
        - Quantity column → 'Quantity' (numeric, optional)
        
        Handles column variations:
        - Dates: sale_date, order_date, transaction_date, invoice_date, date, timestamp
        - Values: sales, revenue, amount, total_sales, sales_amount, total_amount
        - Products: product_name, product, item, item_name, sku
        - Quantities: quantity, qty, units, units_sold, items_sold
        
        Args:
            df: Raw DataFrame from any source
            data_source: Source identifier ("postgresql", "kaggle", "external", etc.)
            
        Returns:
            Standardized DataFrame or None if formatting fails
        """
        try:
            logger.info(f"Standardizing data format from source: {data_source}")
            standardized = df.copy()
            
            # Track what we've mapped
            mapped_columns = {}
            
            # 1. MAP DATE COLUMN
            date_col = None
            date_candidates = [
                'Order Date',  # Already standardized
                'order_date',
                'sale_date',
                'sales_date',
                'transaction_date',
                'invoice_date',
                'date',
                'timestamp',
                'created_at',
                'order_created_at'
            ]
            
            for candidate in date_candidates:
                if candidate in standardized.columns:
                    if candidate != 'Order Date':
                        standardized['Order Date'] = standardized[candidate]
                        mapped_columns['date'] = f"{candidate} → Order Date"
                        logger.info(f"Mapped date column: {candidate} → Order Date")
                    date_col = 'Order Date'
                    break
            
            # If no exact match, look for columns containing 'date'
            if date_col is None:
                for col in standardized.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        standardized['Order Date'] = standardized[col]
                        mapped_columns['date'] = f"{col} → Order Date"
                        logger.info(f"Mapped date column (fuzzy): {col} → Order Date")
                        date_col = 'Order Date'
                        break
            
            if date_col is None:
                logger.error("No date column found in data")
                return None
            
            # 2. MAP VALUE COLUMN (Sales/Revenue)
            value_col = None
            value_candidates = [
                'Sales',  # Already standardized
                'sales',
                'revenue',
                'amount',
                'total_sales',
                'sales_amount',
                'total_amount',
                'total_revenue',
                'net_sales',
                'gross_sales'
            ]
            
            for candidate in value_candidates:
                if candidate in standardized.columns:
                    if candidate != 'Sales':
                        standardized['Sales'] = standardized[candidate]
                        mapped_columns['value'] = f"{candidate} → Sales"
                        logger.info(f"Mapped value column: {candidate} → Sales")
                    value_col = 'Sales'
                    break
            
            # If no sales/revenue column, try quantity columns
            if value_col is None:
                quantity_candidates = ['quantity', 'qty', 'units', 'units_sold', 'items_sold', 'Quantity']
                for candidate in quantity_candidates:
                    if candidate in standardized.columns:
                        standardized['Sales'] = standardized[candidate]
                        mapped_columns['value'] = f"{candidate} → Sales"
                        logger.info(f"Mapped quantity to sales: {candidate} → Sales")
                        value_col = 'Sales'
                        break
            
            # Last resort: fuzzy match on numeric columns
            if value_col is None:
                for col in standardized.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['sale', 'revenue', 'amount', 'total', 'value', 'price']):
                        if pd.api.types.is_numeric_dtype(standardized[col]) or pd.api.types.is_object_dtype(standardized[col]):
                            standardized['Sales'] = pd.to_numeric(standardized[col], errors='coerce')
                            mapped_columns['value'] = f"{col} → Sales"
                            logger.info(f"Mapped value column (fuzzy): {col} → Sales")
                            value_col = 'Sales'
                            break
            
            if value_col is None:
                logger.error("No value column found in data")
                return None
            
            # 3. MAP PRODUCT COLUMN (Optional)
            product_candidates = ['Product Name', 'product_name', 'product', 'item', 'item_name', 'sku', 'product_id']
            for candidate in product_candidates:
                if candidate in standardized.columns:
                    if candidate != 'Product Name':
                        standardized['Product Name'] = standardized[candidate]
                        mapped_columns['product'] = f"{candidate} → Product Name"
                        logger.info(f"Mapped product column: {candidate} → Product Name")
                    break
            
            # 4. MAP QUANTITY COLUMN (Optional, if not already used for Sales)
            if 'Quantity' not in standardized.columns:
                quantity_candidates = ['quantity', 'qty', 'units', 'units_sold', 'items_sold']
                for candidate in quantity_candidates:
                    if candidate in standardized.columns and mapped_columns.get('value', '') != f"{candidate} → Sales":
                        standardized['Quantity'] = standardized[candidate]
                        mapped_columns['quantity'] = f"{candidate} → Quantity"
                        logger.info(f"Mapped quantity column: {candidate} → Quantity")
                        break
            
            # 5. VALIDATE STANDARDIZED FORMAT
            if 'Order Date' not in standardized.columns:
                logger.error("Standardization failed: No Order Date column")
                return None
            
            if 'Sales' not in standardized.columns:
                logger.error("Standardization failed: No Sales column")
                return None
            
            # 6. ENSURE CORRECT DATA TYPES
            try:
                standardized['Order Date'] = pd.to_datetime(standardized['Order Date'], errors='coerce')
                standardized['Sales'] = pd.to_numeric(standardized['Sales'], errors='coerce')
            except Exception as e:
                logger.error(f"Failed to convert data types: {e}")
                return None
            
            # Log summary
            logger.info(f"✓ Data standardization complete:")
            for col_type, mapping in mapped_columns.items():
                logger.info(f"  - {col_type}: {mapping}")
            
            # Keep only relevant columns (drop unnecessary ones to reduce memory)
            keep_columns = ['Order Date', 'Sales']
            if 'Product Name' in standardized.columns:
                keep_columns.append('Product Name')
            if 'Quantity' in standardized.columns:
                keep_columns.append('Quantity')
            
            # Add any other columns that might be useful
            for col in standardized.columns:
                if col not in keep_columns and col.lower() in ['store_id', 'store', 'location', 'region', 'category']:
                    keep_columns.append(col)
            
            standardized = standardized[keep_columns]
            
            return standardized
            
        except Exception as e:
            logger.error(f"Error standardizing data format: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def get_data_source_info(self, session_id: str) -> Dict:
        """
        Get information about available data sources for this session.

        Returns:
            Dict with data source status:
            {
                "postgresql_connected": bool,
                "active_source": str,
                "data_quality": str
            }
        """
        return {
            "postgresql_connected": DatabaseManager.has_user_connection(session_id),
            "active_source": self._determine_active_source(session_id),
            "fallback_available": True  # Kaggle dataset always available
        }

    def _determine_active_source(self, session_id: str) -> str:
        """Determine which data source will be used"""
        if DatabaseManager.has_user_connection(session_id):
            return "postgresql"
        else:
            return "kaggle_demo"
    
    def _enrich_product_names(
        self,
        df: pd.DataFrame,
        session_id: str
    ) -> pd.DataFrame:
        """
        Enrich sales data with product names if they don't exist.
        
        Strategy (dynamic):
        1. Check if product names already in data
        2. If not, try to find products table and join
        3. Handle any dataset structure automatically
        
        Args:
            df: Sales data with product IDs
            session_id: Session for data source context
            
        Returns:
            DataFrame with Product Name column populated
        """
        try:
            # Already has product names
            if 'Product Name' in df.columns and df['Product Name'].notna().any():
                logger.info("✓ Product names already in data")
                return df
            
            # Check if we have product IDs to work with
            product_id_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['product_id', 'product', 'item_id', 'sku']):
                    product_id_col = col
                    break
            
            if not product_id_col:
                logger.warning("No product ID column found, using aggregated forecast")
                df['Product Name'] = 'All Products'
                return df
            
            logger.info(f"Found product ID column: {product_id_col}")
            
            # Try to load products table from same data source
            products_df = self._load_products_table(session_id)
            
            if products_df is None:
                logger.warning("No products table found, using product IDs as names")
                df['Product Name'] = df[product_id_col].astype(str)
                return df
            
            # Find matching columns for join
            product_name_col = None
            for col in products_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['product_name', 'name', 'product', 'item_name', 'title']):
                    product_name_col = col
                    break
            
            if not product_name_col:
                logger.warning("No product name column found in products table")
                df['Product Name'] = df[product_id_col].astype(str)
                return df
            
            # Find ID column in products table
            products_id_col = None
            for col in products_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['product_id', 'id', 'sku', 'product']):
                    products_id_col = col
                    break
            
            if not products_id_col:
                logger.warning("No product ID column found in products table")
                df['Product Name'] = df[product_id_col].astype(str)
                return df
            
            # Perform join
            logger.info(f"Joining {product_id_col} with {products_id_col} to get {product_name_col}")
            df = df.merge(
                products_df[[products_id_col, product_name_col]],
                left_on=product_id_col,
                right_on=products_id_col,
                how='left'
            )
            
            # Rename to standard column name
            df['Product Name'] = df[product_name_col].fillna(df[product_id_col].astype(str))
            
            logger.info(f"✓ Enriched {len(df)} records with product names")
            logger.info(f"  Products found: {df['Product Name'].nunique()}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Product enrichment failed: {e}, using product IDs as names")
            if product_id_col and product_id_col in df.columns:
                df['Product Name'] = df[product_id_col].astype(str)
            else:
                df['Product Name'] = 'All Products'
            return df
    
    def _load_products_table(self, session_id: str) -> Optional[pd.DataFrame]:
        """
        Try to load products/items table from the same data source.
        
        Looks for common table names:
        - products.csv, items.csv, catalog.csv
        - products table in PostgreSQL
        
        Returns:
            Products DataFrame or None
        """
        try:
            # Try Kaggle first (since we're using it for demo)
            if not DatabaseManager.has_user_connection(session_id):
                # Try to load products from Kaggle
                import kagglehub
                from kagglehub import KaggleDatasetAdapter
                import os
                
                dataset_path = kagglehub.dataset_download("amangarg08/apple-retail-sales-dataset")
                all_files = os.listdir(dataset_path)
                
                # Look for products table
                product_files = [f for f in all_files if any(
                    keyword in f.lower() for keyword in ['product', 'item', 'catalog', 'sku']
                )]
                
                if product_files:
                    product_file = product_files[0]
                    logger.info(f"Found products table: {product_file}")
                    
                    products_df = kagglehub.load_dataset(
                        KaggleDatasetAdapter.PANDAS,
                        "amangarg08/apple-retail-sales-dataset",
                        product_file
                    )
                    return products_df
            
            # TODO: Add PostgreSQL products table loading
            # else:
            #     products_data = DatabaseManager.get_data(session_id, "products")
            #     if products_data:
            #         return pd.DataFrame(products_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not load products table: {e}")
            return None
    
    def _determine_products_to_forecast(
        self,
        df: pd.DataFrame,
        forecast_mode: str,
        product_filter: Optional[str],
        top_n: int
    ) -> List[str]:
        """
        Determine which products to forecast based on mode and filters.
        
        Args:
            df: Historical data with Product Name column
            forecast_mode: "aggregate", "single", "multi", "top_n"
            product_filter: Product name(s) or pattern
            top_n: Number of products for top_n mode
            
        Returns:
            List of product names to forecast (empty list = aggregate all)
        """
        if forecast_mode == "aggregate":
            return []  # Empty list means aggregate all products
        
        if forecast_mode == "single":
            if not product_filter:
                logger.warning("Single mode requires product_filter, falling back to aggregate")
                return []
            
            # Find matching product (fuzzy match)
            matching_products = df[df['Product Name'].str.contains(
                product_filter, case=False, na=False, regex=False
            )]['Product Name'].unique()
            
            if len(matching_products) == 0:
                logger.warning(f"No products match filter '{product_filter}', falling back to aggregate")
                return []
            
            if len(matching_products) > 1:
                logger.info(f"Multiple products match '{product_filter}': {list(matching_products)}, using first")
            
            return [matching_products[0]]
        
        if forecast_mode == "multi":
            if not product_filter:
                logger.warning("Multi mode requires product_filter, falling back to top_n")
                return self._get_top_n_products(df, top_n)
            
            # Parse comma-separated list
            product_filters = [p.strip() for p in product_filter.split(',')]
            
            matched_products = []
            for filter_term in product_filters:
                matches = df[df['Product Name'].str.contains(
                    filter_term, case=False, na=False, regex=False
                )]['Product Name'].unique()
                matched_products.extend(matches)
            
            if len(matched_products) == 0:
                logger.warning(f"No products match filters '{product_filter}', falling back to top_n")
                return self._get_top_n_products(df, top_n)
            
            # Remove duplicates, keep order
            unique_products = list(dict.fromkeys(matched_products))
            logger.info(f"Matched {len(unique_products)} products for multi-product forecast")
            return unique_products[:top_n]  # Limit to top_n
        
        if forecast_mode == "top_n":
            return self._get_top_n_products(df, top_n)
        
        # Default fallback
        logger.warning(f"Unknown forecast_mode '{forecast_mode}', falling back to aggregate")
        return []
    
    def _get_top_n_products(self, df: pd.DataFrame, n: int) -> List[str]:
        """Get top N products by sales volume"""
        if 'Product Name' not in df.columns or 'Sales' not in df.columns:
            return []
        
        top_products = (
            df.groupby('Product Name')['Sales']
            .sum()
            .sort_values(ascending=False)
            .head(n)
            .index.tolist()
        )
        
        logger.info(f"Top {len(top_products)} products by volume: {top_products}")
        return top_products
    
    def _run_single_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_params: Dict,
        economic_events: List,
        supply_chain_locations: List,
        product_name: str = "All Products"
    ) -> Dict:
        """
        Run forecast for a single product or aggregated data.
        This is the original forecasting pipeline.
        """
        try:
            logger.info(f"Running forecast for: {product_name}")
            
            # Prepare data
            prepared_data = self.data_pipeline.prepare_for_forecasting(
                historical_data,
                freq=forecast_params.get("frequency", "D"),
                remove_outliers=False
            )
            
            # Prepare Prophet format
            prophet_data = self.forecasting_engine.prepare_data_for_prophet(
                prepared_data,
                date_col='Order Date',
                value_col='Sales',
                freq=forecast_params.get("frequency", "D")
            )
            
            # Run forecast
            forecast_result = self.forecasting_engine.forecast_demand(
                historical_data=prepared_data,
                forecast_periods=forecast_params.get("periods", 30),
                frequency=forecast_params.get("frequency", "D"),
                economic_events=economic_events,
                supply_chain_locations=supply_chain_locations
            )
            
            if "error" in forecast_result:
                return forecast_result
            
            # Validate
            validation_result = self.validation_engine.validate_forecast(
                prophet_forecast=forecast_result,
                historical_data=prophet_data,
                economic_events=economic_events,
                tolerance_thresholds={
                    'mae_threshold': 5000.0,
                    'mse_threshold': 25000000.0,
                    'r2_min': 0.5,
                    'mape_threshold': 50.0
                },
                horizon=f'{forecast_params.get("periods", 30)} {forecast_params.get("frequency", "D").replace('W', 'days').replace('M', 'days')}' # Construct horizon string
            )
            
            
            # Combine confidences
            original_confidence = forecast_result["forecast"]["confidence_score"]
            validation_confidence = validation_result.get("confidence_adjustment", 1.0)
            final_confidence = min(original_confidence, validation_confidence)
            
            if original_confidence == 0 and validation_confidence > 0:
                final_confidence = min(validation_confidence, 0.5)
                logger.warning(f"Prophet confidence is 0, capping at 50%")
            
            forecast_result["forecast"]["confidence_score"] = round(final_confidence, 3)
            forecast_result["forecast"]["validation_adjusted"] = True
            forecast_result["forecast"]["confidence_breakdown"] = {
                "prophet_confidence": round(original_confidence, 3),
                "validation_confidence": round(validation_confidence, 3),
                "final_confidence": round(final_confidence, 3),
                "method": "minimum" if original_confidence > 0 else "validation_capped"
            }
            forecast_result["forecast"]["product_name"] = product_name
            forecast_result["validation"] = validation_result
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Single forecast failed for {product_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": "Forecast failed", "message": str(e)}
    
    def _run_multi_product_forecast(
        self,
        historical_data: pd.DataFrame,
        products: List[str],
        forecast_params: Dict,
        economic_events: List,
        supply_chain_locations: List
    ) -> Dict:
        """
        Run forecasts for multiple products in parallel.
        Returns combined results with comparative insights.
        """
        try:
            logger.info(f"Running multi-product forecast for {len(products)} products")
            
            forecasts = {}
            successful_forecasts = []
            failed_products = []
            
            for product in products:
                logger.info(f"Forecasting product: {product}")
                
                # Filter data for this product
                product_data = historical_data[historical_data['Product Name'] == product]
                
                if len(product_data) < 30:
                    logger.warning(f"Skipping {product}: only {len(product_data)} records")
                    failed_products.append({
                        "product": product,
                        "reason": f"Insufficient data ({len(product_data)} records)"
                    })
                    continue
                
                # Run forecast for this product
                product_forecast = self._run_single_forecast(
                    product_data,
                    forecast_params,
                    economic_events,
                    supply_chain_locations,
                    product_name=product
                )
                
                if "error" not in product_forecast:
                    forecasts[product] = product_forecast
                    successful_forecasts.append(product)
                else:
                    failed_products.append({
                        "product": product,
                        "reason": product_forecast.get("message", "Unknown error")
                    })
            
            if len(forecasts) == 0:
                return {
                    "error": "All product forecasts failed",
                    "message": f"Attempted to forecast {len(products)} products, all failed",
                    "failed_products": failed_products
                }
            
            # Aggregate results
            return self._aggregate_multi_product_results(
                forecasts,
                successful_forecasts,
                failed_products,
                forecast_params
            )
            
        except Exception as e:
            logger.error(f"Multi-product forecast failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": "Multi-product forecast failed", "message": str(e)}
    
    def _aggregate_multi_product_results(
        self,
        forecasts: Dict,
        successful_products: List[str],
        failed_products: List[Dict],
        forecast_params: Dict
    ) -> Dict:
        """
        Aggregate multiple product forecasts into single response.
        """
        try:
            # Calculate summary statistics
            total_demand = sum(
                f["key_insights"]["total_forecast_demand"]
                for f in forecasts.values()
            )
            
            avg_confidence = sum(
                f["forecast"]["confidence_score"]
                for f in forecasts.values()
            ) / len(forecasts)
            
            # Rank products by confidence
            products_by_confidence = sorted(
                forecasts.items(),
                key=lambda x: x[1]["forecast"]["confidence_score"],
                reverse=True
            )
            
            high_confidence = [p for p, f in products_by_confidence if f["forecast"]["confidence_score"] >= 0.6]
            low_confidence = [p for p, f in products_by_confidence if f["forecast"]["confidence_score"] < 0.5]
            
            # Rank by demand
            products_by_demand = sorted(
                forecasts.items(),
                key=lambda x: x[1]["key_insights"]["average_daily_demand"],
                reverse=True
            )
            
            # Build combined response
            return {
                "success": True,
                "forecast_mode": "multi",
                "products_analyzed": len(successful_products),
                "products_failed": len(failed_products),
                "forecasts": {
                    product: {
                        "periods": f["forecast"]["periods"],
                        "confidence_score": f["forecast"]["confidence_score"],
                        "confidence_breakdown": f["forecast"]["confidence_breakdown"],
                        "trend": f["forecast"]["trend"],
                        "seasonality_detected": f["forecast"]["seasonality_detected"],
                        "peak_demand": f["key_insights"]["peak_demand_value"],
                        "average_demand": f["key_insights"]["average_daily_demand"],
                        "total_forecast": f["key_insights"]["total_forecast_demand"]
                    }
                    for product, f in forecasts.items()
                },
                "summary": {
                    "total_demand_forecast": round(total_demand, 2),
                    "average_confidence": round(avg_confidence, 3),
                    "high_confidence_products": high_confidence,
                    "low_confidence_products": low_confidence,
                    "top_products_by_demand": [p for p, _ in products_by_demand[:5]],
                    "successful_products": successful_products,
                    "failed_products": failed_products
                },
                "comparative_insights": self._generate_comparative_insights(forecasts),
                "chart_data": self._build_multi_product_chart_data(forecasts, forecast_params),
                "metadata": {
                    "forecast_generated_at": datetime.now().isoformat(),
                    "forecast_periods": forecast_params.get("periods", 30),
                    "frequency": forecast_params.get("frequency", "D")
                }
            }
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            # Fallback: return first product's forecast
            first_product = list(forecasts.keys())[0]
            return forecasts[first_product]
    
    def _generate_comparative_insights(self, forecasts: Dict) -> Dict:
        """Generate comparative insights across products."""
        growth_leaders = []
        stable_performers = []
        declining = []
        
        for product, forecast in forecasts.items():
            trend = forecast["forecast"]["trend"]
            if "increas" in trend.lower() or "grow" in trend.lower():
                growth_leaders.append(product)
            elif "declin" in trend.lower() or "decreas" in trend.lower():
                declining.append(product)
            else:
                stable_performers.append(product)
        
        return {
            "growth_leaders": growth_leaders,
            "stable_performers": stable_performers,
            "declining": declining
        }
    
    def _build_multi_product_chart_data(self, forecasts: Dict, params: Dict) -> Dict:
        """Build chart data for multiple products."""
        # Use dates from first forecast
        first_forecast = list(forecasts.values())[0]
        dates = [p["date"] for p in first_forecast["forecast"]["periods"]]
        
        series = {}
        for product, forecast in forecasts.items():
            series[product] = [p["predicted_demand"] for p in forecast["forecast"]["periods"]]
        
        return {
            "type": "multi_series",
            "dates": dates,
            "series": series
        }
