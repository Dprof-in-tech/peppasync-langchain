"""
Forecasting Engine - Prophet-based demand forecasting
Part of the CONTEXT LAYER in the demand forecast architecture
"""

from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ForecastingEngine:
    """
    Handles demand forecasting using Facebook Prophet.

    This is the core forecasting component that:
    1. Prepares time series data for Prophet
    2. Integrates economic events as holidays (from Tavily-enriched settings)
    3. Generates forecasts with confidence intervals
    4. Detects trends and seasonality
    5. Calculates supply chain ordering alerts

    Part of the CONTEXT LAYER in the demand forecast system.
    """

    def __init__(self):
        self.model = None
        self.last_forecast = None

    def prepare_data_for_prophet(
        self,
        df: pd.DataFrame,
        date_col: str = 'Order Date',
        value_col: str = 'Sales',
        freq: str = 'D',
        product_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform raw sales data into Prophet's required format.

        Prophet requires:
        - 'ds' column: datetime dates
        - 'y' column: numeric values to forecast

        Args:
            df: Raw sales data (from external sources or Kaggle)
            date_col: Name of the date column
            value_col: Name of the value column (sales/revenue)
            freq: Frequency for date range ('D'=daily, 'W'=weekly, 'M'=monthly)
            product_filter: Optional product name to filter by

        Returns:
            DataFrame in Prophet format with 'ds' and 'y' columns
        """
        try:
            # Make a copy to avoid modifying original
            df_copy = df.copy()

            # Filter by product if specified
            if product_filter and 'Product Name' in df_copy.columns:
                df_copy = df_copy[
                    df_copy['Product Name'].str.contains(product_filter, case=False, na=False)
                ]
                logger.info(f"Filtered to {len(df_copy)} records for product: {product_filter}")

            # Ensure date column is datetime
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

            # Remove any rows with invalid dates
            df_copy = df_copy.dropna(subset=[date_col])

            # Aggregate by date and sum sales
            prophet_df = df_copy.groupby(date_col)[value_col].sum().reset_index()

            # Rename columns to Prophet's required format
            prophet_df.columns = ['ds', 'y']

            # Sort by date
            prophet_df = prophet_df.sort_values('ds')

            # Fill missing dates with zero sales
            date_range = pd.date_range(
                start=prophet_df['ds'].min(),
                end=prophet_df['ds'].max(),
                freq=freq
            )

            # Create complete date range dataframe
            complete_df = pd.DataFrame({'ds': date_range})
            prophet_df = complete_df.merge(prophet_df, on='ds', how='left')
            prophet_df['y'] = prophet_df['y'].fillna(0)

            logger.info(f"Prepared {len(prophet_df)} data points for Prophet")
            return prophet_df

        except Exception as e:
            logger.error(f"Error preparing data for Prophet: {str(e)}")
            raise

    def add_economic_events(
        self,
        events: List[Dict],
        future_only: bool = False
    ) -> pd.DataFrame:
        """
        Convert economic events from user settings (enriched by Tavily) into Prophet holidays format.

        Args:
            events: List of economic events from ForecastSettingsManager:
                [
                    {
                        'name': 'Black Friday',
                        'date': '2024-11-29',
                        'description': '...',
                        'impact_days_before': 7,  # From Tavily
                        'impact_days_after': 3    # From Tavily
                    }
                ]
            future_only: If True, only include future events

        Returns:
            DataFrame in Prophet holidays format
        """
        if not events:
            return pd.DataFrame()

        holidays_list = []
        current_date = pd.Timestamp.now()

        for event in events:
            try:
                event_date = pd.to_datetime(event['date'])

                # Skip past events if future_only is True
                if future_only and event_date < current_date:
                    continue

                # Use Tavily-calculated impact days, or defaults
                impact_before = event.get('impact_days_before', 7)
                impact_after = event.get('impact_days_after', 3)

                holidays_list.append({
                    'holiday': event['name'],
                    'ds': event_date,
                    'lower_window': -impact_before,
                    'upper_window': impact_after,
                })
            except Exception as e:
                logger.warning(f"Error processing event {event.get('name')}: {str(e)}")
                continue

        if holidays_list:
            holidays_df = pd.DataFrame(holidays_list)
            logger.info(f"Added {len(holidays_df)} economic events to forecast")
            return holidays_df

        return pd.DataFrame()

    def forecast_demand(
        self,
        historical_data: pd.DataFrame,
        forecast_periods: int = 30,
        frequency: str = 'D',
        economic_events: Optional[List[Dict]] = None,
        product_filter: Optional[str] = None,
        supply_chain_locations: Optional[List[Dict]] = None,
        auto_tune: bool = True
    ) -> Dict:
        """
        Main forecasting function - generates demand forecast using Prophet.

        This is called by the CONTEXT LAYER with all aggregated data.

        Args:
            historical_data: Sales data (from external sources, Kaggle, or PostgreSQL)
            forecast_periods: Number of periods to forecast ahead (from user prompt)
            frequency: 'D' (daily), 'W' (weekly), 'M' (monthly) (from user prompt)
            economic_events: List of events from settings (Tavily-enriched)
            product_filter: Specific product to forecast
            supply_chain_locations: Array of supply chain timelines (from settings)
            auto_tune: If True, automatically tune hyperparameters (default: True)

        Returns:
            Dict with forecast results matching specified JSON format
        """
        try:
            # Validate minimum data requirement
            if len(historical_data) < 30:
                return {
                    "error": "Insufficient historical data",
                    "message": "At least 30 days of historical data required for accurate forecasting",
                    "min_required": 30,
                    "available": len(historical_data)
                }

            # Step 1: Prepare data for Prophet
            prophet_df = self.prepare_data_for_prophet(
                historical_data,
                date_col='Order Date',
                value_col='Sales',
                freq=frequency,
                product_filter=product_filter
            )

            # Step 2: Auto-tune hyperparameters if enabled and sufficient data
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.8
            }

            if auto_tune and len(prophet_df) >= 100:
                logger.info("Auto-tuning Prophet hyperparameters (fast mode: 18 combinations)...")
                try:
                    tuning_result = self.tune_hyperparameters(
                        historical_data=prophet_df,
                        economic_events=economic_events,
                        test_size=0.2,
                        param_grid={
                            # Reduced grid for faster tuning (18 combinations instead of 135)
                            'changepoint_prior_scale': [0.01, 0.05, 0.1],
                            'seasonality_prior_scale': [5, 10],
                            'holidays_prior_scale': [5, 10],
                            'seasonality_mode': ['additive', 'multiplicative'],
                            'changepoint_range': [0.8]  # Use default
                        }
                    )
                    # 3 × 2 × 2 × 2 × 1 = 24 combinations (balanced speed/accuracy)
                    best_params = tuning_result['best_params']
                    logger.info(f"✓ Auto-tuning complete. Best R²: {tuning_result['best_r2']:.3f}")
                    logger.info(f"✓ Using optimized parameters: {best_params}")
                except Exception as tune_error:
                    logger.warning(f"Auto-tuning failed: {tune_error}. Using default parameters.")
            else:
                if len(prophet_df) < 100:
                    logger.info(f"Insufficient data for auto-tuning ({len(prophet_df)} < 100). Using default parameters.")
                else:
                    logger.info("Auto-tuning disabled. Using default parameters.")

            # Step 3: Initialize Prophet model with best/default parameters
            holidays_df = self.add_economic_events(economic_events) if economic_events else None

            model = Prophet(
                holidays=holidays_df if holidays_df is not None and not holidays_df.empty else None,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=best_params['changepoint_prior_scale'],
                seasonality_prior_scale=best_params['seasonality_prior_scale'],
                holidays_prior_scale=best_params['holidays_prior_scale'],
                seasonality_mode=best_params['seasonality_mode'],
                changepoint_range=best_params['changepoint_range']
            )

            # Step 4: Fit the model
            logger.info(f"Training Prophet model on {len(prophet_df)} data points with tuned parameters")
            model.fit(prophet_df)

            # Step 5: Create future dataframe
            future = model.make_future_dataframe(
                periods=forecast_periods,
                freq=frequency
            )

            # Step 6: Generate forecast
            forecast = model.predict(future)

            # Step 7: Format output for VALIDATION and LLM CALL layers
            forecast_result = self._format_forecast_output(
                forecast,
                prophet_df,
                forecast_periods,
                model,
                supply_chain_locations
            )

            # Add tuning info to metadata
            forecast_result["metadata"]["auto_tuned"] = auto_tune and len(prophet_df) >= 100
            forecast_result["metadata"]["prophet_parameters"] = best_params

            # Store for validation layer
            self.model = model
            self.last_forecast = forecast

            return forecast_result

        except Exception as e:
            logger.error(f"Forecasting error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": "Forecasting failed",
                "message": str(e)
            }

    def _format_forecast_output(
        self,
        forecast: pd.DataFrame,
        historical: pd.DataFrame,
        periods: int,
        model: Prophet,
        supply_chain_locations: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Format Prophet output into the specified JSON structure.

        This output goes to:
        1. VALIDATION CHECK layer (for validation)
        2. LLM CALL layer (for insights generation)
        3. ADVISOR layer (for recommendations)

        Returns:
            Structured forecast data matching the system design spec
        """
        # Get only future predictions (not historical fit)
        future_forecast = forecast.tail(periods)

        # Calculate trend direction
        trend_start = future_forecast['trend'].iloc[0]
        trend_end = future_forecast['trend'].iloc[-1]
        trend_change_pct = ((trend_end - trend_start) / trend_start) * 100 if trend_start != 0 else 0

        if trend_change_pct > 5:
            trend_direction = "increasing"
        elif trend_change_pct < -5:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        # Detect seasonality strength
        if 'yearly' in forecast.columns:
            seasonality_strength = float(forecast['yearly'].std() / forecast['yhat'].std()) if forecast['yhat'].std() != 0 else 0.0
            seasonality_detected = bool(seasonality_strength > 0.1)
        else:
            seasonality_detected = False
            seasonality_strength = 0.0

        # Calculate confidence score
        # Based on prediction interval width (tighter = more confident)
        avg_interval_width = float((future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean())
        avg_prediction = float(future_forecast['yhat'].mean())

        if avg_prediction > 0:
            confidence_score = float(max(0, min(1, 1 - (avg_interval_width / (2 * avg_prediction)))))
        else:
            confidence_score = 0.5

        # Format periods for output
        periods_list = []
        for _, row in future_forecast.iterrows():
            periods_list.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "predicted_demand": round(max(0, row['yhat']), 2),  # Ensure non-negative
                "lower_bound": round(max(0, row['yhat_lower']), 2),
                "upper_bound": round(max(0, row['yhat_upper']), 2)
            })

        # Prepare chart data
        chart_dates = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        chart_values = [round(max(0, v), 2) for v in future_forecast['yhat'].tolist()]
        chart_lower = [round(max(0, v), 2) for v in future_forecast['yhat_lower'].tolist()]
        chart_upper = [round(max(0, v), 2) for v in future_forecast['yhat_upper'].tolist()]

        # Identify peak demand periods (for recommendations)
        peak_demand_date = future_forecast.loc[future_forecast['yhat'].idxmax(), 'ds']
        peak_demand_value = float(future_forecast['yhat'].max())

        # Calculate supply chain ordering alerts for each location
        supply_chain_alerts = []
        if supply_chain_locations:
            supply_chain_alerts = self._generate_supply_chain_alerts(
                future_forecast,
                supply_chain_locations
            )

        # Build final output (ensure all numpy types are converted to Python native types)
        result = {
            "forecast": {
                "periods": periods_list,
                "confidence_score": float(round(confidence_score, 3)),
                "trend": str(trend_direction),
                "trend_change_percent": float(round(trend_change_pct, 2)),
                "seasonality_detected": bool(seasonality_detected),
                "seasonality_strength": float(round(seasonality_strength, 3)),
                "model_used": "Prophet"
            },
            "chart_data": {
                "type": "line_with_confidence",
                "dates": chart_dates,
                "predicted": chart_values,
                "lower_bound": chart_lower,
                "upper_bound": chart_upper
            },
            "key_insights": {
                "peak_demand_date": peak_demand_date.strftime('%Y-%m-%d'),
                "peak_demand_value": float(round(peak_demand_value, 2)),
                "average_daily_demand": float(round(future_forecast['yhat'].mean(), 2)),
                "total_forecast_demand": float(round(future_forecast['yhat'].sum(), 2))
            },
            "supply_chain_alerts": supply_chain_alerts,
            "metadata": {
                "forecast_generated_at": datetime.now().isoformat(),
                "historical_data_points": int(len(historical)),
                "forecast_periods": int(periods),
                "frequency": "daily" if periods < 90 else "weekly"
            }
        }

        return result

    def _generate_supply_chain_alerts(
        self,
        future_forecast: pd.DataFrame,
        supply_chain_locations: List[Dict]
    ) -> List[Dict]:
        """
        Generate ordering alerts based on supply chain timelines.

        For each location, calculates when to order based on:
        - Manufacturing days
        - Logistics days
        - Possible delay days

        Args:
            future_forecast: Prophet forecast DataFrame
            supply_chain_locations: Array of supply chain location configs

        Returns:
            List of ordering alerts with dates and quantities
        """
        alerts = []

        for location in supply_chain_locations:
            # Calculate total lead time for this location
            total_days = (
                location.get('manufacturing_days', 0) +
                location.get('logistics_days', 0) +
                location.get('possible_delay_days', 0)
            )

            if total_days == 0:
                continue

            # For each forecast period, calculate when to order
            for _, row in future_forecast.iterrows():
                demand_date = row['ds']
                order_by_date = demand_date - timedelta(days=total_days)

                # Only include future ordering dates
                if order_by_date >= pd.Timestamp.now():
                    alerts.append({
                        "location": location.get('name', 'Unknown'),
                        "demand_date": demand_date.strftime('%Y-%m-%d'),
                        "order_by_date": order_by_date.strftime('%Y-%m-%d'),
                        "quantity_needed": round(max(0, row['yhat']), 2),
                        "lead_time_days": total_days,
                        "notes": location.get('notes', '')
                    })

        # Sort by order_by_date
        alerts.sort(key=lambda x: x['order_by_date'])

        logger.info(f"Generated {len(alerts)} supply chain ordering alerts")
        return alerts

    def tune_hyperparameters(
        self,
        historical_data: pd.DataFrame,
        economic_events: Optional[List[Dict]] = None,
        test_size: float = 0.2,
        param_grid: Optional[Dict] = None
    ) -> Dict:
        """
        Fine-tune Prophet hyperparameters using grid search on train-test split.

        Based on the approach from "Time Series Forecasting with Facebook's Prophet" article.

        Args:
            historical_data: Prophet-formatted data (ds, y)
            economic_events: Optional economic events
            test_size: Proportion of data for testing (default 0.2)
            param_grid: Optional custom parameter grid, defaults to:
                {
                    'changepoint_prior_scale': [0.01, 0.05, 0.1],
                    'seasonality_prior_scale': [1, 5, 10],
                    'holidays_prior_scale': [1, 5, 10],
                    'seasonality_mode': ['additive', 'multiplicative'],
                    'changepoint_range': [0.7, 0.8, 0.9]
                }

        Returns:
            Dict with best parameters and performance metrics
        """
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.01, 0.05, 0.1],
                'seasonality_prior_scale': [1, 5, 10],
                'holidays_prior_scale': [1, 5, 10],
                'seasonality_mode': ['additive', 'multiplicative'],
                'changepoint_range': [0.7, 0.8, 0.9]
            }

        # Split data
        split_idx = int(len(historical_data) * (1 - test_size))
        train = historical_data.iloc[:split_idx].copy()
        test = historical_data.iloc[split_idx:].copy()

        logger.info(f"Hyperparameter tuning: {len(train)} train, {len(test)} test")

        # Prepare holidays
        holidays_df = self.add_economic_events(economic_events) if economic_events else None

        best_params = None
        best_mse = float('inf')
        best_r2 = -float('inf')
        results = []

        import itertools
        
        # Generate all parameter combinations
        param_combinations = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]

        total_combinations = len(param_combinations)
        logger.info(f"Testing {total_combinations} parameter combinations...")

        for idx, params in enumerate(param_combinations, 1):
            try:
                # Build model with these parameters
                model = Prophet(
                    holidays=holidays_df if holidays_df is not None and not holidays_df.empty else None,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    interval_width=0.95,
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    holidays_prior_scale=params.get('holidays_prior_scale', 10),
                    seasonality_mode=params['seasonality_mode'],
                    changepoint_range=params['changepoint_range']
                )

                # Fit on training data
                model.fit(train)

                # Predict on test data
                future = model.make_future_dataframe(periods=len(test), freq='D')
                forecast = model.predict(future)
                predictions = forecast.tail(len(test))['yhat'].values
                actuals = test['y'].values

                # Calculate metrics
                mse = float(np.mean((actuals - predictions) ** 2))
                mae = float(np.mean(np.abs(actuals - predictions)))
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

                results.append({
                    'params': params,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                })

                # Track best by MSE (prioritize minimizing large errors)
                if mse < best_mse:
                    best_mse = mse
                    best_r2 = r2
                    best_params = params

                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{total_combinations} - Best R²: {best_r2:.3f}")

            except Exception as e:
                logger.warning(f"Failed to test params {params}: {str(e)}")
                continue

        logger.info(f"Hyperparameter tuning complete. Best R²: {best_r2:.3f}, MSE: {best_mse:.2f}")
        logger.info(f"Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_mse': best_mse,
            'best_r2': best_r2,
            'all_results': results
        }

    def get_model_components(self) -> Dict:
        """
        Extract Prophet model components for analysis.
        Used by validation layer to understand forecast behavior.

        Returns:
            Dict with trend, seasonality, and changepoint information
        """
        if not self.model or self.last_forecast is None:
            return {}

        return {
            "has_trend": True,
            "has_yearly_seasonality": self.model.yearly_seasonality,
            "has_weekly_seasonality": self.model.weekly_seasonality,
            "changepoints": self.model.changepoints.tolist() if hasattr(self.model, 'changepoints') else []
        }
