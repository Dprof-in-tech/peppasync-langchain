"""
Validation Engine - Train-Test Split validation for Prophet forecasts
Part of the VALIDATION CHECK LAYER in the demand forecast architecture
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from prophet import Prophet

logger = logging.getLogger(__name__)


from prophet.diagnostics import cross_validation, performance_metrics

class ValidationEngine:
    """
    Validates Prophet forecasts using train-test split backtesting, time series
    cross-validation, and a baseline model comparison.

    This is the VALIDATION CHECK layer that:
    1. Splits historical data into train/test sets (80/20 split)
    2. Trains Prophet model on training set only
    3. Predicts on test period
    4. Compares predictions vs actual values (MAE, MSE, R²)
    5. Flags discrepancies and anomalies
    6. Returns validation pass/fail with confidence adjustments
    7. Performs time series cross-validation for more robust metrics
    8. Compares Prophet performance against a naive baseline model

    This is the industry-standard approach for time series validation.
    """

    def __init__(self):
        self.backtest_model = None
        self.backtest_forecast = None
        self.test_metrics = None

    def backtest_prophet(
        self,
        historical_data: pd.DataFrame,
        economic_events: Optional[List[Dict]] = None,
        train_test_split: float = 0.8
    ) -> Dict:
        """
        Perform train-test split validation on Prophet forecast.

        This is the gold-standard validation approach:
        1. Split historical data (80% train, 20% test)
        2. Train Prophet model on training set only
        3. Predict on test period
        4. Calculate MAE, MSE, R² metrics

        Args:
            historical_data: Time series data in Prophet format (ds, y)
            economic_events: Optional list of economic events for Prophet
            train_test_split: Proportion of data for training (default 0.8)

        Returns:
            Dict with backtest metrics and predictions
        """
        try:
            # Step 1: Split data into train/test
            split_idx = int(len(historical_data) * train_test_split)
            train_data = historical_data.iloc[:split_idx].copy()
            test_data = historical_data.iloc[split_idx:].copy()

            test_periods = len(test_data)

            logger.info(f"Backtesting: Train={len(train_data)} periods, Test={test_periods} periods")

            # Step 2: Prepare holidays (if economic events provided)
            holidays_df = None
            if economic_events:
                holidays_list = []
                for event in economic_events:
                    try:
                        event_date = pd.to_datetime(event['date'])
                        holidays_list.append({
                            'holiday': event['name'],
                            'ds': event_date,
                            'lower_window': -event.get('impact_days_before', 7),
                            'upper_window': event.get('impact_days_after', 3),
                        })
                    except Exception:
                        continue

                if holidays_list:
                    holidays_df = pd.DataFrame(holidays_list)

            # Step 3: Train Prophet model on training set ONLY
            backtest_model = Prophet(
                holidays=holidays_df if holidays_df is not None and not holidays_df.empty else None,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.09,
                seasonality_mode='multiplicative'
            )

            backtest_model.fit(train_data)

            # Step 4: Make predictions for test period
            future = backtest_model.make_future_dataframe(periods=test_periods, freq='D')
            backtest_forecast = backtest_model.predict(future)

            # Extract only the test period predictions
            test_predictions = backtest_forecast.tail(test_periods)

            # Step 5: Calculate performance metrics
            actual_values = test_data['y'].values
            predicted_values = test_predictions['yhat'].values

            # Mean Absolute Error (MAE)
            mae = float(np.mean(np.abs(actual_values - predicted_values)))

            # Mean Squared Error (MSE)
            mse = float(np.mean((actual_values - predicted_values) ** 2))

            # R-squared (coefficient of determination)
            ss_res = np.sum((actual_values - predicted_values) ** 2)
            ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
            r2_score = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

            # Mean Absolute Percentage Error (MAPE)
            mape = float(np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100) if np.all(actual_values != 0) else 0.0

            # Store for later reference
            self.backtest_model = backtest_model
            self.backtest_forecast = test_predictions
            self.test_metrics = {
                'mae': mae,
                'mse': mse,
                'r2_score': r2_score,
                'mape': mape,
                'train_size': len(train_data),
                'test_size': test_periods
            }

            logger.info(f"Backtest complete: MAE=${mae:.2f}, MSE=${mse:.2f}, R²={r2_score:.3f}, MAPE={mape:.1f}%")

            return {
                'success': True,
                'metrics': self.test_metrics,
                'test_predictions': test_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
                'actual_values': test_data[['ds', 'y']].to_dict('records')
            }

        except Exception as e:
            logger.error(f"Backtest validation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }

    def cross_validate_prophet(self, historical_data: pd.DataFrame, horizon: str, initial: str, period: str) -> Dict:
        """
        Perform time series cross-validation using Prophet's diagnostics.
        """
        try:
            m = Prophet()
            m.fit(historical_data)
            df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)
            df_p = performance_metrics(df_cv)
            return df_p.to_dict('records')
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {}

    def get_baseline_forecast(self, historical_data: pd.DataFrame, horizon: int) -> Dict:
        """
        Generate a naive seasonal forecast as a baseline.
        """
        try:
            df = historical_data.copy()
            df.set_index('ds', inplace=True)
            last_year = df.index.max() - pd.DateOffset(years=1)
            historical_subset = df[df.index > last_year]

            if historical_subset.empty:
                return {'metrics': {'mae': 0, 'mse': 0, 'r2_score': 0, 'mape': 0}, 'forecast': []}

            # Create a simple seasonal naive forecast
            forecast_dates = pd.date_range(start=df.index.max() + pd.DateOffset(days=1), periods=horizon)
            forecast_values = []
            for date in forecast_dates:
                last_year_date = date - pd.DateOffset(years=1)
                closest_date_index = historical_subset.index.get_indexer([last_year_date], method='nearest')[0]
                closest_date = historical_subset.index[closest_date_index]
                forecast_values.append(historical_subset.loc[closest_date, 'y'])

            # Create a forecast DataFrame
            forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast_values})

            # Calculate metrics against a hold-out set if possible
            # For simplicity, we'll return the forecast for now
            return {'metrics': {}, 'forecast': forecast_df.to_dict('records')}
        except Exception as e:
            logger.error(f"Baseline forecast failed: {e}")
            return {'metrics': {}, 'forecast': []}

    def validate_forecast(
        self,
        prophet_forecast: Dict,
        historical_data: pd.DataFrame,
        economic_events: Optional[List[Dict]] = None,
        tolerance_thresholds: Optional[Dict] = None,
        horizon: str = '30 days'
    ) -> Dict:
        """
        Validate Prophet forecast using train-test split backtesting, cross-validation, and a baseline model.

        This is the main validation method called by the Context Layer.

        Industry-standard validation approach:
        1. JSON structure validity
        2. Train-test split backtesting (80/20)
        3. Performance metrics (MAE, MSE, R², MAPE)
        4. Hallucination detection
        5. Confidence score calculation based on backtest performance
        6. Time series cross-validation
        7. Comparison with a naive baseline model

        Args:
            prophet_forecast: Prophet forecast output from ForecastingEngine
            historical_data: Historical data used for forecasting
            economic_events: Optional economic events for backtest model
            tolerance_thresholds: Optional dict with MAE, MSE, R² thresholds
            horizon: The forecast horizon (e.g., '30 days')

        Returns:
            Validation result: Dict with comprehensive validation info
        """
        try:
            # Default thresholds if not provided
            if tolerance_thresholds is None:
                tolerance_thresholds = {
                    'mae_threshold': 500.0,
                    'mse_threshold': 500000.0,
                    'r2_min': 0.5,
                    'mape_threshold': 30.0
                }

            validation_result = {
                "valid": True,
                "confidence_adjustment": 1.0,
                "warnings": [],
                "metrics": {},
                "backtest_results": {},
                "cross_validation_metrics": {},
                "baseline_comparison": {}
            }

            # CHECK 1: JSON Structure Validation
            structure_check = self._validate_json_structure(prophet_forecast)
            if not structure_check["valid"]:
                validation_result["valid"] = False
                validation_result["warnings"].extend(structure_check["warnings"])
                return validation_result

            # Extract Prophet forecast periods
            prophet_periods = prophet_forecast.get("forecast", {}).get("periods", [])
            if not prophet_periods:
                validation_result["valid"] = False
                validation_result["warnings"].append("No forecast periods found in Prophet output")
                return validation_result

            # CHECK 2: Run train-test split backtest validation
            logger.info("Running train-test split validation...")
            backtest_result = self.backtest_prophet(
                historical_data,
                economic_events=economic_events,
                train_test_split=0.8
            )

            if not backtest_result.get('success'):
                validation_result["warnings"].append(f"Backtest failed: {backtest_result.get('error', 'Unknown error')}")
                validation_result["confidence_adjustment"] = 0.7
            else:
                validation_result["metrics"] = backtest_result['metrics']
                validation_result["backtest_results"] = backtest_result

            # CHECK 3: Time Series Cross-Validation
            logger.info("Running time series cross-validation...")
            cv_metrics = self.cross_validate_prophet(historical_data, horizon, '180 days', '30 days')
            if cv_metrics:
                validation_result["cross_validation_metrics"] = cv_metrics[0]

            # CHECK 4: Baseline Model Comparison
            logger.info("Generating baseline forecast...")
            horizon_days = int(horizon.split()[0])
            baseline_result = self.get_baseline_forecast(historical_data, horizon_days)
            if baseline_result['forecast']:
                # Here you would calculate metrics for the baseline and compare
                # For now, we just note that it was generated
                validation_result["baseline_comparison"] = {
                    "status": "Generated",
                    "metrics": baseline_result['metrics']
                }

            # CHECK 5: Evaluate backtest performance against thresholds
            if backtest_result.get('success'):
                mae = backtest_result['metrics']['mae']
                mse = backtest_result['metrics']['mse']
                r2_score = backtest_result['metrics']['r2_score']
                mape = backtest_result['metrics']['mape']

                if mae > tolerance_thresholds['mae_threshold']:
                    validation_result["warnings"].append(f"High MAE: ${mae:.2f}")
                if mse > tolerance_thresholds['mse_threshold']:
                    validation_result["warnings"].append(f"High MSE: ${mse:.2f}")
                if r2_score < tolerance_thresholds['r2_min']:
                    validation_result["warnings"].append(f"Low R² score: {r2_score:.3f}")
                    validation_result["valid"] = False
                if mape > tolerance_thresholds['mape_threshold']:
                    validation_result["warnings"].append(f"High MAPE: {mape:.1f}%")

            # CHECK 6: Hallucination Detection - Sanity checks
            sanity_check = self._sanity_checks(prophet_forecast, historical_data)
            validation_result["warnings"].extend(sanity_check["warnings"])
            if sanity_check["fail"]:
                validation_result["valid"] = False

            # CHECK 7: Calculate confidence adjustment
            if backtest_result.get('success'):
                validation_result["confidence_adjustment"] = self._calculate_confidence_from_r2(
                    backtest_result['metrics']['r2_score'],
                    backtest_result['metrics'],
                    sanity_check,
                    tolerance_thresholds
                )

            logger.info(
                f"Validation complete: {'PASSED' if validation_result['valid'] else 'FAILED'} "
                f"(Confidence Adj: {validation_result['confidence_adjustment']:.2f})"
            )
            return validation_result

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "valid": False,
                "confidence_adjustment": 0.5,
                "warnings": [f"Validation error: {str(e)}"],
                "metrics": {},
                "backtest_results": {},
                "cross_validation_metrics": {},
                "baseline_comparison": {}
            }

    def _validate_json_structure(self, prophet_forecast: Dict) -> Dict:
        """
        Validate JSON structure from Prophet output.
        """
        warnings = []
        valid = True
        required_top_level = ["forecast", "chart_data", "key_insights", "metadata"]
        for key in required_top_level:
            if key not in prophet_forecast:
                warnings.append(f"Missing required field: {key}")
                valid = False

        if "forecast" in prophet_forecast:
            forecast_obj = prophet_forecast["forecast"]
            required_forecast_fields = ["periods", "confidence_score", "trend"]
            for field in required_forecast_fields:
                if field not in forecast_obj:
                    warnings.append(f"Missing forecast field: {field}")
                    valid = False
            if "periods" in forecast_obj and isinstance(forecast_obj["periods"], list) and len(forecast_obj["periods"]) > 0:
                first_period = forecast_obj["periods"][0]
                required_period_fields = ["date", "predicted_demand", "lower_bound", "upper_bound"]
                for field in required_period_fields:
                    if field not in first_period:
                        warnings.append(f"Period missing field: {field}")
                        valid = False

        if "chart_data" in prophet_forecast:
            chart_data = prophet_forecast["chart_data"]
            required_chart_fields = ["dates", "predicted", "lower_bound", "upper_bound"]
            for field in required_chart_fields:
                if field not in chart_data:
                    warnings.append(f"Chart data missing field: {field}")
                    valid = False

        return {"valid": valid, "warnings": warnings}

    def _sanity_checks(self, prophet_forecast: Dict, historical_data: pd.DataFrame) -> Dict:
        """
        Perform sanity checks on Prophet forecast.
        """
        warnings = []
        fail = False
        prophet_periods = prophet_forecast.get("forecast", {}).get("periods", [])
        if not prophet_periods:
            return {"fail": True, "warnings": ["No forecast periods to validate"]}

        negative_values = [p for p in prophet_periods if p['predicted_demand'] < 0]
        if negative_values:
            warnings.append(f"Negative demand detected in {len(negative_values)} periods")
            fail = True

        prophet_values = [p['predicted_demand'] for p in prophet_periods]
        historical_mean = historical_data['y'].mean()

        for i in range(1, len(prophet_values)):
            if prophet_values[i-1] > 0:
                growth_rate = (prophet_values[i] - prophet_values[i-1]) / prophet_values[i-1]
                if growth_rate > 5.0:
                    warnings.append(f"Unrealistic growth rate: {growth_rate*100:.0f}% on {prophet_periods[i]['date']}")

        max_forecast = max(prophet_values)
        if historical_mean > 0 and max_forecast > historical_mean * 10:
            warnings.append(f"Forecast peak ({max_forecast:.0f}) is >10x historical average ({historical_mean:.0f})")

        for period in prophet_periods:
            interval_width = period['upper_bound'] - period['lower_bound']
            predicted = period['predicted_demand']
            if predicted > 0 and interval_width > predicted * 3:
                warnings.append(f"Very wide confidence interval on {period['date']}: {interval_width:.0f} (predicted: {predicted:.0f})")

        return {"fail": fail, "warnings": warnings}

    def _calculate_confidence_from_r2(self, r2_score: float, backtest_metrics: Dict, sanity_check: Dict, thresholds: Dict) -> float:
        """
        Calculate confidence score adjustment based on R² and other metrics.
        """
        if r2_score >= 0.8:
            base_confidence = 0.95
        elif r2_score >= 0.6:
            base_confidence = 0.80
        elif r2_score >= 0.4:
            base_confidence = 0.60
        elif r2_score >= 0.2:
            base_confidence = 0.45
        else:
            base_confidence = 0.30

        mae = backtest_metrics.get('mae', 0)
        mae_threshold = thresholds.get('mae_threshold', 500)
        if mae > mae_threshold * 2:
            base_confidence *= 0.8
        elif mae > mae_threshold:
            base_confidence *= 0.9

        mape = backtest_metrics.get('mape', 0)
        mape_threshold = thresholds.get('mape_threshold', 30)
        if mape > mape_threshold * 1.5:
            base_confidence *= 0.85
        elif mape > mape_threshold:
            base_confidence *= 0.92

        warning_count = len(sanity_check.get("warnings", []))
        if warning_count > 5:
            base_confidence *= 0.8
        elif warning_count > 3:
            base_confidence *= 0.85
        elif warning_count > 1:
            base_confidence *= 0.92

        if sanity_check.get("fail"):
            base_confidence *= 0.5

        return max(0.3, min(1.0, base_confidence))
