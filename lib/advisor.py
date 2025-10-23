"""
Advisor Layer - Generates actionable recommendations from forecasts
Part of the ADVISOR LAYER in the demand forecast architecture

Takes validated forecast data and generates:
1. Inventory recommendations
2. Marketing timing suggestions
3. Supply chain preparation actions
4. Risk mitigation strategies
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from lib.config import LLMManager

logger = logging.getLogger(__name__)


class Advisor:
    """
    Generates actionable business recommendations from forecast data.

    This is the ADVISOR layer in your architecture:
    CONTEXT LAYER → LLM CALL → VALIDATION → **ADVISOR** → RESULTS
    """

    def __init__(self):
        self.llm = LLMManager.get_chat_llm()

    def generate_recommendations(
        self,
        forecast_context: Dict,
        use_pinecone: bool = True
    ) -> Dict:
        """
        Generate actionable recommendations from forecast context.

        Args:
            forecast_context: Complete context from Context Layer
            use_pinecone: Whether to use business knowledge from Pinecone

        Returns:
            Dict with recommendations:
            {
                "recommendations": List[str],
                "insights": str,
                "action_items": List[Dict],
                "risk_alerts": List[str]
            }
        """
        try:
            forecast_data = forecast_context.get("forecast_data", {})
            validation_result = forecast_context.get("validation_result", {})
            settings = forecast_context.get("settings", {})

            # Extract key information
            forecast_info = forecast_data.get("forecast", {})
            key_insights = forecast_data.get("key_insights", {})
            supply_chain_alerts = forecast_data.get("supply_chain_alerts", [])

            # Build recommendations based on forecast
            recommendations = {
                "recommendations": [],
                "insights": "",
                "action_items": [],
                "risk_alerts": []
            }

            # 1. Inventory recommendations
            inventory_recs = self._generate_inventory_recommendations(
                forecast_info,
                key_insights,
                supply_chain_alerts
            )
            recommendations["recommendations"].extend(inventory_recs)

            # 2. Marketing timing recommendations
            marketing_recs = self._generate_marketing_recommendations(
                forecast_info,
                key_insights,
                settings.get("economic_events", [])
            )
            recommendations["recommendations"].extend(marketing_recs)

            # 3. Supply chain action items
            supply_chain_actions = self._generate_supply_chain_actions(
                supply_chain_alerts,
                key_insights
            )
            recommendations["action_items"].extend(supply_chain_actions)

            # 4. Risk alerts
            risk_alerts = self._generate_risk_alerts(
                forecast_info,
                validation_result,
                key_insights
            )
            recommendations["risk_alerts"].extend(risk_alerts)

            # 5. Generate natural language insights using LLM
            insights_text = self._generate_insights_text(
                forecast_context,
                recommendations
            )
            recommendations["insights"] = insights_text

            logger.info(f"Generated {len(recommendations['recommendations'])} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "recommendations": [],
                "insights": "Unable to generate recommendations",
                "action_items": [],
                "risk_alerts": [f"Error: {str(e)}"]
            }

    def _generate_inventory_recommendations(
        self,
        forecast_info: Dict,
        key_insights: Dict,
        supply_chain_alerts: List[Dict]
    ) -> List[str]:
        """Generate inventory management recommendations"""
        recommendations = []

        # Trend-based recommendations
        trend = forecast_info.get("trend", "stable")
        trend_change = forecast_info.get("trend_change_percent", 0)

        if trend == "increasing":
            if trend_change > 20:
                recommendations.append(
                    f"📈 Strong upward trend detected ({trend_change:.1f}% increase). "
                    "Consider increasing inventory levels by 25-30% to avoid stockouts."
                )
            else:
                recommendations.append(
                    f"📈 Moderate growth trend ({trend_change:.1f}% increase). "
                    "Plan for 15-20% inventory increase."
                )

        elif trend == "decreasing":
            recommendations.append(
                f"📉 Declining trend detected ({trend_change:.1f}% decrease). "
                "Reduce new orders and focus on clearing existing inventory."
            )

        # Peak demand preparation
        peak_date = key_insights.get("peak_demand_date")
        peak_value = key_insights.get("peak_demand_value")
        avg_demand = key_insights.get("average_daily_demand")

        if peak_value and avg_demand and peak_value > avg_demand * 1.5:
            spike_pct = ((peak_value - avg_demand) / avg_demand) * 100
            recommendations.append(
                f"⚠️ Peak demand expected on {peak_date} ({spike_pct:.0f}% above average). "
                "Ensure adequate stock at least 1 week in advance."
            )

        # Supply chain timing
        if supply_chain_alerts:
            earliest_order = supply_chain_alerts[0]
            recommendations.append(
                f" First order deadline: {earliest_order['order_by_date']} "
                f"for {earliest_order['location']} "
                f"(lead time: {earliest_order['lead_time_days']} days)"
            )

        return recommendations

    def _generate_marketing_recommendations(
        self,
        forecast_info: Dict,
        key_insights: Dict,
        economic_events: List[Dict]
    ) -> List[str]:
        """Generate marketing timing recommendations"""
        recommendations = []

        # Event-based marketing
        for event in economic_events:
            event_date = pd.to_datetime(event['date'])
            days_until = (event_date - pd.Timestamp.now()).days

            if 0 < days_until <= 60:  # Within next 60 days
                impact_before = event.get('impact_days_before', 7)
                recommendations.append(
                    f"🎯 {event['name']} approaching ({days_until} days). "
                    f"Launch marketing campaign {impact_before} days before ({event_date - timedelta(days=impact_before):%Y-%m-%d})."
                )

        # Seasonality-based marketing
        if forecast_info.get("seasonality_detected"):
            recommendations.append(
                "📊 Strong seasonal patterns detected. "
                "Align marketing spend with high-demand periods for better ROI."
            )

        return recommendations

    def _generate_supply_chain_actions(
        self,
        supply_chain_alerts: List[Dict],
        key_insights: Dict
    ) -> List[Dict]:
        """Generate supply chain action items"""
        action_items = []

        # Create actionable tasks for each ordering alert
        for alert in supply_chain_alerts[:5]:  # Top 5 most urgent
            action_items.append({
                "action": f"Place order with {alert['location']}",
                "deadline": alert['order_by_date'],
                "quantity": alert['quantity_needed'],
                "priority": "high" if pd.to_datetime(alert['order_by_date']) < pd.Timestamp.now() + timedelta(days=7) else "medium",
                "notes": alert.get('notes', '')
            })

        return action_items

    def _generate_risk_alerts(
        self,
        forecast_info: Dict,
        validation_result: Dict,
        key_insights: Dict
    ) -> List[str]:
        """Generate risk and warning alerts"""
        alerts = []

        # Low confidence warning
        confidence = forecast_info.get("confidence_score", 1.0)
        if confidence < 0.7:
            alerts.append(
                f"⚠️ Low forecast confidence ({confidence:.1%}). "
                "Predictions may be less reliable. Consider multiple scenarios."
            )

        # Validation warnings
        validation_warnings = validation_result.get("warnings", [])
        if validation_warnings:
            alerts.append(
                f"⚠️ Forecast validation flagged {len(validation_warnings)} concern(s). "
                "Review predictions carefully."
            )

        # Extreme demand spikes
        peak_demand = key_insights.get("peak_demand_value", 0)
        avg_demand = key_insights.get("average_daily_demand", 1)

        if peak_demand > avg_demand * 3:
            alerts.append(
                f"🚨 Extreme demand spike predicted ({peak_demand:.0f} vs avg {avg_demand:.0f}). "
                "Verify with additional data sources before major inventory commitments."
            )

        return alerts

    def _generate_insights_text(
        self,
        forecast_context: Dict,
        recommendations: Dict
    ) -> str:
        """
        Generate natural language insights using LLM.

        This leverages existing Pinecone knowledge for context-aware insights.
        """
        try:
            forecast_data = forecast_context.get("forecast_data", {})
            forecast_info = forecast_data.get("forecast", {})
            key_insights = forecast_data.get("key_insights", {})
            user_query = forecast_context.get("user_query", {})

            # Build prompt for LLM
            prompt = f"""
Based on the demand forecast analysis, provide a concise summary (2-3 sentences) of the key insights and business implications.

Forecast Summary:
- Trend: {forecast_info.get('trend', 'N/A')} ({forecast_info.get('trend_change_percent', 0):.1f}% change)
- Peak demand: {key_insights.get('peak_demand_value', 'N/A')} on {key_insights.get('peak_demand_date', 'N/A')}
- Average daily demand: {key_insights.get('average_daily_demand', 'N/A')}
- Confidence: {forecast_info.get('confidence_score', 0):.1%}
- Seasonality: {'Detected' if forecast_info.get('seasonality_detected') else 'Not detected'}

User Query: {user_query.get('original_prompt', 'forecast demand')}

Provide insights focusing on:
1. What the trend means for the business
2. Key dates to watch
3. Overall recommendation (prepare, maintain, or reduce inventory)

Keep it concise and actionable.
"""

            # Call LLM
            response = self.llm.invoke(prompt)
            insights_text = response.content.strip()

            return insights_text

        except Exception as e:
            logger.error(f"Error generating insights text: {str(e)}")
            return "Forecast analysis complete. Review the recommendations for specific action items."
