"""
Demand Forecast Tool - LangChain tool for demand forecasting
Integrates with existing tool registry and conversation manager
"""

from langchain.tools import BaseTool
from typing import Optional, Dict, Any, ClassVar
from pydantic import Field, BaseModel
import logging
import json

from lib.context_layer import ContextLayer
from lib.advisor import Advisor

logger = logging.getLogger(__name__)


class ForecastInput(BaseModel):
    """Input schema for demand forecast tool"""
    session_id: str = Field(description="User session ID")
    user_prompt: str = Field(description="Natural language forecast query")
    product_filter: Optional[str] = Field(default=None, description="Product name(s) to filter")
    forecast_mode: str = Field(default="aggregate", description="Forecast mode: aggregate, single, multi, or top_n")
    top_n_products: int = Field(default=10, description="Number of products for top_n mode")


class DemandForecastTool(BaseTool):
    """
    LangChain tool for demand forecasting.

    This tool integrates the entire forecast system into the existing
    LangChain/LangGraph workflow, making it available to the conversation manager.

    Usage in chat:
    User: "Forecast iPhone demand for next 30 days"
    → Tool triggered → Full forecast pipeline → Results returned
    """

    name: str = "demand_forecast"
    description: str = """
    Forecasts future demand for products based on historical sales data.

    Use this tool when the user asks about:
    - Future sales predictions or forecasts
    - Demand forecasting for specific periods
    - Inventory planning for future dates
    - "How much should I stock for next month?"
    - "What will my sales look like during [event]?"
    - "Predict demand for [product] for [timeframe]"

    Input should be a JSON string with:
    {
        "session_id": "user_session_id",
        "user_prompt": "natural language forecast query",
        "product_filter": "optional product name"
    }

    Returns comprehensive forecast with insights and recommendations.
    """
    
    # Declare as class variables with Field to avoid Pydantic validation issues
    context_layer: Any = Field(default=None, exclude=True)
    advisor: Any = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        # Initialize after super().__init__() to avoid Pydantic conflicts
        object.__setattr__(self, 'context_layer', ContextLayer())
        object.__setattr__(self, 'advisor', Advisor())

    def _run(
        self,
        session_id: str,
        user_prompt: str,
        product_filter: Optional[str] = None,
        forecast_mode: str = "aggregate",
        top_n_products: int = 10
    ) -> str:
        """
        Execute demand forecast.

        This method is called by LangChain when the tool is invoked.

        Args:
            session_id: User session ID
            user_prompt: Natural language query (e.g., "forecast next 45 days weekly")
            product_filter: Optional product name(s) to filter (comma-separated for multi mode)
            forecast_mode: "aggregate", "single", "multi", or "top_n"
            top_n_products: Number of products for top_n mode

        Returns:
            JSON string with forecast results
        """
        try:
            logger.info(f"Demand forecast tool triggered for session {session_id}")
            logger.info(f"Query: {user_prompt}, Mode: {forecast_mode}")

            # Step 1: Prepare complete forecast context
            # This runs the entire pipeline: data fetch → Prophet → validation
            forecast_context = self.context_layer.prepare_forecast_context(
                session_id=session_id,
                user_prompt=user_prompt,
                product_filter=product_filter,
                forecast_mode=forecast_mode,
                top_n_products=top_n_products
            )

            # Check for errors
            if "error" in forecast_context:
                return json.dumps({
                    "success": False,
                    "error": forecast_context.get("error"),
                    "message": forecast_context.get("message"),
                    "suggestion": "Please ensure you have sufficient historical data (minimum 30 days) and try again."
                }, indent=2)

            # Step 2: Generate recommendations using Advisor
            recommendations = self.advisor.generate_recommendations(forecast_context)

            # Step 3: Build final response
            response = {
                "success": True,
                "forecast": forecast_context["forecast_data"]["forecast"],
                "key_insights": forecast_context["forecast_data"]["key_insights"],
                "chart_data": forecast_context["forecast_data"]["chart_data"],
                "supply_chain_alerts": forecast_context["forecast_data"].get("supply_chain_alerts", []),
                "recommendations": recommendations["recommendations"],
                "insights": recommendations["insights"],
                "action_items": recommendations["action_items"],
                "risk_alerts": recommendations["risk_alerts"],
                "validation": {
                    "passed": forecast_context["validation_result"]["valid"],
                    "confidence_adjustment": forecast_context["validation_result"]["confidence_adjustment"],
                    "warnings": forecast_context["validation_result"]["warnings"]
                },
                "data_source": forecast_context.get("data_summary", {}).get("date_range", {}),
                "metadata": forecast_context["forecast_data"]["metadata"]
            }

            # Return as formatted JSON string for LLM consumption
            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"Demand forecast tool error: {str(e)}")
            return json.dumps({
                "success": False,
                "error": "Forecast execution failed",
                "message": str(e)
            }, indent=2)

    async def _arun(
        self,
        session_id: str,
        user_prompt: str,
        product_filter: Optional[str] = None
    ) -> str:
        """Async version - calls sync method"""
        return self._run(session_id, user_prompt, product_filter)


class DemandForecastDirectTool(BaseTool):
    """
    Simplified forecast tool that returns data directly (for API endpoints).

    This is used by the /forecast endpoint, not by the conversation manager.
    Returns Dict instead of JSON string for direct API consumption.
    """

    name: str = "demand_forecast_direct"
    description: str = "Direct demand forecasting tool for API endpoints"
    args_schema: type[BaseModel] = ForecastInput
    
    # Declare as class variables with Field to avoid Pydantic validation issues
    context_layer: Any = Field(default=None, exclude=True)
    advisor: Any = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        # Initialize after super().__init__() to avoid Pydantic conflicts
        object.__setattr__(self, 'context_layer', ContextLayer())
        object.__setattr__(self, 'advisor', Advisor())


    async def _run(
        self,
        session_id: str,
        user_prompt: str,
        product_filter: Optional[str] = None,
        forecast_mode: str = "aggregate",
        top_n_products: int = 10
    ) -> Dict[str, Any]:
        """
        Execute demand forecast and return Dict directly.

        Args:
            session_id: User session ID
            user_prompt: Natural language query
            product_filter: Optional product name(s)
            forecast_mode: Forecast mode (aggregate, single, multi, top_n)
            top_n_products: Number of products for top_n mode

        Returns:
            Dict with forecast results (not JSON string)
        """
        try:
            forecast_context = await self.context_layer.prepare_forecast_context(
                session_id=session_id,
                user_prompt=user_prompt,
                product_filter=product_filter,
                forecast_mode=forecast_mode,
                top_n_products=top_n_products
            )

            if "error" in forecast_context:
                return {
                    "success": False,
                    "error": forecast_context.get("error"),
                    "message": forecast_context.get("message")
                }

            # Optionally add advisor recommendations here if needed
            return {
                "success": True,
                "forecast": forecast_context.get("forecast_data"),
                "settings": forecast_context.get("settings"),
                "data_summary": forecast_context.get("data_summary"),
                "user_query": forecast_context.get("user_query"),
                "context_generated_at": forecast_context.get("context_generated_at")
            }
        except Exception as e:
            logger.error(f"Demand forecast failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Demand forecast failed"
            }
