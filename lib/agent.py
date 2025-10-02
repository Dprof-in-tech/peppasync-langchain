"""
Unified Business Intelligence Agent
Single agent that orchestrates multiple tools for different business scenarios
"""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class UnifiedBusinessAgent:
    """
    Single agent that can handle all business intelligence tasks
    Uses modular tools that can be composed for different scenarios
    """

    def __init__(self, session_id: str = None):
        # Use ToolRegistry for centralized tool management
        from .tool_registry import ToolRegistry
        self.session_id = session_id
        self.tools = ToolRegistry.get_all_tools()
        # If any tool needs session_id, set it
        if "database" in self.tools and hasattr(self.tools["database"], "session_id"):
            self.tools["database"].session_id = session_id

    async def analyze_direct_query(self, query: str, business_data: Dict[str, Any], conversation_history: List[Dict] = None) -> str:
        """
        Simplified direct analysis: just call the unified analysis tool and return results.
        No more LLM planning, no more reference resolvers, no more multi-step orchestration.
        """
        try:
            logger.info(f"Analyzing query: {query}")

            # Just call the unified analysis tool directly - it does everything
            result = await self.tools["unified_analysis"]._arun(query, business_data, conversation_history or [])

            # Format response
            import json
            response_data = {
                "type": "business_analysis",
                "insights": result.get("insights", ""),
                "recommendations": result.get("recommendations", []),
                "alerts": result.get("alerts", []),
                "suggested_actions": result.get("suggested_actions", []),
                "draft_content": result.get("draft_content"),
                "draft_type": result.get("draft_type"),
                "metadata": {
                    "analysis_type": "Unified Analysis",
                    "business_category": "Auto",
                    "timestamp": int(__import__('time').time())
                }
            }

            logger.info(f"Analysis complete: {len(result.get('recommendations', []))} recommendations, {len(result.get('alerts', []))} alerts, {len(result.get('suggested_actions', []))} suggested actions")
            return json.dumps(response_data, indent=2, default=self._json_serializer)

        except Exception as e:
            logger.error(f"Error in direct query analysis: {e}")
            return self._format_error_response(str(e))

    def _json_serializer(self, obj):
        """Custom JSON serializer for date/decimal objects"""
        import decimal
        import datetime

        if isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    def _format_error_response(self, error_msg: str) -> str:
        """Format error as JSON response"""
        error_response = {
            "type": "error",
            "insights": f"I encountered an error processing your request: {error_msg}",
            "recommendations": [],
            "alerts": [],
            "suggested_actions": [],
            "metadata": {
                "analysis_type": "Error",
                "business_category": "System",
                "timestamp": int(__import__('time').time())
            }
        }

        import json
        return json.dumps(error_response, indent=2)

    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())
