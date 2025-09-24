"""
Tool Registry for PeppaSync Business Intelligence Platform
Centralized registry of all available tools and their configurations
"""
from typing import Dict, Any, List
from .tools.database_tool import DatabaseQueryTool
from .tools.insight_tool import InsightGenerationTool
from .tools.alert_tool import AlertTool
from .tools.recommendation_tool import RecommendationTool

class ToolRegistry:
    """Centralized registry for all business intelligence tools"""
    
    @staticmethod
    def get_all_tools() -> Dict[str, Any]:
        """Get all available tools"""
        return {
            "database": DatabaseQueryTool(),
            "insights": InsightGenerationTool(),
            "alerts": AlertTool(),
            "recommendations": RecommendationTool()
        }
    
    @staticmethod
    def get_tool_descriptions() -> Dict[str, str]:
        """Get descriptions of all tools"""
        return {
            "database": "Query business data (sales, inventory, campaigns, customers)",
            "insights": "Generate AI-powered business insights from data",
            "alerts": "Generate alerts for critical business conditions",
            "recommendations": "Generate actionable business recommendations"
        }
    
    @staticmethod
    def get_workflow_descriptions() -> Dict[str, str]:
        """Get descriptions of available workflows"""
        return {
            "inventory_analysis": "Monitor inventory levels, generate alerts, and recommend reorders",
            "marketing_analysis": "Analyze campaigns, optimize ROAS, and improve marketing ROI",
            "sales_analysis": "Analyze sales performance, identify trends, and forecast revenue",
            "general_analysis": "Comprehensive business analysis across all categories"
        }
    
    @staticmethod
    def get_business_categories() -> List[str]:
        """Get supported business categories"""
        return [
            "sales_revenue",
            "marketing_customer", 
            "pricing_promotions",
            "inventory_operations",
            "channel_performance",
            "customer_behavior",
            "financial_efficiency",
            "strategic_scenarios"
        ]
    
    @staticmethod
    def get_analysis_types() -> List[str]:
        """Get supported analysis types"""
        return [
            "descriptive",
            "diagnostic", 
            "predictive",
            "prescriptive",
            "scenario",
            "comparative"
        ]