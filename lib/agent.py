"""
Unified Business Intelligence Agent
Single agent that orchestrates multiple tools for different business scenarios
"""
import logging
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .tools.database_tool import DatabaseQueryTool
from .tools.insight_tool import InsightGenerationTool
from .tools.alert_tool import AlertTool
from .tools.recommendation_tool import RecommendationTool

logger = logging.getLogger(__name__)

class BusinessAnalysisState(BaseModel):
    """State for business analysis workflow"""
    query: str = ""
    business_category: str = ""
    analysis_type: str = ""
    raw_data: Dict[str, Any] = Field(default_factory=dict)  # Changed from List[Dict] to Dict[str, Any]
    insights: str = ""
    alerts: List[Dict] = Field(default_factory=list)
    recommendations: List[Dict] = Field(default_factory=list)
    final_response: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class UnifiedBusinessAgent:
    """
    Single agent that can handle all business intelligence tasks
    Uses modular tools that can be composed for different scenarios
    """
    
    def __init__(self, session_id: str = None):
        # Initialize all available tools
        self.session_id = session_id
        self.tools = {
            "database": DatabaseQueryTool(session_id=session_id),
            "insights": InsightGenerationTool(),
            "alerts": AlertTool(),
            "recommendations": RecommendationTool()
        }
        
        # Build workflow for different business scenarios
        self.workflows = {
            "inventory_analysis": self._build_inventory_workflow(),
            "marketing_analysis": self._build_marketing_workflow(),
            "sales_analysis": self._build_sales_workflow(),
            "general_analysis": self._build_general_workflow()
        }

    def _build_inventory_workflow(self) -> StateGraph:
        """Workflow for inventory-related analysis"""
        workflow = StateGraph(BusinessAnalysisState)
        
        workflow.add_node("fetch_inventory_data", self._fetch_inventory_data)
        workflow.add_node("generate_inventory_alerts", self._generate_inventory_alerts)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("compile_response", self._compile_response)
        
        workflow.set_entry_point("fetch_inventory_data")
        workflow.add_edge("fetch_inventory_data", "generate_inventory_alerts")
        workflow.add_edge("generate_inventory_alerts", "generate_insights")
        workflow.add_edge("generate_insights", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "compile_response")
        workflow.add_edge("compile_response", END)
        
        return workflow.compile()

    def _build_marketing_workflow(self) -> StateGraph:
        """Workflow for marketing-related analysis"""
        workflow = StateGraph(BusinessAnalysisState)
        
        workflow.add_node("fetch_marketing_data", self._fetch_marketing_data)
        workflow.add_node("generate_marketing_alerts", self._generate_marketing_alerts)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("compile_response", self._compile_response)
        
        workflow.set_entry_point("fetch_marketing_data")
        workflow.add_edge("fetch_marketing_data", "generate_marketing_alerts")
        workflow.add_edge("generate_marketing_alerts", "generate_insights")
        workflow.add_edge("generate_insights", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "compile_response")
        workflow.add_edge("compile_response", END)
        
        return workflow.compile()

    def _build_sales_workflow(self) -> StateGraph:
        """Workflow for sales-related analysis"""
        workflow = StateGraph(BusinessAnalysisState)
        
        workflow.add_node("fetch_sales_data", self._fetch_sales_data)
        workflow.add_node("generate_sales_alerts", self._generate_sales_alerts)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("compile_response", self._compile_response)
        
        workflow.set_entry_point("fetch_sales_data")
        workflow.add_edge("fetch_sales_data", "generate_sales_alerts")
        workflow.add_edge("generate_sales_alerts", "generate_insights")
        workflow.add_edge("generate_insights", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "compile_response")
        workflow.add_edge("compile_response", END)
        
        return workflow.compile()

    def _build_general_workflow(self) -> StateGraph:
        """General workflow for mixed or unspecified analysis"""
        workflow = StateGraph(BusinessAnalysisState)
        
        workflow.add_node("fetch_general_data", self._fetch_general_data)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("compile_response", self._compile_response)
        
        workflow.set_entry_point("fetch_general_data")
        workflow.add_edge("fetch_general_data", "generate_insights")
        workflow.add_edge("generate_insights", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "compile_response")
        workflow.add_edge("compile_response", END)
        
        return workflow.compile()

    # Data fetching nodes
    async def _fetch_inventory_data(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Fetch inventory-related data"""
        try:
            inventory_data = await self.tools["database"]._arun("inventory_data")
            low_stock_data = await self.tools["database"]._arun("low_stock_items")
            
            state.raw_data = {
                "inventory_data": inventory_data,
                "low_stock_items": low_stock_data
            }
            logger.info(f"Fetched inventory data: {len(inventory_data)} items, {len(low_stock_data)} low stock")
            
        except Exception as e:
            logger.error(f"Error fetching inventory data: {e}")
            state.error = f"Data fetch failed: {str(e)}"
        
        return state

    async def _fetch_marketing_data(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Fetch marketing-related data"""
        try:
            campaign_data = await self.tools["database"]._arun("campaign_data")
            underperforming = await self.tools["database"]._arun("underperforming_campaigns")
            
            state.raw_data = {
                "campaign_data": campaign_data,
                "underperforming_campaigns": underperforming
            }
            logger.info(f"Fetched marketing data: {len(campaign_data)} campaigns")
            
        except Exception as e:
            logger.error(f"Error fetching marketing data: {e}")
            state.error = f"Data fetch failed: {str(e)}"
        
        return state

    async def _fetch_sales_data(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Fetch sales-related data"""
        try:
            sales_data = await self.tools["database"]._arun("sales_data")
            
            state.raw_data = {
                "sales_data": sales_data
            }
            logger.info(f"Fetched sales data: {len(sales_data)} records")
            
        except Exception as e:
            logger.error(f"Error fetching sales data: {e}")
            state.error = f"Data fetch failed: {str(e)}"
        
        return state

    async def _fetch_general_data(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Fetch general business data"""
        try:
            sales_data = await self.tools["database"]._arun("sales_data")
            inventory_data = await self.tools["database"]._arun("inventory_data")
            campaign_data = await self.tools["database"]._arun("campaign_data")
            
            state.raw_data = {
                "sales_data": sales_data,
                "inventory_data": inventory_data,
                "campaign_data": campaign_data
            }
            logger.info("Fetched general business data")
            
        except Exception as e:
            logger.error(f"Error fetching general data: {e}")
            state.error = f"Data fetch failed: {str(e)}"
        
        return state

    # Alert generation nodes
    async def _generate_inventory_alerts(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Generate inventory-specific alerts"""
        try:
            inventory_data = state.raw_data.get("inventory_data", [])
            alerts = await self.tools["alerts"]._arun(inventory_data, "inventory_alerts")
            state.alerts = alerts
            logger.info(f"Generated {len(alerts)} inventory alerts")
            
        except Exception as e:
            logger.error(f"Error generating inventory alerts: {e}")
            state.alerts = []
        
        return state

    async def _generate_marketing_alerts(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Generate marketing-specific alerts"""
        try:
            campaign_data = state.raw_data.get("campaign_data", [])
            alerts = await self.tools["alerts"]._arun(campaign_data, "marketing_alerts")
            state.alerts = alerts
            logger.info(f"Generated {len(alerts)} marketing alerts")
            
        except Exception as e:
            logger.error(f"Error generating marketing alerts: {e}")
            state.alerts = []
        
        return state

    async def _generate_sales_alerts(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Generate sales-specific alerts"""
        try:
            sales_data = state.raw_data.get("sales_data", [])
            alerts = await self.tools["alerts"]._arun(sales_data, "sales_alerts")
            state.alerts = alerts
            logger.info(f"Generated {len(alerts)} sales alerts")
            
        except Exception as e:
            logger.error(f"Error generating sales alerts: {e}")
            state.alerts = []
        
        return state

    # Common workflow nodes
    async def _generate_insights(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Generate AI insights from data"""
        try:
            # Determine analysis context
            context = f"Business Category: {state.business_category}, Analysis Type: {state.analysis_type}"
            
            # Use first available data for insights
            data_for_analysis = []
            if state.raw_data:
                for key, value in state.raw_data.items():
                    if isinstance(value, list) and value:
                        data_for_analysis = value
                        break
            
            if data_for_analysis:
                insights = await self.tools["insights"]._arun(
                    data_for_analysis, 
                    state.business_category.split('_')[0],  # Extract main category
                    context
                )
                state.insights = insights
                logger.info("Generated AI insights")
            else:
                state.insights = "No data available for insights generation"
                
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            state.insights = f"Insights generation failed: {str(e)}"
        
        return state

    async def _generate_recommendations(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Generate actionable recommendations"""
        try:
            recommendations = await self.tools["recommendations"]._arun(
                state.raw_data, 
                state.business_category
            )
            state.recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            state.recommendations = []
        
        return state

    async def _compile_response(self, state: BusinessAnalysisState) -> BusinessAnalysisState:
        """Compile final response"""
        try:
            state.final_response = {
                "status": "success" if not state.error else "error",
                "business_category": state.business_category,
                "analysis_type": state.analysis_type,
                "insights": state.insights,
                "alerts": state.alerts,
                "recommendations": state.recommendations,
                "data_summary": {
                    "total_alerts": len(state.alerts),
                    "total_recommendations": len(state.recommendations),
                    "critical_alerts": len([a for a in state.alerts if a.get("priority") == "CRITICAL"])
                },
                "error": state.error
            }
            logger.info("Compiled final response")
            
        except Exception as e:
            logger.error(f"Error compiling response: {e}")
            state.final_response = {"status": "error", "error": str(e)}
        
        return state

    async def analyze(self, query: str, business_category: str, analysis_type: str = "descriptive") -> Dict[str, Any]:
        """
        Main method to analyze business requests
        """
        try:
            # Determine appropriate workflow
            if "inventory" in business_category or "stock" in query.lower():
                workflow_name = "inventory_analysis"
            elif "marketing" in business_category or "campaign" in query.lower():
                workflow_name = "marketing_analysis"
            elif "sales" in business_category or "revenue" in query.lower():
                workflow_name = "sales_analysis"
            else:
                workflow_name = "general_analysis"
            
            # Initialize state
            state = BusinessAnalysisState(
                query=query,
                business_category=business_category,
                analysis_type=analysis_type
            )
            
            # Run appropriate workflow
            workflow = self.workflows[workflow_name]
            result = await workflow.ainvoke(state)
            
            # Handle different return types from LangGraph
            if isinstance(result, dict):
                final_state = result
            else:
                final_state = result.__dict__ if hasattr(result, '__dict__') else result
            
            return final_state.get("final_response", {"status": "error", "error": "No response generated"})
            
        except Exception as e:
            logger.error(f"Error in unified analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "business_category": business_category,
                "analysis_type": analysis_type
            }

    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())

    def get_available_workflows(self) -> List[str]:
        """Get list of available workflows"""
        return list(self.workflows.keys())