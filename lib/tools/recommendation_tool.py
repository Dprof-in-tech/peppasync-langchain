from langchain.tools import BaseTool
from typing import Dict, List, Any
import logging
from ..config import LLMManager
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class RecommendationTool(BaseTool):
    """Generate actionable business recommendations"""
    name: str = "recommendation_generation"
    description: str = "Generate specific, actionable business recommendations"

class RecommendationTool(BaseTool):
    """Generate actionable business recommendations"""
    name: str = "recommendation_generation"
    description: str = "Generate specific, actionable business recommendations"

    def _run(self, analysis_data: Dict[str, Any], business_category: str) -> List[Dict]:
        """Generate recommendations based on analysis"""
        try:
            recommendations = []
            
            if business_category == "inventory_operations":
                # Inventory-specific recommendations
                low_stock_items = analysis_data.get('low_stock_items', [])
                for item in low_stock_items:
                    recommendations.append({
                        "type": "INVENTORY_REORDER",
                        "priority": "HIGH",
                        "action": f"Reorder {item.get('product_name')}",
                        "details": f"Current stock: {item.get('current_stock')}, Reorder level: {item.get('reorder_level')}",
                        "estimated_cost": item.get('reorder_level', 0) * 2 * 50000,  # Rough estimation
                        "timeline": "1-2 weeks"
                    })
                    
            elif business_category == "marketing_customer":
                # Marketing-specific recommendations
                underperforming_campaigns = analysis_data.get('underperforming_campaigns', [])
                for campaign in underperforming_campaigns:
                    if campaign.get('roas', 0) < 1.5:
                        recommendations.append({
                            "type": "MARKETING_PAUSE",
                            "priority": "HIGH",
                            "action": f"Pause or reduce budget for {campaign.get('campaign_name')}",
                            "details": f"Current ROAS: {campaign.get('roas')}, spend: ₦{campaign.get('spend', 0):,}",
                            "estimated_savings": campaign.get('spend', 0) * 0.5,
                            "timeline": "Immediate"
                        })
                    else:
                        recommendations.append({
                            "type": "MARKETING_OPTIMIZE",
                            "priority": "MEDIUM",
                            "action": f"Optimize targeting for {campaign.get('campaign_name')}",
                            "details": f"ROAS can be improved from {campaign.get('roas')} to 3.0+",
                            "estimated_impact": "20-30% ROAS improvement",
                            "timeline": "1-2 weeks"
                        })
                        
            elif business_category == "sales_revenue":
                # Sales-specific recommendations
                sales_data = analysis_data.get('sales_data', [])
                if sales_data:
                    latest_month = sales_data[-1] if sales_data else {}
                    recommendations.append({
                        "type": "SALES_STRATEGY",
                        "priority": "MEDIUM",
                        "action": "Focus on high-performing product categories",
                        "details": f"Current monthly revenue: ₦{latest_month.get('revenue', 0):,}",
                        "estimated_impact": "10-15% revenue increase",
                        "timeline": "1 month"
                    })
            
            # Always add a general AI-generated recommendation
            ai_recommendation = self._generate_ai_recommendation(analysis_data, business_category)
            if ai_recommendation:
                recommendations.append({
                    "type": "AI_STRATEGIC",
                    "priority": "MEDIUM",
                    "action": "AI-Generated Strategic Recommendation",
                    "details": ai_recommendation,
                    "timeline": "2-4 weeks"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return [{"error": str(e), "type": "ERROR"}]

    def _generate_ai_recommendation(self, data: Dict[str, Any], category: str) -> str:
        """Generate AI-powered strategic recommendation"""
        try:
            llm = LLMManager.get_chat_llm()
            
            prompt = f"""
            Based on this {category} analysis data, provide ONE specific, actionable recommendation for a Nigerian retail business:
            
            Analysis Summary: {str(data)[:500]}...
            
            Provide a recommendation that is:
            1. Specific and actionable
            2. Considers Nigerian market conditions
            3. Can be implemented within 2-4 weeks
            4. Focuses on measurable business impact
            
            Keep it under 100 words.
            """
            
            messages = [
                SystemMessage(content="You are a business strategy consultant for Nigerian retail businesses."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"AI recommendation failed: {e}")
            return "Unable to generate AI recommendation"

    async def _arun(self, analysis_data: Dict[str, Any], business_category: str) -> List[Dict]:
        """Async version of recommendation generation"""
        return self._run(analysis_data, business_category)