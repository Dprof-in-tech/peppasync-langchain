import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import centralized configuration
from ..config import LLMManager, DatabaseManager

load_dotenv()
logger = logging.getLogger(__name__)

# LangGraph State for Marketing Agent  
class MarketingState(BaseModel):
    campaign_data: List[Dict] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    opportunities: List[Dict] = Field(default_factory=list)
    recommendations: List[Dict] = Field(default_factory=list)
    optimizations_executed: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    status: str = "initialized"

class MarketingAgent:
    """LangGraph-powered marketing optimization agent"""
    
    def __init__(self):
        self.llm = LLMManager.get_chat_llm()
        self.mock_data = DatabaseManager.get_mock_data()
        
        # Build the LangGraph workflow
        self.workflow = self._build_marketing_graph()

    def _build_marketing_graph(self) -> StateGraph:
        """Build the LangGraph marketing optimization workflow"""
        
        workflow = StateGraph(MarketingState)
        
        # Add nodes
        workflow.add_node("analyze_campaigns", self._analyze_campaigns_node)
        workflow.add_node("identify_opportunities", self._identify_opportunities_node)
        workflow.add_node("generate_recommendations", self._generate_recommendations_node)
        workflow.add_node("execute_optimizations", self._execute_optimizations_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_campaigns")
        
        # Add edges
        workflow.add_edge("analyze_campaigns", "identify_opportunities")
        workflow.add_edge("identify_opportunities", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "execute_optimizations")
        workflow.add_edge("execute_optimizations", END)
        
        return workflow.compile()

    async def _analyze_campaigns_node(self, state: MarketingState) -> MarketingState:
        """Analyze current marketing campaign performance"""
        try:
            logger.info("Analyzing marketing campaign performance...")
            
            # Mock campaign data - in production, this would query marketing platforms APIs
            mock_campaigns = [
                {
                    "campaign_id": "FB_001",
                    "campaign_name": "Summer Electronics Sale - Facebook",
                    "platform": "Facebook",
                    "spend": 150000,  # NGN
                    "impressions": 45000,
                    "clicks": 1200,
                    "conversions": 45,
                    "revenue": 360000,  # NGN
                    "roas": 2.4,
                    "ctr": 2.67,
                    "conversion_rate": 3.75,
                    "campaign_start_date": "2024-01-01",
                    "campaign_status": "ACTIVE",
                    "target_audience": "Electronics enthusiasts, 25-44",
                    "ad_creative": "Product showcase video"
                },
                {
                    "campaign_id": "G_002", 
                    "campaign_name": "Brand Awareness - Google Ads",
                    "platform": "Google",
                    "spend": 220000,  # NGN
                    "impressions": 78000,
                    "clicks": 890,
                    "conversions": 18,
                    "revenue": 290000,  # NGN
                    "roas": 1.32,
                    "ctr": 1.14,
                    "conversion_rate": 2.02,
                    "campaign_start_date": "2023-12-15",
                    "campaign_status": "ACTIVE",
                    "target_audience": "Broad audience, 18-55",
                    "ad_creative": "Brand story carousel"
                },
                {
                    "campaign_id": "IG_003",
                    "campaign_name": "Fashion Collection - Instagram",
                    "platform": "Instagram", 
                    "spend": 85000,  # NGN
                    "impressions": 32000,
                    "clicks": 1450,
                    "conversions": 87,
                    "revenue": 435000,  # NGN
                    "roas": 5.12,
                    "ctr": 4.53,
                    "conversion_rate": 6.0,
                    "campaign_start_date": "2024-01-10",
                    "campaign_status": "ACTIVE",
                    "target_audience": "Fashion lovers, 18-35 F",
                    "ad_creative": "User-generated content"
                },
                {
                    "campaign_id": "TT_004",
                    "campaign_name": "Gen-Z Electronics - TikTok",
                    "platform": "TikTok",
                    "spend": 65000,  # NGN  
                    "impressions": 125000,
                    "clicks": 2200,
                    "conversions": 28,
                    "revenue": 168000,  # NGN
                    "roas": 2.58,
                    "ctr": 1.76,
                    "conversion_rate": 1.27,
                    "campaign_start_date": "2024-01-05",
                    "campaign_status": "ACTIVE",
                    "target_audience": "Gen-Z, 16-24",
                    "ad_creative": "Trending music + product demo"
                }
            ]
            
            state.campaign_data = mock_campaigns
            
            # Calculate overall performance metrics
            total_spend = sum(c['spend'] for c in mock_campaigns)
            total_revenue = sum(c['revenue'] for c in mock_campaigns)
            total_conversions = sum(c['conversions'] for c in mock_campaigns)
            total_clicks = sum(c['clicks'] for c in mock_campaigns)
            total_impressions = sum(c['impressions'] for c in mock_campaigns)
            
            overall_roas = (total_revenue / total_spend) if total_spend > 0 else 0
            avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            avg_conversion_rate = sum(c['conversion_rate'] for c in mock_campaigns) / len(mock_campaigns) if mock_campaigns else 0
            
            state.performance_metrics = {
                "total_spend": total_spend,
                "total_revenue": total_revenue,
                "total_conversions": total_conversions,
                "total_clicks": total_clicks,
                "total_impressions": total_impressions,
                "overall_roas": overall_roas,
                "avg_ctr": avg_ctr,
                "avg_conversion_rate": avg_conversion_rate,
                "campaigns_count": len(mock_campaigns),
                "analysis_period": "Last 30 days"
            }
            
            logger.info(f"Analyzed {len(mock_campaigns)} campaigns with ₦{total_spend:,.2f} total spend")
            
        except Exception as e:
            logger.error(f"Error in analyze_campaigns_node: {e}")
            state.error = f"Campaign analysis failed: {str(e)}"
            state.status = "error"
        
        return state

    async def _identify_opportunities_node(self, state: MarketingState) -> MarketingState:
        """Identify marketing optimization opportunities"""
        try:
            logger.info("Identifying marketing optimization opportunities...")
            
            if state.error:
                return state
            
            opportunities = []
            
            # Define optimization thresholds
            LOW_ROAS_THRESHOLD = 2.0
            HIGH_ROAS_THRESHOLD = 4.0
            LOW_CTR_THRESHOLD = 1.5
            HIGH_CTR_THRESHOLD = 3.0
            LOW_CONVERSION_THRESHOLD = 2.5
            HIGH_SPEND_THRESHOLD = 100000  # NGN
            
            for campaign in state.campaign_data:
                campaign_id = campaign['campaign_id']
                campaign_name = campaign['campaign_name']
                roas = campaign['roas']
                ctr = campaign['ctr']
                conversion_rate = campaign['conversion_rate']
                spend = campaign['spend']
                platform = campaign['platform']
                
                # Low ROAS campaigns - reduce spend or pause
                if roas < LOW_ROAS_THRESHOLD and spend > 50000:  # NGN 50k threshold
                    opportunities.append({
                        "type": "LOW_ROAS",
                        "priority": "HIGH",
                        "campaign_id": campaign_id,
                        "campaign_name": campaign_name,
                        "platform": platform,
                        "current_roas": roas,
                        "current_spend": spend,
                        "recommendation": "REDUCE_SPEND_OR_PAUSE",
                        "description": f"Campaign ROAS of {roas:.2f} is below target of {LOW_ROAS_THRESHOLD}",
                        "potential_impact": f"Save up to ₦{spend * 0.5:,.0f} in wasted ad spend",
                        "suggested_action": f"Reduce daily budget by 50% or pause if ROAS doesn't improve within 7 days"
                    })
                
                # High performing campaigns - scale up
                if roas > HIGH_ROAS_THRESHOLD and spend < HIGH_SPEND_THRESHOLD:
                    potential_increase = min(spend * 0.3, HIGH_SPEND_THRESHOLD - spend)
                    expected_revenue_increase = potential_increase * roas
                    
                    opportunities.append({
                        "type": "SCALE_OPPORTUNITY",
                        "priority": "MEDIUM",
                        "campaign_id": campaign_id,
                        "campaign_name": campaign_name,
                        "platform": platform,
                        "current_roas": roas,
                        "current_spend": spend,
                        "recommendation": "INCREASE_BUDGET",
                        "description": f"High-performing campaign with ROAS of {roas:.2f}",
                        "potential_impact": f"Potential revenue increase: ₦{expected_revenue_increase:,.0f}",
                        "suggested_action": f"Increase daily budget to ₦{spend + potential_increase:,.0f} (+30%)"
                    })
                
                # Low CTR campaigns - improve creative/targeting
                if ctr < LOW_CTR_THRESHOLD and spend > 30000:  # NGN 30k threshold
                    opportunities.append({
                        "type": "LOW_CTR", 
                        "priority": "MEDIUM",
                        "campaign_id": campaign_id,
                        "campaign_name": campaign_name,
                        "platform": platform,
                        "current_ctr": ctr,
                        "recommendation": "IMPROVE_CREATIVE_OR_TARGETING",
                        "description": f"CTR of {ctr:.2f}% is below optimal threshold of {LOW_CTR_THRESHOLD}%",
                        "potential_impact": "Improve ad relevance and reduce cost per click by 20-40%",
                        "suggested_action": f"A/B test new ad creatives or refine {platform} audience targeting"
                    })
                
                # Low conversion rate - optimize landing page/funnel
                if conversion_rate < LOW_CONVERSION_THRESHOLD and ctr > LOW_CTR_THRESHOLD:
                    opportunities.append({
                        "type": "LOW_CONVERSION",
                        "priority": "MEDIUM", 
                        "campaign_id": campaign_id,
                        "campaign_name": campaign_name,
                        "platform": platform,
                        "current_conversion_rate": conversion_rate,
                        "current_ctr": ctr,
                        "recommendation": "OPTIMIZE_LANDING_PAGE",
                        "description": f"Good CTR ({ctr:.2f}%) but low conversion rate ({conversion_rate:.2f}%)",
                        "potential_impact": "Increase conversion rate by 25-50% with better landing page",
                        "suggested_action": "A/B test landing page design, checkout flow, or product descriptions"
                    })
                
                # Platform-specific opportunities
                if platform == "TikTok" and conversion_rate < 2.0 and ctr > 1.5:
                    opportunities.append({
                        "type": "TIKTOK_OPTIMIZATION",
                        "priority": "LOW",
                        "campaign_id": campaign_id,
                        "campaign_name": campaign_name,
                        "platform": platform,
                        "recommendation": "TIKTOK_CREATIVE_REFRESH",
                        "description": f"TikTok campaigns need frequent creative refresh for sustained performance",
                        "potential_impact": "Maintain CTR and prevent audience fatigue",
                        "suggested_action": "Create 3-5 new video creatives following current trending formats"
                    })
            
            # Sort opportunities by priority
            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            opportunities.sort(key=lambda x: priority_order.get(x.get('priority', 'LOW'), 2))
            
            state.opportunities = opportunities
            
            logger.info(f"Identified {len(opportunities)} marketing optimization opportunities")
            
        except Exception as e:
            logger.error(f"Error in identify_opportunities_node: {e}")
            state.error = f"Opportunity identification failed: {str(e)}"
        
        return state

    async def _generate_recommendations_node(self, state: MarketingState) -> MarketingState:
        """Generate AI-powered marketing recommendations"""
        try:
            logger.info("Generating marketing recommendations...")
            
            if state.error:
                return state
            
            recommendations = []
            
            if not state.opportunities:
                recommendations.append({
                    "type": "NO_ISSUES",
                    "title": "Campaign Performance Status",
                    "content": "All campaigns are performing within acceptable ranges. Continue monitoring and consider testing new creative variations to prevent ad fatigue.",
                    "priority": "LOW"
                })
            else:
                # Group opportunities by priority
                high_priority_ops = [op for op in state.opportunities if op.get('priority') == 'HIGH']
                medium_priority_ops = [op for op in state.opportunities if op.get('priority') == 'MEDIUM']
                
                # Generate comprehensive AI recommendations
                performance_metrics = state.performance_metrics
                
                recommendation_prompt = f"""
Marketing Campaign Analysis Summary (Nigerian Market):
- Total Ad Spend: ₦{performance_metrics.get('total_spend', 0):,.2f}
- Total Revenue: ₦{performance_metrics.get('total_revenue', 0):,.2f}  
- Overall ROAS: {performance_metrics.get('overall_roas', 0):.2f}
- Average CTR: {performance_metrics.get('avg_ctr', 0):.2f}%
- Average Conversion Rate: {performance_metrics.get('avg_conversion_rate', 0):.2f}%
- Campaigns Analyzed: {performance_metrics.get('campaigns_count', 0)}

Optimization Opportunities Identified:
- High Priority Issues: {len(high_priority_ops)}
- Medium Priority Issues: {len(medium_priority_ops)}

Top Issues Requiring Immediate Attention:
{self._format_opportunities_for_ai(state.opportunities[:5])}

Provide strategic marketing recommendations specifically for the Nigerian market:
1. Budget reallocation strategy considering local purchasing power
2. Platform-specific optimization for Nigerian audience behavior  
3. Creative testing framework for local cultural relevance
4. Audience targeting improvements for Nigerian demographics
5. Seasonal optimization for local holidays and events
6. Currency considerations (NGN) for budget planning
7. Performance monitoring and KPI adjustments

Focus on actionable strategies that can be implemented within 1 week, considering Nigerian market dynamics.
"""
                
                system_message = "You are a digital marketing expert specializing in the Nigerian market. Consider local consumer behavior, purchasing power, cultural preferences, and platform usage patterns in Nigeria."
                
                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=recommendation_prompt)
                ]
                
                ai_response = await self.llm.ainvoke(messages)
                
                recommendations.append({
                    "type": "AI_STRATEGIC",
                    "title": "Strategic Marketing Recommendations for Nigerian Market",
                    "content": ai_response.content,
                    "timestamp": int(time.time()),
                    "priority": "HIGH" if high_priority_ops else "MEDIUM"
                })
                
                # Add specific campaign recommendations
                for opportunity in state.opportunities[:3]:  # Top 3 opportunities
                    estimated_impact = self._calculate_opportunity_impact(opportunity, performance_metrics)
                    
                    recommendations.append({
                        "type": "SPECIFIC_CAMPAIGN",
                        "campaign_id": opportunity["campaign_id"],
                        "campaign_name": opportunity["campaign_name"],
                        "platform": opportunity.get("platform", "Unknown"),
                        "issue_type": opportunity["type"],
                        "recommendation": opportunity["recommendation"],
                        "description": opportunity["description"],
                        "suggested_action": opportunity["suggested_action"],
                        "potential_impact": opportunity["potential_impact"],
                        "estimated_roi": estimated_impact,
                        "priority": opportunity["priority"],
                        "implementation_timeline": "1-7 days"
                    })
            
            state.recommendations = recommendations
            
            logger.info(f"Generated {len(recommendations)} marketing recommendations")
            
        except Exception as e:
            logger.error(f"Error in generate_recommendations_node: {e}")
            state.error = f"Recommendation generation failed: {str(e)}"
        
        return state

    async def _execute_optimizations_node(self, state: MarketingState) -> MarketingState:
        """Execute automatic marketing optimizations"""
        try:
            logger.info("Executing marketing optimizations...")
            
            if state.error:
                return state
            
            optimizations_executed = []
            
            # Execute safe, automated optimizations
            high_priority_ops = [op for op in state.opportunities if op.get('priority') == 'HIGH']
            
            for opportunity in high_priority_ops:
                if opportunity["type"] == "LOW_ROAS":
                    # Log low ROAS campaign for review (would integrate with ad platform APIs)
                    optimization_log = {
                        "action": "FLAGGED_FOR_BUDGET_REDUCTION",
                        "campaign_id": opportunity["campaign_id"],
                        "campaign_name": opportunity["campaign_name"],
                        "platform": opportunity.get("platform"),
                        "current_roas": opportunity["current_roas"],
                        "recommended_action": "Reduce spend by 50%",
                        "potential_savings": opportunity["current_spend"] * 0.5,
                        "timestamp": int(time.time())
                    }
                    
                    logger.warning(f"LOW_ROAS_CAMPAIGN_FLAGGED: {json.dumps(optimization_log)}")
                    optimizations_executed.append(f"Flagged {opportunity['campaign_name']} for budget reduction - potential savings: ₦{optimization_log['potential_savings']:,.0f}")
                
                elif opportunity["type"] == "SCALE_OPPORTUNITY":
                    # Log scaling opportunity
                    scaling_log = {
                        "action": "SCALING_OPPORTUNITY_IDENTIFIED",
                        "campaign_id": opportunity["campaign_id"],
                        "campaign_name": opportunity["campaign_name"],
                        "platform": opportunity.get("platform"),
                        "current_spend": opportunity["current_spend"],
                        "current_roas": opportunity["current_roas"],
                        "recommended_increase": "20-30%",
                        "estimated_additional_revenue": opportunity["current_spend"] * 0.25 * opportunity["current_roas"],
                        "timestamp": int(time.time())
                    }
                    
                    logger.info(f"SCALING_OPPORTUNITY_IDENTIFIED: {json.dumps(scaling_log)}")
                    optimizations_executed.append(f"Identified scaling opportunity for {opportunity['campaign_name']} - potential additional revenue: ₦{scaling_log['estimated_additional_revenue']:,.0f}")
            
            # Generate optimization summary for marketing team
            if high_priority_ops:
                optimization_summary = {
                    "summary_type": "MARKETING_OPTIMIZATION_ALERT",
                    "high_priority_issues": len(high_priority_ops),
                    "total_potential_savings": sum(op.get("current_spend", 0) * 0.3 for op in high_priority_ops if op["type"] == "LOW_ROAS"),
                    "total_potential_revenue": sum(op.get("current_spend", 0) * 0.25 * op.get("current_roas", 0) for op in high_priority_ops if op["type"] == "SCALE_OPPORTUNITY"),
                    "campaigns_flagged": [op["campaign_name"] for op in high_priority_ops],
                    "timestamp": int(time.time())
                }
                
                logger.info(f"MARKETING_OPTIMIZATION_SUMMARY: {json.dumps(optimization_summary)}")
                optimizations_executed.append(f"Generated optimization summary for {len(high_priority_ops)} high-priority campaigns")
            
            # Create automated reporting for marketing dashboard
            performance_report = {
                "report_type": "AUTOMATED_PERFORMANCE_REPORT",
                "total_spend": state.performance_metrics.get("total_spend", 0),
                "total_revenue": state.performance_metrics.get("total_revenue", 0),
                "overall_roas": state.performance_metrics.get("overall_roas", 0),
                "opportunities_count": len(state.opportunities),
                "campaigns_analyzed": len(state.campaign_data),
                "generated_at": int(time.time())
            }
            
            logger.info(f"PERFORMANCE_REPORT_GENERATED: {json.dumps(performance_report)}")
            optimizations_executed.append(f"Generated automated performance report for {len(state.campaign_data)} campaigns")
            
            state.optimizations_executed = optimizations_executed
            state.status = "completed"
            
            logger.info(f"Executed {len(optimizations_executed)} marketing optimizations")
            
        except Exception as e:
            logger.error(f"Error in execute_optimizations_node: {e}")
            state.error = f"Optimization execution failed: {str(e)}"
            state.status = "error"
        
        return state

    def _format_opportunities_for_ai(self, opportunities: List[Dict]) -> str:
        """Format opportunities for AI analysis"""
        if not opportunities:
            return "No major issues identified"
        
        formatted = []
        for i, op in enumerate(opportunities, 1):
            formatted.append(f"{i}. {op.get('type')}: {op.get('description', 'No description')}")
        
        return "\n".join(formatted)

    def _calculate_opportunity_impact(self, opportunity: Dict, performance_metrics: Dict) -> Dict[str, Any]:
        """Calculate estimated impact of optimization opportunity"""
        try:
            if opportunity["type"] == "LOW_ROAS":
                potential_savings = opportunity.get("current_spend", 0) * 0.5
                return {
                    "type": "cost_savings",
                    "estimated_amount": potential_savings,
                    "confidence": "high",
                    "timeframe": "immediate"
                }
            elif opportunity["type"] == "SCALE_OPPORTUNITY":
                current_spend = opportunity.get("current_spend", 0)
                current_roas = opportunity.get("current_roas", 0)
                potential_revenue = current_spend * 0.3 * current_roas
                return {
                    "type": "revenue_increase",
                    "estimated_amount": potential_revenue,
                    "confidence": "medium",
                    "timeframe": "1-4 weeks"
                }
            else:
                return {
                    "type": "performance_improvement",
                    "estimated_amount": 0,
                    "confidence": "low",
                    "timeframe": "2-8 weeks"
                }
        except Exception:
            return {"type": "unknown", "estimated_amount": 0}

    async def run_marketing_optimization(self) -> Dict[str, Any]:
        """
        Main marketing optimization workflow using LangGraph
        """
        try:
            # Initialize state
            state = MarketingState()
            
            # Run the LangGraph workflow
            result = await self.workflow.ainvoke(state)
            
            # Handle different return types from LangGraph versions
            if isinstance(result, dict):
                # Newer LangGraph versions return dict
                final_state = result
            else:
                # Older versions might return state object
                final_state = result.__dict__ if hasattr(result, '__dict__') else result
            
            # Format response
            return {
                "status": final_state.get('status', 'completed'),
                "timestamp": int(time.time()),
                "campaign_analysis": {
                    "data": final_state.get('campaign_data', []),
                    "performance_metrics": final_state.get('performance_metrics', {})
                },
                "opportunities": final_state.get('opportunities', []),
                "recommendations": final_state.get('recommendations', []),
                "optimizations_executed": final_state.get('optimizations_executed', []),
                "summary": {
                    "campaigns_analyzed": len(final_state.get('campaign_data', [])),
                    "opportunities_found": len(final_state.get('opportunities', [])),
                    "recommendations_generated": len(final_state.get('recommendations', [])),
                    "optimizations_executed": len(final_state.get('optimizations_executed', []))
                },
                "error": final_state.get('error', None)
            }
            
        except Exception as e:
            logger.error(f"Error in marketing optimization workflow: {e}")
            return {
                "status": "error",
                "timestamp": int(time.time()),
                "error": str(e),
                "message": "Marketing optimization failed"
            }

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current marketing agent status"""
        return {
            "agent": "MarketingAgent",
            "status": "active",
            "llm_model": "gpt-4o-mini",
            "capabilities": [
                "campaign_performance_analysis",
                "roas_optimization",
                "budget_reallocation",
                "creative_testing_recommendations",
                "audience_targeting_optimization",
                "platform_specific_insights",
                "nigerian_market_specialization"
            ],
            "supported_platforms": ["Facebook", "Instagram", "Google", "TikTok", "Twitter"],
            "optimization_types": ["ROAS", "CTR", "Conversion Rate", "Budget Allocation", "Creative Performance"],
            "market_focus": "Nigerian market (NGN currency, local consumer behavior)"
        }

    def get_mock_campaign_data(self) -> List[Dict]:
        """Get mock campaign data for testing"""
        return [
            {
                "campaign_id": "FB_001",
                "campaign_name": "Summer Electronics Sale",
                "platform": "Facebook", 
                "spend": 150000,
                "revenue": 360000,
                "roas": 2.4,
                "status": "Needs Optimization"
            },
            {
                "campaign_id": "IG_003", 
                "campaign_name": "Fashion Collection",
                "platform": "Instagram",
                "spend": 85000,
                "revenue": 435000,
                "roas": 5.12,
                "status": "High Performer - Scale"
            }
        ]