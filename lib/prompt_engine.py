import os
import json
import logging
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import re

# Import centralized configuration
from .config import LLMManager, AppConfig

load_dotenv()
logger = logging.getLogger(__name__)

class PromptClassificationState(BaseModel):
    """State for prompt classification and routing"""
    original_prompt: str = ""
    prompt_category: str = ""
    analysis_type: str = ""
    confidence: float = 0.0
    requires_forecasting: bool = False
    requires_scenario_analysis: bool = False
    key_metrics: List[str] = Field(default_factory=list)
    time_horizon: str = ""
    business_context: Dict[str, Any] = Field(default_factory=dict)

class PeppaPromptEngine:
    """Advanced prompt understanding and routing engine for retail BI"""
    
    def __init__(self):
        self.llm = LLMManager.get_chat_llm()
        
        # Define prompt categories and their characteristics
        self.prompt_categories = {
            "sales_revenue": {
                "keywords": ["sales", "revenue", "profit", "growth", "increase", "boost", "projected sales"],
                "requires_forecasting": True,
                "typical_metrics": ["revenue", "sales volume", "growth rate", "profit margin"]
            },
            "marketing_customer": {
                "keywords": ["marketing", "ad spend", "ROAS", "customer acquisition", "traffic", "conversion"],
                "requires_scenario_analysis": True,
                "typical_metrics": ["ROAS", "CAC", "conversion rate", "traffic", "CTR"]
            },
            "pricing_promotions": {
                "keywords": ["price", "discount", "promotion", "margin", "bundle", "free shipping"],
                "requires_scenario_analysis": True,
                "typical_metrics": ["profit margin", "AOV", "sales volume", "price elasticity"]
            },
            "inventory_operations": {
                "keywords": ["inventory", "stock", "reorder", "warehouse", "supply chain", "lead time"],
                "requires_forecasting": True,
                "typical_metrics": ["stock levels", "turnover", "holding costs", "stockout rate"]
            },
            "channel_performance": {
                "keywords": ["store", "online", "channel", "marketplace", "omnichannel", "footfall"],
                "requires_scenario_analysis": True,
                "typical_metrics": ["sales by channel", "transaction value", "customer retention"]
            },
            "customer_behavior": {
                "keywords": ["customer", "churn", "retention", "loyalty", "CLV", "segment"],
                "requires_forecasting": True,
                "typical_metrics": ["churn rate", "CLV", "retention rate", "satisfaction"]
            },
            "financial_efficiency": {
                "keywords": ["profit", "cost", "ROI", "efficiency", "expense", "operational"],
                "requires_scenario_analysis": True,
                "typical_metrics": ["profit margin", "ROI", "operational costs", "efficiency ratio"]
            },
            "strategic_scenarios": {
                "keywords": ["what if", "scenario", "impact", "strategy", "competitive", "launch"],
                "requires_scenario_analysis": True,
                "typical_metrics": ["comprehensive business metrics"]
            }
        }
        
        # Build the classification workflow
        self.workflow = self._build_classification_graph()

    def _build_classification_graph(self) -> StateGraph:
        """Build LangGraph workflow for prompt classification and analysis"""
        workflow = StateGraph(PromptClassificationState)
        
        # Add nodes
        workflow.add_node("classify_prompt", self._classify_prompt_node)
        workflow.add_node("extract_metrics", self._extract_metrics_node)
        workflow.add_node("determine_analysis", self._determine_analysis_node)
        workflow.add_node("prepare_context", self._prepare_context_node)
        
        # Set entry point and edges
        workflow.set_entry_point("classify_prompt")
        workflow.add_edge("classify_prompt", "extract_metrics")
        workflow.add_edge("extract_metrics", "determine_analysis")
        workflow.add_edge("determine_analysis", "prepare_context")
        workflow.add_edge("prepare_context", END)
        
        return workflow.compile()

    async def _classify_prompt_node(self, state: PromptClassificationState) -> PromptClassificationState:
        """Classify the prompt into business categories"""
        try:
            # Handle dict vs object state
            if isinstance(state, dict):
                prompt = state.get('original_prompt', '').lower()
            else:
                prompt = state.original_prompt.lower()
            
            # Score each category based on keyword matching
            category_scores = {}
            for category, info in self.prompt_categories.items():
                score = 0
                for keyword in info["keywords"]:
                    if keyword in prompt:
                        score += 1
                
                # Boost score for exact phrase matches
                for keyword in info["keywords"]:
                    if len(keyword.split()) > 1 and keyword in prompt:
                        score += 2
                
                category_scores[category] = score
            
            # Get the highest scoring category
            best_category = max(category_scores.items(), key=lambda x: x[1])
            
            # Update state (handle both dict and object)
            if isinstance(state, dict):
                if best_category[1] > 0:
                    state['prompt_category'] = best_category[0]
                    state['confidence'] = min(best_category[1] / 5.0, 1.0)  # Normalize to 0-1
                    
                    # Set analysis requirements based on category
                    category_info = self.prompt_categories[best_category[0]]
                    state['requires_forecasting'] = category_info.get("requires_forecasting", False)
                    state['requires_scenario_analysis'] = category_info.get("requires_scenario_analysis", False)
                else:
                    # Use AI for complex classification
                    await self._ai_classify_prompt_dict(state)
            else:
                if best_category[1] > 0:
                    state.prompt_category = best_category[0]
                    state.confidence = min(best_category[1] / 5.0, 1.0)  # Normalize to 0-1
                    
                    # Set analysis requirements based on category
                    category_info = self.prompt_categories[best_category[0]]
                    state.requires_forecasting = category_info.get("requires_forecasting", False)
                    state.requires_scenario_analysis = category_info.get("requires_scenario_analysis", False)
                else:
                    # Use AI for complex classification
                    await self._ai_classify_prompt(state)
            
            category = state.get('prompt_category') if isinstance(state, dict) else state.prompt_category
            confidence = state.get('confidence') if isinstance(state, dict) else state.confidence
            logger.info(f"Classified prompt as: {category} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Error in classify_prompt_node: {e}")
            if isinstance(state, dict):
                state['prompt_category'] = "general_analysis"
                state['confidence'] = 0.5
            else:
                state.prompt_category = "general_analysis"
                state.confidence = 0.5
        
        return state

    async def _ai_classify_prompt(self, state: PromptClassificationState):
        """Use AI for complex prompt classification"""
        try:
            classification_prompt = f"""
            Analyze this retail business intelligence prompt and classify it:
            
            Prompt: "{state.original_prompt}"
            
            Categories:
            1. sales_revenue - Sales performance, revenue forecasting, growth analysis
            2. marketing_customer - Marketing campaigns, customer acquisition, advertising
            3. pricing_promotions - Pricing strategies, discounts, promotions
            4. inventory_operations - Stock management, supply chain, operations
            5. channel_performance - Store performance, online/offline channels
            6. customer_behavior - Customer analytics, retention, segmentation
            7. financial_efficiency - Profitability, costs, operational efficiency
            8. strategic_scenarios - What-if analysis, strategic planning
            
            Respond with JSON:
            {{
                "category": "category_name",
                "confidence": 0.0-1.0,
                "requires_forecasting": true/false,
                "requires_scenario_analysis": true/false,
                "reasoning": "brief explanation"
            }}
            """
            
            messages = [
                SystemMessage(content="You are a business intelligence prompt classifier. Respond only with valid JSON."),
                HumanMessage(content=classification_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            result = json.loads(response.content.strip())
            
            state.prompt_category = result.get("category", "general_analysis")
            state.confidence = result.get("confidence", 0.5)
            state.requires_forecasting = result.get("requires_forecasting", False)
            state.requires_scenario_analysis = result.get("requires_scenario_analysis", False)
            
        except Exception as e:
            logger.error(f"Error in AI classification: {e}")
            state.prompt_category = "general_analysis"

    async def _ai_classify_prompt_dict(self, state: dict):
        """Use AI for complex prompt classification (dict version)"""
        try:
            classification_prompt = f"""
            Classify this business intelligence prompt into one of these retail categories:
            
            Prompt: "{state.get('original_prompt', '')}"
            
            Categories:
            1. sales_revenue - Sales performance, revenue forecasting, growth analysis
            2. marketing_customer - Marketing campaigns, customer acquisition, advertising
            3. pricing_promotions - Pricing strategies, discounts, promotions
            4. inventory_operations - Stock management, supply chain, operations
            5. channel_performance - Store performance, online/offline channels
            6. customer_behavior - Customer analytics, retention, segmentation
            7. financial_efficiency - Profitability, costs, operational efficiency
            8. strategic_scenarios - What-if analysis, strategic planning
            
            Respond with JSON:
            {{
                "category": "category_name",
                "confidence": 0.0-1.0,
                "requires_forecasting": true/false,
                "requires_scenario_analysis": true/false,
                "reasoning": "brief explanation"
            }}
            """
            
            messages = [
                SystemMessage(content="You are a business intelligence prompt classifier. Respond only with valid JSON."),
                HumanMessage(content=classification_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            result = json.loads(response.content.strip())
            
            state['prompt_category'] = result.get("category", "general_analysis")
            state['confidence'] = result.get("confidence", 0.5)
            state['requires_forecasting'] = result.get("requires_forecasting", False)
            state['requires_scenario_analysis'] = result.get("requires_scenario_analysis", False)
            
        except Exception as e:
            logger.error(f"Error in AI classification: {e}")
            state['prompt_category'] = "general_analysis"

    async def _extract_metrics_node(self, state: PromptClassificationState) -> PromptClassificationState:
        """Extract key metrics and numerical values from the prompt"""
        try:
            prompt = state.original_prompt
            
            # Extract numerical values and their contexts
            numerical_patterns = [
                (r'(\d+)%', 'percentage'),
                (r'\$(\d+(?:,\d{3})*)', 'currency_usd'),
                (r'₦(\d+(?:,\d{3})*)', 'currency_ngn'),
                (r'(\d+) days?', 'days'),
                (r'(\d+) weeks?', 'weeks'),
                (r'(\d+) months?', 'months'),
                (r'Q(\d)', 'quarter'),
                (r'(\d+) members?', 'people'),
                (r'(\d+)x|(\d+) times', 'multiplier')
            ]
            
            extracted_values = {}
            for pattern, value_type in numerical_patterns:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if matches:
                    extracted_values[value_type] = matches
            
            # Extract common business metrics mentioned
            metric_keywords = [
                'revenue', 'sales', 'profit', 'margin', 'ROAS', 'ROI', 'conversion rate',
                'customer acquisition cost', 'CAC', 'lifetime value', 'CLV', 'churn rate',
                'inventory turnover', 'stock levels', 'average order value', 'AOV',
                'traffic', 'impressions', 'clicks', 'CTR'
            ]
            
            mentioned_metrics = []
            prompt_lower = prompt.lower()
            for metric in metric_keywords:
                if metric.lower() in prompt_lower:
                    mentioned_metrics.append(metric)
            
            state.key_metrics = mentioned_metrics
            state.business_context['extracted_values'] = extracted_values
            
            # Determine time horizon
            if any(term in prompt_lower for term in ['next quarter', 'q4', 'q1', 'q2', 'q3']):
                state.time_horizon = 'quarterly'
            elif any(term in prompt_lower for term in ['next 6 months', '6 months', 'half year']):
                state.time_horizon = '6_months'
            elif any(term in prompt_lower for term in ['next year', '12 months', 'annual']):
                state.time_horizon = 'annual'
            elif any(term in prompt_lower for term in ['30 days', 'next month', 'monthly']):
                state.time_horizon = 'monthly'
            elif any(term in prompt_lower for term in ['week', 'weekly']):
                state.time_horizon = 'weekly'
            else:
                state.time_horizon = 'unspecified'
                
            logger.info(f"Extracted metrics: {mentioned_metrics}, Time horizon: {state.time_horizon}")
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
        
        return state

    async def _determine_analysis_node(self, state: PromptClassificationState) -> PromptClassificationState:
        """Determine the specific type of analysis needed"""
        try:
            prompt_lower = state.original_prompt.lower()
            
            # Determine analysis type based on prompt characteristics
            if any(term in prompt_lower for term in ['forecast', 'predict', 'project', 'expect', 'will be']):
                state.analysis_type = 'predictive'
            elif any(term in prompt_lower for term in ['what if', 'scenario', 'impact of', 'would happen']):
                state.analysis_type = 'scenario'
            elif any(term in prompt_lower for term in ['why did', 'what caused', 'reason for', 'because']):
                state.analysis_type = 'diagnostic' 
            elif any(term in prompt_lower for term in ['recommend', 'should we', 'best strategy', 'optimize']):
                state.analysis_type = 'prescriptive'
            elif any(term in prompt_lower for term in ['compare', 'vs', 'versus', 'difference between']):
                state.analysis_type = 'comparative'
            else:
                state.analysis_type = 'descriptive'
                
            logger.info(f"Determined analysis type: {state.analysis_type}")
            
        except Exception as e:
            logger.error(f"Error determining analysis: {e}")
            state.analysis_type = 'descriptive'
        
        return state

    async def _prepare_context_node(self, state: PromptClassificationState) -> PromptClassificationState:
        """Prepare additional context for analysis"""
        try:
            # Add Nigerian retail context
            state.business_context.update({
                'market': 'Nigerian retail',
                'currency': 'NGN',
                'typical_profit_margins': {
                    'electronics': 0.15,
                    'fashion': 0.45,
                    'home_goods': 0.35
                },
                'seasonal_factors': {
                    'Q4': 1.4,  # Holiday boost
                    'Q1': 0.8,  # Post-holiday dip
                    'Q2': 1.1,  # Summer season
                    'Q3': 0.9   # Back-to-school
                },
                'market_conditions': {
                    'inflation_rate': 0.12,
                    'growth_rate': 0.08,
                    'competition_level': 'high'
                }
            })
            
            logger.info("Prepared business context for analysis")
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
        
        return state

    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Main method to analyze and route retail prompts"""
        try:
            # Initialize state
            state = PromptClassificationState(original_prompt=prompt)
            
            # Run the classification workflow
            result = await self.workflow.ainvoke(state)
            
            # Handle different return types from LangGraph versions
            if isinstance(result, dict):
                # Newer LangGraph versions return dict
                final_state = result
            else:
                # Older versions might return state object
                final_state = result.__dict__ if hasattr(result, '__dict__') else result
            
            return {
                'original_prompt': final_state.get('original_prompt', prompt),
                'category': final_state.get('prompt_category', 'general_analysis'),
                'analysis_type': final_state.get('analysis_type', 'descriptive'),
                'confidence': final_state.get('confidence', 0.5),
                'requires_forecasting': final_state.get('requires_forecasting', False),
                'requires_scenario_analysis': final_state.get('requires_scenario_analysis', False),
                'key_metrics': final_state.get('key_metrics', []),
                'time_horizon': final_state.get('time_horizon', 'unspecified'),
                'business_context': final_state.get('business_context', {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            return {
                'original_prompt': prompt,
                'category': 'general_analysis',
                'analysis_type': 'descriptive',
                'confidence': 0.5,
                'error': str(e)
            }

    async def generate_sophisticated_response(self, prompt_analysis: Dict[str, Any]) -> str:
        """Generate sophisticated business intelligence responses"""
        try:
            category = prompt_analysis.get('category', 'general_analysis')
            analysis_type = prompt_analysis.get('analysis_type', 'descriptive')
            original_prompt = prompt_analysis.get('original_prompt', '')
            context = prompt_analysis.get('business_context', {})
            
            # Create specialized system message based on category
            system_messages = {
                'sales_revenue': "You are a retail sales analyst specializing in revenue optimization and growth strategies for Nigerian businesses.",
                'marketing_customer': "You are a digital marketing expert focusing on customer acquisition and retention in the Nigerian market.",
                'pricing_promotions': "You are a pricing strategist with expertise in promotional campaigns and profit optimization.",
                'inventory_operations': "You are a supply chain and inventory management expert specializing in retail operations.",
                'channel_performance': "You are an omnichannel retail expert analyzing store and online performance.",
                'customer_behavior': "You are a customer analytics expert specializing in behavior analysis and segmentation.",
                'financial_efficiency': "You are a retail finance expert focusing on profitability and operational efficiency.",
                'strategic_scenarios': "You are a strategic business analyst specializing in scenario planning and competitive analysis."
            }
            
            system_message = system_messages.get(category, 
                "You are a comprehensive business intelligence analyst specializing in Nigerian retail.")
            
            # Create analysis framework based on type
            analysis_frameworks = {
                'predictive': """
                Provide a comprehensive predictive analysis including:
                1. FORECAST: Specific numerical predictions with confidence intervals
                2. ASSUMPTIONS: Key assumptions underlying the forecast
                3. SCENARIOS: Best case, expected, and worst case scenarios
                4. TIMELINE: When these outcomes are expected
                5. RISK FACTORS: What could affect the predictions
                6. RECOMMENDATIONS: Actions to achieve desired outcomes
                """,
                'scenario': """
                Provide detailed scenario analysis including:
                1. IMPACT ANALYSIS: Direct and indirect effects of the proposed change
                2. QUANTIFIED OUTCOMES: Specific metrics and expected changes
                3. COMPARATIVE SCENARIOS: Multiple options with trade-offs
                4. IMPLEMENTATION: Step-by-step approach
                5. SUCCESS METRICS: How to measure results
                6. CONTINGENCY PLANS: What to do if results differ from expectations
                """,
                'diagnostic': """
                Provide thorough diagnostic analysis including:
                1. ROOT CAUSE: Primary factors causing the observed trend/issue
                2. CONTRIBUTING FACTORS: Secondary influences
                3. DATA CORRELATION: Relationships between different metrics
                4. INDUSTRY CONTEXT: How this compares to market norms
                5. HISTORICAL PATTERNS: Similar situations and outcomes
                6. CORRECTIVE ACTIONS: Specific steps to address the issue
                """,
                'prescriptive': """
                Provide actionable prescriptive recommendations including:
                1. STRATEGIC RECOMMENDATIONS: High-level strategic actions
                2. TACTICAL STEPS: Specific implementation actions
                3. PRIORITIZATION: Which actions to take first and why
                4. RESOURCE REQUIREMENTS: What's needed to execute
                5. SUCCESS METRICS: KPIs to track progress
                6. OPTIMIZATION OPPORTUNITIES: Continuous improvement suggestions
                """
            }
            
            framework = analysis_frameworks.get(analysis_type, 
                "Provide comprehensive analysis with diagnostic, predictive, and prescriptive insights.")
            
            # Create comprehensive prompt
            comprehensive_prompt = f"""
            BUSINESS CONTEXT:
            - Market: Nigerian retail sector
            - Currency: Nigerian Naira (₦)
            - Business Context: {json.dumps(context, indent=2)}
            
            ANALYSIS REQUEST:
            {original_prompt}
            
            ANALYSIS FRAMEWORK:
            {framework}
            
            REQUIREMENTS:
            - Use specific numerical projections where possible
            - Consider Nigerian market conditions (inflation, seasonality, consumer behavior)
            - Provide actionable insights that can be implemented
            - Include confidence levels for predictions
            - Address both short-term and long-term implications
            - Consider competitive landscape and market dynamics
            
            Provide a detailed, professional analysis that directly answers the question with specific, actionable insights.
            """
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=comprehensive_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error analyzing your request. Please try rephrasing your question. Error: {str(e)}"

    def get_supported_categories(self) -> Dict[str, List[str]]:
        """Get all supported business categories and sample questions"""
        return {
            category: {
                "description": f"Analysis for {category.replace('_', ' ')} scenarios",
                "keywords": info["keywords"][:5],  # First 5 keywords as examples
                "sample_questions": self._get_sample_questions(category)
            }
            for category, info in self.prompt_categories.items()
        }

    def _get_sample_questions(self, category: str) -> List[str]:
        """Get sample questions for each category"""
        samples = {
            "sales_revenue": [
                "What would be the projected sales increase for Q4 if we offered a 15% discount?",
                "What's the forecasted revenue if we increase our average order value by 10%?",
                "Project our total sales for the upcoming holiday season."
            ],
            "marketing_customer": [
                "What is the projected increase in website traffic if we double our ad spend?",
                "If we allocate ₦500,000 to a new influencer campaign, what's the expected ROAS?",
                "Forecast new customers from a 'refer a friend' campaign with ₦1,000 credit."
            ],
            "pricing_promotions": [
                "What's the forecasted change in profit margin if we increase prices by 8%?",
                "Impact of offering free shipping on all orders over ₦5,000?",
                "Optimal discount level for clearance section to maximize profit?"
            ],
            "inventory_operations": [
                "If we reduce safety stock by 10%, what's the projected decrease in holding costs?",
                "Optimal number of specific products to hold for next six weeks?",
                "Impact of consolidating warehouses on operational expenses?"
            ],
            "channel_performance": [
                "Expected lift in in-store sales from promoting exclusive products online?",
                "Revenue projection for new pop-up store in first 3 months?",
                "Impact of 'buy online, pick up in-store' on online sales?"
            ],
            "customer_behavior": [
                "Projected decrease in churn with personalized email campaign?",
                "Expected CLV for new subscription box service subscribers?",
                "Impact of gamifying loyalty program on engagement?"
            ],
            "financial_efficiency": [
                "Projected profit margin if we reduce operational expenses by 5%?",
                "Impact of 10% inventory turnover increase on profitability?",
                "ROI projection for ₦1,000,000 investment in analytics dashboard?"
            ],
            "strategic_scenarios": [
                "Impact on sales if major competitor lowers prices by 15%?",
                "Effect of launching subscription model for popular product?",
                "Combined impact of increasing ad spend and launching loyalty program?"
            ]
        }
        return samples.get(category, ["Analysis questions for this category"])