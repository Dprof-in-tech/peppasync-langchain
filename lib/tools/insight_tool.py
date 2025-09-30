from langchain.tools import BaseTool
from typing import Dict, List, Any
import logging
import json
from datetime import date, datetime
from decimal import Decimal
from ..config import LLMManager
from ..peppagenbi import GenBISQL
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

def json_serializer(obj):
    """Custom JSON serializer for date/datetime/decimal objects"""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class InsightGenerationTool(BaseTool):
    """Unified business analysis tool - generates insights, recommendations, and alerts in one pass"""
    name: str = "unified_business_analysis"
    description: str = "Analyze business data and generate comprehensive insights, actionable recommendations, and critical alerts"

    def _run(self, query: str, business_data: Dict[str, Any], conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Unified analysis: generates insights, recommendations, and alerts in one pass.
        No more complex orchestration or reference resolving needed.
        """
        try:
            llm = LLMManager.get_chat_llm()

            # Extract sales data (primary data source for most queries)
            sales_data = business_data.get("sales_data", [])
            inventory_data = business_data.get("inventory_data", [])
            campaign_data = business_data.get("campaign_data", [])

            # Aggregate sales data by product (30-day and 14-day summaries)
            product_summary = self._aggregate_sales_by_product(sales_data)

            # Get expert knowledge from vector store
            expert_insights = self._retrieve_expert_knowledge(query, sales_data)

            # Format conversation history if available
            conversation_context = ""
            if conversation_history:
                conversation_context = "PREVIOUS CONVERSATION:\n"
                for exchange in conversation_history[-3:]:  # Last 3 exchanges for context
                    # Handle both formats: {'query': '...', 'response': '...'} or {'role': '...', 'content': '...'}
                    if 'query' in exchange and 'response' in exchange:
                        user_query = exchange['query']
                        assistant_response = exchange['response']
                        # If response is JSON, extract the insights
                        if isinstance(assistant_response, str) and assistant_response.strip().startswith('{'):
                            try:
                                response_json = json.loads(assistant_response)
                                assistant_response = response_json.get('insights', assistant_response)
                            except:
                                pass
                        conversation_context += f"USER: {user_query}\nASSISTANT: {assistant_response}\n\n"
                    elif 'role' in exchange and 'content' in exchange:
                        role = exchange.get('role', 'unknown')
                        content = exchange.get('content', '')
                        conversation_context += f"{role.upper()}: {content}\n"
                conversation_context += "\n"

            # Analyze the data and generate everything in one prompt
            analysis_prompt = f"""
            You are a business intelligence analyst. Analyze the following query and business data, then provide a comprehensive response.

            {conversation_context}CURRENT USER QUERY: "{query}"

            PRODUCT PERFORMANCE SUMMARY (last 30 days):
            {json.dumps(product_summary, indent=2, default=json_serializer)}

            INVENTORY DATA:
            {json.dumps(inventory_data, indent=2, default=json_serializer)}

            CAMPAIGN DATA:
            {json.dumps(campaign_data, indent=2, default=json_serializer)}

            EXPERT MARKETING KNOWLEDGE:
            {expert_insights[:500]}...

            RESPOND WITH A JSON OBJECT IN THIS EXACT FORMAT:
            {{
                "insights": "Brief summary of the current situation based on the data (2-3 sentences max)",
                "recommendations": [
                    {{
                        "type": "STRATEGIC/OPERATIONAL/TACTICAL",
                        "priority": "HIGH/MEDIUM/LOW",
                        "action": "Specific action to take",
                        "details": "Why this matters and expected impact",
                        "timeline": "Implementation timeframe"
                    }}
                ],
                "alerts": [
                    {{
                        "severity": "CRITICAL/WARNING/INFO",
                        "message": "Alert message",
                        "details": "More context about the alert",
                        "action_required": "What needs to be done"
                    }}
                ]
            }}

            IMPORTANT FORMATTING RULES:
            - If user asks "how to improve" or "what should I do", PUT YOUR ANSWER IN THE RECOMMENDATIONS ARRAY, NOT just in insights
            - insights = brief data summary (what IS happening)
            - recommendations = actionable advice (what to DO about it)
            - alerts = critical issues requiring immediate attention

            CRITICAL INSTRUCTIONS:
            - IF THERE IS A PREVIOUS CONVERSATION: Use context to understand what the user is referring to (e.g., "this product", "improve sales for this")
            - ANSWER THE ACTUAL QUESTION ASKED: If user asks "how to improve", provide actionable recommendations, not just stats
            - Base insights ONLY on the ACTUAL DATA provided - DO NOT make up or hallucinate numbers
            - Use the exact numbers from the PRODUCT PERFORMANCE SUMMARY provided above
            - If user asks about "last 2 weeks" or "last 14 days", use the *_14d fields
            - If user asks about "last month" or "last 30 days", use the *_30d fields
            - Include specific numbers, product names, and metrics FROM THE DATA
            - Recommendations should incorporate expert marketing knowledge and be ACTIONABLE
            - Generate 2-4 recommendations when appropriate
            - Only create alerts for truly critical issues (low stock, failing campaigns, etc.)
            - Respond ONLY with valid JSON, no markdown or extra text
            - DO NOT HALLUCINATE OR INVENT NUMBERS - only use what's in the data provided
            """

            messages = [
                SystemMessage(content="You are a business intelligence analyst. Respond only with valid JSON."),
                HumanMessage(content=analysis_prompt)
            ]

            response = llm.invoke(messages)

            # Parse JSON response
            try:
                result = json.loads(response.content.strip())
                logger.info(f"Unified analysis complete: {len(result.get('recommendations', []))} recommendations, {len(result.get('alerts', []))} alerts")
                logger.info(f"Full LLM response: {json.dumps(result, indent=2, default=json_serializer)}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                # Return fallback structure
                return {
                    "insights": response.content,
                    "recommendations": [],
                    "alerts": []
                }

        except Exception as e:
            logger.error(f"Error in unified analysis: {e}")
            return {
                "insights": f"Analysis failed: {str(e)}",
                "recommendations": [],
                "alerts": [{"severity": "CRITICAL", "message": "Analysis error", "details": str(e), "action_required": "Check logs"}]
            }

    def _aggregate_sales_by_product(self, sales_data: List[Dict]) -> List[Dict]:
        """
        Aggregate sales data by product with both 30-day and 14-day summaries.
        Returns list of products sorted by 30-day revenue.
        """
        from collections import defaultdict
        from datetime import datetime, timedelta

        product_stats = defaultdict(lambda: {
            'total_revenue_30d': 0,
            'total_units_30d': 0,
            'transaction_count_30d': 0,
            'total_revenue_14d': 0,
            'total_units_14d': 0,
            'transaction_count_14d': 0,
            'product_name': None,
            'product_id': None
        })

        # Calculate date thresholds
        now = datetime.now()
        two_weeks_ago = now - timedelta(days=14)

        for item in sales_data:
            if not isinstance(item, dict):
                continue

            # Extract product identifier
            product_id = item.get('product_id') or item.get('id') or 'unknown'
            product_name = item.get('product_name') or item.get('name') or f"Product {product_id}"

            # Extract revenue
            revenue = float(item.get('total_amount') or item.get('sales_amount') or item.get('amount') or 0)

            # Extract units
            units = int(item.get('quantity') or item.get('units_sold') or item.get('qty') or 1)

            # Extract and parse date
            sale_date_str = item.get('sale_date') or item.get('order_date') or item.get('created_at')
            is_recent = False
            if sale_date_str:
                try:
                    # Handle different date formats
                    sale_date = datetime.fromisoformat(str(sale_date_str).replace('Z', '+00:00'))
                    is_recent = sale_date >= two_weeks_ago
                except:
                    pass

            # Aggregate for 30-day period (all data)
            product_stats[product_id]['product_id'] = product_id
            product_stats[product_id]['product_name'] = product_name
            product_stats[product_id]['total_revenue_30d'] += revenue
            product_stats[product_id]['total_units_30d'] += units
            product_stats[product_id]['transaction_count_30d'] += 1

            # Aggregate for 14-day period (recent only)
            if is_recent:
                product_stats[product_id]['total_revenue_14d'] += revenue
                product_stats[product_id]['total_units_14d'] += units
                product_stats[product_id]['transaction_count_14d'] += 1

        # Convert to list and sort by 30-day revenue
        result = list(product_stats.values())
        result.sort(key=lambda x: x['total_revenue_30d'], reverse=True)

        return result

    def _retrieve_expert_knowledge(self, query: str, sales_data: List[Dict] = None) -> str:
        """Retrieve relevant expert marketing insights from vector database"""
        try:
            # Create a more comprehensive query for vector search
            enhanced_query = self._enhance_query_for_retrieval(query, sales_data)
            
            # Direct Pinecone similarity search instead of going through GenBISQL
            from ..peppagenbi import GenBISQL
            genbi = GenBISQL()
            
            if genbi.vector_store:
                # Use direct similarity search from Pinecone
                docs = genbi.vector_store.similarity_search(enhanced_query, k=5)
                
                if docs:
                    # Combine the retrieved documents into expert insights
                    expert_insights = "\n\n".join([doc.page_content for doc in docs])
                    logger.info(f"Successfully retrieved {len(docs)} expert insights for query: {query[:50]}...")
                    return expert_insights
                else:
                    logger.warning("No expert insights found in vector search")
                    return self._get_default_marketing_insights(query)
            else:
                logger.warning("Vector store not available, using default knowledge")
                return self._get_default_marketing_insights(query)
                
        except Exception as e:
            logger.error(f"Error retrieving expert knowledge: {e}")
            return self._get_default_marketing_insights(query)

    # Remove the async method since we're using direct similarity search
    # async def _async_retrieve_knowledge(self, enhanced_query: str):

    def _enhance_query_for_retrieval(self, query: str, sales_data: List[Dict] = None) -> str:
        """Enhance the query with context for better vector retrieval"""
        query_lower = query.lower()
        
        # Identify the type of business question to retrieve relevant expert insights
        if any(phrase in query_lower for phrase in ['top performing', 'best performing', 'highest revenue']):
            enhanced_query = f"product performance optimization strategies revenue growth top products {query}"
        elif any(phrase in query_lower for phrase in ['improve sales', 'increase sales', 'boost sales']):
            enhanced_query = f"sales improvement strategies marketing tactics revenue optimization {query}"
        elif any(phrase in query_lower for phrase in ['marketing', 'campaign', 'advertising']):
            enhanced_query = f"marketing strategies advertising campaigns customer acquisition {query}"
        elif any(phrase in query_lower for phrase in ['customer', 'retention', 'loyalty']):
            enhanced_query = f"customer retention strategies loyalty programs customer lifetime value {query}"
        else:
            enhanced_query = f"business growth strategies marketing optimization {query}"
        
        # Add data context if available
        if sales_data:
            total_revenue = sum(item.get('total_amount', 0) for item in sales_data[:5])
            enhanced_query += f" revenue analysis business intelligence data-driven decisions total revenue {total_revenue}"
        
        return enhanced_query

    def _get_default_marketing_insights(self, query: str) -> str:
        """Provide default marketing insights when vector retrieval fails"""
        return """
        Key Marketing Principles to Consider:
        
        1. Customer-Centric Approach: Focus on understanding customer needs and preferences to drive product development and marketing strategies.
        
        2. Data-Driven Decision Making: Use analytics and performance metrics to guide strategic decisions and optimize marketing spend.
        
        3. Multi-Channel Marketing: Leverage various marketing channels including digital, social media, and traditional methods for maximum reach.
        
        4. Product Positioning: Clearly communicate unique value propositions and differentiate from competitors in the market.
        
        5. Customer Retention: Implement loyalty programs and exceptional customer service to increase lifetime value and reduce churn.
        """

    async def _arun(self, query: str, business_data: Dict[str, Any], conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Async version of unified analysis"""
        return self._run(query, business_data, conversation_history)