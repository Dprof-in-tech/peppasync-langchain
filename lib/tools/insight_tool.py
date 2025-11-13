from langchain.tools import BaseTool
from typing import Dict, List, Any
import logging
import json
from datetime import date, datetime
from decimal import Decimal
from ..config import LLMManager
from ..peppagenbi import GenBISQL
from ..action_handler import ActionHandler
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
        Also handles action confirmations (draft emails, reports, etc.)
        """
        try:
            # First, check if user is confirming a suggested action
            if conversation_history and ActionHandler.detect_confirmation(query):
                pending_action = ActionHandler.extract_pending_action(conversation_history, query)
                if pending_action:
                    logger.info(f"User confirmed action: {pending_action['action'].get('action_type')}")
                    # Note: This sync method can't call async _handle_action_confirmation
                    # It should only be used in sync contexts. Use _arun for async contexts.
                    raise NotImplementedError("Action confirmation requires async context. Use _arun instead.")

            llm = LLMManager.get_chat_llm()

            # Extract sales data (primary data source for most queries)
            sales_data = business_data.get("sales_data", [])
            inventory_data = business_data.get("inventory_data", [])
            campaign_data = business_data.get("campaign_data", [])

            # Limit inventory data to top 20 items to reduce LLM context
            inventory_data_limited = inventory_data[:20] if len(inventory_data) > 20 else inventory_data

            # Get expert knowledge from vector store (skip for simple queries)
            expert_insights = self._retrieve_expert_knowledge(query, sales_data) if self._needs_expert_insights(query) else ""

            # Format conversation history if available
            conversation_context = ""
            if conversation_history:
                conversation_context = "PREVIOUS CONVERSATION:\n"
                for exchange in conversation_history[-5:]:  # Last 5 exchanges for context
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

            {conversation_context} CURRENT USER QUERY: "{query}"

            SALES DATA:
            {sales_data}

            INVENTORY DATA (top {len(inventory_data_limited)} items):
            {json.dumps(inventory_data_limited, indent=2, default=json_serializer)}

            CAMPAIGN DATA:
            {json.dumps(campaign_data, indent=2, default=json_serializer)}
            {("EXPERT SALES AND MARKETING KNOWLEDGE:\n" + expert_insights[:500] + "...") if expert_insights else ""}

            RESPOND WITH A JSON OBJECT IN THIS EXACT FORMAT:
            {{
                "insights": "The answer to the question here. Can include insights or a very brief summary about the situation but must contain the answer to the question asked. this must be CONCISE and straight to the point, we dont want the users to get lost in the context.")",
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
                ],
                "suggested_actions": [
                    {{
                        "action_type": "draft_email|create_report|generate_forecast|create_purchase_order | other | would you like to know more about why this is happening?",
                        "description": "What you can help with",
                        "prompt": "Question to ask user (e.g., 'Would you like me to draft a reorder email?', 'Would you like to know more about why this is happening?')"
                    }}
                ]
            }}

            Here is an example of a good recommendation:
            "Bundle Product 16 with Product 23 in a promotional offer to boost sales of the underperforming item. This tactic leverages the high sales volume of Product 23 to increase visibility and sales of Product 16. Expected impact: 15-20% increase in sales for Product 16 over the next month. Implementation timeframe: 1-2 weeks to set up the promotion and marketing materials."

            IMPORTANT FORMATTING RULES:
            - If user asks "how to improve" or "what should I do", PUT YOUR ANSWER IN THE RECOMMENDATIONS ARRAY, NOT just in insights
            - insights = brief data summary (what IS happening) + the direct answer to the question asked
            - recommendations = actionable advice (what to DO about it) with clear and specific numeric projections of what can happen when that is done.
            - alerts = critical issues requiring immediate attention
            - suggested_actions = proactive offers to help either with an action or to explain more (e.g., "low stock alert" → suggest "draft_email" to supplier, suggest "would you like to know more about why this is happening?")
            - When you generate an alert, also add a suggested_action asking if user wants help with it

            CRITICAL INSTRUCTIONS:
            - IF THERE IS A PREVIOUS CONVERSATION: Use context to understand what the user is referring to (e.g., "this product", "improve sales for this")
            - ANSWER THE ACTUAL QUESTION ASKED: If user asks "how to improve", provide actionable recommendations, not just stats
            - Base insights ONLY on the ACTUAL DATA provided - DO NOT make up or hallucinate numbers
            - Use the exact numbers from the PRODUCT PERFORMANCE SUMMARY provided above
            - If user asks about "last 2 weeks" or "last 14 days", use the *_14d fields
            - If user asks about "last month" or "last 30 days", use the *_30d fields
            - Include specific numbers, product names, and metrics FROM THE DATA
            - Recommendations should incorporate expert marketing knowledge and be ACTIONABLE
            - EXPERT MARKETING KNOWLEDGE / INSIGHTS should be used to inform recommendations, especially for strategic questions. they are there to help you provide better advice by leveraging proven marketing principles and tactics.
            - Generate 2-4 recommendations when appropriate
            - Only create alerts for truly critical issues (low stock, failing campaigns, etc.)
            - Respond ONLY with valid JSON - NO markdown code blocks, NO ```json, just pure JSON
            - DO NOT HALLUCINATE OR INVENT NUMBERS - only use what's in the data provided
            - ONLY ANSWER QUESTIONS RELATED TO SALES AND MARKETING. IF the question asked by the user is not related to sales and marketing , respond that the question is out of scope and do not generate any recommendations or any alerts.
            - ALWAYS CHECK MY CURRENT PERFORMANCE before giving any answer or making any recommendations to how i can improve my sales and marketing performance.
            - RECOMMENDATIONS MUST BE FACTUAL, REALISTIC AND SPECIFIC. Don't make vague suggestions like improve marketing or increase sales - be specific and provide a valid tactic or strategy with specific numbers, targets and projected results.
            - INSIGHTS SHOULD BE DATA-DRIVEN AND OBJECTIVE. Avoid personal opinions or subjective statements - focus on the facts and figures from the data provided.
            - Be more SPECIFIC on where to spend MARKETING $ and WHY you should spend it.
            - ALL ANSWERS AND OUTCOMES MUST BE SPECIFIC, NUMERIC AND MEASURABLE WHEREVER POSSIBLE. Vague, generic or non-numeric answers are not acceptable.
            - ALWAYS ANSWER THE QUESTIONS IN THE INSIGHTS SECTION IF IT IS A SIMPLE DATA QUESTION. If the user is just asking for data or stats, provide that in insights. Only use recommendations if the user is asking for advice or what to do.
            - THE ANSWER MUST BE VERY SIMPLE AND CONCISE. DO NOT OVER-COMPLICATE OR OVER-EXPLAIN. THE USER WANTS A STRAIGHTFORWARD ANSWER TO THEIR QUESTION, NOT A DETAILED REPORT.

            IMPORTANT: 
            - Your response must be valid JSON that can be parsed directly. Do NOT wrap it in markdown.
            - If i ask a trick question about improving my sales or marketing performance to a lower target than what i currently have, do NOT fall for it - always check my current performance first and call it out if the target is lower than current.
            """

            messages = [
                SystemMessage(content="You are a business intelligence analyst. Respond ONLY with valid JSON. Do NOT use markdown code blocks or any formatting - just pure JSON."),
                HumanMessage(content=analysis_prompt)
            ]

            response = llm.invoke(messages)

            # Parse JSON response
            try:
                result = json.loads(response.content.strip())
                logger.info(f"Unified analysis complete: {len(result.get('recommendations', []))} recommendations, {len(result.get('alerts', []))} alerts, {len(result.get('suggested_actions', []))} suggested actions")
                # logger.info(f"Full LLM response: {json.dumps(result, indent=2, default=json_serializer)}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                # Return fallback structure
                return {
                    "insights": response.content,
                    "recommendations": response.get('recommendations', []),
                    "alerts": response.get('alerts', []),
                    "suggested_actions": response.get('suggested_actions', [])
                }

        except Exception as e:
            logger.error(f"Error in unified analysis: {e}")
            return {
                "insights": f"Analysis failed: {str(e)}",
                "recommendations": [],
                "alerts": [{"severity": "CRITICAL", "message": "Analysis error", "details": str(e), "action_required": "Check logs"}],
                "suggested_actions": []
            }

    def _needs_expert_insights(self, query: str) -> bool:
        """
        Determine if query needs expert marketing insights.
        Skip for simple data queries to speed up response.
        """
        query_lower = query.lower()

        # Skip expert insights for simple data queries
        simple_patterns = [
            "top performing", "worst performing", "show me", "what are",
            "list", "how many", "total", "sales", "revenue"
        ]

        # Needs expert insights for strategic/marketing questions
        expert_patterns = [
            "how to improve", "how can i", "what should i do",
            "recommend", "strategy", "marketing", "grow", "increase"
        ]

        # If it's asking for advice/recommendations, use expert insights
        if any(pattern in query_lower for pattern in expert_patterns):
            return True

        # If it's just a simple data query, skip expert insights
        if any(pattern in query_lower for pattern in simple_patterns):
            return False

        # Default: use expert insights
        return True
    
    def _aggregate_sales_by_product(self, sales_data: List[Dict]) -> List[Dict]:
        """
        Aggregate sales data by product for both Shopify (nested line_items) and Postgres (flat) formats.
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

        now = datetime.now()
        two_weeks_ago = now - timedelta(days=14)

        for item in sales_data:
            if not isinstance(item, dict):
                continue

            # Shopify order: has 'line_items' (list of products in the order)
            if 'line_items' in item and isinstance(item['line_items'], list):
                sale_date_str = item.get('created_at') or item.get('order_date') or item.get('sale_date')
                is_recent = False
                if sale_date_str:
                    try:
                        sale_date = datetime.fromisoformat(str(sale_date_str).replace('Z', '+00:00'))
                        is_recent = sale_date >= two_weeks_ago
                    except:
                        pass
                for li in item['line_items']:
                    product_id = li.get('product_id') or li.get('id') or 'unknown'
                    product_name = li.get('title') or li.get('name') or li.get('product_name') or f"Product {product_id}"
                    # Shopify: price * quantity per line item
                    try:
                        revenue = float(li.get('price') or li.get('total_amount') or li.get('sales_amount') or li.get('amount') or 0) * int(li.get('quantity') or 1)
                    except Exception:
                        revenue = 0
                    units = int(li.get('quantity') or li.get('units_sold') or li.get('qty') or 1)

                    product_stats[product_id]['product_id'] = product_id
                    product_stats[product_id]['product_name'] = product_name
                    product_stats[product_id]['total_revenue_30d'] += revenue
                    product_stats[product_id]['total_units_30d'] += units
                    product_stats[product_id]['transaction_count_30d'] += 1
                    if is_recent:
                        product_stats[product_id]['total_revenue_14d'] += revenue
                        product_stats[product_id]['total_units_14d'] += units
                        product_stats[product_id]['transaction_count_14d'] += 1
            else:
                # Postgres/flat format
                product_id = item.get('product_id') or item.get('id') or 'unknown'
                product_name = item.get('product_name') or item.get('name') or f"Product {product_id}"
                try:
                    revenue = float(item.get('total_amount') or item.get('sales_amount') or item.get('amount') or 0)
                except Exception:
                    revenue = 0
                units = int(item.get('quantity') or item.get('units_sold') or item.get('qty') or 1)
                sale_date_str = item.get('sale_date') or item.get('order_date') or item.get('created_at')
                is_recent = False
                if sale_date_str:
                    try:
                        sale_date = datetime.fromisoformat(str(sale_date_str).replace('Z', '+00:00'))
                        is_recent = sale_date >= two_weeks_ago
                    except:
                        pass
                product_stats[product_id]['product_id'] = product_id
                product_stats[product_id]['product_name'] = product_name
                product_stats[product_id]['total_revenue_30d'] += revenue
                product_stats[product_id]['total_units_30d'] += units
                product_stats[product_id]['transaction_count_30d'] += 1
                if is_recent:
                    product_stats[product_id]['total_revenue_14d'] += revenue
                    product_stats[product_id]['total_units_14d'] += units
                    product_stats[product_id]['transaction_count_14d'] += 1

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

    async def _handle_action_confirmation(self, query: str, pending_action: Dict, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle when user confirms a suggested action.
        Generate the actual output (email draft, report, etc.) via LLM.
        """
        action_type = pending_action['action'].get('action_type')
        context = pending_action['context']

        logger.info(f"Generating {action_type} for user...")

        try:
            # Extract relevant data based on action type
            sales_data = business_data.get("sales_data", [])
            inventory_data = business_data.get("inventory_data", [])
            product_summary = self._aggregate_sales_by_product(sales_data)

            # Generate the appropriate output via LLM
            if action_type == 'draft_email':
                # Use ActionHandler to generate email
                draft = await ActionHandler.generate_draft_email(
                    product_data=product_summary[0] if product_summary else {},
                    inventory_data=inventory_data,
                    context=context
                )

                return {
                    "insights": f"I've drafted a requisition email based on the stock alert. You can copy and send this to your supplier:",
                    "draft_content": draft,
                    "draft_type": "email",
                    "recommendations": [],
                    "alerts": [],
                    "suggested_actions": []
                }

            elif action_type == 'create_report':
                draft = await ActionHandler.generate_sales_report(
                    product_data=product_summary[0] if product_summary else {},
                    context=context
                )

                return {
                    "insights": "Here's your sales performance report:",
                    "draft_content": draft,
                    "draft_type": "report",
                    "recommendations": [],
                    "alerts": [],
                    "suggested_actions": []
                }

            elif action_type == 'generate_forecast':
                draft = await ActionHandler.generate_forecast(
                    product_data=product_summary[0] if product_summary else {},
                    sales_history=sales_data,
                    context=context
                )

                return {
                    "insights": "Here's your sales forecast:",
                    "draft_content": draft,
                    "draft_type": "forecast",
                    "recommendations": [],
                    "alerts": [],
                    "suggested_actions": []
                }

            elif action_type == 'create_purchase_order':
                draft = await ActionHandler.generate_purchase_order(
                    product_data=product_summary[0] if product_summary else {},
                    inventory_data=inventory_data,
                    context=context
                )

                return {
                    "insights": "Here's your purchase order draft:",
                    "draft_content": draft,
                    "draft_type": "purchase_order",
                    "recommendations": [],
                    "alerts": [],
                    "suggested_actions": []
                }

            else:
                logger.warning(f"Unknown action type: {action_type}")
                return {
                    "insights": f"I'm not sure how to handle that action type: {action_type}",
                    "recommendations": [],
                    "alerts": [],
                    "suggested_actions": []
                }

        except Exception as e:
            logger.error(f"Error generating action output: {e}")
            return {
                "insights": f"I encountered an error generating the {action_type}: {str(e)}",
                "recommendations": [],
                "alerts": [],
                "suggested_actions": []
            }

    async def _arun(self, query: str, business_data: Dict[str, Any], conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Async version of unified analysis"""
        try:
            # First, check if user is confirming a suggested action
            if conversation_history and ActionHandler.detect_confirmation(query):
                pending_action = ActionHandler.extract_pending_action(conversation_history, query)
                if pending_action:
                    logger.info(f"User confirmed action: {pending_action['action'].get('action_type')}")
                    return await self._handle_action_confirmation(query, pending_action, business_data)

            # For non-action-confirmation queries, delegate to sync _run
            return self._run(query, business_data, conversation_history)
        except NotImplementedError:
            # If _run raised NotImplementedError for action confirmation, handle it here
            if conversation_history and ActionHandler.detect_confirmation(query):
                pending_action = ActionHandler.extract_pending_action(conversation_history, query)
                if pending_action:
                    return await self._handle_action_confirmation(query, pending_action, business_data)
            raise