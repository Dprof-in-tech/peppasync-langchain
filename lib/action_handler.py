"""
Action Handler - Detects confirmations and generates action outputs via LLM
"""
import logging
import json
from typing import Dict, Any, List, Optional
from langchain.schema import HumanMessage, SystemMessage
from .config import LLMManager

logger = logging.getLogger(__name__)

class ActionHandler:
    """Handle proactive action requests from users"""

    @staticmethod
    def detect_confirmation(query: str) -> bool:
        """Detect if user is confirming a suggested action"""
        affirmative = [
            "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead",
            "do it", "please", "absolutely", "definitely", "sounds good",
            "let's do it", "proceed", "confirm", "approve", "draft it"
        ]
        query_lower = query.lower().strip()
        return any(word in query_lower for word in affirmative)

    @staticmethod
    def extract_pending_action(conversation_history: List[Dict], current_query: str = "") -> Optional[Dict]:
        """
        Extract the most relevant suggested action from conversation history.
        Matches user's current query against action descriptions.

        Args:
            conversation_history: Previous conversation exchanges
            current_query: User's current confirmation message

        Returns action details if found, None otherwise.
        """
        if not conversation_history:
            return None

        # Check last exchange for suggested_actions
        last_exchange = conversation_history[-1]
        if 'response' not in last_exchange:
            return None

        try:
            response_data = json.loads(last_exchange['response'])
            suggested_actions = response_data.get('suggested_actions', [])

            if not suggested_actions:
                return None

            # If there's only one action, return it
            if len(suggested_actions) == 1:
                return {
                    'action': suggested_actions[0],
                    'context': {
                        'insights': response_data.get('insights', ''),
                        'alerts': response_data.get('alerts', []),
                        'recommendations': response_data.get('recommendations', [])
                    }
                }

            # If multiple actions, try to match user's query to the right one
            if current_query:
                query_lower = current_query.lower()

                # Keywords for different action types
                action_keywords = {
                    'draft_email': ['email', 'mail', 'draft', 'send', 'write email'],
                    'create_report': ['report', 'analysis', 'document', 'summary report'],
                    'generate_forecast': ['forecast', 'predict', 'projection', 'future'],
                    'create_purchase_order': ['purchase', 'order', 'po', 'buy']
                }

                # Try to find best match
                best_match = None
                best_score = 0

                for action in suggested_actions:
                    action_type = action.get('action_type', '')
                    keywords = action_keywords.get(action_type, [])

                    # Count keyword matches
                    score = sum(1 for keyword in keywords if keyword in query_lower)

                    if score > best_score:
                        best_score = score
                        best_match = action

                # If we found a good match, use it
                if best_match and best_score > 0:
                    logger.info(f"Matched user query to action: {best_match.get('action_type')}")
                    return {
                        'action': best_match,
                        'context': {
                            'insights': response_data.get('insights', ''),
                            'alerts': response_data.get('alerts', []),
                            'recommendations': response_data.get('recommendations', [])
                        }
                    }

            # Default: return first action if no match found
            logger.info(f"No specific match, using first suggested action: {suggested_actions[0].get('action_type')}")
            return {
                'action': suggested_actions[0],
                'context': {
                    'insights': response_data.get('insights', ''),
                    'alerts': response_data.get('alerts', []),
                    'recommendations': response_data.get('recommendations', [])
                }
            }
        except Exception as e:
            logger.error(f"Error extracting pending action: {e}")

        return None

    @staticmethod
    async def generate_draft_email(product_data: Dict, inventory_data: List[Dict], context: Dict) -> str:
        """
        Generate a requisition email draft using LLM.

        Args:
            product_data: Product info from alerts/context
            inventory_data: Inventory data for supplier info
            context: Full context from previous analysis (insights, alerts)
        """
        llm = LLMManager.get_chat_llm()

        prompt = f"""
        You are drafting a professional requisition email to a supplier.

        CONTEXT FROM ANALYSIS:
        {context.get('insights', '')}

        ALERTS:
        {json.dumps(context.get('alerts', []), indent=2)}

        PRODUCT/INVENTORY DATA:
        {json.dumps(product_data, indent=2)}

        TASK:
        Draft a professional, concise email to the supplier requesting urgent reorder of the product(s) mentioned in the alerts.

        Include:
        - Professional subject line
        - Product details (name, ID, current stock level)
        - Quantity needed (suggest appropriate amount based on stock level)
        - Urgency indication if stock is critically low
        - Request for confirmation and delivery timeline

        Format as a complete, ready-to-send email. Do NOT include placeholders like [Your Name] - make it ready to copy/paste.
        Use "Procurement Team" as the sender.
        """

        messages = [
            SystemMessage(content="You are a professional business email writer. Write clear, concise, actionable emails."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    @staticmethod
    async def generate_sales_report(product_data: Dict, context: Dict) -> str:
        """Generate a sales performance report via LLM"""
        llm = LLMManager.get_chat_llm()

        prompt = f"""
        You are generating a sales performance report.

        CONTEXT FROM ANALYSIS:
        {context.get('insights', '')}

        PRODUCT PERFORMANCE DATA:
        {json.dumps(product_data, indent=2)}

        TASK:
        Create a professional sales performance report for this product.

        Include:
        - Executive summary
        - 30-day and 14-day performance metrics
        - Trend analysis
        - Key insights
        - Actionable recommendations

        Format as a clear, professional report ready to share with stakeholders.
        """

        messages = [
            SystemMessage(content="You are a business analyst creating professional reports. Be data-driven and actionable."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    @staticmethod
    async def generate_forecast(product_data: Dict, sales_history: List[Dict], context: Dict) -> str:
        """Generate a sales forecast via LLM"""
        llm = LLMManager.get_chat_llm()

        prompt = f"""
        You are generating a sales forecast.

        HISTORICAL PERFORMANCE:
        {json.dumps(product_data, indent=2)}

        CONTEXT:
        {context.get('insights', '')}

        TASK:
        Create a sales forecast for the next 7 and 30 days based on historical data.

        Include:
        - Historical trends summary
        - Forecast for next 7 days
        - Forecast for next 30 days
        - Confidence intervals
        - Recommended stock levels
        - Risk factors to consider

        Be realistic and data-driven. Format as a professional forecast document.
        """

        messages = [
            SystemMessage(content="You are a data analyst creating sales forecasts. Use historical data to make realistic projections."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    @staticmethod
    async def generate_purchase_order(product_data: Dict, inventory_data: List[Dict], context: Dict) -> str:
        """Generate a purchase order draft via LLM"""
        llm = LLMManager.get_chat_llm()

        prompt = f"""
        You are drafting a purchase order.

        CONTEXT:
        {context.get('insights', '')}

        PRODUCT/INVENTORY DATA:
        {json.dumps(product_data, indent=2)}

        TASK:
        Create a professional purchase order draft.

        Include:
        - PO number (format: PO-YYYYMMDD-###)
        - Date
        - Supplier information
        - Product details
        - Quantity and pricing
        - Total cost calculation
        - Payment terms
        - Delivery requirements

        Format as a formal purchase order ready for review and submission.
        """

        messages = [
            SystemMessage(content="You are creating professional purchase orders. Be precise with numbers and formatting."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()
