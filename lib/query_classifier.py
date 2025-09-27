"""
Intelligent Query Classifier for Business Questions
Determines whether a user question needs database context or can be answered with general advice
"""
import logging
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from .config import LLMManager
import json

logger = logging.getLogger(__name__)

class QueryClassifier:
    """
    Classifies user queries to determine:
    1. Whether they need specific business data from user's database
    2. Whether they can be answered with general business advice
    3. What type of data would be helpful if needed
    """

    def __init__(self):
        self.llm = LLMManager.get_chat_llm()

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify a user query to determine data needs

        Returns:
        {
            "needs_data": bool,
            "query_type": str,  # "specific_data", "general_advice", "mixed"
            "data_requirements": List[str],  # What data would be helpful
            "can_answer_without_data": bool,
            "confidence": float,
            "response_strategy": str
        }
        """
        try:
            classification_prompt = f"""
            Analyze this business question to determine if it needs specific business data or can be answered with general advice:

            Query: "{query}"

            Classification criteria:

            NEEDS SPECIFIC DATA (needs_data: true):
            - Questions about "my sales", "our inventory", "my customers"
            - Requests for specific numbers, trends, or analysis from user's business
            - Questions like "What are my top selling products?", "How much revenue did I make last month?"
            - Analysis of specific business performance metrics

            GENERAL ADVICE (needs_data: false):
            - Questions about general business strategies, market trends, best practices
            - "What might cause declining sales?", "How to improve customer retention?"
            - Industry advice, general recommendations, strategic guidance
            - Questions that can be answered with business knowledge without specific data

            MIXED (needs_data: false, but suggest data would help):
            - General questions that would benefit from specific data for better answers
            - "How to optimize pricing?" (general advice possible, but user's pricing data would help)

            Data types that might be helpful:
            - sales_data: Transaction history, revenue, product performance
            - customer_data: Customer demographics, behavior, retention
            - inventory_data: Stock levels, turnover, supply chain
            - marketing_data: Campaign performance, ROAS, traffic
            - financial_data: Costs, margins, profitability
            - operational_data: Store performance, efficiency metrics

            Respond with JSON:
            {{
                "needs_data": true/false,
                "query_type": "specific_data" | "general_advice" | "mixed",
                "data_requirements": ["sales_data", "inventory_data", etc.],
                "can_answer_without_data": true/false,
                "confidence": 0.0-1.0,
                "response_strategy": "request_data" | "provide_general_advice" | "offer_both",
                "reasoning": "brief explanation of classification"
            }}
            """

            messages = [
                SystemMessage(content="You are a business query classifier. Analyze whether questions need specific user data or can be answered with general business advice. Respond only with valid JSON."),
                HumanMessage(content=classification_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            result = json.loads(response.content.strip())

            # Validate and set defaults
            classification = {
                "needs_data": result.get("needs_data", False),
                "query_type": result.get("query_type", "general_advice"),
                "data_requirements": result.get("data_requirements", []),
                "can_answer_without_data": result.get("can_answer_without_data", True),
                "confidence": result.get("confidence", 0.5),
                "response_strategy": result.get("response_strategy", "provide_general_advice"),
                "reasoning": result.get("reasoning", "")
            }

            logger.info(f"Query classified as: {classification['query_type']} (confidence: {classification['confidence']})")
            return classification

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to general advice if classification fails
            return {
                "needs_data": False,
                "query_type": "general_advice",
                "data_requirements": [],
                "can_answer_without_data": True,
                "confidence": 0.3,
                "response_strategy": "provide_general_advice",
                "reasoning": f"Classification failed: {str(e)}"
            }

    async def generate_data_request(self, classification: Dict[str, Any], original_query: str) -> str:
        """
        Generate a helpful message requesting user to connect their database
        """
        try:
            data_types = classification.get("data_requirements", [])

            data_descriptions = {
                "sales_data": "sales transactions, revenue, and product performance",
                "customer_data": "customer demographics, purchase history, and behavior patterns",
                "inventory_data": "stock levels, product availability, and supply chain metrics",
                "marketing_data": "campaign performance, advertising spend, and conversion metrics",
                "financial_data": "costs, profit margins, and financial performance",
                "operational_data": "store performance, operational efficiency, and logistics"
            }

            needed_data = [data_descriptions.get(dt, dt) for dt in data_types]

            request_prompt = f"""
            The user asked: "{original_query}"

            This question requires specific business data to provide accurate insights.
            Data needed: {', '.join(needed_data)}

            Generate a helpful, professional one liner message that:
            1. Acknowledges their question
            2. Explains why specific data would help provide better insights
            3. Asks them to connect their database or provide relevant data
            4. Offers to provide general advice in the meantime if helpful
            5. Suggests what specific data would be most valuable

            Keep it concise and helpful, not pushy.
            """

            messages = [
                SystemMessage(content="You are a helpful business analyst. Generate a polite request for data to better help the user."),
                HumanMessage(content=request_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Error generating data request: {e}")
            return "To provide more specific insights for your question, I'd need access to your business data. Would you like to connect your database, or shall I provide some general recommendations instead?"

    async def generate_general_advice(self, query: str, classification: Dict[str, Any]) -> str:
        """
        Generate general business advice without requiring specific data
        """
        try:
            advice_prompt = f"""
            User Question: "{query}"
            Query Type: {classification.get('query_type', 'general_advice')}

            Provide helpful, actionable business advice for this question without requiring specific user data.

            Guidelines:
            1. Give practical, implementable recommendations
            2. Consider common business scenarios and best practices
            3. Provide specific strategies and tactics where possible
            4. Include relevant metrics to track and benchmarks when helpful
            5. Consider different business sizes and types
            6. Be specific enough to be actionable but general enough to apply broadly

            If this is about a specific business scenario (like declining sales), provide:
            - Common causes and factors to investigate
            - Proven strategies to address the issue
            - Steps to diagnose the specific situation
            - Metrics to track improvement

            Make it comprehensive and valuable even without their specific data.
            """

            messages = [
                SystemMessage(content="You are an experienced business consultant providing practical advice. Focus on actionable strategies and best practices."),
                HumanMessage(content=advice_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Error generating general advice: {e}")
            return "I'd be happy to help with your business question. Could you provide a bit more context about your specific situation so I can give you more targeted advice?"