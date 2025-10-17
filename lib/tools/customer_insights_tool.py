
from langchain.tools import BaseTool
from typing import Optional, Dict, Any, List
from pydantic import Field, BaseModel
import logging
import json
import re
from lib.config import LLMManager, DatabaseManager

logger = logging.getLogger(__name__)

class CustomerInsightsInput(BaseModel):
    """Input schema for CustomerInsightsTool."""
    session_id: str = Field(description="User session ID")
    query: str = Field(description="Natural language query for customer insights")

class CustomerInsightsTool(BaseTool):
    """Tool for answering questions about customer insights."""

    name: str = "customer_insights"
    description: str = """
    Analyzes customer data to provide insights on sales and customer intelligence.
    """
    args_schema: type[BaseModel] = CustomerInsightsInput

    def _run(
        self,
        session_id: str,
        query: str,
    ) -> str:
        """Execute the customer insights tool."""
        try:
            logger.info(f"Customer insights tool triggered for session {session_id}")
            logger.info(f"Query: {query}")

            # Simple routing based on keywords in the query
            if "top" in query and "valuable" in query and "clients" in query:
                return self._handle_top_valuable_clients_query(session_id, query)
            else:
                return self._handle_generic_customer_query(session_id, query)

        except Exception as e:
            logger.error(f"Customer insights tool error: {str(e)}")
            return json.dumps({
                "success": False,
                "error": "Customer insights tool failed",
                "message": str(e)
            }, indent=2)

    def _handle_top_valuable_clients_query(self, session_id: str, query: str) -> str:
        """Handles queries about top valuable clients."""
        # 1. Parse the user's prompt to extract parameters
        llm = LLMManager.get_chat_llm()
        prompt = f"""Extract the number of clients and the time frame from the following query: "{query}"

        Respond with a JSON object with the following keys:
        - "top_n": integer (default to 10 if not specified)
        - "time_frame_days": integer (default to 365 if not specified)
        """
        response = llm.invoke(prompt)
        params = json.loads(response.content)
        top_n = params.get("top_n", 10)
        time_frame_days = params.get("time_frame_days", 365)

        # 2. Construct the SQL query
        # This is a simplified query and assumes a certain database schema.
        # In a real-world scenario, this would be much more complex and would be
        # generated based on the detected schema of the user's database.
        sql_query = f"""
        SELECT
            c.customer_id,
            c.name,
            c.email,
            c.signup_date,
            c.total_orders,
            c.total_spent
        FROM
            customers c
        JOIN
            orders o ON c.customer_id = o.customer_id
        WHERE
            o.order_date >= NOW() - INTERVAL '{time_frame_days} days'
        GROUP BY
            c.customer_id
        ORDER BY
            SUM(o.total_amount) DESC
        LIMIT {top_n};
        """

        # 3. Execute the SQL query
        db_manager = DatabaseManager()
        results = db_manager._query_postgres_database(db_manager.get_user_connection(session_id)['database_url'], "customer_data", session_id)

        # 4. Analyze the results and generate insights
        if not results:
            return json.dumps({"success": True, "insights": "No data found for the specified criteria."}, indent=2)

        insights_prompt = f"""Analyze the following list of top {top_n} most valuable clients from the past {time_frame_days} days and identify their common characteristics. What defines a 'whale' customer based on this data?

        Data:
        {json.dumps(results, indent=2)}

        Provide a summary of your findings.
        """
        insights_response = llm.invoke(insights_prompt)

        response = {
            "success": True,
            "insights": insights_response.content,
            "data": results
        }
        return json.dumps(response, indent=2)

    def _handle_generic_customer_query(self, session_id: str, query: str) -> str:
        """Handles generic customer-related queries."""
        # For now, this is a placeholder. In a real implementation, this would
        # also use an LLM to parse the query and construct a SQL query.

        db_manager = DatabaseManager()
        db_schema = db_manager.get_database_schema(session_id=session_id)

        llm = LLMManager.get_chat_llm()
        prompt = f"""Understand what this query is actually talking about and what data is needed to answer the query. "{query}"

        You will also be given the database schema "{db_schema}"
        generate the appropriate query required to retrieve the needed data from the database so you can answer this question.

        Respond with a JSON object with the following keys:
        - "user_query_summary": a detailed summary of what the user prompt actually means with possible guides to answering the user question.
        - "db_query": the query to use to fetch the required data from the db if needed. 
        """
        response = llm.invoke(prompt)

        results = db_manager._query_postgres_database(db_manager.get_user_connection(session_id)['database_url'], "customer_data", session_id)

        prompt1 = f"""Here is the response of the first llm call to understand the user query and fetch the required data from the database.
        "{response}", "{results}", 
        Here is the original query.
        "{query}"

        using all the information above, answer the users query in a very detailed and targeted manner. do not hallucinate or give wrong answers.
        Do not make up any data, only use the data provided in the results above to answer the query.
        Be straight to the point and concise when answering the query. 

        return the response in json format with the following keys:
        - "success": boolean, true if the query was answered successfully, false otherwise.
        - "insights": string, the detailed but concise answer to the user's query.

        """
        main_response = llm.invoke(prompt1)
        return json.dumps({
            "success": True,
            "insights": main_response.content
        }, indent=2)

    async def _arun(
        self,
        session_id: str,
        query: str,
    ) -> str:
        """Async version - calls sync method."""
        return self._run(session_id, query)
