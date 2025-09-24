from langchain.tools import BaseTool
from typing import Dict, List, Any
import logging
from ..config import LLMManager
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class InsightGenerationTool(BaseTool):
    """Generate AI-powered business insights from data"""
    name: str = "insight_generation"
    description: str = "Generate comprehensive business insights using AI analysis"

    def _run(self, data: List[Dict], analysis_type: str, business_context: str = "") -> str:
        """Generate insights from provided data"""
        try:
            llm = LLMManager.get_chat_llm()
            
            # Create analysis prompt based on type
            prompts = {
                "sales": "Analyze this sales data for trends, opportunities, and risks. Focus on Nigerian market conditions.",
                "inventory": "Analyze inventory levels and provide reorder recommendations. Consider Nigerian supply chain factors.",
                "marketing": "Analyze campaign performance and suggest optimizations. Focus on Nigerian digital marketing landscape.",
                "customer": "Analyze customer behavior and segmentation opportunities.",
                "financial": "Analyze financial performance and efficiency opportunities."
            }
            
            base_prompt = prompts.get(analysis_type, "Analyze this business data and provide insights.")
            
            analysis_prompt = f"""
            {base_prompt}
            
            Data to analyze:
            {data[:3]}  # Limit data size for prompt
            
            Business Context: {business_context}
            
            Provide:
            1. Key findings
            2. Trends and patterns
            3. Specific recommendations
            4. Risk factors
            5. Nigerian market considerations
            
            Be specific and actionable in your analysis.
            """
            
            messages = [
                SystemMessage(content="You are a business intelligence analyst specializing in Nigerian retail markets."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return f"Unable to generate insights: {str(e)}"

    async def _arun(self, data: List[Dict], analysis_type: str, business_context: str = "") -> str:
        """Async version of insight generation"""
        return self._run(data, analysis_type, business_context)