import os
import time
import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio
from dotenv import load_dotenv
from .prompt_engine import PeppaPromptEngine
from .config import LLMManager, DatabaseManager
from .utils.common import ValidationUtils, LoggingUtils, ResponseFormatter
from .agent import UnifiedBusinessAgent

load_dotenv()
logger = logging.getLogger(__name__)

# LangGraph State - Simplified
class ConversationState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str = ""
    query: str = ""
    original_query: str = ""  # Keep track of original query
    context: str = ""
    response: str = ""
    conversation_history: List[Dict] = Field(default_factory=list)
    citations: List[Dict] = Field(default_factory=list)
    error: Optional[str] = None
    status: str = "initialized"

class ConversationManager:
    """LangGraph-powered conversation manager for contextual RAG"""
    
    def __init__(self):
        self.llm = LLMManager.get_chat_llm()

        # Initialize advanced prompt engine  
        self.prompt_engine = PeppaPromptEngine()

        # Initialize unified business agent
        self.business_agent = UnifiedBusinessAgent()

        # In-memory session storage (in production, use Redis or database)
        self.sessions: Dict[str, Dict] = {}

        # Build the simplified LangGraph workflow
        self.workflow = self._build_conversation_graph()

    def _build_conversation_graph(self) -> StateGraph:
        """Build the simplified LangGraph conversation workflow"""
        
        # Define the workflow steps - simplified flow
        workflow = StateGraph(ConversationState)
        
        # Add nodes for conversation flow
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("enhance_query", self._enhance_query_node)
        workflow.add_node("generate_response", self._simplified_generate_response_node)
        workflow.add_node("save_session", self._save_session_node)

        # Set entry point and edges - direct path without classification
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "load_context")
        workflow.add_edge("load_context", "enhance_query")
        workflow.add_edge("enhance_query", "generate_response")
        workflow.add_edge("generate_response", "save_session")
        workflow.add_edge("save_session", END)
        
        return workflow.compile()

    async def _analyze_query_node(self, state: ConversationState) -> ConversationState:
        """Simplified query analysis - just prep the query for processing"""
        try:
            logger.info(f"Analyzing query for session: {state.session_id}")
            
            # Store original query
            state.original_query = state.query
            
            # Load conversation history from session if available
            session_data = self.sessions.get(state.session_id, {})
            state.conversation_history = session_data.get('history', [])
            
            logger.info(f"Query prepared for direct processing: {state.query}")
            return state
            
        except Exception as e:
            logger.error(f"Error in analyze_query_node: {e}")
            state.error = f"Query analysis failed: {str(e)}"
            return state



    async def _classify_prompt_node(self, state: ConversationState) -> ConversationState:
        """Classify the prompt using advanced prompt engine"""
        try:
            logger.info(f"Classifying prompt: {state.query[:100]}...")
            
            # Use the original query (before context enhancement) for classification
            original_query = state.query
            if "Current question:" in state.query:
                # Extract original query if it was enhanced with context
                parts = state.query.split("Current question:")
                if len(parts) > 1:
                    original_query = parts[-1].strip()
            
            # Analyze the prompt
            prompt_analysis = await self.prompt_engine.analyze_prompt(original_query)
            
            # Update state with analysis results
            state.prompt_analysis = prompt_analysis
            state.business_category = prompt_analysis.get('category', 'general_analysis')
            state.analysis_type = prompt_analysis.get('analysis_type', 'descriptive')
            
            # Determine if advanced analysis is needed
            state.requires_advanced_analysis = (
                prompt_analysis.get('requires_forecasting', False) or
                prompt_analysis.get('requires_scenario_analysis', False) or
                prompt_analysis.get('confidence', 0) > 0.7
            )
            
            logger.info(f"Classified as {state.business_category} - {state.analysis_type} (confidence: {prompt_analysis.get('confidence', 0)})")
            
        except Exception as e:
            logger.error(f"Error in classify_prompt_node: {e}")
            state.error = f"Prompt classification failed: {str(e)}"
            state.business_category = "general_analysis"
        
        return state

    async def _load_context_node(self, state: ConversationState) -> ConversationState:
        """Load conversation history - simplified"""
        try:
            logger.info(f"Loading context for session: {state.session_id}")
            # Context is already loaded in analyze_query_node
            return state
                
        except Exception as e:
            logger.error(f"Error in load_context_node: {e}")
            state.error = f"Context loading failed: {str(e)}"
            return state

    async def _enhance_query_node(self, state: ConversationState) -> ConversationState:
        """Enhanced query handling - simplified"""
        try:
            logger.info("Query enhancement - pass through for direct processing")
            # Query is passed through without modification for direct processing
            return state
            
        except Exception as e:
            logger.error(f"Error in enhance_query_node: {e}")
            state.error = f"Query enhancement failed: {str(e)}"
            return state

    async def _simplified_generate_response_node(self, state: ConversationState) -> ConversationState:
        """Generate response directly using the unified agent without classification"""
        try:
            logger.info(f"Processing query directly: {state.query}")
            
            # Load all available data and let the agent decide what it needs
            database_manager = DatabaseManager()
            
            # Check if user has database connected
            if DatabaseManager.has_user_connection(state.session_id):
                logger.info("User has database connected - using unified agent with user data")
                
                # Get comprehensive business data using the available methods
                business_data = {
                    "sales_data": database_manager.get_data(state.session_id, "sales_data"),
                    "product_data": database_manager.get_data(state.session_id, "product_data"),
                    "inventory_data": database_manager.get_data(state.session_id, "inventory_data"),
                    "customer_data": database_manager.get_data(state.session_id, "customer_data"),
                    "revenue_data": database_manager.get_data(state.session_id, "revenue_data"),
                }
                
                # Create unified agent with all tools available
                unified_agent = UnifiedBusinessAgent(session_id=state.session_id)
                
                # Direct query to agent with full context
                response_data = await unified_agent.analyze_direct_query(
                    query=state.query,
                    business_data=business_data,
                    conversation_history=state.conversation_history
                )
                
                state.response = response_data
                
            else:
                logger.info("No database connected - providing general business advice")
                
                # Create a simple general advice response
                general_advice = f"I'd be happy to help with your question: '{state.query}'. To provide specific insights about your business, I would need access to your business data. You can connect your database to get personalized analysis, or I can provide general business advice based on industry best practices."
                
                # Structure as JSON for consistency
                structured_response = {
                    "type": "general_advice",
                    "insights": general_advice,
                    "recommendations": [
                        {
                            "title": "Connect Your Database",
                            "description": "Link your business database to get specific insights about your products, sales, and customers.",
                            "priority": "HIGH"
                        }
                    ],
                    "metadata": {
                        "analysis_type": "General Business Advice",
                        "business_category": "Advisory", 
                        "timestamp": int(time.time())
                    }
                }
                
                import json
                state.response = json.dumps(structured_response, indent=2)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in simplified response generation: {e}")
            
            error_response = {
                "type": "error",
                "insights": "I encountered an error processing your request. Please try again.",
                "recommendations": [],
                "metadata": {
                    "analysis_type": "Error",
                    "business_category": "System",
                    "timestamp": int(time.time())
                }
            }
            
            import json
            state.response = json.dumps(error_response, indent=2)
            state.error = f"Response generation failed: {str(e)}"
            return state



    async def _save_session_node(self, state: ConversationState) -> ConversationState:
        """Save conversation exchange to session memory"""
        try:
            if state.session_id not in self.sessions:
                self.sessions[state.session_id] = {
                    'created_at': int(time.time()),
                    'history': []
                }
            
            # Add new exchange to history
            new_exchange = {
                'query': state.query,
                'response': state.response,
                'timestamp': int(time.time())
            }
            
            self.sessions[state.session_id]['history'].append(new_exchange)
            self.sessions[state.session_id]['last_updated'] = int(time.time())
            
            # Keep only last 10 exchanges to manage memory
            if len(self.sessions[state.session_id]['history']) > 10:
                self.sessions[state.session_id]['history'] = self.sessions[state.session_id]['history'][-10:]
            
            logger.info(f"Saved conversation exchange for session: {state.session_id}")
            
        except Exception as e:
            logger.error(f"Error in save_session_node: {e}")
            state.error = f"Session saving failed: {str(e)}"
            
        return state

    async def _llm_analyze_context_need(self, query: str, history: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze if query needs context"""
        try:
            if not history:
                return {'needs_context': False, 'is_followup': False}
            
            # Get recent history
            recent_history = history[-2:] if len(history) >= 2 else history
            history_text = ""
            
            for exchange in recent_history:
                history_text += f"User: {exchange.get('query', '')}\n"
                history_text += f"Assistant: {exchange.get('response', '')[:150]}...\n\n"
            
            prompt = f"""
Recent conversation:
{history_text}

New user query: "{query}"

Analyze if this new query refers to or builds upon the previous conversation.

Respond with a JSON object:
{{
    "needs_context": true/false,
    "is_followup": true/false,
    "reasoning": "brief explanation"
}}
"""
            
            system_msg = "You are a conversation analyzer. Respond only with valid JSON."
            
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                analysis = json.loads(response.content.strip())
                return analysis
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM context analysis response: {response.content}")
                return {'needs_context': False, 'is_followup': False}
                
        except Exception as e:
            logger.error(f"Error in LLM context analysis: {e}")
            return {'needs_context': False, 'is_followup': False}

    async def process_conversation(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Main method to process conversation using LangGraph workflow
        """
        session_id = ValidationUtils.validate_session_id(session_id)
        start_time = time.time()
        
        try:
            # Validate input
            if not ValidationUtils.validate_prompt(query):
                return ResponseFormatter.format_error(
                    "Invalid prompt provided",
                    session_id,
                    context_used=False,
                    is_followup=False,
                    citations=[]
                )
            
            # Initialize state
            state = ConversationState(
                session_id=session_id,
                query=query
            )
            
            # Run the LangGraph workflow
            result = await self.workflow.ainvoke(state)
            
            # Handle different return types from LangGraph versions
            if isinstance(result, dict):
                # Newer LangGraph versions return dict
                final_state = result
            else:
                # Older versions might return state object
                final_state = result.__dict__ if hasattr(result, '__dict__') else result
            
            # Format successful response
            response_data = {
                'output': final_state.get('response', 'No response generated'),
                'citations': final_state.get('citations', []),
                'context_used': False,  # Simplified - no context analysis
                'is_followup': False,   # Simplified - no followup analysis  
                'error': final_state.get('error', None)
            }
            
            LoggingUtils.log_performance("process_conversation", start_time, session_id=session_id)
            return ResponseFormatter.format_success(response_data, session_id)
            
        except Exception as e:
            LoggingUtils.log_error("process_conversation", e, session_id=session_id)
            return ResponseFormatter.format_error(
                'I encountered an error processing your request. Please try again.',
                session_id,
                context_used=False,
                is_followup=False,
                citations=[]
            )

    async def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Cleared session: {session_id}")
                return True
            else:
                logger.warning(f"Session not found: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation session"""
        try:
            if session_id not in self.sessions:
                return {'exists': False, 'exchange_count': 0}
            
            session_data = self.sessions[session_id]
            history = session_data.get('history', [])
            
            return {
                'exists': True,
                'exchange_count': len(history),
                'first_query': history[0].get('query', '') if history else '',
                'last_updated': session_data.get('last_updated', 0),
                'created_at': session_data.get('created_at', 0),
                'session_age_hours': (time.time() - session_data.get('created_at', time.time())) / 3600 if history else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            return {'exists': False, 'exchange_count': 0, 'error': str(e)}

    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than specified hours"""
        try:
            current_time = time.time()
            sessions_to_remove = []
            
            for session_id, session_data in self.sessions.items():
                session_age = (current_time - session_data.get('created_at', current_time)) / 3600
                if session_age > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
            
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            return len(sessions_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific numbered section from insights text"""
        try:
            if not text:
                return ""
            
            # Map section names to their typical numbers
            section_numbers = {
                "Key Findings": "1",
                "Trends and Patterns": "2", 
                "Specific Recommendations": "3",
                "Risk Factors": "4"
            }
            
            section_num = section_numbers.get(section_name, "")
            if not section_num:
                return ""
            
            # Look for patterns like "1. Key Findings" or "1 Key Findings"
            import re
            pattern = rf"{section_num}\.?\s*{re.escape(section_name)}\s*(.*?)(?=\n\d\.|\n\d\s|$)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if match:
                return match.group(1).strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting section {section_name}: {e}")
            return ""