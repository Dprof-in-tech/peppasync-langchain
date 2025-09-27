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
from .query_classifier import QueryClassifier

load_dotenv()
logger = logging.getLogger(__name__)

# LangGraph State
class ConversationState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str = ""
    query: str = ""
    context: str = ""
    response: str = ""
    needs_context: bool = False
    is_followup: bool = False
    citations: List[Dict] = Field(default_factory=list)
    error: Optional[str] = None
    # Enhanced fields for prompt classification
    prompt_analysis: Dict[str, Any] = Field(default_factory=dict)
    business_category: str = ""
    analysis_type: str = ""
    requires_advanced_analysis: bool = False
    # Query classification fields
    query_classification: Dict[str, Any] = Field(default_factory=dict)
    needs_data: bool = False
    data_requirements: List[str] = Field(default_factory=list)
    can_answer_without_data: bool = True
    response_strategy: str = "provide_general_advice"

class ConversationManager:
    """LangGraph-powered conversation manager for contextual RAG"""
    
    def __init__(self):
        self.llm = LLMManager.get_chat_llm()

        # Initialize advanced prompt engine
        self.prompt_engine = PeppaPromptEngine()

        # Initialize unified business agent
        self.business_agent = UnifiedBusinessAgent()

        # Initialize query classifier
        self.query_classifier = QueryClassifier()

        # In-memory session storage (in production, use Redis or database)
        self.sessions: Dict[str, Dict] = {}

        # Build the LangGraph workflow
        self.workflow = self._build_conversation_graph()

    def _build_conversation_graph(self) -> StateGraph:
        """Build the LangGraph conversation workflow"""
        
        # Define the workflow steps
        workflow = StateGraph(ConversationState)
        
        # Add nodes for conversation flow
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("classify_query_needs", self._classify_query_needs_node)
        workflow.add_node("classify_prompt", self._classify_prompt_node)
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("enhance_query", self._enhance_query_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("save_session", self._save_session_node)

        # Set entry point and edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "classify_query_needs")
        workflow.add_edge("classify_query_needs", "classify_prompt")
        workflow.add_edge("classify_prompt", "load_context")
        workflow.add_edge("load_context", "enhance_query")
        workflow.add_edge("enhance_query", "generate_response")
        workflow.add_edge("generate_response", "save_session")
        workflow.add_edge("save_session", END)
        
        return workflow.compile()

    async def _analyze_query_node(self, state: ConversationState) -> ConversationState:
        """Analyze if the query needs conversational context"""
        try:
            logger.info(f"Analyzing query for session: {state.session_id}")
            
            # Simple heuristic analysis first
            follow_up_indicators = [
                'what about', 'how about', 'what if', 'also show', 'and what', 
                'compare that', 'similar to', 'different from', 'tell me more',
                'explain that', 'why is that', 'how so', 'elaborate'
            ]
            
            context_indicators = [
                'that', 'this', 'it', 'they', 'those', 'these', 'same',
                'previous', 'earlier', 'before', 'above', 'mentioned'
            ]
            
            query_lower = state.query.lower()
            
            # Check for follow-up patterns
            is_followup = any(indicator in query_lower for indicator in follow_up_indicators)
            needs_context = any(indicator in query_lower for indicator in context_indicators)
            
            # If we have session history and indicators suggest context needed
            session_data = self.sessions.get(state.session_id, {})
            has_history = len(session_data.get('history', [])) > 0
            
            if has_history and (is_followup or needs_context or len(state.query.split()) < 8):
                # Use LLM for more sophisticated analysis
                analysis = await self._llm_analyze_context_need(state.query, session_data.get('history', []))
                state.needs_context = analysis.get('needs_context', False)
                state.is_followup = analysis.get('is_followup', False)
            else:
                state.needs_context = needs_context and has_history
                state.is_followup = is_followup and has_history
            
            logger.info(f"Query analysis: needs_context={state.needs_context}, is_followup={state.is_followup}")
            
        except Exception as e:
            logger.error(f"Error in analyze_query_node: {e}")
            state.error = f"Query analysis failed: {str(e)}"
        
        return state

    async def _classify_query_needs_node(self, state: ConversationState) -> ConversationState:
        """Classify if query needs database context or can be answered with general advice"""
        try:
            logger.info(f"Classifying query needs for: {state.query[:100]}...")

            # Use query classifier to determine data needs
            classification = await self.query_classifier.classify_query(state.query)

            # Update state with classification results
            state.query_classification = classification
            state.needs_data = classification.get("needs_data", False)
            state.data_requirements = classification.get("data_requirements", [])
            state.can_answer_without_data = classification.get("can_answer_without_data", True)
            state.response_strategy = classification.get("response_strategy", "provide_general_advice")

            logger.info(f"Query classified: needs_data={state.needs_data}, strategy={state.response_strategy}")

        except Exception as e:
            logger.error(f"Error in classify_query_needs_node: {e}")
            state.error = f"Query classification failed: {str(e)}"
            # Default to general advice on error
            state.needs_data = False
            state.can_answer_without_data = True
            state.response_strategy = "provide_general_advice"

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
        """Load conversation history if needed"""
        try:
            if state.needs_context:
                session_data = self.sessions.get(state.session_id, {})
                history = session_data.get('history', [])
                
                if history:
                    # Get last 3 exchanges for context
                    recent_history = history[-3:] if len(history) >= 3 else history
                    
                    context_parts = []
                    for exchange in recent_history:
                        context_parts.append(f"User: {exchange.get('query', '')}")
                        response_snippet = exchange.get('response', '')[:200]
                        if len(exchange.get('response', '')) > 200:
                            response_snippet += "..."
                        context_parts.append(f"Assistant: {response_snippet}")
                    
                    state.context = "\n".join(context_parts)
                    logger.info(f"Loaded context for session: {state.session_id}")
                else:
                    state.context = ""
            else:
                state.context = ""
                
        except Exception as e:
            logger.error(f"Error in load_context_node: {e}")
            state.error = f"Context loading failed: {str(e)}"
            
        return state

    async def _enhance_query_node(self, state: ConversationState) -> ConversationState:
        """Enhance query with context if needed"""
        try:
            if state.needs_context and state.context:
                enhanced_query = f"""
Previous conversation context:
{state.context}

Current question: {state.query}

Please answer the current question considering the conversation context where relevant.
"""
                state.query = enhanced_query
                logger.info("Enhanced query with conversation context")
            
        except Exception as e:
            logger.error(f"Error in enhance_query_node: {e}")
            state.error = f"Query enhancement failed: {str(e)}"
            
        return state

    async def _generate_response_node(self, state: ConversationState) -> ConversationState:
        """Generate response based on query classification and data availability"""
        try:
            # Check if user needs specific data but doesn't have it connected
            if state.needs_data and not DatabaseManager.has_user_connection(state.session_id):
                logger.info("Query needs data but no user database connected - requesting connection")

                # Generate database connection request
                connection_prompt = await self.query_classifier.generate_data_request(
                    state.query_classification,
                    state.query
                )

                # Also offer general advice option
                general_advice = await self.query_classifier.generate_general_advice(
                    state.query,
                    state.query_classification
                )

                state.response = f"{connection_prompt}\n\n---\n\n**Alternative: General Business Advice**\n\n{general_advice}"

            elif state.response_strategy == "provide_general_advice" or state.can_answer_without_data:
                logger.info("Providing general business advice")

                # Generate general advice without requiring specific data
                state.response = await self.query_classifier.generate_general_advice(
                    state.query,
                    state.query_classification
                )

            elif state.needs_data and DatabaseManager.has_user_connection(state.session_id):
                logger.info("User has database connected - using unified agent with user data")

                # Use unified business agent with user's actual data
                # Create session-specific business agent with database access
                session_business_agent = UnifiedBusinessAgent(session_id=state.session_id)
                business_result = await session_business_agent.analyze(
                    query=state.query,
                    business_category=state.business_category,
                    analysis_type=state.analysis_type
                )

                if business_result.get("status") == "success":
                    # Format the business analysis response
                    insights = business_result.get("insights", "")
                    alerts = business_result.get("alerts", [])
                    recommendations = business_result.get("recommendations", [])

                    response_parts = []

                    # Add insights
                    if insights:
                        response_parts.append(f"## Business Analysis:\n{insights}")

                    # Add critical alerts
                    critical_alerts = [a for a in alerts if a.get("priority") == "CRITICAL"]
                    if critical_alerts:
                        response_parts.append("\n## 🚨 Critical Alerts:")
                        for alert in critical_alerts:
                            response_parts.append(f"- {alert.get('message', 'Alert')}")

                    # Add top recommendations
                    high_priority_recs = [r for r in recommendations if r.get("priority") == "HIGH"][:3]
                    if high_priority_recs:
                        response_parts.append("\n## 💡 Key Recommendations:")
                        for rec in high_priority_recs:
                            response_parts.append(f"- {rec.get('action', 'Recommendation')}: {rec.get('details', '')}")

                    # Add summary
                    summary = business_result.get("data_summary", {})
                    if summary:
                        response_parts.append(f"\n## Summary:")
                        response_parts.append(f"- Total Alerts: {summary.get('total_alerts', 0)}")
                        response_parts.append(f"- Recommendations: {summary.get('total_recommendations', 0)}")
                        if summary.get('critical_alerts', 0) > 0:
                            response_parts.append(f"- Critical Issues: {summary.get('critical_alerts', 0)}")

                    state.response = "\n".join(response_parts)

                    # Add analysis metadata
                    state.response += f"\n\n---\n*Analysis Type: {state.analysis_type.title()} | Category: {state.business_category.replace('_', ' ').title()}*"

                else:
                    # Fallback to general advice if business analysis fails
                    logger.warning(f"Business analysis failed, providing general advice: {business_result.get('error')}")
                    state.response = await self.query_classifier.generate_general_advice(
                        state.query,
                        state.query_classification
                    )

            else:
                # Fallback to traditional RAG for edge cases
                logger.info("Using traditional RAG fallback")
                await self._fallback_to_rag(state)

        except Exception as e:
            logger.error(f"Error in generate_response_node: {e}")
            state.error = f"Response generation failed: {str(e)}"
            state.response = "I encountered an error generating a response. Please try rephrasing your question."

        return state

    async def _fallback_to_rag(self, state: ConversationState):
        """Fallback to traditional RAG for simpler queries"""
        try:
            logger.info("Using traditional RAG analysis")
            from .peppagenbi import GenBISQL
            
            genbi = GenBISQL()
            rag_response = await genbi.retrieve_and_generate(state.query, state.session_id)
            
            if rag_response:
                state.response = rag_response.get('output', '')
                state.citations = rag_response.get('citations', [])
                logger.info(f"Generated traditional RAG response for session: {state.session_id}")
            else:
                state.response = "I couldn't generate a response. Please try again."
                state.citations = []
                
        except Exception as e:
            logger.error(f"RAG fallback failed: {e}")
            state.response = "I encountered an error. Please try again."

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
                'query': state.query if not state.needs_context else state.query.split("Current question: ")[-1].strip(),
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
                'context_used': final_state.get('needs_context', False),
                'is_followup': final_state.get('is_followup', False),
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