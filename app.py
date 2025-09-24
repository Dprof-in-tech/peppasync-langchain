from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import os
import logging
import uuid
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from lib.peppagenbi import GenBISQL
from lib.utils.utils import parse_api_response
from lib.conversation_manager import ConversationManager
from lib.analytics_engine import SimpleAnalyticsEngine
from lib.agents.inventory_agent import InventoryAgent
from lib.agents.marketing_agent import MarketingAgent
from lib.prompt_engine import PeppaPromptEngine
from lib.config import AppConfig, LLMManager

# Load environment variables
load_dotenv()

# Set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
if not AppConfig.validate_config():
    logger.error("Configuration validation failed. Please check your environment variables.")
    exit(1)

# Environment variables from config
database_config = AppConfig.DATABASE_CONFIG

app = FastAPI(title="PeppaSync LangChain", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
genbiapp = GenBISQL()
conversation_manager = ConversationManager()
analytics_engine = SimpleAnalyticsEngine()
inventory_agent = InventoryAgent()
marketing_agent = MarketingAgent()
prompt_engine = PeppaPromptEngine()

# Pydantic models
class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class AnalyticsRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = {}

class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class AdvancedAnalysisRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    include_forecasting: Optional[bool] = True
    include_scenarios: Optional[bool] = True


@app.post("/chat")
async def enhanced_chat(request: ChatRequest):
    """Enhanced chat endpoint with conversation context using LangGraph"""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Use conversation manager for contextual responses
        result = await conversation_manager.process_conversation(request.prompt, session_id)
        
        logger.info(f"Enhanced chat response for session: {session_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced chat: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to process conversation',
                'sessionId': session_id,
                'output': 'I encountered an error. Please try again.'
            }
        )


@app.post("/analytics/{analysis_type}")
async def run_analytics(analysis_type: str, request: AnalyticsRequest):
    """Run analytics using the simple analytics engine"""
    try:
        result = await analytics_engine.execute_analysis(analysis_type, request.filters)
        
        logger.info(f"Analytics executed: {analysis_type}")
        return result
        
    except Exception as e:
        logger.error(f"Error in analytics endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': f'Analytics failed for {analysis_type}',
                'message': str(e)
            }
        )


@app.post("/agents/inventory/run")
async def run_inventory_agent():
    """Run the inventory monitoring agent"""
    try:
        result = await inventory_agent.run_inventory_monitoring()
        
        logger.info(f"Inventory agent executed: {result.get('status')}")
        return result
        
    except Exception as e:
        logger.error(f"Error running inventory agent: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Inventory agent failed',
                'message': str(e),
                'status': 'error'
            }
        )


@app.post("/agents/marketing/run")
async def run_marketing_agent():
    """Run the marketing optimization agent"""
    try:
        result = await marketing_agent.run_marketing_optimization()
        
        logger.info(f"Marketing agent executed: {result.get('status')}")
        return result
        
    except Exception as e:
        logger.error(f"Error running marketing agent: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Marketing agent failed',
                'message': str(e),
                'status': 'error'
            }
        )


@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        status = {
            'inventory_agent': await inventory_agent.get_agent_status(),
            'marketing_agent': await marketing_agent.get_agent_status(),
            'timestamp': int(time.time())
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(
            status_code=500,
            detail={'error': 'Failed to get agent status'}
        )


@app.post("/analyze-prompt")
async def analyze_business_prompt(request: ChatRequest):
    """Analyze and classify business intelligence prompts"""
    try:
        # Analyze the prompt
        analysis = await prompt_engine.analyze_prompt(request.prompt)
        
        # Generate sophisticated response if it's a complex business query
        if analysis.get('confidence', 0) > 0.6:
            sophisticated_response = await prompt_engine.generate_sophisticated_response(analysis)
            analysis['generated_response'] = sophisticated_response
        
        logger.info(f"Analyzed prompt: {analysis.get('category')} - {analysis.get('analysis_type')}")
        return {
            "prompt_analysis": analysis,
            "timestamp": int(time.time()),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to analyze prompt',
                'message': str(e)
            }
        )


@app.get("/prompt-capabilities")
async def get_prompt_capabilities():
    """Get supported business intelligence categories and sample questions"""
    try:
        capabilities = prompt_engine.get_supported_categories()
        
        return {
            "supported_categories": capabilities,
            "total_categories": len(capabilities),
            "analysis_types": [
                "predictive", "scenario", "diagnostic", 
                "prescriptive", "comparative", "descriptive"
            ],
            "specializations": [
                "Nigerian retail market",
                "Sales forecasting", 
                "Marketing optimization",
                "Inventory management",
                "Customer analytics",
                "Financial analysis"
            ],
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail={'error': 'Failed to get capabilities'}
        )


@app.post("/sophisticated-analysis")
async def sophisticated_business_analysis(request: ChatRequest):
    """Endpoint for sophisticated retail business analysis"""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # First analyze and classify the prompt
        analysis = await prompt_engine.analyze_prompt(request.prompt)
        
        # Generate sophisticated response
        sophisticated_response = await prompt_engine.generate_sophisticated_response(analysis)
        
        # Get supporting data from RAG system
        rag_response = await genbiapp.retrieve_and_generate(request.prompt, session_id)
        
        return {
            "sophisticated_analysis": sophisticated_response,
            "prompt_classification": {
                "category": analysis.get('category'),
                "analysis_type": analysis.get('analysis_type'),
                "confidence": analysis.get('confidence'),
                "key_metrics": analysis.get('key_metrics', []),
                "time_horizon": analysis.get('time_horizon')
            },
            "supporting_data": rag_response.get('output', '') if rag_response else '',
            "citations": rag_response.get('citations', []) if rag_response else [],
            "sessionId": session_id,
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error(f"Error in sophisticated analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Sophisticated analysis failed',
                'sessionId': session_id,
                'message': str(e)
            }
        )


@app.delete("/session/{session_id}")
async def clear_conversation_session(session_id: str):
    """Clear a conversation session"""
    try:
        success = await conversation_manager.clear_session(session_id)
        
        if success:
            return {'message': f'Session {session_id} cleared successfully'}
        else:
            raise HTTPException(
                status_code=404,
                detail={'message': f'Session {session_id} not found or already cleared'}
            )
            
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=500,
            detail={'error': 'Failed to clear session'}
        )


# Keep original endpoints for backward compatibility
@app.post("/retrieve_and_visualize")
async def create_visuals(request: PromptRequest):
    """Original visualization endpoint - maintained for backward compatibility"""
    try:
        response = await genbiapp.retrieve_and_visualize(request.prompt)
        resp = {"user": response}
        
        logger.info(f"Response from retrieve_and_visualize: {len(str(response))} chars")
        return resp
        
    except Exception as e:
        logger.error(f"Error in retrieve_and_visualize: {e}")
        raise HTTPException(
            status_code=500,
            detail={'error': 'Visualization failed', 'message': str(e)}
        )


@app.post("/retrieve_and_generate")
async def generate_insights(request: PromptRequest):
    """Original insights endpoint - maintained for backward compatibility"""
    try:
        insight = await genbiapp.retrieve_and_generate(request.prompt, request.session_id)
        response = {"user": insight}

        logger.info(f"Response from retrieve_and_generate: session {request.session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in retrieve_and_generate: {e}")
        raise HTTPException(
            status_code=500,
            detail={'error': 'Insight generation failed', 'message': str(e)}
        )


@app.get("/")
async def root():
    """Health check endpoint with capabilities overview"""
    return {
        "message": "PeppaSync LangChain API",
        "version": "2.0.0",
        "status": "running",
        "capabilities": {
            "conversational_bi": "Context-aware business intelligence chat",
            "advanced_analytics": "6 types of business analysis", 
            "autonomous_agents": "Inventory & marketing optimization",
            "sophisticated_prompts": "100+ retail scenario prompts supported",
            "market_specialization": "Nigerian retail market focus"
        },
        "new_endpoints": {
            "/analyze-prompt": "Analyze and classify business prompts",
            "/prompt-capabilities": "Get supported categories and samples", 
            "/sophisticated-analysis": "Advanced retail business analysis"
        },
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)