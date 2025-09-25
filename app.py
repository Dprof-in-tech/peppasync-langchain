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
from lib.agent import UnifiedBusinessAgent
from lib.prompt_engine import PeppaPromptEngine
from lib.config import AppConfig, LLMManager
from lib.tool_registry import ToolRegistry

# Import the new database components and user router
from lib.db.database import Base, engine # Base and engine are needed for table creation
from lib.api.routes.user import user_router

# Load environment variables
load_dotenv()

# Set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
if not AppConfig.validate_config():
    logger.error("Configuration validation failed. Please check your environment variables.")
    exit(1)

# Environment variables from config (this line is technically not needed here anymore as AppConfig.DATABASE_URL is preferred)
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
business_agent = UnifiedBusinessAgent()
prompt_engine = PeppaPromptEngine()

# Create database tables if they don't exist
# This is for initial setup during development. In production, use Alembic migrations.
Base.metadata.create_all(bind=engine)

# Include the new user authentication router
app.include_router(user_router, prefix="/auth", tags=["Authentication"])

# Pydantic models (existing ones from app.py)
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
    """Run analytics using the unified business agent"""
    try:
        # Map analysis type to business category
        category_mapping = {
            "sales_performance": "sales_revenue",
            "inventory_analysis": "inventory_operations",
            "marketing_performance": "marketing_customer",
            "customer_segmentation": "customer_behavior",
            "revenue_trends": "sales_revenue",
            "product_performance": "sales_revenue"
        }

        business_category = category_mapping.get(analysis_type, "general_analysis")

        result = await business_agent.analyze(
            query=f"Analyze {analysis_type} for business insights",
            business_category=business_category,
            analysis_type="descriptive"
        )

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
    """Run inventory monitoring using unified business agent"""
    try:
        result = await business_agent.analyze(
            query="Monitor inventory levels and generate alerts for low stock items",
            business_category="inventory_operations",
            analysis_type="diagnostic"
        )

        logger.info(f"Inventory analysis executed: {result.get('status')}")
        return result

    except Exception as e:
        logger.error(f"Error running inventory analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Inventory analysis failed',
                'message': str(e),
                'status': 'error'
            }
        )


@app.post("/agents/marketing/run")
async def run_marketing_agent():
    """Run marketing optimization using unified business agent"""
    try:
        result = await business_agent.analyze(
            query="Analyze marketing campaign performance and optimize ROAS",
            business_category="marketing_customer",
            analysis_type="prescriptive"
        )

        logger.info(f"Marketing analysis executed: {result.get('status')}")
        return result

    except Exception as e:
        logger.error(f"Error running marketing analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Marketing analysis failed',
                'message': str(e),
                'status': 'error'
            }
        )



# Internal function for prompt analysis (used by other endpoints)
async def _analyze_business_prompt_internal(prompt: str):
    """Internal function to analyze and classify business intelligence prompts"""
    try:
        analysis = await prompt_engine.analyze_prompt(prompt)

        # Generate sophisticated response if it's a complex business query
        if analysis.get('confidence', 0) > 0.6:
            sophisticated_response = await prompt_engine.generate_sophisticated_response(analysis)
            analysis['generated_response'] = sophisticated_response

        logger.info(f"Analyzed prompt: {analysis.get('category')} - {analysis.get('analysis_type')}")
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}")
        return {
            'error': 'Failed to analyze prompt',
            'message': str(e),
            'category': 'general_analysis',
            'analysis_type': 'descriptive',
            'confidence': 0.3
        }


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


@app.post("/unified-analysis")
async def unified_business_analysis(request: ChatRequest):
    """New unified business analysis endpoint showcasing the refactored architecture"""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # First classify the prompt using internal function
        analysis = await _analyze_business_prompt_internal(request.prompt)

        # Use unified business agent for comprehensive analysis
        business_result = await business_agent.analyze(
            query=request.prompt,
            business_category=analysis.get('category', 'general_analysis'),
            analysis_type=analysis.get('analysis_type', 'descriptive')
        )

        return {
            "unified_analysis": {
                "status": business_result.get("status"),
                "insights": business_result.get("insights"),
                "alerts": business_result.get("alerts", []),
                "recommendations": business_result.get("recommendations", []),
                "data_summary": business_result.get("data_summary", {})
            },
            "prompt_classification": {
                "category": analysis.get('category'),
                "analysis_type": analysis.get('analysis_type'),
                "confidence": analysis.get('confidence'),
                "key_metrics": analysis.get('key_metrics', []),
                "time_horizon": analysis.get('time_horizon')
            },
            "system_info": {
                "architecture": "Unified Business Agent",
                "tools_used": business_agent.get_available_tools(),
                "workflow_executed": f"{analysis.get('category', 'general')}_analysis"
            },
            "sessionId": session_id,
            "timestamp": int(time.time())
        }

    except Exception as e:
        logger.error(f"Error in unified analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Unified analysis failed',
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
        "message": "PeppaSync LangChain API - Unified Agent Architecture",
        "version": "2.1.0",
        "status": "running",
        "architecture": {
            "type": "Unified Business Agent",
            "description": "Single intelligent agent with modular tools",
            "benefits": ["Reduced redundancy", "Consistent responses", "Easier maintenance"]
        },
        "capabilities": {
            "conversational_bi": "Context-aware business intelligence chat",
            "unified_analytics": "Single agent for all business analysis types",
            "modular_tools": "Reusable tools for database, insights, alerts, recommendations",
            "sophisticated_prompts": "100+ retail scenario prompts supported",
            "market_specialization": "Nigerian retail market focus"
        },
        "available_tools": business_agent.get_available_tools(),
        "available_workflows": business_agent.get_available_workflows(),
        "endpoints": {
            "/chat": "Enhanced conversational BI with unified agent",
            "/analytics/{type}": "Business analytics using unified workflows",
            "/agents/inventory/run": "Inventory monitoring and alerts",
            "/agents/marketing/run": "Marketing optimization and ROAS analysis",
            "/analyze-prompt": "Analyze and classify business prompts",
            "/sophisticated-analysis": "Advanced retail business analysis",
            "/auth/signup": "Register a new user and send OTP",
            "/auth/verify-otp": "Verify OTP for user registration",
            "/auth/login": "Login user after email verification"
        },
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
