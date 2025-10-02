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
from functools import lru_cache

from lib.peppagenbi import GenBISQL
from lib.utils.utils import parse_api_response
from lib.conversation_manager import ConversationManager
from lib.agent import UnifiedBusinessAgent
from lib.prompt_engine import PeppaPromptEngine
from lib.config import AppConfig, LLMManager, DatabaseManager
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

# Startup event to populate knowledge base
@app.on_event("startup")
async def startup_event():
    """Populate knowledge base with expert content and web insights on startup"""
    logger.info("Server starting up - initializing knowledge base...")
    try:
        # Trigger knowledge base population on first retrieval
        await genbiapp._ensure_knowledge_base_populated()
        logger.info("Knowledge base initialization completed")
    except Exception as e:
        logger.error(f"Knowledge base initialization failed: {e}")
        logger.info("Server will continue with fallback content")

    # Initialize Redis session manager
    logger.info("Initializing Redis session manager...")
    try:
        from lib.redis_session import redis_session_manager
        redis_health = redis_session_manager.health_check()
        if redis_health['available']:
            logger.info(f"✅ Redis session manager ready - {redis_health.get('redis_version', 'unknown version')}")
        else:
            logger.warning(f"⚠️  Redis not available: {redis_health.get('message', 'unknown error')}")
            logger.warning("Sessions will use in-memory fallback storage")
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}")
        logger.warning("Sessions will use in-memory fallback storage")

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

class AdvancedAnalysisRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    include_forecasting: Optional[bool] = True
    include_scenarios: Optional[bool] = True

class DatabaseConnectionRequest(BaseModel):
    database_url: str
    session_id: Optional[str] = None

class DatabaseTestRequest(BaseModel):
    database_url: str


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


# Simple cache for dashboard analytics to prevent excessive calls
dashboard_cache = {}
CACHE_DURATION = 30  # 30 seconds

@app.post("/analytics/dashboard")
async def get_dashboard_analytics(request: AnalyticsRequest):
    """Get dashboard analytics data for connected database or mock data"""
    try:
        session_id = request.filters.get('session_id') if request.filters else None
        
        # Check cache first
        cache_key = f"dashboard_{session_id}"
        current_time = time.time()
        
        if cache_key in dashboard_cache:
            cached_data, cache_time = dashboard_cache[cache_key]
            if current_time - cache_time < CACHE_DURATION:
                logger.info(f"Returning cached dashboard data for session: {session_id}")
                return cached_data
        
        # Get specific data for dashboard widgets with fallback logic (no AI calls needed)
        sales_data = DatabaseManager.get_data(session_id=session_id, query_type="sales_data", use_mock=False)
        
        # If no real sales data, fallback to mock data for demo purposes
        if not sales_data:
            logger.info("No real sales data found, using mock data for dashboard")
            sales_data = DatabaseManager.get_data(session_id=session_id, query_type="sales_data", use_mock=True)
        
        inventory_data = DatabaseManager.get_data(session_id=session_id, query_type="inventory_data", use_mock=False)
        if not inventory_data:
            inventory_data = DatabaseManager.get_data(session_id=session_id, query_type="inventory_data", use_mock=True)
            
        campaign_data = DatabaseManager.get_data(session_id=session_id, query_type="campaign_data", use_mock=False)
        if not campaign_data:
            campaign_data = DatabaseManager.get_data(session_id=session_id, query_type="campaign_data", use_mock=True)

        # Debug logging
        logger.info(f"Dashboard analytics - Session: {session_id}")
        logger.info(f"Sales data count: {len(sales_data)}")
        logger.info(f"Sample sales data: {sales_data[:2] if sales_data else 'No data'}")
        logger.info(f"Inventory data count: {len(inventory_data)}")
        logger.info(f"Campaign data count: {len(campaign_data)}")

        # Calculate key metrics with flexible field mapping
        def get_sales_amount(item):
            # Try different possible field names for sales amount
            return (item.get('sales_amount') or 
                   item.get('amount') or 
                   item.get('total_amount') or 
                   item.get('revenue') or 
                   item.get('total') or 0)

        def get_units_sold(item):
            # Try different possible field names for units sold
            return (item.get('units_sold') or 
                   item.get('quantity') or 
                   item.get('qty') or 
                   item.get('units') or 0)

        total_revenue = sum(get_sales_amount(item) for item in sales_data)
        total_units = sum(get_units_sold(item) for item in sales_data)
        avg_order_value = total_revenue / len(sales_data) if sales_data else 0
        
        logger.info(f"Calculated metrics - Revenue: {total_revenue}, Units: {total_units}, AOV: {avg_order_value}")
        
        # Low stock items
        low_stock_items = [
            item for item in inventory_data 
            if item.get('current_stock', 0) < item.get('reorder_level', 0)
        ]
        
        # Campaign performance with flexible field mapping
        def get_roas(item):
            return (item.get('roas') or 
                   item.get('return_on_ad_spend') or 
                   (item.get('revenue', 0) / max(item.get('spend', 1), 1)) or 0)

        def get_spend(item):
            return (item.get('spend') or 
                   item.get('ad_spend') or 
                   item.get('cost') or 0)

        avg_roas = sum(get_roas(item) for item in campaign_data) / len(campaign_data) if campaign_data else 0
        total_ad_spend = sum(get_spend(item) for item in campaign_data)

        response_data = {
            "status": "success",
            "metrics": {
                "total_revenue": total_revenue,
                "total_units_sold": total_units,
                "average_order_value": avg_order_value,
                "low_stock_count": len(low_stock_items),
                "average_roas": avg_roas,
                "total_ad_spend": total_ad_spend
            },
            "sales_data": sales_data[-10:],  # Last 10 sales
            "inventory_alerts": low_stock_items[:5],  # Top 5 low stock items
            "top_campaigns": sorted(campaign_data, key=lambda x: x.get('roas', 0), reverse=True)[:3],
            "last_updated": int(time.time()),
            "session_id": session_id
        }

        # Cache the response
        dashboard_cache[cache_key] = (response_data, current_time)
        logger.info(f"Cached dashboard data for session: {session_id}")

        return response_data

    except Exception as e:
        logger.error(f"Error getting dashboard analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Dashboard analytics failed',
                'message': str(e)
            }
        )


@app.post("/analytics/{analysis_type}")
async def run_analytics(analysis_type: str, request: AnalyticsRequest):
    """Run analytics using the unified business agent"""
    try:
        session_id = request.filters.get("session_id", "default-session") if request.filters else "default-session"

        # Get business data for analysis
        business_data = {
            "sales_data": DatabaseManager.get_data(session_id, "sales_data"),
            "inventory_data": DatabaseManager.get_data(session_id, "inventory_data"),
            "campaign_data": DatabaseManager.get_data(session_id, "campaign_data"),
        }

        # Use analyze_direct_query with proper business data
        result_str = await business_agent.analyze_direct_query(
            query=f"Analyze {analysis_type} for business insights",
            business_data=business_data,
            conversation_history=[]
        )

        # Parse JSON response
        import json
        result = json.loads(result_str)

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
        session_id = "default-session"  # Could be passed as query param if needed

        # Get business data for analysis
        business_data = {
            "inventory_data": DatabaseManager.get_data(session_id, "inventory_data"),
            "low_stock_items": DatabaseManager.get_data(session_id, "low_stock_items"),
        }

        # Use analyze_direct_query
        result_str = await business_agent.analyze_direct_query(
            query="Monitor inventory levels and generate alerts for low stock items",
            business_data=business_data,
            conversation_history=[]
        )

        # Parse JSON response
        import json
        result = json.loads(result_str)

        logger.info(f"Inventory analysis executed: {result.get('type')}")
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
        session_id = "default-session"  # Could be passed as query param if needed

        # Get business data for analysis
        business_data = {
            "campaign_data": DatabaseManager.get_data(session_id, "campaign_data"),
            "underperforming_campaigns": DatabaseManager.get_data(session_id, "underperforming_campaigns"),
        }

        # Use analyze_direct_query
        result_str = await business_agent.analyze_direct_query(
            query="Analyze marketing campaign performance and optimize ROAS",
            business_data=business_data,
            conversation_history=[]
        )

        # Parse JSON response
        import json
        result = json.loads(result_str)

        logger.info(f"Marketing analysis executed: {result.get('type')}")
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
                "Global ecommerce market",
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


@app.post("/database/test")
async def test_database_connection(request: DatabaseTestRequest):
    """Test a PostgreSQL database connection"""
    try:
        result = DatabaseManager.test_database_connection(request.database_url)

        if result["success"]:
            logger.info(f"Database connection test successful: {result['database_name']}")
            return {
                "status": "success",
                "message": result["message"],
                "database_info": {
                    "database_name": result["database_name"],
                    "host": result["host"],
                    "available_tables": result["available_tables"],
                    "detected_data_types": result["detected_data_types"],
                    "table_count": len(result["available_tables"])
                },
                "table_details": result["table_info"]
            }
        else:
            logger.error(f"Database connection test failed: {result['message']}")
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": result["message"],
                    "error": result.get("error", "Unknown error")
                }
            )

    except Exception as e:
        logger.error(f"Error testing database connection: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Database connection test failed",
                "error": str(e)
            }
        )


@app.post("/database/connect")
async def connect_database(request: DatabaseConnectionRequest):
    """Connect a PostgreSQL database to a session"""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # First test the connection
        test_result = DatabaseManager.test_database_connection(request.database_url)

        if not test_result["success"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Database connection failed: {test_result['message']}",
                    "session_id": session_id
                }
            )

        # Store the connection information
        connection_info = {
            "database_url": request.database_url,
            "connected_at": int(time.time()),
            "database_name": test_result["database_name"],
            "host": test_result["host"],
            "available_tables": test_result["available_tables"],
            "detected_data_types": test_result["detected_data_types"],
            "table_info": test_result["table_info"]
        }

        DatabaseManager.set_user_connection(session_id, connection_info)

        # Note: Pinecone is pre-populated with expert marketing/sales knowledge
        # User's database provides the data, Pinecone provides the expertise
        logger.info(f"Database connected for session {session_id}: {test_result['database_name']}")

        return {
            "status": "success",
            "message": "Database connected successfully",
            "session_id": session_id,
            "database_info": {
                "database_name": test_result["database_name"],
                "host": test_result["host"],
                "available_tables": test_result["available_tables"],
                "detected_data_types": test_result["detected_data_types"],
                "table_count": len(test_result["available_tables"])
            },
            "next_steps": [
                "You can now ask questions about your business data",
                "The system will automatically query your database for relevant information",
                f"Available data types: {', '.join(test_result['detected_data_types'])}"
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting database: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to connect database",
                "session_id": session_id,
                "error": str(e)
            }
        )


@app.get("/database/status/{session_id}")
async def get_database_status(session_id: str):
    """Get database connection status for a session"""
    try:
        if DatabaseManager.has_user_connection(session_id):
            connection_info = DatabaseManager.get_user_connection(session_id)
            return {
                "status": "connected",
                "database_name": connection_info.get("database_name", "Unknown"),
                "host": connection_info.get("host", "Unknown"),
                "connected_at": connection_info.get("connected_at"),
                "available_tables": connection_info.get("available_tables", []),
                "detected_data_types": connection_info.get("detected_data_types", []),
                "table_count": len(connection_info.get("available_tables", []))
            }
        else:
            return {
                "status": "not_connected",
                "message": "No database connection found for this session"
            }

    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to get database status",
                "error": str(e)
            }
        )


@app.delete("/database/disconnect/{session_id}")
async def disconnect_database(session_id: str):
    """Disconnect database for a session"""
    try:
        if DatabaseManager.has_user_connection(session_id):
            DatabaseManager.remove_user_connection(session_id)
            logger.info(f"Database disconnected for session {session_id}")
            return {
                "status": "success",
                "message": f"Database disconnected for session {session_id}"
            }
        else:
            return {
                "status": "not_connected",
                "message": "No database connection found for this session"
            }

    except Exception as e:
        logger.error(f"Error disconnecting database: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to disconnect database",
                "error": str(e)
            }
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
        insight = await genbiapp.retrieve_and_generate(request.prompt)
        response = {"user": insight}

        logger.info(f"Response from retrieve_and_generate: {len(str(insight))} chars")
        return response

    except Exception as e:
        logger.error(f"Error in retrieve_and_generate: {e}")
        raise HTTPException(
            status_code=500,
            detail={'error': 'Insight generation failed', 'message': str(e)}
        )


@app.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring/deployment platforms"""
    # Check Redis health
    redis_status = {"available": False, "status": "not initialized"}
    try:
        from lib.redis_session import redis_session_manager
        redis_health = redis_session_manager.health_check()
        redis_status = {
            "available": redis_health.get('available', False),
            "status": redis_health.get('status', 'unknown')
        }
    except Exception as e:
        redis_status = {"available": False, "status": f"error: {str(e)}"}

    return {
        "status": "healthy",
        "service": "peppasync-langchain",
        "timestamp": time.time(),
        "redis": redis_status
    }

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
            "market_specialization": "Global ecommerce market focus"
        },
        "available_tools": business_agent.get_available_tools(),
        "available_workflows": business_agent.get_available_workflows(),
        "endpoints": {
            "/chat": "Enhanced conversational BI with unified agent",
            "/analytics/{type}": "Business analytics using unified workflows",
            "/agents/inventory/run": "Inventory monitoring and alerts",
            "/agents/marketing/run": "Marketing optimization and ROAS analysis",
            "/database/test": "Test PostgreSQL database connection",
            "/database/connect": "Connect PostgreSQL database to session",
            "/database/status/{session_id}": "Get database connection status",
            "/database/disconnect/{session_id}": "Disconnect database from session",
            "/auth/signup": "Register a new user and send OTP",
            "/auth/verify-otp": "Verify OTP for user registration",
            "/auth/login": "Login user after email verification"
        },
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
