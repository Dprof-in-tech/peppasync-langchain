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
from lib.config import AppConfig, DatabaseManager
from lib.tool_registry import ToolRegistry
from lib.forecast_settings import ForecastSettingsManager
from lib.tools.forecast_tool import DemandForecastDirectTool

# Import the new database components and user router
from lib.db.database import Base, engine
from lib.api.routes.user import user_router

# Load environment variables
load_dotenv()

# Set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FUNDAM_API_KEY = os.getenv("FUNDAM_API_KEY")

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
            logger.info(f"Redis session manager ready - {redis_health.get('redis_version', 'unknown version')}")
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
    data_source: Optional[str] = "postgres"  # "postgres" or "shopify"

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

class ForecastSettingsRequest(BaseModel):
    session_id: Optional[str] = None
    economic_events: Optional[List[Dict[str, Any]]] = []
    supply_chain_locations: Optional[List[Dict[str, Any]]] = []
    enrich_events: Optional[bool] = True

class ForecastRequest(BaseModel):
    session_id: Optional[str] = None
    user_prompt: str
    product_filter: Optional[str] = None
    forecast_mode: str = "aggregate"  # "aggregate", "single", "multi", "top_n"
    top_n_products: int = 10  # For top_n mode


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
    """Get dashboard analytics data for connected database or Shopify"""
    try:
        session_id = request.filters.get('session_id') if request.filters else None
        data_source = request.data_source or "postgres"
        cache_key = f"dashboard_{data_source}_{session_id}"
        current_time = time.time()

        # BACKEND GUARD: Check if session is connected before proceeding
        is_connected = False
        if data_source == "shopify":
            connection = DatabaseManager.get_shopify_connection(session_id)
            is_connected = connection is not None
        else:
            is_connected = DatabaseManager.has_user_connection(session_id)

        if not is_connected:
            logger.warning(f"Dashboard analytics requested for disconnected session: {session_id} [{data_source}]")
            raise HTTPException(
                status_code=403,
                detail={
                    'error': 'Session is not connected. Please connect your data source first.',
                    'session_id': session_id,
                    'data_source': data_source
                }
            )

        if cache_key in dashboard_cache:
            cached_data, cache_time = dashboard_cache[cache_key]
            if current_time - cache_time < CACHE_DURATION:
                logger.info(f"Returning cached dashboard data for session: {session_id} [{data_source}]")
                return cached_data

        # Branch logic for data source
        if data_source == "shopify":
            # Get Shopify orders and normalize to sales_data
            shopify_orders = DatabaseManager.get_shopify_orders(session_id)
            # Normalize Shopify orders to sales_data structure
            sales_data = [
                {
                    'product_name': o.get('line_items', [{}])[0].get('title', 'Unknown') if o.get('line_items') else 'Unknown',
                    'sales_amount': float(o.get('total_price', 0)),
                    'units_sold': sum([li.get('quantity', 0) for li in o.get('line_items', [])]),
                    'category': o.get('line_items', [{}])[0].get('product_type', 'Shopify') if o.get('line_items') else 'Shopify',
                    'profit_margin': None,
                    'sale_date': o.get('created_at')
                }
                for o in shopify_orders
            ]
            # Inventory and campaign data from Shopify not implemented (could be added if available)
            inventory_data = await DatabaseManager.get_data(session_id=session_id, query_type="inventory_data", use_mock=False)
            campaign_data = []
        else:
            # Default: Postgres
            sales_data = await DatabaseManager.get_data(session_id=session_id, query_type="sales_data", use_mock=False)
            if not sales_data:
                logger.info("No real sales data found, using mock data for dashboard")
                sales_data = await DatabaseManager.get_data(session_id=session_id, query_type="sales_data", use_mock=True)
            inventory_data = await DatabaseManager.get_data(session_id=session_id, query_type="inventory_data", use_mock=False)
            if not inventory_data:
                inventory_data = await DatabaseManager.get_data(session_id=session_id, query_type="inventory_data", use_mock=True)
            campaign_data = await DatabaseManager.get_data(session_id=session_id, query_type="campaign_data", use_mock=False)
            if not campaign_data:
                campaign_data = await DatabaseManager.get_data(session_id=session_id, query_type="campaign_data", use_mock=True)

        # Calculate key metrics with flexible field mapping
        def get_sales_amount(item):
            amount = (
                item.get('sales_amount') or
                item.get('amount') or
                item.get('total_amount') or
                item.get('revenue') or
                item.get('total')
            )
            if not amount and ('price' in item and 'quantity' in item):
                amount = item['price'] * item['quantity']
            return amount or 0

        def get_units_sold(item):
            return (
                item.get('units_sold') or
                item.get('quantity') or
                item.get('qty') or
                item.get('units') or 0
            )

        total_revenue = sum(get_sales_amount(item) for item in sales_data)
        total_units = sum(get_units_sold(item) for item in sales_data)
        avg_order_value = total_revenue / len(sales_data) if sales_data else 0

        logger.info(f"Calculated metrics - Revenue: {total_revenue}, Units: {total_units}, AOV: {avg_order_value} [{data_source}]")

        # Low stock items (only for Postgres for now)
        low_stock_items = []
        if data_source == "postgres":
            low_stock_items = [
                item for item in inventory_data
                if item.get('quantity', 0) < item.get('reorder_level', 10)
            ]

        # Campaign performance (only for Postgres for now)
        def get_revenue(item):
            try:
                return float(item.get('revenue') or 0)
            except ValueError:
                return 0

        def get_spend(item):
            try:
                return float(item.get('spend') or item.get('ad_spend') or item.get('cost') or 0)
            except ValueError:
                return 0

        def get_roas(item):
            spend = get_spend(item)
            revenue = get_revenue(item)
            if spend > 0:
                return revenue / spend
            return 0

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
            "sales_data": sales_data[-10:],
            "inventory_alerts": low_stock_items[:5],
            "top_campaigns": sorted(
                [
                    {**c, "roas": round(get_roas(c), 2)}
                    for c in campaign_data
                ],
                key=lambda x: x["roas"],
                reverse=True
            )[:3],
            "last_updated": int(time.time()),
            "session_id": session_id,
            "data_source": data_source
        }

        dashboard_cache[cache_key] = (response_data, current_time)
        logger.info(f"Cached dashboard data for session: {session_id} [{data_source}]")
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


@app.post("/customer-insights")
async def customer_insights(request: ChatRequest):
    """Endpoint for customer insights analysis."""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        customer_insights_tool = ToolRegistry.get_all_tools()["customer_insights"]
        result = customer_insights_tool._run(
            session_id=session_id,
            query=request.prompt
        )
        return json.loads(result)

    except Exception as e:
        logger.error(f"Error in customer insights: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Customer insights analysis failed',
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
        # Clear dashboard cache for this session/postgres
        cache_key = f"dashboard_postgres_{session_id}"
        if cache_key in dashboard_cache:
            del dashboard_cache[cache_key]
            logger.info(f"Dashboard cache cleared for session {session_id} [postgres]")

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


# ===========================
# SHOPIFY INTEGRATION ENDPOINTS
# ===========================

class ShopifyConnectRequest(BaseModel):
    shop_name: str
    session_id: str
    redirect_url: str
    bearer_token: Optional[str] = None


@app.post("/shopify/connect")
async def connect_shopify(request: ShopifyConnectRequest):
    """Connect to Shopify store and sync data"""
    try:
        from lib.shopify_service import get_shopify_service

        logger.info(f"Connecting Shopify store: {request.shop_name} for session {request.session_id}")

        shopify_service = get_shopify_service()

        # Call external connector API
        result = await shopify_service.connect_shopify_store(
            shop_name=request.shop_name,
            redirect_url=request.redirect_url,
            bearer_token=request.bearer_token
        )

        if result.get("success"):
            connection_id = await shopify_service.get_connection_id_by_shop(
                shop_name=request.shop_name
            )
            # Store connection info
            connection_data = {
                "shop_name": request.shop_name,
                "connected_at": time.time(),
                "connection_id": connection_id,
                "redirect_url": request.redirect_url
            }
            DatabaseManager.store_shopify_connection(
                request.session_id,
                request.shop_name,
                connection_data
            )

            return {
                "success": True,
                "message": f"Shopify store {request.shop_name} connected successfully",
                "session_id": request.session_id,
                "auth_url": result.get("data", {}).get("auth_url"),
                "shop_name": request.shop_name
            }
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": result.get("error", "Failed to connect Shopify store"),
                    "session_id": request.session_id
                }
            )

    except Exception as e:
        logger.error(f"Error connecting Shopify: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to connect Shopify store",
                "error": str(e),
                "session_id": request.session_id
            }
        )


@app.get("/shopify/status/{session_id}")
async def get_shopify_status(session_id: str):
    """Get Shopify connection status for a session and auto-sync if needed"""
    try:
        connection = DatabaseManager.get_shopify_connection(session_id)

        if connection:
            orders = DatabaseManager.get_shopify_orders(session_id)
            shop_name = connection.get("shop_name")

            # Auto-sync orders if:
            # 1. Never synced before (no last_sync timestamp), OR
            # 2. Last sync was more than 1 hour ago
            should_sync = False
            last_sync = connection.get("last_sync")

            if not last_sync:
                # Never synced before - attempt first sync
                should_sync = True
                logger.info(f"No sync history for {shop_name}, will attempt initial sync")
            elif last_sync:
                time_since_sync = time.time() - last_sync
                if time_since_sync > 3600:  # 1 hour
                    should_sync = True
                    logger.info(f"Last sync was {time_since_sync/60:.1f} minutes ago, will attempt re-sync")

            # Attempt auto-sync if needed
            if should_sync and shop_name:
                try:
                    from lib.shopify_service import get_shopify_service
                    shopify_service = get_shopify_service()

                    # Check if we have access_token in connection data
                    # access_token = connection.get("access_token")
                    access_token = FUNDAM_API_KEY

                    if access_token:
                        logger.info(f"Auto-syncing orders for {shop_name}")
                        orders_synced = await shopify_service.auto_sync_orders(
                            shop_name=shop_name,
                            access_token=access_token,
                            session_id=session_id,
                            days_back=90,
                            limit=250
                        )

                        # Update last_sync timestamp
                        connection["last_sync"] = time.time()
                        DatabaseManager.store_shopify_connection(session_id, shop_name, connection)

                        # Refresh orders after sync
                        orders = DatabaseManager.get_shopify_orders(session_id)
                        logger.info(f"Auto-sync completed: {orders_synced} orders synced")
                    else:
                        logger.info(f"No access_token available yet for {shop_name}. User may need to complete OAuth.")

                except Exception as sync_err:
                    logger.warning(f"Auto-sync failed for {shop_name}: {sync_err}")
                    # Don't fail the status check if sync fails

            return {
                "connected": True,
                "shop_name": shop_name,
                "connected_at": connection.get("connected_at"),
                "orders_count": len(orders),
                "last_sync": connection.get("last_sync"),
                "products_count": connection.get("products_count", 0)
            }
        else:
            return {
                "connected": False,
                "message": "No Shopify connection found for this session"
            }

    except Exception as e:
        logger.error(f"Error getting Shopify status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get Shopify status",
                "message": str(e)
            }
        )


@app.post("/shopify/oauth-callback")
async def shopify_oauth_callback(request: Request):
    """
    Callback endpoint for Shopify OAuth completion.
    The external connector should POST to this endpoint with session_id, shop_name, and access_token.
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        shop_name = data.get("shop_name")
        access_token = data.get("access_token")

        if not session_id or not shop_name or not access_token:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": "Missing required fields: session_id, shop_name, access_token"
                }
            )

        logger.info(f"OAuth callback received for shop: {shop_name}, session: {session_id}")

        # Get existing connection or create new one
        connection = DatabaseManager.get_shopify_connection(session_id) or {}

        # Update connection with access_token
        connection.update({
            "shop_name": shop_name,
            "access_token": access_token,
            "oauth_completed": True,
            "oauth_completed_at": time.time()
        })

        # Store updated connection
        DatabaseManager.store_shopify_connection(session_id, shop_name, connection)

        logger.info(f"Access token stored for {shop_name}")

        # Trigger auto-sync immediately
        try:
            from lib.shopify_service import get_shopify_service
            shopify_service = get_shopify_service()

            orders_synced = await shopify_service.auto_sync_orders(
                shop_name=shop_name,
                access_token=access_token,
                session_id=session_id,
                days_back=90,
                limit=250
            )

            connection["last_sync"] = time.time()
            DatabaseManager.store_shopify_connection(session_id, shop_name, connection)

            logger.info(f"OAuth callback: {orders_synced} orders synced for {shop_name}")

            return {
                "success": True,
                "message": f"OAuth completed and {orders_synced} orders synced",
                "session_id": session_id,
                "shop_name": shop_name,
                "orders_synced": orders_synced
            }

        except Exception as sync_err:
            logger.warning(f"OAuth callback: sync failed for {shop_name}: {sync_err}")
            return {
                "success": True,
                "message": "OAuth completed but sync failed. Will retry on next status check.",
                "session_id": session_id,
                "shop_name": shop_name,
                "sync_error": str(sync_err)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OAuth callback: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "OAuth callback failed",
                "error": str(e)
            }
        )


@app.post("/shopify/disconnect/{session_id}")
async def disconnect_shopify(session_id: str):
    """Disconnect Shopify store from session"""
    try:
        # Clear dashboard cache for this session/shopify
        cache_key = f"dashboard_shopify_{session_id}"
        if cache_key in dashboard_cache:
            del dashboard_cache[cache_key]
            logger.info(f"Dashboard cache cleared for session {session_id} [shopify]")

        connection = DatabaseManager.get_shopify_connection(session_id)

        if connection:
            DatabaseManager.remove_shopify_session(session_id)
            logger.info(f"Shopify disconnected for session {session_id}")

            return {
                "success": True,
                "message": f"Shopify store disconnected successfully",
                "session_id": session_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "message": "No Shopify connection found for this session",
                    "session_id": session_id
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disconnecting Shopify: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to disconnect Shopify store",
                "error": str(e),
                "session_id": session_id
            }
        )


# ===========================
# DEMAND FORECASTING ENDPOINTS
# ===========================

@app.post("/forecast/settings")
async def save_forecast_settings(request: ForecastSettingsRequest):
    """
    Save or update forecast settings for a session.

    Single endpoint for all forecast configuration:
    - Economic events (will be auto-enriched with Tavily if enrich_events=True)
    - Supply chain locations (multiple locations with manufacturing/logistics/delay days)

    Note: forecast_horizon and frequency are NOT in settings - extracted from user prompt.
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        settings = {
            "economic_events": request.economic_events or [],
            "supply_chain_locations": request.supply_chain_locations or []
        }

        # Auto-enrich economic events with Tavily if requested
        if request.enrich_events and settings["economic_events"]:
            logger.info(f"Enriching {len(settings['economic_events'])} economic events with Tavily...")
            try:
                from lib.mcp.tavily_fetcher import TavilyFetcher
                tavily = TavilyFetcher()

                enriched_events = []
                for event in settings["economic_events"]:
                    enriched = tavily.enrich_event(event)
                    if enriched:
                        enriched_events.append(enriched)
                        logger.info(f"Enriched event: {enriched['name']} - impact: {enriched.get('impact_days_before', 'N/A')} days before")
                    else:
                        # Keep original if enrichment fails
                        enriched_events.append(event)

                settings["economic_events"] = enriched_events
                logger.info(f"Successfully enriched {len(enriched_events)} events")

            except Exception as e:
                logger.warning(f"Event enrichment failed: {e}. Using original events.")

        # Save settings
        ForecastSettingsManager.save_settings(session_id, settings)

        logger.info(f"Forecast settings saved for session {session_id}")

        return {
            "status": "success",
            "message": "Forecast settings saved successfully",
            "session_id": session_id,
            "settings": {
                "economic_events_count": len(settings["economic_events"]),
                "supply_chain_locations_count": len(settings["supply_chain_locations"]),
                "events_enriched": request.enrich_events
            },
            "settings_detail": settings
        }

    except Exception as e:
        logger.error(f"Error saving forecast settings: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to save forecast settings",
                "session_id": session_id,
                "error": str(e)
            }
        )


@app.get("/forecast/settings/{session_id}")
async def get_forecast_settings(session_id: str):
    """Get current forecast settings for a session"""
    try:
        settings = ForecastSettingsManager.get_settings(session_id)

        if settings:
            return {
                "status": "success",
                "session_id": session_id,
                "settings": settings,
                "economic_events_count": len(settings.get("economic_events", [])),
                "supply_chain_locations_count": len(settings.get("supply_chain_locations", []))
            }
        else:
            return {
                "status": "not_found",
                "message": "No forecast settings found for this session",
                "session_id": session_id,
                "default_settings": ForecastSettingsManager.get_default_settings()
            }

    except Exception as e:
        logger.error(f"Error getting forecast settings: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to get forecast settings",
                "error": str(e)
            }
        )


@app.post("/forecast/tune")
async def tune_forecast_hyperparameters(request: ForecastRequest):
    """
    Hyperparameter tuning endpoint for Prophet forecasting.

    Uses grid search to find optimal parameters based on train-test split validation.
    This may take several minutes depending on the grid size.

    Returns best parameters and performance metrics (MAE, MSE, R²).
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(f"Starting hyperparameter tuning for session {session_id}")

        from lib.context_layer import ContextLayer
        from lib.forecasting_engine import ForecastingEngine
        from lib.data_pipeline import DataPipeline
        from lib.forecast_settings import ForecastSettingsManager

        context_layer = ContextLayer()
        forecasting_engine = ForecastingEngine()
        data_pipeline = DataPipeline()

        # Get settings
        settings = ForecastSettingsManager.get_settings(session_id)
        if not settings:
            settings = ForecastSettingsManager.get_default_settings()

        economic_events = settings.get("economic_events", [])

        # Fetch historical data
        historical_data = await context_layer._fetch_historical_data(
            session_id,
            request.product_filter,
            lookback_days=180  # Use more data for tuning
        )

        if historical_data is None or len(historical_data) < 100:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for hyperparameter tuning (need at least 100 records)"
            )

        # Prepare data
        prepared_data = data_pipeline.prepare_for_forecasting(
            historical_data,
            product_filter=request.product_filter,
            freq='D',
            remove_outliers=True
        )

        # Convert to Prophet format
        prophet_data = forecasting_engine.prepare_data_for_prophet(
            prepared_data,
            date_col='Order Date',
            value_col='Sales',
            freq='D',
            product_filter=request.product_filter
        )

        # Run comprehensive grid search (135 combinations)
        logger.info("Running comprehensive hyperparameter grid search (135 combinations)...")
        tuning_results = forecasting_engine.tune_hyperparameters(
            historical_data=prophet_data,
            economic_events=economic_events,
            test_size=0.2,
            param_grid={
                'changepoint_prior_scale': [0.01, 0.05, 0.1],
                'seasonality_prior_scale': [1, 5, 10],
                'holidays_prior_scale': [1, 5, 10],
                'seasonality_mode': ['additive', 'multiplicative'],
                'changepoint_range': [0.7, 0.8, 0.9]
            }
        )

        logger.info("Hyperparameter tuning complete")

        return {
            "success": True,
            "session_id": session_id,
            "best_parameters": tuning_results['best_params'],
            "performance": {
                "mse": tuning_results['best_mse'],
                "r2_score": tuning_results['best_r2']
            },
            "tested_combinations": len(tuning_results['all_results']),
            "recommendation": "Use these parameters in your Prophet model for better accuracy"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Hyperparameter tuning failed",
                "error": str(e)
            }
        )


@app.post("/forecast")
async def run_demand_forecast(request: ForecastRequest):
    """
    Main demand forecasting endpoint.

    Executes the complete forecast pipeline:
    1. Extract forecast parameters from user_prompt (periods, frequency)
    2. Retrieve user settings (economic events, supply chain)
    3. Fetch historical data (PostgreSQL → Kaggle fallback)
    4. Run Prophet forecast
    5. Validate with train-test split backtesting
    6. Generate recommendations via Advisor
    7. Return comprehensive results

    Examples:
    - user_prompt: "forecast next 45 days"
    - user_prompt: "predict demand for 8 weeks weekly"
    - user_prompt: "monthly forecast for iPhone for 3 months"
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(f"Starting demand forecast for session {session_id}")
        logger.info(f"User prompt: {request.user_prompt}")

        # Use the direct forecast tool (returns Dict, not JSON string)
        forecast_tool = DemandForecastDirectTool()

        result = await forecast_tool._run(
            session_id=session_id,
            user_prompt=request.user_prompt,
            product_filter=request.product_filter,
            forecast_mode=request.forecast_mode,
            top_n_products=request.top_n_products
        )

        # Check if forecast succeeded
        if not result.get("success"):
            logger.error(f"Forecast failed: {result.get('error')}")
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": result.get("message", "Forecast execution failed"),
                    "error": result.get("error"),
                    "suggestion": result.get("suggestion"),
                    "session_id": session_id
                }
            )

        # Add session info to response
        result["session_id"] = session_id
        result["timestamp"] = int(time.time())

        logger.info(f"Demand forecast completed successfully for session {session_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running demand forecast: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Demand forecast failed: {str(e)}",
                "session_id": session_id,
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
        "endpoints": {
            "/chat": "Enhanced conversational BI with unified agent",
            "/database/test": "Test PostgreSQL database connection",
            "/database/connect": "Connect PostgreSQL database to session",
            "/database/status/{session_id}": "Get database connection status",
            "/database/disconnect/{session_id}": "Disconnect database from session",

            "/forecast/settings": "Configure forecast settings (economic events, supply chain)",
            "/forecast/settings/{session_id}": "Get forecast settings for session",
            "/forecast": "Run demand forecast with Prophet + validation",
            "/auth/signup": "Register a new user and send OTP",
            "/auth/verify-otp": "Verify OTP for user registration",
            "/auth/login": "Login user after email verification",
            "/shopify/connect": "Connect Shopify store and initiate OAuth",
            "/shopify/status/{session_id}": "Get Shopify connection status (auto-syncs if needed)",
            "/shopify/oauth-callback": "OAuth callback endpoint (receives access_token after authorization)",
            "/shopify/disconnect/{session_id}": "Disconnect Shopify store"
        },
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
