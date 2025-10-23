"""
Shopify Integration Routes
Single endpoint for frontend: /shopify/connect
Auto-syncs data internally after connection
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from lib.shopify_service import get_shopify_service
from lib.config import DatabaseManager

logger = logging.getLogger(__name__)

shopify_router = APIRouter(prefix="/shopify", tags=["shopify"])

"""
Shopify Integration Routes
Single endpoint for frontend: /shopify/connect
Auto-syncs data internally after connection
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from lib.shopify_service import get_shopify_service
from lib.config import DatabaseManager

logger = logging.getLogger(__name__)

shopify_router = APIRouter(prefix="/shopify", tags=["shopify"])


class ShopifyConnectRequest(BaseModel):
    shop_name: str  # e.g., "thediamondstoreprnz"
    session_id: str
    redirect_url: Optional[str] = "http://localhost:3000/shopify/callback"


@shopify_router.post("/connect")
async def connect_shopify(request: ShopifyConnectRequest):
    """
    Single endpoint for Shopify connection (frontend-facing).

    Flow:
    1. Frontend sends shop_name
    2. Backend initiates OAuth with connector API
    3. User authorizes on Shopify
    4. Connector handles callback and provides access token
    5. Backend auto-syncs data (internal)
    6. Returns success with connection status

    Args:
        shop_name: Shopify store name (e.g., "mystore" from mystore.myshopify.com)
        session_id: User session ID
        redirect_url: Where to redirect after OAuth

    Returns:
        Connection status and auth URL
    """
    try:
        logger.info(f"🔌 Shopify connection request: {request.shop_name}")

        shopify_service = get_shopify_service()

        # Step 1: Initiate OAuth via connector
        result = await shopify_service.connect_shopify_store(
            shop_name=request.shop_name,
            redirect_url=request.redirect_url
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Connection failed")
            )

        # Step 2: Store connection info in session
        DatabaseManager.store_shopify_connection(
            session_id=request.session_id,
            shop_name=request.shop_name,
            connection_data=result
        )

        logger.info(f"Shopify connection initiated for {request.shop_name}")

        # Step 3: Auto-sync orders if access token is available
        access_token = result.get("data", {}).get("access_token")
        shop_url = f"{request.shop_name}.myshopify.com"
        orders_synced = 0
        if access_token:
            try:
                orders = await shopify_service.fetch_shopify_orders(
                    shop_url=shop_url,
                    access_token=access_token,
                    days_back=90,
                    limit=250
                )
                DatabaseManager.store_shopify_orders(request.session_id, orders)
                orders_synced = len(orders)
                logger.info(f"Synced {orders_synced} orders for {request.shop_name}")
            except Exception as sync_err:
                logger.error(f"❌ Error syncing Shopify orders: {sync_err}")

        return {
            "success": True,
            "message": "Shopify OAuth initiated",
            "shop_name": request.shop_name,
            "auth_url": result.get("data", {}).get("auth_url"),
            "orders_synced": orders_synced,
            "next_step": "User will be redirected to Shopify for authorization"
        }

    except Exception as e:
        logger.error(f"❌ Shopify connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@shopify_router.get("/status/{session_id}")
async def get_shopify_status(session_id: str):
    """
    Check if Shopify is connected and get data counts.

    Args:
        session_id: User session ID

    Returns:
        Connection status and data availability
    """
    try:
        connection_info = DatabaseManager.get_shopify_connection(session_id)

        if not connection_info:
            return {
                "connected": False,
                "shop_name": None,
                "message": "No Shopify store connected"
            }

        return {
            "connected": True,
            "shop_name": connection_info.get("shop_name"),
            "shop_url": f"{connection_info.get('shop_name')}.myshopify.com",
            "last_sync": connection_info.get("last_sync"),
            "data_counts": connection_info.get("data_counts", {
                "orders": 0,
                "products": 0,
                "customers": 0
            }),
            "data_sources": ["forecasting", "analytics", "customer_insights"]
        }

    except Exception as e:
        logger.error(f"❌ Error checking Shopify status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
