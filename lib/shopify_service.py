"""
Shopify Integration Service
Handles connection, authentication, and data syncing with Shopify stores via external connector
"""

import httpx
import logging
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)


class ShopifyService:
    """
    Service for integrating with Shopify stores via external connector API.

    External Connector API: https://connector.fundam.ng/api
    """

    def __init__(self, connector_base_url: str = "https://connector.fundam.ng/api"):
        self.connector_base_url = connector_base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def connect_shopify_store(
        self,
        shop_name: str,
        redirect_url: str,
        bearer_token: Optional[str] = '15|dIKpifYjKg4jlutB5mZidgG9x4x1iDsdHNLPEfgR9f09d70d',
        session_id: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Initiate Shopify connection via external connector.

        Args:
            shop_name: Shopify store name (e.g., "thediamondstoreprnz")
            redirect_url: URL to redirect after auth
            bearer_token: Optional bearer token for auth

        Returns:
            Connection result with auth URL
        """
        try:
            logger.info(f"🔌 Connecting to Shopify store: {shop_name}")

            url = f"{self.connector_base_url}/connect/shopify"

            payload = {
                "store": shop_name,
                "redirect_url": redirect_url
            }

            headers = {}
            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"

            response = await self.client.post(url, json=payload, headers=headers)
            logger.info(f"Connector response: {response.status_code} - {response.text}")

            if response.status_code == 200:
                logger.info(f"✅ Shopify connection initiated for {shop_name}")

                # Parse response data
                response_data = response.json() if response.text else {}

                # NOTE: Auto-sync is NOT done here because:
                # 1. User hasn't authorized via OAuth yet (they need to visit auth_url)
                # 2. access_token is only available AFTER OAuth authorization
                # 3. Attempting to sync now would fail because we don't have permissions yet
                #
                # Auto-sync should happen:
                # - After user completes OAuth flow (in a callback handler)
                # - Or when user manually triggers sync after authorization

                return {
                    "success": True,
                    "message": "Shopify connection initiated. Please authorize via auth_url.",
                    "shop": shop_name,
                    "data": response_data
                }
            else:
                logger.error(f"❌ Shopify connection failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Connection failed: {response.status_code}",
                    "details": response.text
                }

        except Exception as e:
            logger.error(f"❌ Shopify connection error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_connections(
        self,
        bearer_token: str = '15|dIKpifYjKg4jlutB5mZidgG9x4x1iDsdHNLPEfgR9f09d70d'
    ) -> List[Dict[str, Any]]:
        """
        Get list of all Shopify connections for this user.

        Args:
            bearer_token: Bearer token for auth

        Returns:
            List of connections with their IDs
        """
        try:
            logger.info(f"📋 Fetching Shopify connections")

            url = f"{self.connector_base_url}/connections"

            headers = {"Authorization": f"Bearer {bearer_token}"}

            response = await self.client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json() if response.text else {}
                # Response structure: {'success': True, 'connections': [...]}
                connections = data.get("connections", [])
                logger.info(f"✅ Retrieved {len(connections)} connections")
                return connections
            else:
                logger.error(f"❌ Connections fetch failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ Connections fetch error: {str(e)}")
            return []

    async def get_connection_id_by_shop(
        self,
        shop_name: str,
        bearer_token: str = '15|dIKpifYjKg4jlutB5mZidgG9x4x1iDsdHNLPEfgR9f09d70d'
    ) -> Optional[int]:
        """
        Get connection ID for a specific shop name.

        Args:
            shop_name: Shopify store name (e.g., "dewdrop-labs-2")
            bearer_token: Bearer token for auth

        Returns:
            Connection ID or None if not found
        """
        try:
            connections = await self.get_connections(bearer_token)
            logger.info(f"🔍 Searching for connection ID for shop: {shop_name}")
            logger.debug(f"Available connections: {connections}")

            for conn in connections:
                # Connection structure: {'id': 7, 'slug': 'dewdrop-labs-2.myshopify.com', 'platform': 'shopify', ...}
                # Match by slug (which contains the shop name)
                slug = conn.get("slug", "")

                # Check if shop_name is in the slug (e.g., "dewdrop-labs-2" in "dewdrop-labs-2.myshopify.com")
                if shop_name in slug:
                    connection_id = conn.get("id")
                    logger.info(f"✅ Found connection ID {connection_id} for {shop_name} (slug: {slug})")
                    return connection_id

            logger.warning(f"⚠️  No connection found for shop: {shop_name}")
            logger.warning(f"Available slugs: {[c.get('slug') for c in connections]}")
            return None

        except Exception as e:
            logger.error(f"❌ Error getting connection ID: {str(e)}")
            return None

    async def get_customer(
        self,
        customer_id: int,
        connection_id: int,
        bearer_token: Optional[str] = '15|dIKpifYjKg4jlutB5mZidgG9x4x1iDsdHNLPEfgR9f09d70d'
    ) -> Dict[str, Any]:
        """
        Get single customer from Shopify via connector.

        Args:
            customer_id: Shopify customer ID
            connection_id: Connection ID from connector
            bearer_token: Bearer token for auth

        Returns:
            Customer data
        """
        try:
            logger.info(f"📊 Fetching Shopify customer: {customer_id}")

            url = f"{self.connector_base_url}/shopify/customer/{customer_id}"

            payload = {"connectionId": connection_id}

            headers = {}
            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"

            response = await self.client.get(url, json=payload, headers=headers)

            if response.status_code == 200:
                logger.info(f"✅ Customer {customer_id} retrieved")
                return {
                    "success": True,
                    "customer": response.json() if response.text else {}
                }
            else:
                logger.error(f"❌ Customer fetch failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Failed to fetch customer: {response.status_code}"
                }

        except Exception as e:
            logger.error(f"❌ Customer fetch error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def fetch_shopify_orders(
        self,
        connection_id: int,
        bearer_token: str = '15|dIKpifYjKg4jlutB5mZidgG9x4x1iDsdHNLPEfgR9f09d70d'
    ) -> List[Dict[str, Any]]:
        """
        Fetch orders via connector API.

        Args:
            connection_id: Connection ID from connector
            bearer_token: Bearer token for auth

        Returns:
            List of orders
        """
        try:
            logger.info(f"📦 Fetching Shopify orders via connector (connection {connection_id})")

            url = f"{self.connector_base_url}/shopify/fetch/orders/{connection_id}"

            headers = {"Authorization": f"Bearer {bearer_token}"}

            response = await self.client.get(url, headers=headers)

            if response.status_code != 200:
                logger.error(f"❌ Connector API error: {response.status_code} - {response.text}")
                return []

            data = response.json() if response.text else {}

            # The connector might return orders directly or nested in a data field
            orders = data.get("orders") or data.get("data") or (data if isinstance(data, list) else [])

            logger.info(f"✅ Total orders fetched: {len(orders)}")
            return orders

        except Exception as e:
            logger.error(f"❌ Error fetching Shopify orders: {str(e)}")
            return []

    async def fetch_shopify_products(
        self,
        connection_id: int,
        bearer_token: str = '15|dIKpifYjKg4jlutB5mZidgG9x4x1iDsdHNLPEfgR9f09d70d'
    ) -> List[Dict[str, Any]]:
        """
        Fetch products via connector API.

        Args:
            connection_id: Connection ID from connector
            bearer_token: Bearer token for auth

        Returns:
            List of products
        """
        try:
            logger.info(f"🛍️  Fetching Shopify products via connector (connection {connection_id})")

            url = f"{self.connector_base_url}/shopify/fetch/products/{connection_id}"

            headers = {"Authorization": f"Bearer {bearer_token}"}

            response = await self.client.get(url, headers=headers)

            if response.status_code != 200:
                logger.error(f"❌ Connector API error: {response.status_code} - {response.text}")
                return []

            data = response.json() if response.text else {}

            # The connector might return products directly or nested in a data field
            products = data.get("products") or data.get("data") or (data if isinstance(data, list) else [])

            logger.info(f"✅ Total products fetched: {len(products)}")
            return products

        except Exception as e:
            logger.error(f"❌ Error fetching Shopify products: {str(e)}")
            return []

    async def fetch_shopify_customers(
        self,
        connection_id: int,
        bearer_token: str = '15|dIKpifYjKg4jlutB5mZidgG9x4x1iDsdHNLPEfgR9f09d70d'
    ) -> List[Dict[str, Any]]:
        """
        Fetch customers via connector API.

        Args:
            connection_id: Connection ID from connector
            bearer_token: Bearer token for auth

        Returns:
            List of customers
        """
        try:
            logger.info(f"👥 Fetching Shopify customers via connector (connection {connection_id})")

            url = f"{self.connector_base_url}/shopify/customers/{connection_id}"

            headers = {"Authorization": f"Bearer {bearer_token}"}

            response = await self.client.get(url, headers=headers)

            if response.status_code != 200:
                logger.error(f"❌ Connector API error: {response.status_code} - {response.text}")
                return []

            data = response.json() if response.text else {}

            # The connector might return customers directly or nested in a data field
            customers = data.get("customers") or data.get("data") or (data if isinstance(data, list) else [])

            logger.info(f"✅ Total customers fetched: {len(customers)}")
            return customers

        except Exception as e:
            logger.error(f"❌ Error fetching Shopify customers: {str(e)}")
            return []

    async def auto_sync_orders(
        self,
        shop_name: str,
        access_token: str,
        session_id: str,
        days_back: int = 90,
        limit: int = 250
    ) -> int:
        """
        Auto-sync orders from Shopify via connector and store them in the database.

        Args:
            shop_name: Shopify store name (e.g., "dewdrop-labs-2")
            access_token: Bearer token for connector API (not Shopify access_token)
            session_id: User session ID
            days_back: Not used with connector API
            limit: Not used with connector API

        Returns:
            Number of orders synced
        """
        try:
            logger.info(f"🔄 Auto-syncing orders for {shop_name}")

            # Step 1: Get connection ID from connector
            connection_id = await self.get_connection_id_by_shop(shop_name, access_token)

            if not connection_id:
                logger.error(f"❌ No connection ID found for {shop_name}")
                return 0

            # Step 2: Fetch orders using connection ID
            orders = await self.fetch_shopify_orders(connection_id, access_token)

            # Step 3: Store orders in database
            if orders:
                from lib.config import DatabaseManager
                DatabaseManager.store_shopify_orders(session_id, orders)
                logger.info(f"✅ Auto-sync completed: {len(orders)} orders stored for {shop_name}")
            else:
                logger.warning(f"⚠️ No orders fetched from {shop_name}")

            return len(orders)

        except Exception as e:
            logger.error(f"❌ Error in auto_sync_orders for {shop_name}: {str(e)}")
            raise

    def transform_orders_for_forecasting(self, orders: List[Dict]) -> List[Dict[str, Any]]:
        """
        Transform Shopify orders to format expected by forecasting engine.

        Args:
            orders: Raw Shopify orders

        Returns:
            Standardized order data for forecasting
        """
        standardized_orders = []

        for order in orders:
            try:
                # Extract order details
                order_data = {
                    "order_id": order.get("id"),
                    "order_name": order.get("name"),
                    "customer_id": order.get("customer", {}).get("id"),
                    "customer_name": f"{order.get('customer', {}).get('first_name', '')} {order.get('customer', {}).get('last_name', '')}".strip() or "Guest",
                    "email": order.get("email") or order.get("customer", {}).get("email"),
                    "order_date": order.get("created_at"),
                    "total_amount": float(order.get("total_price", 0)),
                    "currency": order.get("currency", "USD"),
                    "line_items_count": len(order.get("line_items", [])),
                    "financial_status": order.get("financial_status"),
                    "fulfillment_status": order.get("fulfillment_status")
                }

                # Add line items
                for item in order.get("line_items", []):
                    order_data_with_item = order_data.copy()
                    order_data_with_item.update({
                        "product_id": item.get("product_id"),
                        "product_name": item.get("title"),
                        "variant_id": item.get("variant_id"),
                        "quantity": item.get("quantity", 0),
                        "price": float(item.get("price", 0)),
                        "sku": item.get("sku")
                    })
                    standardized_orders.append(order_data_with_item)

            except Exception as e:
                logger.warning(f"⚠️  Failed to transform order {order.get('id')}: {str(e)}")
                continue

        logger.info(f"✅ Transformed {len(standardized_orders)} order line items")
        return standardized_orders

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Singleton instance
_shopify_service = None

def get_shopify_service() -> ShopifyService:
    """Get singleton Shopify service instance."""
    global _shopify_service
    if _shopify_service is None:
        _shopify_service = ShopifyService()
    return _shopify_service
