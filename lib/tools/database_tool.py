from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional
import logging
import json
from ..config import DatabaseManager

logger = logging.getLogger(__name__)

class DatabaseQueryTool(BaseTool):
    """Unified database query tool for all business data needs"""
    name: str = "database_query"
    description: str = "Query database for business data (sales, inventory, campaigns, customers)"
    session_id: Optional[str] = None

    def __init__(self, session_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id

    def _run(self, query_type: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Execute database query based on type and filters"""
        try:
            filters = filters or {}

            # Use real database data if connected, otherwise fall back to mock
            data = DatabaseManager.get_data(
                session_id=self.session_id,
                query_type=query_type,
                use_mock=False
            )

            if query_type == "low_stock_items":
                # Special case: filter inventory for low stock
                if query_type != "inventory_data":
                    inventory_data = DatabaseManager.get_data(
                        session_id=self.session_id,
                        query_type="inventory_data",
                        use_mock=False
                    )
                else:
                    inventory_data = data
                return [item for item in inventory_data if item.get('current_stock', 0) < item.get('reorder_level', 0)]
            elif query_type == "underperforming_campaigns":
                # Special case: filter campaigns for low ROAS
                if query_type != "campaign_data":
                    campaign_data = DatabaseManager.get_data(
                        session_id=self.session_id,
                        query_type="campaign_data",
                        use_mock=False
                    )
                else:
                    campaign_data = data
                return [camp for camp in campaign_data if camp.get('roas', 0) < 2.0]
            else:
                return data

        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return [{"error": str(e)}]

    async def _arun(self, query_type: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Async version of database query"""
        return self._run(query_type, filters)