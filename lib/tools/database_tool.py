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

class DatabaseQueryTool(BaseTool):
    """Unified database query tool for all business data needs"""
    name: str = "database_query"
    description: str = "Query database for business data (sales, inventory, campaigns, customers)"

    def _run(self, query_type: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Execute database query based on type and filters"""
        try:
            filters = filters or {}
            
            # Use mock data for now, but this can be easily switched to real DB
            mock_data = DatabaseManager.get_mock_data()
            
            if query_type == "sales_data":
                return mock_data.get('sales_data', [])
            elif query_type == "inventory_data":
                return mock_data.get('inventory_data', [])
            elif query_type == "campaign_data":
                return mock_data.get('campaign_data', [])
            elif query_type == "low_stock_items":
                inventory = mock_data.get('inventory_data', [])
                return [item for item in inventory if item['current_stock'] < item['reorder_level']]
            elif query_type == "underperforming_campaigns":
                campaigns = mock_data.get('campaign_data', [])
                return [camp for camp in campaigns if camp['roas'] < 2.0]
            else:
                logger.warning(f"Unknown query type: {query_type}")
                return []
                
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return [{"error": str(e)}]

    async def _arun(self, query_type: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Async version of database query"""
        return self._run(query_type, filters)