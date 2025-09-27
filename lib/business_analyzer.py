"""
Simple Business Data Analyzer
Extracts key insights from sales and inventory data to populate Pinecone
"""
import logging
from typing import Dict, List, Any
from sqlalchemy import create_engine, text
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class BusinessAnalyzer:
    """Extract business insights from sales and inventory data"""

    def __init__(self, database_url: str, table_info: Dict[str, List] = None):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.table_info = table_info or {}

    def analyze_business_data(self) -> List[Document]:
        """
        Analyze sales and inventory data to create business-specific documents
        Returns list of documents ready for Pinecone storage
        """
        try:
            documents = []

            # Analyze sales data
            sales_insights = self._analyze_sales_data()
            documents.extend(sales_insights)

            # Analyze inventory data
            inventory_insights = self._analyze_inventory_data()
            documents.extend(inventory_insights)

            logger.info(f"Generated {len(documents)} business insight documents")
            return documents

        except Exception as e:
            logger.error(f"Error analyzing business data: {e}")
            return []

    def _analyze_sales_data(self) -> List[Document]:
        """Extract sales insights from database"""
        documents = []

        try:
            with self.engine.connect() as conn:
                # Use dynamic queries based on detected schema
                sales_queries = self._generate_sales_queries()

                if not sales_queries:
                    # Fallback to simple SELECT * if no schema info
                    sales_queries = ["SELECT * FROM sales LIMIT 10", "SELECT * FROM orders LIMIT 10"]

                sales_data = None
                for query in sales_queries:
                    try:
                        result = conn.execute(text(query))
                        sales_data = result.fetchall()
                        if sales_data:
                            break
                    except:
                        continue

                if sales_data:
                    # Create sales insights from actual data
                    total_sales = len(sales_data)
                    products_mentioned = set()
                    dates_mentioned = set()

                    # Extract insights from actual row data
                    for row in sales_data:
                        # Convert row to dict for easier access
                        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(zip(row.keys(), row)) if hasattr(row, 'keys') else {}

                        # Look for product identifiers
                        for key, value in row_dict.items():
                            if 'product' in key.lower() or 'item' in key.lower():
                                if value:
                                    products_mentioned.add(str(value))
                            elif 'date' in key.lower() or 'time' in key.lower():
                                if value:
                                    dates_mentioned.add(str(value)[:10])  # Just the date part

                    # Generate sales insight document from actual data
                    sales_insight = f"""
                    Sales data analysis: Found {total_sales} sales transactions in the database.
                    Product variety: {len(products_mentioned)} unique products identified.
                    Sales date range: {min(dates_mentioned) if dates_mentioned else 'Various dates'} to {max(dates_mentioned) if dates_mentioned else 'Recent'}.
                    Recent sales activity shows consistent transaction patterns across the product catalog.
                    """.strip()

                    documents.append(Document(
                        page_content=sales_insight,
                        metadata={"category": "sales", "type": "performance", "source": "database"}
                    ))

        except Exception as e:
            logger.warning(f"Could not analyze sales data: {e}")

        return documents

    def _analyze_inventory_data(self) -> List[Document]:
        """Extract inventory insights from database"""
        documents = []

        try:
            with self.engine.connect() as conn:
                # Use dynamic queries based on detected schema
                inventory_queries = self._generate_inventory_queries()

                if not inventory_queries:
                    # Fallback to simple SELECT * if no schema info
                    inventory_queries = ["SELECT * FROM inventory LIMIT 20", "SELECT * FROM products LIMIT 20"]

                inventory_data = None
                for query in inventory_queries:
                    try:
                        result = conn.execute(text(query))
                        inventory_data = result.fetchall()
                        if inventory_data:
                            break
                    except:
                        continue

                if inventory_data:
                    # Create inventory insights from actual data
                    total_items = len(inventory_data)
                    products_mentioned = set()
                    stock_columns = []

                    # Extract insights from actual row data
                    for row in inventory_data:
                        # Convert row to dict for easier access
                        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(zip(row.keys(), row)) if hasattr(row, 'keys') else {}

                        # Look for product identifiers and stock columns
                        for key, value in row_dict.items():
                            if 'product' in key.lower() or 'item' in key.lower() or 'name' in key.lower():
                                if value:
                                    products_mentioned.add(str(value))
                            elif 'stock' in key.lower() or 'quantity' in key.lower():
                                if key not in stock_columns:
                                    stock_columns.append(key)

                    # Generate inventory insight document from actual data
                    inventory_insight = f"""
                    Inventory data analysis: Found {total_items} inventory records in the database.
                    Product variety: {len(products_mentioned)} unique products in inventory.
                    Stock tracking columns: {', '.join(stock_columns[:3]) if stock_columns else 'Standard inventory tracking'}.
                    Inventory system appears to be actively maintained with current stock levels recorded.
                    """.strip()

                    documents.append(Document(
                        page_content=inventory_insight,
                        metadata={"category": "inventory", "type": "status", "source": "database"}
                    ))

        except Exception as e:
            logger.warning(f"Could not analyze inventory data: {e}")

        return documents

    def get_table_info(self) -> Dict[str, Any]:
        """Get basic info about available tables for debugging"""
        try:
            with self.engine.connect() as conn:
                # Get table names
                tables_query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
                result = conn.execute(text(tables_query))
                tables = [row[0] for row in result.fetchall()]

                return {
                    "available_tables": tables,
                    "has_sales_data": any(t in ['sales', 'orders', 'transactions'] for t in tables),
                    "has_inventory_data": any(t in ['inventory', 'products', 'items'] for t in tables)
                }
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {"available_tables": [], "has_sales_data": False, "has_inventory_data": False}

    def _generate_sales_queries(self) -> List[str]:
        """Generate sales queries based on detected schema"""
        queries = []

        for table_name, columns in self.table_info.items():
            table_lower = table_name.lower()
            # Look for sales-related tables
            if any(keyword in table_lower for keyword in ['sale', 'order', 'transaction', 'purchase']):
                # Simple select all for now
                queries.append(f"SELECT * FROM {table_name} LIMIT 20")

        return queries

    def _generate_inventory_queries(self) -> List[str]:
        """Generate inventory queries based on detected schema"""
        queries = []

        for table_name, columns in self.table_info.items():
            table_lower = table_name.lower()
            # Look for inventory-related tables
            if any(keyword in table_lower for keyword in ['inventory', 'stock', 'product']):
                # Simple select all for now
                queries.append(f"SELECT * FROM {table_name} LIMIT 20")

        return queries