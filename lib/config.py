"""
Centralized configuration for PeppaSync LangChain Application
All shared configurations, LLM instances, and database connections
"""
import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import urllib.parse

# Load environment variables once
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppConfig:
    """Centralized application configuration"""

    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # NeonDB/Postgres Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL')  # Example: postgresql+psycopg2://<user>:<password>@<host>/<db>

    # Web Content Fetching Configuration (always enabled, runs once on startup)
    MAX_WEB_ARTICLES_PER_SOURCE = int(os.getenv('MAX_WEB_ARTICLES_PER_SOURCE', '3'))  # Reduced for startup performance

    # For backwards compatibility (if DATABASE_URL is not present, fallback to these):
    DATABASE_CONFIG = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': os.getenv('DATABASE_PORT', '5432'),
        'database': os.getenv('DATABASE_NAME', 'peppagenbi'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD', '')
    }

    # LLM Configuration
    LLM_CONFIG = {
        'model': 'gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 2000
    }

    # Global Market Context
    GLOBAL_MARKET_CONTEXT = {
        'currency': 'USD',
        'typical_profit_margins': {
            'electronics': 0.12,
            'fashion': 0.50,
            'home_goods': 0.35,
            'health_beauty': 0.40,
            'automotive': 0.20
        },
        'seasonal_factors': {
            'Q1': 0.85,
            'Q2': 1.0,
            'Q3': 0.95,
            'Q4': 1.35
        },
        'market_conditions': {
            'avg_inflation_rate': 0.04,
            'ecommerce_growth_rate': 0.15,
            'competition_level': 'high'
        }
    }

    # Pinecone Vector Store Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'peppasync')
    USE_PINECONE_VECTOR = os.getenv('USE_PINECONE_VECTOR', 'true').lower() == 'true'
    VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', '1536'))  # OpenAI embeddings

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        if not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set")
            return False
        if not cls.DATABASE_URL:
            logger.error("DATABASE_URL (NeonDB/Postgres) is not set")
            return False
        if cls.USE_PINECONE_VECTOR and not cls.PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY is required when USE_PINECONE_VECTOR is enabled")
            return False
        return True

class LLMManager:
    """Singleton manager for LLM instances"""

    _chat_llm_instance = None
    _embeddings_instance = None

    @classmethod
    def get_chat_llm(cls, **kwargs) -> ChatOpenAI:
        """Get shared ChatOpenAI instance"""
        if cls._chat_llm_instance is None:
            config = AppConfig.LLM_CONFIG.copy()
            config.update(kwargs)
            cls._chat_llm_instance = ChatOpenAI(
                model=config['model'],
                temperature=config['temperature'],
                max_tokens=config.get('max_tokens'),
                api_key=AppConfig.OPENAI_API_KEY
            )
        return cls._chat_llm_instance

    @classmethod
    def get_embeddings(cls) -> OpenAIEmbeddings:
        """Get shared OpenAI embeddings instance"""
        if cls._embeddings_instance is None:
            cls._embeddings_instance = OpenAIEmbeddings(
                api_key=AppConfig.OPENAI_API_KEY
            )
        return cls._embeddings_instance

    @classmethod
    def reset_instances(cls):
        """Reset instances (useful for testing)"""
        cls._chat_llm_instance = None
        cls._embeddings_instance = None

class DatabaseManager:
    """Database connection manager with support for user database connections"""

    _user_connections: Dict[str, Dict[str, Any]] = {}  # session_id -> connection_info

    @staticmethod
    def get_connection_string() -> str:
        """Get NeonDB/Postgres connection string"""
        # Prefer DATABASE_URL (NeonDB); fallback to manual config
        if AppConfig.DATABASE_URL:
            return AppConfig.DATABASE_URL
        config = AppConfig.DATABASE_CONFIG
        return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

    @classmethod
    def set_user_connection(cls, session_id: str, connection_info: Dict[str, Any]):
        """Set user database connection for a session"""
        cls._user_connections[session_id] = connection_info

    @classmethod
    def get_user_connection(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Get user database connection for a session"""
        return cls._user_connections.get(session_id)

    @classmethod
    def has_user_connection(cls, session_id: str) -> bool:
        """Check if session has a user database connection"""
        return session_id in cls._user_connections

    @classmethod
    def remove_user_connection(cls, session_id: str):
        """Remove user database connection for a session"""
        if session_id in cls._user_connections:
            del cls._user_connections[session_id]

    @classmethod
    def get_data(cls, session_id: str, query_type: str, use_mock: bool = False) -> List[Dict[str, Any]]:
        """
        Get data based on session and query type

        Args:
            session_id: Session identifier
            query_type: Type of data requested (sales_data, inventory_data, etc.)
            use_mock: Force use of mock data for testing

        Returns:
            List of data records
        """
        # If user has connected their database and use_mock is False
        if not use_mock and cls.has_user_connection(session_id):
            return cls._get_user_data(session_id, query_type)
        else:
            # Fall back to mock data
            return cls._get_mock_data_by_type(query_type)

    @classmethod
    def _get_user_data(cls, session_id: str, query_type: str) -> List[Dict[str, Any]]:
        """Get data from user's connected database"""
        try:
            connection_info = cls.get_user_connection(session_id)
            if not connection_info:
                logger.warning(f"No user connection found for session {session_id}")
                return []

            # Extract database URL from connection info
            db_url = connection_info.get('database_url')
            if not db_url:
                logger.warning(f"No database URL found in connection info for session {session_id}")
                return []

            # Connect to PostgreSQL and query data
            return cls._query_postgres_database(db_url, query_type, session_id)

        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return []

    @classmethod
    def _query_postgres_database(cls, db_url: str, query_type: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Query PostgreSQL database for specific data type"""
        try:
            # Parse the database URL
            parsed_url = urllib.parse.urlparse(db_url)

            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=parsed_url.hostname,
                port=parsed_url.port or 5432,
                database=parsed_url.path.lstrip('/'),
                user=parsed_url.username,
                password=parsed_url.password
            )

            # Use RealDictCursor to get results as dictionaries
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Define queries for different data types using session schema
            queries = cls._get_data_queries(query_type, session_id)

            results = []
            for query in queries:
                try:
                    cursor.execute(query)
                    query_results = cursor.fetchall()
                    # Convert RealDictRow to regular dict
                    results.extend([dict(row) for row in query_results])
                    logger.info(f"Executed query for {query_type}: {len(query_results)} rows returned")
                except psycopg2.Error as e:
                    logger.warning(f"Query failed for {query_type}: {e}")
                    continue  # Try next query if one fails

            cursor.close()
            conn.close()

            return results

        except psycopg2.Error as e:
            logger.error(f"PostgreSQL connection error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error querying PostgreSQL database: {e}")
            return []

    @classmethod
    def _get_data_queries(cls, query_type: str, session_id: str = None) -> List[str]:
        """Get SQL queries for different data types using detected schema"""

        # Try to get the detected schema for this session
        if session_id and session_id in cls._user_connections:
            table_info = cls._user_connections[session_id].get('table_info', {})
            return cls._generate_dynamic_queries(query_type, table_info)

        # Fallback to template queries if no schema detected

        # Common table name variations for each data type
        query_templates = {
            "sales_data": [
                # Try common sales table names and structures
                """
                SELECT
                    product_name,
                    sales_amount,
                    units_sold,
                    category,
                    profit_margin,
                    sale_date
                FROM sales
                ORDER BY sale_date DESC
                LIMIT 1000
                """,
                """
                SELECT
                    product_name,
                    amount as sales_amount,
                    quantity as units_sold,
                    category,
                    margin as profit_margin,
                    created_at as sale_date
                FROM orders
                ORDER BY created_at DESC
                LIMIT 1000
                """,
                """
                SELECT
                    name as product_name,
                    total_amount as sales_amount,
                    qty as units_sold,
                    product_category as category,
                    profit_margin,
                    order_date as sale_date
                FROM transactions
                ORDER BY order_date DESC
                LIMIT 1000
                """
            ],
            "inventory_data": [
                """
                SELECT
                    product_name,
                    current_stock,
                    reorder_level,
                    category,
                    supplier
                FROM inventory
                ORDER BY product_name
                """,
                """
                SELECT
                    name as product_name,
                    stock_quantity as current_stock,
                    min_stock as reorder_level,
                    category,
                    supplier_name as supplier
                FROM products
                ORDER BY name
                """,
                """
                SELECT
                    product_name,
                    quantity_on_hand as current_stock,
                    reorder_point as reorder_level,
                    product_category as category,
                    vendor as supplier
                FROM stock
                ORDER BY product_name
                """
            ],
            "campaign_data": [
                """
                SELECT
                    campaign_id,
                    campaign_name,
                    platform,
                    spend,
                    revenue,
                    roas
                FROM campaigns
                ORDER BY campaign_id
                """,
                """
                SELECT
                    id as campaign_id,
                    name as campaign_name,
                    platform,
                    ad_spend as spend,
                    revenue,
                    (revenue/ad_spend) as roas
                FROM marketing_campaigns
                ORDER BY id
                """,
                """
                SELECT
                    campaign_id,
                    title as campaign_name,
                    channel as platform,
                    cost as spend,
                    total_revenue as revenue,
                    return_on_ad_spend as roas
                FROM ads
                ORDER BY campaign_id
                """
            ],
            "customer_data": [
                """
                SELECT
                    customer_id,
                    name,
                    email,
                    signup_date,
                    total_orders,
                    total_spent
                FROM customers
                ORDER BY signup_date DESC
                LIMIT 1000
                """,
                """
                SELECT
                    id as customer_id,
                    customer_name as name,
                    email,
                    created_at as signup_date,
                    order_count as total_orders,
                    lifetime_value as total_spent
                FROM users
                ORDER BY created_at DESC
                LIMIT 1000
                """,
                """
                SELECT
                    customer_id,
                    first_name || ' ' || last_name as name,
                    email,
                    registration_date as signup_date,
                    purchase_count as total_orders,
                    total_purchase_amount as total_spent
                FROM client_data
                ORDER BY registration_date DESC
                LIMIT 1000
                """
            ],
            "low_stock_items": [
                """
                SELECT
                    product_name,
                    current_stock,
                    reorder_level,
                    category,
                    supplier
                FROM inventory
                WHERE current_stock < reorder_level
                ORDER BY (current_stock::float / reorder_level) ASC
                """,
                """
                SELECT
                    name as product_name,
                    stock_quantity as current_stock,
                    min_stock as reorder_level,
                    category,
                    supplier_name as supplier
                FROM products
                WHERE stock_quantity < min_stock
                ORDER BY (stock_quantity::float / min_stock) ASC
                """
            ],
            "underperforming_campaigns": [
                """
                SELECT
                    campaign_id,
                    campaign_name,
                    platform,
                    spend,
                    revenue,
                    roas
                FROM campaigns
                WHERE roas < 2.0
                ORDER BY roas ASC
                """,
                """
                SELECT
                    id as campaign_id,
                    name as campaign_name,
                    platform,
                    ad_spend as spend,
                    revenue,
                    (revenue/ad_spend) as roas
                FROM marketing_campaigns
                WHERE (revenue/ad_spend) < 2.0
                ORDER BY (revenue/ad_spend) ASC
                """
            ]
        }

        return query_templates.get(query_type, [])

    @classmethod
    def _generate_dynamic_queries(cls, query_type: str, table_info: Dict) -> List[str]:
        """Generate SQL queries based on detected database schema"""
        queries = []

        if query_type == "sales_data":
            # Look for sales-related tables
            for table_name, columns in table_info.items():
                table_lower = table_name.lower()
                if any(keyword in table_lower for keyword in ['sale', 'order', 'transaction', 'purchase']):
                    column_names = [col['name'] for col in columns]

                    # Try to find a date column for filtering and ordering
                    date_cols = [col for col in column_names if any(date_word in col.lower() for date_word in ['date', 'time', 'created', 'updated'])]

                    if date_cols:
                        # Add 30-day filter and order by date
                        query = f"SELECT * FROM {table_name} WHERE {date_cols[0]} >= NOW() - INTERVAL '30 days' ORDER BY {date_cols[0]} DESC LIMIT 1000"
                    else:
                        # No date column found, just limit results
                        query = f"SELECT * FROM {table_name} ORDER BY {column_names[0]} LIMIT 1000"

                    queries.append(query)

        elif query_type == "inventory_data":
            # Look for inventory-related tables
            for table_name, columns in table_info.items():
                table_lower = table_name.lower()
                if any(keyword in table_lower for keyword in ['inventory', 'stock', 'product']):
                    query = f"SELECT * FROM {table_name} LIMIT 100"
                    queries.append(query)

        return queries

    @classmethod
    def test_database_connection(cls, db_url: str) -> Dict[str, Any]:
        """Test database connection and return available tables/schemas"""
        try:
            # Parse the database URL
            parsed_url = urllib.parse.urlparse(db_url)

            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=parsed_url.hostname,
                port=parsed_url.port or 5432,
                database=parsed_url.path.lstrip('/'),
                user=parsed_url.username,
                password=parsed_url.password
            )

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get list of tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            tables = [row['table_name'] for row in cursor.fetchall()]

            # Try to detect what data types are available
            detected_data_types = []
            table_info = {}

            for table in tables:
                try:
                    # Get column info for each table
                    cursor.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = %s AND table_schema = 'public'
                        ORDER BY ordinal_position
                    """, (table,))
                    columns = [{'name': row['column_name'], 'type': row['data_type']} for row in cursor.fetchall()]
                    table_info[table] = columns

                    # Detect data types based on table names and columns
                    table_lower = table.lower()
                    column_names = [col['name'].lower() for col in columns]

                    if any(keyword in table_lower for keyword in ['sale', 'order', 'transaction', 'purchase']):
                        if 'sales_data' not in detected_data_types:
                            detected_data_types.append('sales_data')

                    if any(keyword in table_lower for keyword in ['inventory', 'stock', 'product']):
                        if 'inventory_data' not in detected_data_types:
                            detected_data_types.append('inventory_data')

                    if any(keyword in table_lower for keyword in ['campaign', 'marketing', 'ad']):
                        if 'campaign_data' not in detected_data_types:
                            detected_data_types.append('campaign_data')

                    if any(keyword in table_lower for keyword in ['customer', 'user', 'client']):
                        if 'customer_data' not in detected_data_types:
                            detected_data_types.append('customer_data')

                except Exception as e:
                    logger.warning(f"Could not analyze table {table}: {e}")
                    continue

            cursor.close()
            conn.close()

            return {
                "success": True,
                "message": "Database connection successful",
                "available_tables": tables,
                "detected_data_types": detected_data_types,
                "table_info": table_info,
                "database_name": parsed_url.path.lstrip('/'),
                "host": parsed_url.hostname
            }

        except psycopg2.Error as e:
            return {
                "success": False,
                "message": f"Database connection failed: {str(e)}",
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection test failed: {str(e)}",
                "error": str(e)
            }

    @classmethod
    def _get_mock_data_by_type(cls, query_type: str) -> List[Dict[str, Any]]:
        """Get mock data by specific type"""
        mock_data = cls.get_mock_data()
        return mock_data.get(query_type, [])

    @staticmethod
    def get_mock_data() -> Dict[str, Any]:
        """Get mock data for development/testing"""
        return {
            'sales_data': [
                {
                    'product_name': 'iPhone 15 Pro',
                    'sales_amount': 94750,
                    'units_sold': 95,
                    'category': 'Electronics',
                    'profit_margin': 0.12
                },
                {
                    'product_name': 'Nike Air Max',
                    'sales_amount': 6750,
                    'units_sold': 45,
                    'category': 'Fashion',
                    'profit_margin': 0.50
                },
                {
                    'product_name': 'Samsung Smart TV 55"',
                    'sales_amount': 19800,
                    'units_sold': 20,
                    'category': 'Electronics',
                    'profit_margin': 0.18
                }
            ],
            'inventory_data': [
                {
                    'product_name': 'iPhone 15 Pro',
                    'current_stock': 5,
                    'reorder_level': 20,
                    'category': 'Electronics',
                    'supplier': 'Apple Inc.'
                },
                {
                    'product_name': 'Samsung Galaxy S24',
                    'current_stock': 12,
                    'reorder_level': 15,
                    'category': 'Electronics',
                    'supplier': 'Samsung Electronics'
                },
                {
                    'product_name': 'Nike Air Max',
                    'current_stock': 2,
                    'reorder_level': 10,
                    'category': 'Fashion',
                    'supplier': 'Nike Inc.'
                },
                {
                    'product_name': 'MacBook Pro 14',
                    'current_stock': 3,
                    'reorder_level': 8,
                    'category': 'Electronics',
                    'supplier': 'Apple Inc.'
                },
                {
                    'product_name': 'Adidas Sneakers',
                    'current_stock': 25,
                    'reorder_level': 20,
                    'category': 'Fashion',
                    'supplier': 'Adidas AG'
                }
            ],
            'campaign_data': [
                {
                    'campaign_id': 'FB_001',
                    'campaign_name': 'Electronics Flash Sale',
                    'platform': 'Facebook',
                    'spend': 1500,
                    'revenue': 4500,
                    'roas': 3.0
                },
                {
                    'campaign_id': 'IG_002',
                    'campaign_name': 'Fashion Week Promo',
                    'platform': 'Instagram',
                    'spend': 2000,
                    'revenue': 6200,
                    'roas': 3.1
                },
                {
                    'campaign_id': 'G_002',
                    'campaign_name': 'Brand Awareness - Google Ads',
                    'platform': 'Google',
                    'spend': 2200,
                    'revenue': 2900,
                    'roas': 1.32
                },
                {
                    'campaign_id': 'TT_001',
                    'campaign_name': 'Youth Fashion - TikTok',
                    'platform': 'TikTok',
                    'spend': 800,
                    'revenue': 1800,
                    'roas': 2.25
                }
            ]
        }
