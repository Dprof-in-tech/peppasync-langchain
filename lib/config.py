"""
Centralized configuration for PeppaSync LangChain Application
All shared configurations, LLM instances, and database connections
"""
import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

    # Nigerian Market Context
    NIGERIAN_MARKET_CONTEXT = {
        'currency': 'NGN',
        'typical_profit_margins': {
            'electronics': 0.15,
            'fashion': 0.45,
            'home_goods': 0.35
        },
        'seasonal_factors': {
            'Q1': 0.8,
            'Q2': 1.1,
            'Q3': 0.9,
            'Q4': 1.4
        },
        'market_conditions': {
            'inflation_rate': 0.12,
            'growth_rate': 0.08,
            'competition_level': 'high'
        }
    }

    # Vector Store Configuration
    VECTOR_STORE_PATH = './vector_store'

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        if not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set")
            return False
        if not cls.DATABASE_URL:
            logger.error("DATABASE_URL (NeonDB/Postgres) is not set")
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
    """Database connection manager"""

    @staticmethod
    def get_connection_string() -> str:
        """Get NeonDB/Postgres connection string"""
        # Prefer DATABASE_URL (NeonDB); fallback to manual config
        if AppConfig.DATABASE_URL:
            return AppConfig.DATABASE_URL
        config = AppConfig.DATABASE_CONFIG
        return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

    @staticmethod
    def get_mock_data() -> Dict[str, Any]:
        """Get mock data for development/testing"""
        return {
            'sales_data': [
                {
                    'product_name': 'iPhone 15 Pro',
                    'sales_amount': 2850000,
                    'units_sold': 95,
                    'category': 'Electronics',
                    'profit_margin': 0.15
                },
                {
                    'product_name': 'Nike Air Max',
                    'sales_amount': 675000,
                    'units_sold': 45,
                    'category': 'Fashion',
                    'profit_margin': 0.45
                },
                {
                    'product_name': 'Samsung Smart TV 55"',
                    'sales_amount': 1200000,
                    'units_sold': 20,
                    'category': 'Electronics',
                    'profit_margin': 0.20
                }
            ],
            'inventory_data': [
                {
                    'product_name': 'iPhone 15 Pro',
                    'current_stock': 5,
                    'reorder_level': 20,
                    'category': 'Electronics',
                    'supplier': 'Apple Nigeria'
                },
                {
                    'product_name': 'Samsung Galaxy S24',
                    'current_stock': 12,
                    'reorder_level': 15,
                    'category': 'Electronics',
                    'supplier': 'Samsung Nigeria'
                },
                {
                    'product_name': 'Nike Air Max',
                    'current_stock': 2,
                    'reorder_level': 10,
                    'category': 'Fashion',
                    'supplier': 'Nike Nigeria'
                },
                {
                    'product_name': 'MacBook Pro 14',
                    'current_stock': 3,
                    'reorder_level': 8,
                    'category': 'Electronics',
                    'supplier': 'Apple Distributor NG'
                },
                {
                    'product_name': 'Adidas Sneakers',
                    'current_stock': 25,
                    'reorder_level': 20,
                    'category': 'Fashion',
                    'supplier': 'Adidas Nigeria'
                }
            ],
            'campaign_data': [
                {
                    'campaign_id': 'FB_001',
                    'campaign_name': 'Electronics Flash Sale',
                    'platform': 'Facebook',
                    'spend': 150000,
                    'revenue': 450000,
                    'roas': 3.0
                },
                {
                    'campaign_id': 'IG_002',
                    'campaign_name': 'Fashion Week Promo',
                    'platform': 'Instagram',
                    'spend': 200000,
                    'revenue': 620000,
                    'roas': 3.1
                },
                {
                    'campaign_id': 'G_002',
                    'campaign_name': 'Brand Awareness - Google Ads',
                    'platform': 'Google',
                    'spend': 220000,
                    'revenue': 290000,
                    'roas': 1.32
                },
                {
                    'campaign_id': 'TT_001',
                    'campaign_name': 'Youth Fashion - TikTok',
                    'platform': 'TikTok',
                    'spend': 80000,
                    'revenue': 180000,
                    'roas': 2.25
                }
            ]
        }
