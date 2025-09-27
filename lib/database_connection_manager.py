"""
Database Connection Manager
Handles prompting users for database connections and managing connection status
"""
import logging
from typing import Dict, List, Any, Optional
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    NOT_CONNECTED = "not_connected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"

class DatabaseConnectionManager:
    """
    Manages database connections and prompts users when data is needed
    """

    def __init__(self):
        # In-memory storage for connection status (in production, use Redis/database)
        self.connection_status: Dict[str, Dict[str, Any]] = {}

    def get_connection_status(self, session_id: str) -> Dict[str, Any]:
        """Get current connection status for a session"""
        return self.connection_status.get(session_id, {
            "status": ConnectionStatus.NOT_CONNECTED.value,
            "connected_data_types": [],
            "last_check": None,
            "connection_details": {}
        })

    def set_connection_status(self, session_id: str, status: ConnectionStatus,
                            data_types: List[str] = None, details: Dict[str, Any] = None):
        """Update connection status for a session"""
        if session_id not in self.connection_status:
            self.connection_status[session_id] = {}

        self.connection_status[session_id].update({
            "status": status.value,
            "connected_data_types": data_types or [],
            "last_check": int(time.time()),
            "connection_details": details or {}
        })

    def has_required_data(self, session_id: str, required_data_types: List[str]) -> bool:
        """Check if session has all required data types connected"""
        status = self.get_connection_status(session_id)
        if status["status"] != ConnectionStatus.CONNECTED.value:
            return False

        connected_types = set(status["connected_data_types"])
        required_types = set(required_data_types)

        return required_types.issubset(connected_types)

    def generate_connection_prompt(self, required_data_types: List[str],
                                 original_query: str) -> Dict[str, Any]:
        """Generate a prompt for database connection"""

        data_type_descriptions = {
            "sales_data": {
                "name": "Sales Data",
                "description": "Transaction history, revenue, product performance",
                "examples": "Sales transactions, order history, product sales data"
            },
            "customer_data": {
                "name": "Customer Data",
                "description": "Customer demographics, purchase history, behavior",
                "examples": "Customer profiles, purchase patterns, customer segments"
            },
            "inventory_data": {
                "name": "Inventory Data",
                "description": "Stock levels, product availability, supply chain",
                "examples": "Product inventory, stock levels, reorder points"
            },
            "marketing_data": {
                "name": "Marketing Data",
                "description": "Campaign performance, advertising metrics, traffic",
                "examples": "Ad campaign results, website traffic, conversion rates"
            },
            "financial_data": {
                "name": "Financial Data",
                "description": "Costs, profit margins, financial performance",
                "examples": "Cost data, profit margins, financial statements"
            },
            "operational_data": {
                "name": "Operational Data",
                "description": "Store performance, operational efficiency",
                "examples": "Store metrics, operational costs, efficiency data"
            }
        }

        required_descriptions = []
        for data_type in required_data_types:
            if data_type in data_type_descriptions:
                desc = data_type_descriptions[data_type]
                required_descriptions.append({
                    "type": data_type,
                    "name": desc["name"],
                    "description": desc["description"],
                    "examples": desc["examples"]
                })

        return {
            "message_type": "database_connection_required",
            "original_query": original_query,
            "required_data": required_descriptions,
            "connection_options": [
                {
                    "type": "database",
                    "name": "Connect Database",
                    "description": "Connect your existing database (MySQL, PostgreSQL, SQLite, etc.)",
                    "setup_steps": [
                        "Provide database connection string",
                        "Verify connection and permissions",
                        "Map your tables to our data types",
                        "Test data access"
                    ]
                },
                {
                    "type": "csv_upload",
                    "name": "Upload CSV Files",
                    "description": "Upload CSV files containing your business data",
                    "setup_steps": [
                        "Prepare CSV files with your data",
                        "Upload files through the interface",
                        "Map columns to our data schema",
                        "Validate data format"
                    ]
                },
                {
                    "type": "api_integration",
                    "name": "API Integration",
                    "description": "Connect through APIs (Shopify, WooCommerce, etc.)",
                    "setup_steps": [
                        "Select your platform",
                        "Provide API credentials",
                        "Configure data sync settings",
                        "Test integration"
                    ]
                }
            ],
            "alternative_help": {
                "message": "Would you like me to provide general business advice for your question instead?",
                "can_provide_general_advice": True
            }
        }

    def generate_connection_instructions(self, connection_type: str) -> Dict[str, Any]:
        """Generate detailed connection instructions for a specific type"""

        instructions = {
            "database": {
                "title": "Database Connection Setup",
                "description": "Connect your existing database to enable data-driven insights",
                "requirements": [
                    "Database connection string (host, port, database name)",
                    "Read-only database credentials",
                    "Network access to your database",
                    "Knowledge of your database schema"
                ],
                "supported_databases": [
                    "PostgreSQL", "MySQL", "SQLite", "SQL Server", "Oracle"
                ],
                "security_notes": [
                    "We only require read-only access",
                    "Connections are encrypted",
                    "No data is stored permanently",
                    "You can revoke access anytime"
                ],
                "example_connection_string": "postgresql://username:password@host:port/database"
            },
            "csv_upload": {
                "title": "CSV Upload Setup",
                "description": "Upload your business data in CSV format for analysis",
                "requirements": [
                    "CSV files with your business data",
                    "Column headers that match our schema",
                    "Clean, formatted data",
                    "Files under 100MB each"
                ],
                "data_format_examples": {
                    "sales_data": "date,product_name,quantity,price,customer_id",
                    "customer_data": "customer_id,name,email,signup_date,total_orders",
                    "inventory_data": "product_id,product_name,current_stock,reorder_level"
                },
                "tips": [
                    "Use consistent date formats (YYYY-MM-DD)",
                    "Ensure no missing required fields",
                    "Use UTF-8 encoding for special characters",
                    "Remove sensitive information like full names/addresses"
                ]
            },
            "api_integration": {
                "title": "API Integration Setup",
                "description": "Connect through your platform's API for real-time data",
                "supported_platforms": [
                    "Shopify", "WooCommerce", "BigCommerce", "Magento",
                    "Stripe", "PayPal", "Square", "Google Analytics"
                ],
                "requirements": [
                    "Admin access to your platform",
                    "API credentials or tokens",
                    "Platform-specific permissions",
                    "Active subscription to your platform"
                ],
                "setup_process": [
                    "Select your platform from the list",
                    "Follow platform-specific authentication",
                    "Configure data sync preferences",
                    "Test connection and data flow"
                ]
            }
        }

        return instructions.get(connection_type, {
            "title": "Connection Setup",
            "description": "Please select a valid connection type",
            "error": "Unknown connection type"
        })

    def simulate_connection_test(self, connection_details: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate testing a database connection (replace with real implementation)"""

        # This would be replaced with actual connection testing logic
        connection_type = connection_details.get("type", "unknown")

        if connection_type == "database":
            return {
                "success": True,
                "message": "Database connection successful",
                "available_tables": ["sales", "customers", "products", "orders"],
                "detected_data_types": ["sales_data", "customer_data", "inventory_data"]
            }
        elif connection_type == "csv_upload":
            return {
                "success": True,
                "message": "CSV files uploaded and validated",
                "processed_files": connection_details.get("files", []),
                "detected_data_types": ["sales_data"]
            }
        elif connection_type == "api_integration":
            return {
                "success": True,
                "message": "API integration configured successfully",
                "platform": connection_details.get("platform", "unknown"),
                "detected_data_types": ["sales_data", "customer_data"]
            }
        else:
            return {
                "success": False,
                "message": "Unknown connection type",
                "error": "Please specify a valid connection type"
            }

    def get_connection_status_message(self, session_id: str) -> str:
        """Get a human-readable connection status message"""
        status = self.get_connection_status(session_id)

        if status["status"] == ConnectionStatus.NOT_CONNECTED.value:
            return "No database connected. Connect your data to get personalized insights."
        elif status["status"] == ConnectionStatus.CONNECTING.value:
            return "Setting up your database connection..."
        elif status["status"] == ConnectionStatus.CONNECTED.value:
            data_types = ", ".join(status["connected_data_types"])
            return f"Database connected. Available data: {data_types}"
        elif status["status"] == ConnectionStatus.FAILED.value:
            return "Database connection failed. Please check your connection details."
        else:
            return "Unknown connection status."