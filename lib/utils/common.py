"""
Common utility functions shared across the PeppaSync application
"""
import time
import uuid
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ValidationUtils:
    """Validation utilities"""
    
    @staticmethod
    def validate_session_id(session_id: Optional[str]) -> str:
        """Validate and return a proper session ID"""
        if not session_id:
            return str(uuid.uuid4())
        
        # Basic validation - ensure it's a valid UUID format or reasonable string
        if len(session_id) < 6:
            logger.warning(f"Session ID too short: {session_id}, generating new one")
            return str(uuid.uuid4())
        
        return session_id
    
    @staticmethod
    def validate_prompt(prompt: str) -> bool:
        """Validate that prompt is not empty and reasonable length"""
        if not prompt or not prompt.strip():
            return False
        if len(prompt) > 10000:  # Reasonable max length
            return False
        return True
    
    @staticmethod
    def sanitize_string(text: str) -> str:
        """Basic string sanitization"""
        if not text:
            return ""
        return text.strip()[:1000]  # Trim to reasonable length

class StateHandler:
    """Handle LangGraph state conversions"""
    
    @staticmethod
    def to_dict(result: Any) -> Dict[str, Any]:
        """Convert LangGraph result to dictionary"""
        if isinstance(result, dict):
            return result
        elif hasattr(result, '__dict__'):
            return result.__dict__
        else:
            logger.warning(f"Unexpected result type: {type(result)}")
            return {}

class ResponseFormatter:
    """Format API responses consistently"""
    
    @staticmethod
    def format_success(data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Format successful response"""
        response = {
            'sessionId': session_id,
            'timestamp': int(time.time())
        }
        response.update(data)
        return response
    
    @staticmethod
    def format_error(message: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """Format error response"""
        return {
            'output': message,
            'sessionId': session_id,
            'error': True,
            'timestamp': int(time.time()),
            **kwargs
        }
    
    @staticmethod
    def format_analysis_response(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format prompt analysis response"""
        return {
            'prompt_analysis': analysis,
            'timestamp': int(time.time()),
            'status': 'success'
        }

class LoggingUtils:
    """Logging utilities"""
    
    @staticmethod
    def log_performance(operation: str, start_time: float, **context):
        """Log operation performance"""
        duration = time.time() - start_time
        logger.info(f"Operation {operation} completed in {duration:.2f}s", extra=context)
    
    @staticmethod
    def log_error(operation: str, error: Exception, **context):
        """Log error with context"""
        logger.error(f"Error in {operation}: {str(error)}", extra=context, exc_info=True)
    
    @staticmethod
    def log_business_event(event_type: str, data: Dict[str, Any]):
        """Log business events for analytics"""
        logger.info(f"BUSINESS_EVENT_{event_type}", extra={'event_data': data})

class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def safe_json_loads(json_str: str, default: Any = None) -> Any:
        """Safely parse JSON string"""
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return default
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract numerical values from text"""
        import re
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    @staticmethod
    def format_currency(amount: float, currency: str = 'NGN') -> str:
        """Format currency amounts"""
        if currency == 'NGN':
            return f"₦{amount:,.2f}"
        elif currency == 'USD':
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100

class TimeUtils:
    """Time and date utilities"""
    
    @staticmethod
    def get_current_timestamp() -> int:
        """Get current UTC timestamp"""
        return int(time.time())
    
    @staticmethod
    def format_timestamp(timestamp: int) -> str:
        """Format timestamp to readable string"""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    @staticmethod
    def get_quarter_from_timestamp(timestamp: int) -> str:
        """Get quarter string from timestamp"""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        quarter = (dt.month - 1) // 3 + 1
        return f"Q{quarter} {dt.year}"

class BusinessMetrics:
    """Business calculation utilities"""
    
    @staticmethod
    def calculate_roas(revenue: float, spend: float) -> float:
        """Calculate Return on Ad Spend"""
        if spend == 0:
            return 0.0
        return revenue / spend
    
    @staticmethod
    def calculate_margin(revenue: float, cost: float) -> float:
        """Calculate profit margin percentage"""
        if revenue == 0:
            return 0.0
        return ((revenue - cost) / revenue) * 100
    
    @staticmethod
    def calculate_growth_rate(current: float, previous: float) -> float:
        """Calculate growth rate percentage"""
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def apply_seasonal_factor(base_value: float, quarter: str) -> float:
        """Apply seasonal adjustments to values"""
        from ..config import AppConfig
        factors = AppConfig.NIGERIAN_MARKET_CONTEXT['seasonal_factors']
        factor = factors.get(quarter.upper(), 1.0)
        return base_value * factor

class CacheUtils:
    """Simple in-memory caching utilities"""
    
    _cache = {}
    
    @classmethod
    def get(cls, key: str) -> Any:
        """Get value from cache"""
        if key in cls._cache:
            data, timestamp = cls._cache[key]
            # Simple 5-minute TTL
            if time.time() - timestamp < 300:
                return data
            else:
                del cls._cache[key]
        return None
    
    @classmethod
    def set(cls, key: str, value: Any):
        """Set value in cache"""
        cls._cache[key] = (value, time.time())
    
    @classmethod
    def clear(cls):
        """Clear all cache"""
        cls._cache.clear()