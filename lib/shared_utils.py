"""
Shared utilities for PeppaSync LangChain Application
Common functions used across multiple modules
"""
import json
import time
import uuid
import re
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StateHandler:
    """Handle LangGraph state objects vs dictionaries across different versions"""
    
    @staticmethod
    def get_value(state: Union[Dict, object], key: str, default: Any = None) -> Any:
        """Get value from state whether it's a dict or object"""
        if isinstance(state, dict):
            return state.get(key, default)
        else:
            return getattr(state, key, default)
    
    @staticmethod
    def set_value(state: Union[Dict, object], key: str, value: Any) -> None:
        """Set value in state whether it's a dict or object"""
        if isinstance(state, dict):
            state[key] = value
        else:
            setattr(state, key, value)
    
    @staticmethod
    def to_dict(state: Union[Dict, object]) -> Dict[str, Any]:
        """Convert state to dictionary format"""
        if isinstance(state, dict):
            return state
        elif hasattr(state, '__dict__'):
            return state.__dict__
        else:
            return {}

class ResponseFormatter:
    """Format API responses consistently"""
    
    @staticmethod
    def format_error(error_message: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Format error responses consistently"""
        response = {
            'error': str(error_message),
            'timestamp': int(time.time()),
            'status': 'error'
        }
        if session_id:
            response['sessionId'] = session_id
        response.update(kwargs)
        return response
    
    @staticmethod
    def format_success(data: Dict[str, Any], session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Format success responses consistently"""
        response = {
            'timestamp': int(time.time()),
            'status': 'success'
        }
        if session_id:
            response['sessionId'] = session_id
        response.update(data)
        response.update(kwargs)
        return response
    
    @staticmethod
    def format_agent_response(status: str, data: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
        """Format agent execution responses"""
        response = {
            'status': status,
            'timestamp': int(time.time())
        }
        response.update(data)
        if error:
            response['error'] = error
        return response

class TextProcessor:
    """Process and analyze text content"""
    
    @staticmethod
    def extract_numerical_values(text: str) -> Dict[str, List[str]]:
        """Extract numerical values and their contexts from text"""
        patterns = [
            (r'(\d+)%', 'percentage'),
            (r'\$(\d+(?:,\d{3})*)', 'currency_usd'),
            (r'₦(\d+(?:,\d{3})*)', 'currency_ngn'),
            (r'(\d+) days?', 'days'),
            (r'(\d+) weeks?', 'weeks'),
            (r'(\d+) months?', 'months'),
            (r'Q(\d)', 'quarter'),
            (r'(\d+) members?', 'people'),
            (r'(\d+)x|(\d+) times', 'multiplier')
        ]
        
        extracted = {}
        for pattern, value_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[value_type] = [match if isinstance(match, str) else match[0] for match in matches]
        
        return extracted
    
    @staticmethod
    def extract_time_horizon(text: str) -> str:
        """Extract time horizon from text"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['quarter', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
            return 'quarterly'
        elif any(term in text_lower for term in ['month', 'monthly']):
            return 'monthly'
        elif any(term in text_lower for term in ['week', 'weekly']):
            return 'weekly'
        elif any(term in text_lower for term in ['day', 'daily']):
            return 'daily'
        elif any(term in text_lower for term in ['year', 'yearly', 'annual']):
            return 'yearly'
        else:
            return 'unspecified'
    
    @staticmethod
    def extract_metrics(text: str) -> List[str]:
        """Extract key business metrics from text"""
        metric_keywords = [
            'sales', 'revenue', 'profit', 'margin', 'roas', 'roi', 'aov', 'clv',
            'conversion', 'traffic', 'engagement', 'retention', 'churn',
            'inventory', 'stock', 'turnover', 'cost', 'price', 'discount'
        ]
        
        found_metrics = []
        text_lower = text.lower()
        
        for metric in metric_keywords:
            if metric in text_lower:
                found_metrics.append(metric)
        
        # Look for compound metrics
        compound_metrics = [
            ('average order value', 'AOV'),
            ('customer lifetime value', 'CLV'),
            ('customer acquisition cost', 'CAC'),
            ('return on ad spend', 'ROAS'),
            ('return on investment', 'ROI')
        ]
        
        for phrase, abbrev in compound_metrics:
            if phrase in text_lower or abbrev.lower() in text_lower:
                found_metrics.append(abbrev)
        
        return list(set(found_metrics))

class BusinessContextBuilder:
    """Build business context for analysis"""
    
    @staticmethod
    def build_nigerian_context(extracted_values: Dict[str, Any]) -> Dict[str, Any]:
        """Build Nigerian market context"""
        from .config import AppConfig
        
        context = AppConfig.NIGERIAN_MARKET_CONTEXT.copy()
        context['extracted_values'] = extracted_values
        context['market'] = 'Nigerian retail'
        
        return context
    
    @staticmethod
    def determine_analysis_type(text: str, category: str) -> str:
        """Determine the type of analysis required"""
        text_lower = text.lower()
        
        # Predictive indicators
        if any(term in text_lower for term in [
            'forecast', 'predict', 'project', 'expect', 'will', 'future',
            'next quarter', 'next month', 'next year', 'upcoming'
        ]):
            return 'predictive'
        
        # Scenario indicators
        if any(term in text_lower for term in [
            'what if', 'if we', 'scenario', 'suppose', 'assume', 'would happen'
        ]):
            return 'scenario'
        
        # Diagnostic indicators
        if any(term in text_lower for term in [
            'why', 'what caused', 'reason', 'analyze', 'explain', 'understand'
        ]):
            return 'diagnostic'
        
        # Prescriptive indicators
        if any(term in text_lower for term in [
            'recommend', 'suggest', 'should', 'optimize', 'improve', 'strategy'
        ]):
            return 'prescriptive'
        
        # Comparative indicators
        if any(term in text_lower for term in [
            'compare', 'versus', 'vs', 'difference', 'better than', 'worse than'
        ]):
            return 'comparative'
        
        # Default to descriptive
        return 'descriptive'

class ValidationUtils:
    """Validation utilities"""
    
    @staticmethod
    def validate_session_id(session_id: Optional[str]) -> str:
        """Validate and generate session ID if needed"""
        if not session_id:
            return str(uuid.uuid4())
        
        # Basic validation
        if len(session_id) > 100:
            return str(uuid.uuid4())
        
        return session_id
    
    @staticmethod
    def validate_api_response(response: Dict[str, Any]) -> bool:
        """Validate API response structure"""
        required_fields = ['timestamp', 'status']
        return all(field in response for field in required_fields)

class LoggingUtils:
    """Logging utilities"""
    
    @staticmethod
    def log_performance(operation: str, start_time: float, **kwargs):
        """Log performance metrics"""
        duration = time.time() - start_time
        logger.info(f"Performance: {operation} completed in {duration:.2f}s", extra=kwargs)
    
    @staticmethod
    def log_agent_action(agent_type: str, action: str, details: Dict[str, Any]):
        """Log agent actions consistently"""
        logger.info(f"Agent Action: {agent_type} - {action}", extra=details)
    
    @staticmethod
    def log_classification(prompt: str, category: str, confidence: float):
        """Log prompt classification results"""
        logger.info(f"Classified '{prompt[:50]}...' as {category} (confidence: {confidence:.2f})")

def parse_api_response(response: str, fallback: str = "No response generated") -> str:
    """Parse API response - moved from utils.py for backward compatibility"""
    if not response:
        return fallback
    
    # Handle JSON responses
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and 'content' in parsed:
            return parsed['content']
        elif isinstance(parsed, dict) and 'message' in parsed:
            return parsed['message']
    except json.JSONDecodeError:
        pass
    
    return response.strip() if response.strip() else fallback