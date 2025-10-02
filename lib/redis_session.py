"""
Redis Session Manager for PeppaSync
Handles session persistence using Redis with automatic expiration
"""
import os
import json
import logging
from typing import Dict, Any, Optional
import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

logger = logging.getLogger(__name__)

class RedisSessionManager:
    """Manages user sessions using Redis for persistence"""

    _instance = None
    _redis_client = None
    _is_connected = False

    # Session configuration
    DEFAULT_SESSION_TTL = 86400  # 24 hours in seconds
    SESSION_KEY_PREFIX = "session:"

    def __new__(cls):
        """Singleton pattern to ensure one Redis connection"""
        if cls._instance is None:
            cls._instance = super(RedisSessionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Redis connection (called once due to singleton)"""
        if not self._is_connected:
            self._connect()

    def _connect(self):
        """Connect to Redis server"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            redis_password = os.getenv('REDIS_PASSWORD', None)

            # Parse Redis URL and create client
            self._redis_client = redis.from_url(
                redis_url,
                password=redis_password,
                decode_responses=True,  # Automatically decode responses to strings
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            self._redis_client.ping()
            self._is_connected = True
            logger.info(f"✅ Redis connected successfully: {redis_url}")

        except (RedisError, RedisConnectionError) as e:
            self._is_connected = False
            logger.warning(f"⚠️  Redis connection failed: {e}")
            logger.warning("Sessions will fall back to in-memory storage")
        except Exception as e:
            self._is_connected = False
            logger.error(f"Unexpected error connecting to Redis: {e}")

    def is_available(self) -> bool:
        """Check if Redis is connected and available"""
        if not self._is_connected or not self._redis_client:
            return False

        try:
            self._redis_client.ping()
            return True
        except (RedisError, RedisConnectionError):
            self._is_connected = False
            return False

    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"{self.SESSION_KEY_PREFIX}{session_id}"

    def set_session(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store session data in Redis with TTL

        Args:
            session_id: Unique session identifier
            session_data: Dictionary containing session information
            ttl: Time to live in seconds (default: 24 hours)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Redis not available for set_session")
            return False

        try:
            key = self._get_session_key(session_id)
            value = json.dumps(session_data)
            ttl = ttl or int(os.getenv('SESSION_TTL', self.DEFAULT_SESSION_TTL))

            # Set with expiration
            self._redis_client.setex(key, ttl, value)
            logger.info(f"Session stored in Redis: {session_id} (TTL: {ttl}s)")
            return True

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error setting session in Redis: {e}")
            return False

    def get_session(self, session_id: str, refresh_ttl: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from Redis

        Args:
            session_id: Unique session identifier
            refresh_ttl: If True, refresh the session TTL on access

        Returns:
            Session data dictionary or None if not found
        """
        if not self.is_available():
            logger.warning("Redis not available for get_session")
            return None

        try:
            key = self._get_session_key(session_id)
            value = self._redis_client.get(key)

            if value is None:
                logger.debug(f"Session not found in Redis: {session_id}")
                return None

            # Parse JSON
            session_data = json.loads(value)

            # Refresh TTL if requested (keeps active sessions alive)
            if refresh_ttl:
                self.refresh_ttl(session_id)

            logger.debug(f"Session retrieved from Redis: {session_id}")
            return session_data

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting session from Redis: {e}")
            return None

    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists in Redis

        Args:
            session_id: Unique session identifier

        Returns:
            True if session exists, False otherwise
        """
        if not self.is_available():
            return False

        try:
            key = self._get_session_key(session_id)
            return self._redis_client.exists(key) > 0

        except RedisError as e:
            logger.error(f"Error checking session existence: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from Redis

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_available():
            logger.warning("Redis not available for delete_session")
            return False

        try:
            key = self._get_session_key(session_id)
            deleted = self._redis_client.delete(key)

            if deleted:
                logger.info(f"Session deleted from Redis: {session_id}")
                return True
            else:
                logger.debug(f"Session not found for deletion: {session_id}")
                return False

        except RedisError as e:
            logger.error(f"Error deleting session from Redis: {e}")
            return False

    def refresh_ttl(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """
        Refresh session TTL to keep it alive

        Args:
            session_id: Unique session identifier
            ttl: New TTL in seconds (default: 24 hours)

        Returns:
            True if refreshed, False otherwise
        """
        if not self.is_available():
            return False

        try:
            key = self._get_session_key(session_id)
            ttl = ttl or int(os.getenv('SESSION_TTL', self.DEFAULT_SESSION_TTL))

            # Set new expiration
            refreshed = self._redis_client.expire(key, ttl)

            if refreshed:
                logger.debug(f"Session TTL refreshed: {session_id} (TTL: {ttl}s)")
                return True
            else:
                logger.warning(f"Failed to refresh TTL, session may not exist: {session_id}")
                return False

        except RedisError as e:
            logger.error(f"Error refreshing session TTL: {e}")
            return False

    def get_all_session_ids(self) -> list:
        """
        Get all active session IDs (for debugging/admin)

        Returns:
            List of session IDs
        """
        if not self.is_available():
            return []

        try:
            pattern = f"{self.SESSION_KEY_PREFIX}*"
            keys = self._redis_client.keys(pattern)

            # Remove prefix to get session IDs
            session_ids = [key.replace(self.SESSION_KEY_PREFIX, '') for key in keys]
            return session_ids

        except RedisError as e:
            logger.error(f"Error getting session IDs: {e}")
            return []

    def clear_all_sessions(self) -> int:
        """
        Clear all sessions (use with caution!)

        Returns:
            Number of sessions deleted
        """
        if not self.is_available():
            return 0

        try:
            pattern = f"{self.SESSION_KEY_PREFIX}*"
            keys = self._redis_client.keys(pattern)

            if keys:
                deleted = self._redis_client.delete(*keys)
                logger.warning(f"Cleared {deleted} sessions from Redis")
                return deleted

            return 0

        except RedisError as e:
            logger.error(f"Error clearing sessions: {e}")
            return 0

    def get_session_ttl(self, session_id: str) -> Optional[int]:
        """
        Get remaining TTL for a session

        Args:
            session_id: Unique session identifier

        Returns:
            Remaining TTL in seconds, or None if session doesn't exist
        """
        if not self.is_available():
            return None

        try:
            key = self._get_session_key(session_id)
            ttl = self._redis_client.ttl(key)

            if ttl == -2:  # Key doesn't exist
                return None
            elif ttl == -1:  # Key exists but has no expiration
                return -1
            else:
                return ttl

        except RedisError as e:
            logger.error(f"Error getting session TTL: {e}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection

        Returns:
            Dictionary with health status
        """
        try:
            if not self._redis_client:
                return {
                    'status': 'disconnected',
                    'available': False,
                    'message': 'Redis client not initialized'
                }

            # Ping Redis
            ping_result = self._redis_client.ping()

            # Get info
            info = self._redis_client.info('server')

            return {
                'status': 'healthy',
                'available': True,
                'ping': ping_result,
                'redis_version': info.get('redis_version'),
                'uptime_seconds': info.get('uptime_in_seconds'),
                'connected_clients': info.get('connected_clients')
            }

        except (RedisError, RedisConnectionError) as e:
            self._is_connected = False
            return {
                'status': 'error',
                'available': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 'error',
                'available': False,
                'error': f'Unexpected error: {str(e)}'
            }

    def close(self):
        """Close Redis connection"""
        if self._redis_client:
            try:
                self._redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self._is_connected = False
                self._redis_client = None


# Create singleton instance
redis_session_manager = RedisSessionManager()
