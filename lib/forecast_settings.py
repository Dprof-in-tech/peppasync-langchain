"""
Forecast Settings Manager - Single endpoint for all forecast configuration
Part of the SETTINGS component in the demand forecast architecture
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ForecastSettingsManager:
    """
    Manages forecast settings for users.

    Stores:
    - Economic events that might affect business
    - Supply chain timeline (manufacturing + logistics days)
    - Forecast preferences

    All settings are stored per session and persist using Redis/in-memory fallback.
    """

    _settings_store: Dict[str, Dict[str, Any]] = {}  # In-memory fallback
    _use_redis = False
    _redis_manager = None

    @classmethod
    def _get_redis_manager(cls):
        """Get or initialize Redis session manager"""
        if cls._redis_manager is None:
            try:
                from lib.redis_session import redis_session_manager
                cls._redis_manager = redis_session_manager
                cls._use_redis = cls._redis_manager.is_available()
                if cls._use_redis:
                    logger.info("Using Redis for forecast settings storage")
            except Exception as e:
                logger.warning(f"Redis not available, using in-memory storage: {e}")
                cls._use_redis = False
        return cls._redis_manager

    @classmethod
    def save_settings(cls, session_id: str, settings_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save all forecast settings in one call (single endpoint design).

        Args:
            session_id: User session identifier
            settings_data: Dictionary containing all settings:
                {
                    "economic_events": [
                        {
                            "name": "Chinese New Year",
                            "date": "2024-02-10",
                            "description": "Spike in sales during celebration period"
                            # impact_days auto-calculated by Tavily
                        }
                    ],
                    "supply_chain_locations": [
                        {
                            "name": "China Factory",
                            "manufacturing_days": 30,
                            "logistics_days": 15,
                            "possible_delay_days": 5,
                            "notes": "Affected by Chinese New Year"
                        },
                        {
                            "name": "Vietnam Warehouse",
                            "manufacturing_days": 20,
                            "logistics_days": 10,
                            "possible_delay_days": 3
                        }
                    ]
                    # forecast_horizon and frequency come from user prompt
                }

        Returns:
            Dict with success status and saved settings
        """
        try:
            # Validate settings
            validation_result = cls._validate_settings(settings_data)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "message": f"Invalid settings: {validation_result['errors']}"
                }

            # Enrich economic events with Tavily data (if available)
            if settings_data.get("economic_events"):
                settings_data["economic_events"] = cls._enrich_economic_events(
                    settings_data["economic_events"]
                )

            # Add metadata
            settings_data["updated_at"] = datetime.now().isoformat()
            settings_data["session_id"] = session_id

            # Store in Redis or in-memory
            redis_manager = cls._get_redis_manager()
            if redis_manager and cls._use_redis:
                success = redis_manager.set_session(
                    f"forecast_settings_{session_id}",
                    settings_data,
                    ttl=2592000  # 30 days
                )
                if success:
                    logger.info(f"Forecast settings saved to Redis for session {session_id}")
                else:
                    logger.warning("Failed to save to Redis, falling back to memory")
                    cls._settings_store[session_id] = settings_data
            else:
                cls._settings_store[session_id] = settings_data
                logger.info(f"Forecast settings saved to memory for session {session_id}")

            return {
                "success": True,
                "message": "Settings saved successfully",
                "settings": settings_data
            }

        except Exception as e:
            logger.error(f"Error saving forecast settings: {str(e)}")
            return {
                "success": False,
                "message": f"Error saving settings: {str(e)}"
            }

    @classmethod
    def get_settings(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve forecast settings for a session.

        Args:
            session_id: User session identifier

        Returns:
            Settings dictionary or None if not found
        """
        try:
            # Try Redis first
            redis_manager = cls._get_redis_manager()
            if redis_manager and cls._use_redis:
                settings = redis_manager.get_session(
                    f"forecast_settings_{session_id}",
                    refresh_ttl=True
                )
                if settings:
                    logger.debug(f"Settings retrieved from Redis for session {session_id}")
                    return settings

            # Fallback to in-memory
            settings = cls._settings_store.get(session_id)
            if settings:
                logger.debug(f"Settings retrieved from memory for session {session_id}")
            return settings

        except Exception as e:
            logger.error(f"Error retrieving settings: {str(e)}")
            return None

    @classmethod
    def has_settings(cls, session_id: str) -> bool:
        """Check if session has saved settings"""
        return cls.get_settings(session_id) is not None

    @classmethod
    def delete_settings(cls, session_id: str):
        """Delete forecast settings for a session"""
        try:
            # Try Redis first
            redis_manager = cls._get_redis_manager()
            if redis_manager and cls._use_redis:
                redis_manager.delete_session(f"forecast_settings_{session_id}")

            # Also delete from in-memory
            if session_id in cls._settings_store:
                del cls._settings_store[session_id]

            logger.info(f"Settings deleted for session {session_id}")

        except Exception as e:
            logger.error(f"Error deleting settings: {str(e)}")

    @classmethod
    def _validate_settings(cls, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate settings data structure and values.

        Returns:
            Dict with validation result: {"valid": bool, "errors": list}
        """
        errors = []

        # Validate economic events
        if "economic_events" in settings:
            if not isinstance(settings["economic_events"], list):
                errors.append("economic_events must be a list")
            else:
                for idx, event in enumerate(settings["economic_events"]):
                    if not isinstance(event, dict):
                        errors.append(f"Event {idx} must be a dictionary")
                        continue

                    # Validate required fields
                    if "name" not in event:
                        errors.append(f"Event {idx} missing 'name' field")
                    if "date" not in event:
                        errors.append(f"Event {idx} missing 'date' field")
                    else:
                        # Validate date format
                        try:
                            datetime.fromisoformat(str(event["date"]))
                        except ValueError:
                            errors.append(f"Event {idx} has invalid date format (use YYYY-MM-DD)")

        # Validate supply chain locations (NEW: array structure)
        if "supply_chain_locations" in settings:
            if not isinstance(settings["supply_chain_locations"], list):
                errors.append("supply_chain_locations must be a list")
            else:
                for idx, location in enumerate(settings["supply_chain_locations"]):
                    if not isinstance(location, dict):
                        errors.append(f"Location {idx} must be a dictionary")
                        continue

                    # Validate required fields
                    if "name" not in location:
                        errors.append(f"Location {idx} missing 'name' field")

                    # Validate manufacturing days
                    if "manufacturing_days" in location:
                        if not isinstance(location["manufacturing_days"], (int, float)):
                            errors.append(f"Location {idx} manufacturing_days must be a number")
                        elif location["manufacturing_days"] < 0:
                            errors.append(f"Location {idx} manufacturing_days cannot be negative")

                    # Validate logistics days
                    if "logistics_days" in location:
                        if not isinstance(location["logistics_days"], (int, float)):
                            errors.append(f"Location {idx} logistics_days must be a number")
                        elif location["logistics_days"] < 0:
                            errors.append(f"Location {idx} logistics_days cannot be negative")

                    # Validate possible delay days
                    if "possible_delay_days" in location:
                        if not isinstance(location["possible_delay_days"], (int, float)):
                            errors.append(f"Location {idx} possible_delay_days must be a number")
                        elif location["possible_delay_days"] < 0:
                            errors.append(f"Location {idx} possible_delay_days cannot be negative")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    @classmethod
    def _enrich_economic_events(cls, events: List[Dict]) -> List[Dict]:
        """
        Enrich economic events with additional context from Tavily MCP.

        This will be called when user adds events - Tavily will fetch
        details about the event from web search.

        Args:
            events: List of economic events

        Returns:
            List of enriched events with additional context
        """
        try:
            from lib.mcp.tavily_fetcher import TavilyFetcher

            fetcher = TavilyFetcher()
            enriched_events = []

            for event in events:
                # Try to enrich the event
                enriched = fetcher.enrich_event(event)
                enriched_events.append(enriched if enriched else event)

            logger.info(f"Enriched {len(enriched_events)} economic events with Tavily data")
            return enriched_events

        except Exception as e:
            logger.warning(f"Could not enrich events with Tavily: {str(e)}")
            # Return original events if enrichment fails
            return events

    @classmethod
    def get_economic_events(cls, session_id: str) -> List[Dict]:
        """Get only economic events for a session"""
        settings = cls.get_settings(session_id)
        if settings:
            return settings.get("economic_events", [])
        return []

    @classmethod
    def get_supply_chain_locations(cls, session_id: str) -> List[Dict]:
        """
        Get supply chain locations for a session.

        Returns:
            List of supply chain locations with manufacturing/logistics timelines
        """
        settings = cls.get_settings(session_id)
        if settings:
            return settings.get("supply_chain_locations", [])
        return []

    @classmethod
    def calculate_max_supply_chain_days(cls, session_id: str) -> int:
        """
        Calculate maximum supply chain days across all locations.

        Used for determining when to place orders for peak demand periods.

        Returns:
            Maximum total days (manufacturing + logistics + possible delays)
        """
        locations = cls.get_supply_chain_locations(session_id)
        if not locations:
            return 45  # Default fallback

        max_days = 0
        for location in locations:
            total = (
                location.get("manufacturing_days", 0) +
                location.get("logistics_days", 0) +
                location.get("possible_delay_days", 0)
            )
            max_days = max(max_days, total)

        return max_days

    @classmethod
    def get_default_settings(cls) -> Dict[str, Any]:
        """
        Get default settings for new users.

        Returns:
            Dict with default forecast settings
        """
        return {
            "economic_events": [
                {
                    "name": "Black Friday",
                    "date": "2024-11-29",
                    "description": "Major retail shopping event with increased sales volume"
                    # impact_days will be auto-calculated by Tavily
                },
                {
                    "name": "Christmas",
                    "date": "2024-12-25",
                    "description": "Holiday season peak demand period"
                    # impact_days will be auto-calculated by Tavily
                }
            ],
            "supply_chain_locations": [
                {
                    "name": "Primary Factory",
                    "manufacturing_days": 30,
                    "logistics_days": 15,
                    "possible_delay_days": 5,
                    "notes": "Standard manufacturing and shipping timeline"
                }
            ]
            # forecast_horizon and frequency extracted from user prompt
        }
