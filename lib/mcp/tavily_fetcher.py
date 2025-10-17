"""
Tavily MCP Integration - Web search for economic event enrichment
Part of the SYSTEM component in the demand forecast architecture
"""

import os
import logging
from typing import Dict, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TavilyFetcher:
    """
    Handles web search via Tavily MCP to enrich economic events.

    When user adds an economic event like "Black Friday 2024", this class:
    1. Searches the web for details about the event
    2. Extracts relevant information (date, impact, context)
    3. Returns enriched event data for Prophet forecasting

    This is part of the SYSTEM layer that fetches external context.
    """

    def __init__(self):
        self.api_key = os.getenv('TAVILY_API_KEY')
        self.cache = {}  # Simple in-memory cache to avoid redundant searches

    def is_available(self) -> bool:
        """Check if Tavily API is configured and available"""
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set in environment variables")
            return False
        return True

    def enrich_event(self, event: Dict) -> Optional[Dict]:
        """
        Enrich an economic event with web search data.

        Args:
            event: Economic event dict with at least 'name' and 'date'
                {
                    "name": "Black Friday",
                    "date": "2024-11-29"
                }

        Returns:
            Enriched event with additional context:
                {
                    "name": "Black Friday",
                    "date": "2024-11-29",
                    "description": "Major retail shopping event...",
                    "impact_days_before": 7,
                    "impact_days_after": 3,
                    "source": "tavily",
                    "enriched_at": "2024-10-10T..."
                }
        """
        try:
            if not self.is_available():
                logger.warning("Tavily not available, skipping enrichment")
                return event

            event_name = event.get('name')
            event_date = event.get('date')

            if not event_name:
                return event

            # Check cache first
            cache_key = f"{event_name}_{event_date}"
            if cache_key in self.cache:
                logger.debug(f"Using cached data for event: {event_name}")
                return self.cache[cache_key]

            # Search for event information
            search_query = f"{event_name} {event_date} business impact retail sales"
            search_results = self._search_tavily(search_query)

            if not search_results:
                logger.warning(f"No search results for event: {event_name}")
                return event

            # Extract and enrich event data
            enriched_event = event.copy()
            enriched_event.update({
                "description": self._extract_description(search_results, event_name),
                "impact_days_before": self._estimate_impact_before(event_name, search_results),
                "impact_days_after": self._estimate_impact_after(event_name, search_results),
                "source": "tavily",
                "enriched_at": datetime.now().isoformat(),
                "search_summary": self._summarize_results(search_results)
            })

            # Cache the result
            self.cache[cache_key] = enriched_event

            logger.info(f"Successfully enriched event: {event_name}")
            return enriched_event

        except Exception as e:
            logger.error(f"Error enriching event: {str(e)}")
            return event

    def _search_tavily(self, query: str, max_results: int = 5) -> Optional[List[Dict]]:
        """
        Perform web search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results or None if search fails
        """
        try:
            # Try to import tavily client
            try:
                from tavily import TavilyClient
            except ImportError:
                logger.warning("tavily-python package not installed")
                return None

            # Initialize client
            client = TavilyClient(api_key=self.api_key)

            # Perform search
            response = client.search(
                query=query,
                search_depth="advanced",  # or "advanced" for more detailed results
                max_results=max_results
            )

            results = response.get('results', [])
            logger.debug(f"Tavily search returned {len(results)} results for: {query}")

            return results

        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return None

    def _extract_description(self, search_results: List[Dict], event_name: str) -> str:
        """
        Extract a concise description from search results.

        Args:
            search_results: List of search results
            event_name: Name of the event

        Returns:
            Description string
        """
        if not search_results:
            return f"Economic event: {event_name}"

        # Get the first few sentences from top results
        descriptions = []
        for result in search_results[:3]:
            content = result.get('content', '')
            # Take first 200 characters
            if content:
                descriptions.append(content[:200])

        if descriptions:
            return " ".join(descriptions)

        return f"Economic event: {event_name}"

    def _estimate_impact_before(self, event_name: str, search_results: List[Dict]) -> int:
        """
        Estimate how many days before the event businesses should prepare.

        Uses heuristics based on event type and search results.

        Args:
            event_name: Name of the event
            search_results: Search results

        Returns:
            Number of days before event to consider impact
        """
        # Default heuristics based on event type
        event_lower = event_name.lower()

        if any(keyword in event_lower for keyword in ['black friday', 'cyber monday']):
            return 14  # 2 weeks before

        if any(keyword in event_lower for keyword in ['christmas', 'holiday']):
            return 30  # 1 month before

        if any(keyword in event_lower for keyword in ['prime day', 'sale']):
            return 7  # 1 week before

        if any(keyword in event_lower for keyword in ['new year', 'easter']):
            return 14  # 2 weeks before

        # Look for keywords in search results
        content = " ".join([r.get('content', '') for r in search_results]).lower()

        if 'major' in content or 'significant' in content:
            return 14

        if 'prepare' in content or 'planning' in content:
            return 10

        # Default
        return 7

    def _estimate_impact_after(self, event_name: str, search_results: List[Dict]) -> int:
        """
        Estimate how many days after the event the impact continues.

        Args:
            event_name: Name of the event
            search_results: Search results

        Returns:
            Number of days after event to consider impact
        """
        event_lower = event_name.lower()

        if any(keyword in event_lower for keyword in ['black friday', 'cyber monday']):
            return 7  # Sales continue for a week

        if any(keyword in event_lower for keyword in ['christmas', 'holiday']):
            return 14  # Returns and exchanges period

        if any(keyword in event_lower for keyword in ['prime day', 'sale']):
            return 3  # Short-term impact

        # Look for keywords in search results
        content = " ".join([r.get('content', '') for r in search_results]).lower()

        if 'extended' in content or 'continues' in content:
            return 7

        # Default
        return 3

    def _summarize_results(self, search_results: List[Dict]) -> str:
        """
        Create a brief summary of search results.

        Args:
            search_results: List of search results

        Returns:
            Summary string
        """
        if not search_results:
            return "No additional context available"

        # Extract titles and key points
        summaries = []
        for result in search_results[:3]:
            title = result.get('title', '')
            if title:
                summaries.append(title)

        return " | ".join(summaries) if summaries else "Economic event information"

    def batch_enrich_events(self, events: List[Dict]) -> List[Dict]:
        """
        Enrich multiple events in batch.

        Args:
            events: List of economic events

        Returns:
            List of enriched events
        """
        enriched_events = []

        for event in events:
            enriched = self.enrich_event(event)
            enriched_events.append(enriched if enriched else event)

        logger.info(f"Batch enriched {len(enriched_events)} events")
        return enriched_events

    def clear_cache(self):
        """Clear the event cache"""
        self.cache = {}
        logger.info("Tavily cache cleared")
