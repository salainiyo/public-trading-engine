"""Event bus and WebSocket broadcaster for decoupled event handling."""
import asyncio
import logging
from typing import Callable, Dict, Set

logger = logging.getLogger(__name__)


class EventBus:
    """Central event bus for dispatching and listening to events."""

    def __init__(self):
        self._handlers: Dict[str, Set[Callable]] = {}
        self._subscribers: Dict[str, Set[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = set()
        self._handlers[event_type].add(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type].discard(handler)

    def publish(self, event_type: str, data: Dict) -> None:
        """Publish an event to all subscribers."""
        if event_type not in self._handlers:
            return
        for handler in self._handlers[event_type]:
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")


class WebSocketBroadcaster:
    """Handles WebSocket communication with the dashboard."""

    def __init__(self, ws_url: str = "ws://localhost:8080"):
        self.ws_url = ws_url
        self._connected = False
        self._broadcast_queue = asyncio.Queue()

    async def start(self) -> None:
        """Start the WebSocket server."""
        # In a real implementation, this would start an async WebSocket server
        # For now, we just track the state
        self._connected = True
        logger.info(f"WebSocket broadcaster started on {self.ws_url}")

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients."""
        if not self._connected:
            return
        # In a real implementation, this would send the message via WebSocket
        logger.debug(f"Broadcasting: {message}")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        logger.info("WebSocket broadcaster stopped")
