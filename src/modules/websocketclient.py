"""
Asynchronous websocket client for Binance

### Todo:
- [ ] Make it more generic to support multiple websocker providers
- [ ] Maybe separate subscription management from the websocket client

### Example usage
```python
async def main():
    # Create client instance for Binance WebSocket
    client = WebSocketClient("wss://stream.binance.com:9443/ws")

    # Define event handlers
    def on_message_received(data: Dict[str, Any]):  # MODIFIED: Now receives dict
        print(f"Stream data received: {data}")

        # Handle different event types
        if data.get("e") == "aggTrade":
            print(
                f"Aggregate trade: {data['s']} - Price: {data['p']}, Qty: {data['q']}"
            )
        elif data.get("e") == "24hrTicker":
            print(f"24hr ticker: {data['s']} - Price change: {data['P']}%")

    def on_response_received(
        data: Dict[str, Any],
    ):  # NEW: Handle subscription responses
        print(f"Subscription response: {data}")

    def on_connected():
        print("Connected to Binance WebSocket!")

    def on_disconnected():
        print("Disconnected from Binance WebSocket!")

    def on_error_occurred(error: Exception):
        print(f"Error occurred: {error}")

    # Set event handlers
    client.set_message_handler(on_message_received)
    client.set_response_handler(on_response_received)  # NEW: Set response handler
    client.set_connect_handler(on_connected)
    client.set_disconnect_handler(on_disconnected)
    client.set_error_handler(on_error_occurred)

    # Start the client in a separate task
    client_task = asyncio.create_task(client.run_with_reconnect())

    # Wait a bit for connection
    await asyncio.sleep(2)

    # NEW: Send subscription requests using the new API
    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@aggTrade", "ethusdt@aggTrade"], id=1
    )
    print(f"Subscription result: {result.info}")

    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@ticker"], id=2
    )
    print(f"Subscription result: {result.info}")

    # Wait and check subscription statuses
    await asyncio.sleep(3)
    print(f"Subscription 1 status: {client.get_subscription_status(1)}")
    print(f"Subscription 2 status: {client.get_subscription_status(2)}")

    # Get all active subscriptions
    active_subs = client.get_subscriptions_by_status(SubscriptionStatus.ACTIVE)
    print(f"Active subscriptions: {len(active_subs)}")

    # Let it run for a while to see data
    await asyncio.sleep(10)

    # Gracefully shutdown
    await client.disconnect()
    client_task.cancel()

    try:
        await client_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
```

"""

import asyncio
import json
import ssl
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from modules.logger import get_logger


# NEW: Enums and dataclasses for subscription management
@dataclass
class SendSubscriptionResult:
    """Subscription request result

    Returned by WebSocketClient.send_subscription_request
    """

    result: bool
    """Result"""
    info: str
    """Message returned by the server on subscription request"""


class SubscriptionStatus(Enum):
    """State of the subscription

    Used in Subscription.status
    """

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    DISCONNECTED = "DISCONNECTED"  # NEW: For connection drops


class RequestType(Enum):
    """Binance Requests types"""

    SUBSCRIBE = "SUBSCRIBE"
    UNSUBSCRIBE = "UNSUBSCRIBE"
    LIST = "LIST_SUBSCRIPTIONS"
    SET_PROPERTY = "SET_PROPERTY"
    GET_PROPERTY = "GET_PROPERTY"


@dataclass
class Subscription:
    """Subscription data type

    Adapted to Binance websocket subscription management
    """

    id: int
    subscriptions: List[str]  # NEW: List to handle batch requests
    status: SubscriptionStatus


class WebSocketClient:
    """
    A robust WebSocket client with reconnection, heartbeat, and message handling.
    Enhanced with Binance-style subscription management.

    Features:
    - Manage reconnection
    - Heartbeat
    - Event Handlers:
        - on connection
        - on disconnection
        - on error
        - on response (management message from the server)
        - on message (data messages from the server)
    - Send messages to webserver
        - str messages
        - JSON messages
    - Subscription Management
    """

    def __init__(self, uri: str):
        self.uri = uri
        """URI of the server"""
        self.websocket: Optional[websockets.ClientConnection] = None
        self.running = False
        """State of the connection"""
        self.reconnect_interval = 5  # seconds
        """Reconnection period in seconds"""
        self.heartbeat_interval = 30  # seconds
        """Period of heartbeats in seconds"""
        self.max_reconnect_attempts = 10
        """Max reconnection attempts number"""
        self.reconnect_attempts = 0
        """Number of reconnections"""

        # Event handlers
        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = (
            None  # MODIFIED: Now receives dict
        )
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_response: Optional[Callable[[Dict[str, Any]], None]] = (
            None  # NEW: For subscription responses
        )

        # NEW: Subscription management
        self.subscriptions: Dict[int, Subscription] = {}

        # Setup logging
        self.logger = get_logger(__name__)

    async def connect(
        self,
        ssl_context: Optional[ssl.SSLContext] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Connect to the WebSocket server.

        # Args:
        - ssl_context: SSL context for secure connections
        - extra_headers: Additional headers to send with the connection

        # Returns:
        - bool: True if connection successful, False otherwise

        # Handlers:
        - WebSocketClient.on_connect
        - WebSocketClient.on_error
        """
        try:
            self.logger.info(f"Connecting to {self.uri}")

            # Create SSL context for wss:// URLs if not provided
            if self.uri.startswith("wss://") and ssl_context is None:
                ssl_context = ssl.create_default_context()

            connect_kwargs = {
                "ssl": ssl_context,
                "ping_interval": 20,
                "ping_timeout": 10,
            }
            # Add extra_headers only if supported (websockets >= 8.0)
            if extra_headers:
                try:
                    self.websocket = await websockets.connect(
                        self.uri, extra_headers=extra_headers, **connect_kwargs
                    )
                except TypeError:
                    # Fallback for older versions without extra_headers support
                    self.logger.warning(
                        "extra_headers not supported in this websockets version, connecting without them"
                    )
                    self.websocket = await websockets.connect(
                        self.uri, **connect_kwargs
                    )
            else:
                self.websocket = await websockets.connect(self.uri, **connect_kwargs)

            # Set reconnection attempts to 0
            self.reconnect_attempts = 0
            self.logger.info("WebSocket connection established")

            # Call on_connect handler if defined
            if self.on_connect:
                self.on_connect()

            return True

        # Connection error
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            # Call on_error handler if defined
            if self.on_error:
                self.on_error(e)
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server.

        # Handlers
        - WebSocketClient.on_disconnect
        """
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.logger.info("WebSocket connection closed")

            if self.on_disconnect:
                self.on_disconnect()

    async def send_message(self, message: str) -> bool:
        """Send a string message to the WebSocket server.

        # Args
        - message: string message to send

        # Returns
        - bool: True if message sent successfully, False otherwise
        """
        if not self.websocket:
            self.logger.error("Cannot send message: not connected")
            return False

        try:
            await self.websocket.send(message)
            self.logger.debug(f"Sent message: {message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            if self.on_error:
                self.on_error(e)
            return False

    async def send_json(self, data: Dict[str, Any]) -> bool:
        """
        Send JSON data to the WebSocket server.

        Args:
            data: Dictionary to send as JSON

        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            message = json.dumps(data)
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"Failed to encode JSON: {e}")
            if self.on_error:
                self.on_error(e)
            return False

    # NEW: Subscription management methods
    async def send_subscription_request(
        self, request_type: RequestType, params: List[str], id: int
    ) -> SendSubscriptionResult:
        """
        Send a subscription management request to the WebSocket server.
        Adapted to Binance requests

        Args:
            request_type: Type of request (SUBSCRIBE, UNSUBSCRIBE, etc.)
            params: List of subscription parameters
            id: Request ID

        Returns:
            SendSubscriptionResult: Result of the operation
        """
        if not self.websocket or not self.running:
            self.logger.error("Not connected or not running")
            return SendSubscriptionResult(False, "Not connected or not running")

        message = {"method": request_type.value, "params": params, "id": id}

        try:
            if await self.send_json(message):
                subscription = Subscription(id, params, SubscriptionStatus.PENDING)
                self.subscriptions[id] = subscription
                self.logger.info(
                    "Request sent",
                    request_type=request_type.value,
                    id=id,
                    params=params,
                )
                return SendSubscriptionResult(True, "Subscription message sent")
            else:
                self.logger.error("Failed to send message")
                return SendSubscriptionResult(False, "Failed to send message")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return SendSubscriptionResult(False, f"Error: {e}")

    def get_subscription_status(self, id: int) -> Optional[SubscriptionStatus]:
        """Get the status of a subscription by ID."""
        subscription = self.subscriptions.get(id)
        if subscription:
            return subscription.status
        else:
            self.logger.warning(f"No subscription with id {id}")
            return None

    def get_subscriptions_by_status(
        self, status: SubscriptionStatus
    ) -> List[Subscription]:
        """Get all subscriptions with a specific status."""
        return [
            subscription
            for subscription in self.subscriptions.values()
            if subscription.status == status
        ]

    def _connection_drop_cleanup(self):
        """Clean up subscriptions when connection is lost."""
        for subscription in self.subscriptions.values():
            if subscription.status == SubscriptionStatus.ACTIVE:
                subscription.status = SubscriptionStatus.DISCONNECTED

    async def _reestablish_subscriptions(self):
        """Re-establish subscriptions after reconnection."""
        for subscription in self.get_subscriptions_by_status(
            SubscriptionStatus.DISCONNECTED
        ):
            await self.send_subscription_request(
                RequestType.SUBSCRIBE, subscription.subscriptions, subscription.id
            )

    def handle_response(self, data: Dict[str, Any]):
        """Handle subscription management responses from the server.

        Adapted to Binance server

        Args:
            data: Dict[str, Any] json response from the server
        """
        if "code" in data:  # Error case
            self.logger.error(f"Request error: code {data['code']} - {data['msg']}")
            subscription = self.subscriptions.get(data["id"])
            if subscription:
                subscription.status = SubscriptionStatus.FAILED
            else:
                self.logger.warning("Subscription not in subscription dict")
        elif "result" in data:  # Success case
            if data["result"] is None:
                subscription = self.subscriptions.get(data["id"])
                if subscription:
                    subscription.status = SubscriptionStatus.ACTIVE
                    self.logger.info(
                        "Processed server response",
                        id=subscription.id,
                        status=subscription.status.value,
                        subscriptions=subscription.subscriptions,
                    )
                else:
                    self.logger.warning("Subscription not in subscription dict")
        if self.on_response:
            self.on_response(data)

    async def listen(self):
        """Listen for incoming messages from the WebSocket server.

        Check connection status
        listen to incoming messages
        Convert incoming messages from JSON to dict
        route messages based on message type: response message, error message, data message
            based on Binance

        """
        if not self.websocket:
            self.logger.error("Cannot listen: not connected")
            return

        try:
            async for message in self.websocket:
                self.logger.debug(f"Received message: {message}")

                # MODIFIED: Added JSON parsing and message routing
                try:
                    data = json.loads(message)

                    # Route messages based on content
                    if "result" in data or "code" in data:
                        # Subscription management response
                        self.handle_response(data)
                    elif "e" in data and self.on_message:
                        # Stream data message
                        self.on_message(data)

                except Exception as e:
                    self.logger.error(f"JSON deserialization error: {e}")

        except ConnectionClosed:
            self.logger.warning("WebSocket connection closed by server")
            # NEW: Cleanup subscriptions on connection drop
            self._connection_drop_cleanup()
        except WebSocketException as e:
            self.logger.error(f"WebSocket error: {e}")
            if self.on_error:
                self.on_error(e)
        except Exception as e:
            self.logger.error(f"Unexpected error in listen: {e}")
            if self.on_error:
                self.on_error(e)

    async def heartbeat(self):
        """Send periodic heartbeat messages to keep connection alive."""
        while self.running and self.websocket:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if self.websocket and self.running:
                    await self.websocket.ping()
                    self.logger.debug("Sent heartbeat ping")
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                break

    async def run_with_reconnect(
        self,
        ssl_context: Optional[ssl.SSLContext] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Run the WebSocket client with automatic reconnection.

        Connect to the websocket server
        spawn hearbeat task
        spawn reestablish subscriptions task
        Listen to incomming messages
        cancel spawned tasks when connection is lost
        Handle reconnection

        Args:
            ssl_context: SSL context for secure connections
            extra_headers: Additional headers to send with the connection
        """
        self.running = True

        while self.running:
            try:
                # Connect to the server
                if await self.connect(ssl_context, extra_headers):
                    # Start heartbeat task
                    heartbeat_task = asyncio.create_task(self.heartbeat())

                    # MODIFIED: Start subscription reestablishment task
                    reestablish_task = asyncio.create_task(
                        self._reestablish_subscriptions()
                    )

                    # Start listening for messages
                    await self.listen()

                    # Cancel tasks when connection is lost
                    heartbeat_task.cancel()
                    reestablish_task.cancel()  # NEW: Cancel reestablishment task
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await reestablish_task
                    except asyncio.CancelledError:
                        pass

                # Handle reconnection
                if (
                    self.running
                    and self.reconnect_attempts < self.max_reconnect_attempts
                ):
                    self.reconnect_attempts += 1
                    self.logger.info(
                        f"Reconnecting in {self.reconnect_interval} seconds... "
                        f"(attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
                    )
                    await asyncio.sleep(self.reconnect_interval)
                elif self.reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error("Max reconnection attempts reached")
                    break

            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                if self.on_error:
                    self.on_error(e)
                await asyncio.sleep(self.reconnect_interval)

    def set_message_handler(
        self, handler: Callable[[Dict[str, Any]], None]
    ):  # MODIFIED: Dict parameter
        """Set the message handler function."""
        self.on_message = handler

    def set_connect_handler(self, handler: Callable[[], None]):
        """Set the connect handler function."""
        self.on_connect = handler

    def set_disconnect_handler(self, handler: Callable[[], None]):
        """Set the disconnect handler function."""
        self.on_disconnect = handler

    def set_error_handler(self, handler: Callable[[Exception], None]):
        """Set the error handler function."""
        self.on_error = handler

    # NEW: Handler for subscription responses
    def set_response_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Set the response handler function for subscription management."""
        self.on_response = handler


# MODIFIED: Updated example usage for Binance-style subscriptions
async def main():
    """Example usage of the enhanced WebSocket client for Binance."""

    # Create client instance for Binance WebSocket
    client = WebSocketClient("wss://stream.binance.com:9443/ws")

    # Define event handlers
    def on_message_received(data: Dict[str, Any]):  # MODIFIED: Now receives dict
        print(f"Stream data received: {data}")

        # Handle different event types
        if data.get("e") == "aggTrade":
            print(
                f"Aggregate trade: {data['s']} - Price: {data['p']}, Qty: {data['q']}"
            )
        elif data.get("e") == "24hrTicker":
            print(f"24hr ticker: {data['s']} - Price change: {data['P']}%")

    def on_response_received(
        data: Dict[str, Any],
    ):  # NEW: Handle subscription responses
        print(f"Subscription response: {data}")

    def on_connected():
        print("Connected to Binance WebSocket!")

    def on_disconnected():
        print("Disconnected from Binance WebSocket!")

    def on_error_occurred(error: Exception):
        print(f"Error occurred: {error}")

    # Set event handlers
    client.set_message_handler(on_message_received)
    client.set_response_handler(on_response_received)  # NEW: Set response handler
    client.set_connect_handler(on_connected)
    client.set_disconnect_handler(on_disconnected)
    client.set_error_handler(on_error_occurred)

    # Start the client in a separate task
    client_task = asyncio.create_task(client.run_with_reconnect())

    # Wait a bit for connection
    await asyncio.sleep(2)

    # NEW: Send subscription requests using the new API
    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@aggTrade", "ethusdt@aggTrade"], id=1
    )
    print(f"Subscription result: {result.info}")

    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@ticker"], id=2
    )
    print(f"Subscription result: {result.info}")

    # Wait and check subscription statuses
    await asyncio.sleep(3)
    print(f"Subscription 1 status: {client.get_subscription_status(1)}")
    print(f"Subscription 2 status: {client.get_subscription_status(2)}")

    # Get all active subscriptions
    active_subs = client.get_subscriptions_by_status(SubscriptionStatus.ACTIVE)
    print(f"Active subscriptions: {len(active_subs)}")

    # Let it run for a while to see data
    await asyncio.sleep(10)

    # Gracefully shutdown
    await client.disconnect()
    client_task.cancel()

    try:
        await client_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
