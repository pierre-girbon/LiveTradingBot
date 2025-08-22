import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosed

# Import your WebSocket client classes
from src.websocket.websocketclient import (RequestType, SendSubscriptionResult,
                                           Subscription, SubscriptionStatus,
                                           WebSocketClient)


class TestWebSocketClientBasic:
    """Test basic WebSocket client functionality."""

    @pytest.fixture
    def client(self):
        """Create a WebSocket client instance for testing."""
        return WebSocketClient("wss://test.example.com/ws")

    def test_client_initialization(self, client):
        """Test client initializes with correct default values."""
        assert client.uri == "wss://test.example.com/ws"
        assert client.websocket is None
        assert client.running is False
        assert client.subscriptions == {}
        assert client.reconnect_interval == 5
        assert client.heartbeat_interval == 30
        assert client.max_reconnect_attempts == 10
        assert client.reconnect_attempts == 0

    def test_handler_setters(self, client):
        """Test that all handler setters work correctly."""
        message_handler = MagicMock()
        connect_handler = MagicMock()
        disconnect_handler = MagicMock()
        error_handler = MagicMock()
        response_handler = MagicMock()

        client.set_message_handler(message_handler)
        client.set_connect_handler(connect_handler)
        client.set_disconnect_handler(disconnect_handler)
        client.set_error_handler(error_handler)
        client.set_response_handler(response_handler)

        assert client.on_message == message_handler
        assert client.on_connect == connect_handler
        assert client.on_disconnect == disconnect_handler
        assert client.on_error == error_handler
        assert client.on_response == response_handler


class TestSubscriptionManagement:
    """Test subscription management functionality."""

    @pytest.fixture
    def client(self):
        return WebSocketClient("wss://stream.binance.com:9443/ws")

    @pytest.fixture
    def mock_websocket(self):
        mock_ws = AsyncMock()
        return mock_ws

    @pytest.mark.asyncio
    async def test_send_subscription_request_not_connected(self, client):
        """Test sending subscription request when not connected."""
        result = await client.send_subscription_request(
            RequestType.SUBSCRIBE, ["btcusdt@aggTrade"], 1
        )

        assert isinstance(result, SendSubscriptionResult)
        assert result.result is False
        assert "Not connected or not running" in result.info
        assert len(client.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_send_subscription_request_success(self, client, mock_websocket):
        """Test successful subscription request."""
        client.websocket = mock_websocket
        client.running = True

        with patch.object(client, "send_json", return_value=True) as mock_send:
            result = await client.send_subscription_request(
                RequestType.SUBSCRIBE, ["btcusdt@aggTrade", "ethusdt@aggTrade"], 1
            )

        # Verify result
        assert result.result is True
        assert "Subscription message sent" in result.info

        # Verify send_json was called with correct data
        expected_message = {
            "method": "SUBSCRIBE",
            "params": ["btcusdt@aggTrade", "ethusdt@aggTrade"],
            "id": 1,
        }
        mock_send.assert_called_once_with(expected_message)

        # Verify subscription tracking
        assert len(client.subscriptions) == 1
        assert 1 in client.subscriptions
        subscription = client.subscriptions[1]
        assert subscription.id == 1
        assert subscription.subscriptions == ["btcusdt@aggTrade", "ethusdt@aggTrade"]
        assert subscription.status == SubscriptionStatus.PENDING

    @pytest.mark.asyncio
    async def test_send_subscription_request_send_failure(self, client, mock_websocket):
        """Test subscription request when send fails."""
        client.websocket = mock_websocket
        client.running = True

        with patch.object(client, "send_json", return_value=False) as mock_send:
            result = await client.send_subscription_request(
                RequestType.UNSUBSCRIBE, ["btcusdt@aggTrade"], 2
            )

        # Verify result
        assert result.result is False
        assert "Failed to send message" in result.info

        # Verify no subscription was added
        assert len(client.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_send_subscription_request_exception(self, client, mock_websocket):
        """Test subscription request when exception occurs."""
        client.websocket = mock_websocket
        client.running = True

        with patch.object(client, "send_json", side_effect=Exception("Network error")):
            result = await client.send_subscription_request(RequestType.LIST, [], 3)

        # Verify result
        assert result.result is False
        assert "Error: Network error" in result.info

        # Verify no subscription was added
        assert len(client.subscriptions) == 0


class TestSubscriptionTracking:
    """Test subscription status tracking and queries."""

    @pytest.fixture
    def client_with_subscriptions(self):
        client = WebSocketClient("wss://test.example.com")

        # Add some test subscriptions
        client.subscriptions[1] = Subscription(
            1, ["btcusdt@aggTrade"], SubscriptionStatus.PENDING
        )
        client.subscriptions[2] = Subscription(
            2, ["ethusdt@aggTrade"], SubscriptionStatus.ACTIVE
        )
        client.subscriptions[3] = Subscription(
            3, ["adausdt@ticker"], SubscriptionStatus.FAILED
        )
        client.subscriptions[4] = Subscription(
            4, ["bnbusdt@depth"], SubscriptionStatus.DISCONNECTED
        )

        return client

    def test_get_subscription_status_exists(self, client_with_subscriptions):
        """Test getting status of existing subscription."""
        status = client_with_subscriptions.get_subscription_status(2)
        assert status == SubscriptionStatus.ACTIVE

    def test_get_subscription_status_not_exists(self, client_with_subscriptions):
        """Test getting status of non-existing subscription."""
        status = client_with_subscriptions.get_subscription_status(999)
        assert status is None

    def test_get_subscriptions_by_status_pending(self, client_with_subscriptions):
        """Test getting subscriptions by PENDING status."""
        pending_subs = client_with_subscriptions.get_subscriptions_by_status(
            SubscriptionStatus.PENDING
        )

        assert len(pending_subs) == 1
        assert pending_subs[0].id == 1
        assert pending_subs[0].subscriptions == ["btcusdt@aggTrade"]

    def test_get_subscriptions_by_status_active(self, client_with_subscriptions):
        """Test getting subscriptions by ACTIVE status."""
        active_subs = client_with_subscriptions.get_subscriptions_by_status(
            SubscriptionStatus.ACTIVE
        )

        assert len(active_subs) == 1
        assert active_subs[0].id == 2

    def test_get_subscriptions_by_status_none(self, client_with_subscriptions):
        """Test getting subscriptions by status that doesn't exist."""
        # Add a new status that doesn't exist in our test data
        empty_subs = client_with_subscriptions.get_subscriptions_by_status(
            SubscriptionStatus.PENDING
        )
        # Change the assertion to look for a status we know has no matches
        client_with_subscriptions.subscriptions.clear()
        empty_subs = client_with_subscriptions.get_subscriptions_by_status(
            SubscriptionStatus.ACTIVE
        )

        assert len(empty_subs) == 0


class TestResponseHandling:
    """Test response message handling and status updates."""

    @pytest.fixture
    def client_with_pending_subscription(self):
        client = WebSocketClient("wss://test.example.com")
        client.subscriptions[1] = Subscription(
            1, ["btcusdt@aggTrade"], SubscriptionStatus.PENDING
        )
        client.subscriptions[2] = Subscription(
            2, ["ethusdt@aggTrade"], SubscriptionStatus.PENDING
        )
        return client

    def test_on_response_success(self, client_with_pending_subscription):
        """Test successful subscription response."""
        response_data = {"result": None, "id": 1}

        client_with_pending_subscription.handle_response(response_data)

        # Verify subscription status updated
        assert (
            client_with_pending_subscription.subscriptions[1].status
            == SubscriptionStatus.ACTIVE
        )
        # Verify other subscription unchanged
        assert (
            client_with_pending_subscription.subscriptions[2].status
            == SubscriptionStatus.PENDING
        )

    def test_on_response_error(self, client_with_pending_subscription):
        """Test error subscription response."""
        response_data = {"code": 2, "msg": "Invalid symbol", "id": 2}

        client_with_pending_subscription.handle_response(response_data)

        # Verify subscription status updated to failed
        assert (
            client_with_pending_subscription.subscriptions[2].status
            == SubscriptionStatus.FAILED
        )
        # Verify other subscription unchanged
        assert (
            client_with_pending_subscription.subscriptions[1].status
            == SubscriptionStatus.PENDING
        )

    def test_on_response_unknown_id(self, client_with_pending_subscription):
        """Test response with unknown subscription ID."""
        response_data = {"result": None, "id": 999}

        # This should not raise an exception
        client_with_pending_subscription.handle_response(response_data)

        # Verify existing subscriptions unchanged
        assert (
            client_with_pending_subscription.subscriptions[1].status
            == SubscriptionStatus.PENDING
        )
        assert (
            client_with_pending_subscription.subscriptions[2].status
            == SubscriptionStatus.PENDING
        )


class TestMessageRouting:
    """Test message parsing and routing functionality."""

    @pytest.fixture
    def client(self):
        client = WebSocketClient("wss://test.example.com")
        client.on_message = MagicMock()
        client.on_response = MagicMock()
        return client

    @pytest.fixture
    def mock_websocket_messages(self):
        """Mock websocket that yields test messages."""
        mock_ws = AsyncMock()

        # Create test messages
        subscription_response = json.dumps({"result": None, "id": 1})
        error_response = json.dumps({"code": 2, "msg": "Invalid symbol", "id": 2})
        trade_data = json.dumps(
            {"e": "aggTrade", "s": "BTCUSDT", "p": "50000.00", "q": "0.1"}
        )
        invalid_json = "invalid json"

        messages = [subscription_response, error_response, trade_data, invalid_json]
        mock_ws.__aiter__.return_value = iter(messages)

        return mock_ws

    @pytest.mark.asyncio
    async def test_listen_message_routing(self, client, mock_websocket_messages):
        """Test that listen() correctly routes different message types."""
        client.websocket = mock_websocket_messages

        # Mock the response handler to avoid calling the actual method
        client.on_response = MagicMock()

        await client.listen()

        # Verify subscription response was routed correctly
        assert client.on_response.call_count == 2  # Success and error response

        # Verify first call (success response)
        first_call_args = client.on_response.call_args_list[0][0][0]
        assert first_call_args["result"] is None
        assert first_call_args["id"] == 1

        # Verify second call (error response)
        second_call_args = client.on_response.call_args_list[1][0][0]
        assert second_call_args["code"] == 2
        assert second_call_args["id"] == 2

        # Verify trade data was routed to message handler
        client.on_message.assert_called_once()
        message_args = client.on_message.call_args[0][0]
        assert message_args["e"] == "aggTrade"
        assert message_args["s"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_listen_connection_closed_cleanup(self, client):
        """Test that connection drop triggers subscription cleanup."""
        # Set up client with active subscriptions
        client.subscriptions[1] = Subscription(
            1, ["btcusdt@aggTrade"], SubscriptionStatus.ACTIVE
        )
        client.subscriptions[2] = Subscription(
            2, ["ethusdt@aggTrade"], SubscriptionStatus.PENDING
        )

        mock_ws = AsyncMock()
        mock_ws.__aiter__.side_effect = ConnectionClosed(None, None)
        client.websocket = mock_ws

        await client.listen()

        # Verify active subscription was marked as disconnected
        assert client.subscriptions[1].status == SubscriptionStatus.DISCONNECTED
        # Verify pending subscription unchanged
        assert client.subscriptions[2].status == SubscriptionStatus.PENDING


class TestConnectionManagement:
    """Test connection drop cleanup and reestablishment."""

    @pytest.fixture
    def client_with_mixed_subscriptions(self):
        client = WebSocketClient("wss://test.example.com")
        client.subscriptions[1] = Subscription(
            1, ["btcusdt@aggTrade"], SubscriptionStatus.ACTIVE
        )
        client.subscriptions[2] = Subscription(
            2, ["ethusdt@aggTrade"], SubscriptionStatus.PENDING
        )
        client.subscriptions[3] = Subscription(
            3, ["adausdt@ticker"], SubscriptionStatus.FAILED
        )
        client.subscriptions[4] = Subscription(
            4, ["bnbusdt@depth"], SubscriptionStatus.DISCONNECTED
        )
        return client

    def test_connection_drop_cleanup(self, client_with_mixed_subscriptions):
        """Test connection drop cleanup only affects active subscriptions."""
        client_with_mixed_subscriptions._connection_drop_cleanup()

        # Verify only active subscription was marked as disconnected
        assert (
            client_with_mixed_subscriptions.subscriptions[1].status
            == SubscriptionStatus.DISCONNECTED
        )
        # Verify others unchanged
        assert (
            client_with_mixed_subscriptions.subscriptions[2].status
            == SubscriptionStatus.PENDING
        )
        assert (
            client_with_mixed_subscriptions.subscriptions[3].status
            == SubscriptionStatus.FAILED
        )
        assert (
            client_with_mixed_subscriptions.subscriptions[4].status
            == SubscriptionStatus.DISCONNECTED
        )

    @pytest.mark.asyncio
    async def test_reestablish_subscriptions(self, client_with_mixed_subscriptions):
        """Test reestablishing disconnected subscriptions."""
        with patch.object(
            client_with_mixed_subscriptions, "send_subscription_request"
        ) as mock_send:
            mock_send.return_value = SendSubscriptionResult(True, "Success")

            await client_with_mixed_subscriptions._reestablish_subscriptions()

            # Verify only disconnected subscription was re-sent
            mock_send.assert_called_once_with(
                RequestType.SUBSCRIBE,
                ["bnbusdt@depth"],  # The disconnected subscription
                4,
            )


class TestEnumsAndDataClasses:
    """Test enum and dataclass definitions."""

    def test_subscription_status_enum(self):
        """Test SubscriptionStatus enum values."""
        assert SubscriptionStatus.PENDING.value == "PENDING"
        assert SubscriptionStatus.ACTIVE.value == "ACTIVE"
        assert SubscriptionStatus.FAILED.value == "FAILED"
        assert SubscriptionStatus.DISCONNECTED.value == "DISCONNECTED"

    def test_request_type_enum(self):
        """Test RequestType enum values."""
        assert RequestType.SUBSCRIBE.value == "SUBSCRIBE"
        assert RequestType.UNSUBSCRIBE.value == "UNSUBSCRIBE"
        assert RequestType.LIST.value == "LIST_SUBSCRIPTIONS"
        assert RequestType.SET_PROPERTY.value == "SET_PROPERTY"
        assert RequestType.GET_PROPERTY.value == "GET_PROPERTY"

    def test_subscription_dataclass(self):
        """Test Subscription dataclass."""
        sub = Subscription(1, ["btcusdt@aggTrade"], SubscriptionStatus.PENDING)

        assert sub.id == 1
        assert sub.subscriptions == ["btcusdt@aggTrade"]
        assert sub.status == SubscriptionStatus.PENDING

    def test_send_subscription_result_dataclass(self):
        """Test SendSubscriptionResult dataclass."""
        result = SendSubscriptionResult(True, "Success")

        assert result.result is True
        assert result.info == "Success"


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.fixture
    def client(self):
        return WebSocketClient("wss://stream.binance.com:9443/ws")

    @pytest.mark.asyncio
    async def test_full_subscription_flow(self, client):
        """Test complete subscription flow: request -> response -> status update."""
        # Mock websocket and connection
        mock_ws = AsyncMock()
        client.websocket = mock_ws
        client.running = True

        # Mock send_json to succeed
        with patch.object(client, "send_json", return_value=True):
            # Send subscription request
            result = await client.send_subscription_request(
                RequestType.SUBSCRIBE, ["btcusdt@aggTrade"], 1
            )

            assert result.result is True
            assert client.subscriptions[1].status == SubscriptionStatus.PENDING

        # Simulate successful response
        response_data = {"result": None, "id": 1}
        client.handle_response(response_data)

        # Verify status updated
        assert client.subscriptions[1].status == SubscriptionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_reconnection_scenario(self, client):
        """Test full reconnection scenario with subscription reestablishment."""
        # Set up client with active subscription
        client.subscriptions[1] = Subscription(
            1, ["btcusdt@aggTrade"], SubscriptionStatus.ACTIVE
        )

        # Simulate connection drop
        client._connection_drop_cleanup()
        assert client.subscriptions[1].status == SubscriptionStatus.DISCONNECTED

        # Mock successful reconnection and reestablishment
        with patch.object(client, "send_subscription_request") as mock_send:
            mock_send.return_value = SendSubscriptionResult(True, "Success")

            await client._reestablish_subscriptions()

            # Verify reestablishment was attempted
            mock_send.assert_called_once_with(
                RequestType.SUBSCRIBE, ["btcusdt@aggTrade"], 1
            )


# Run tests with: pytest test_websocket_client.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
