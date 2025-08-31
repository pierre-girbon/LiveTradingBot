"""
Unit tests for OrderManager module

Tests cover:
- Order validation
- Order placement (market, limit, stop)
- Order tracking and state management
- Order execution simulation
- Database persistence
- Integration with PortfolioManager
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__name__), "src/"))


import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from modules.order_manager import (
    LimitOrder,
    MarketOrder,
    OrderManager,
    OrderStatus,
    OrderType,
    OrderValidationResult,
    StopOrder,
)
from modules.portfolio_manager import PortfolioManager, TradeEvent, TradeType


class TestOrderValidationResult:
    """Test OrderValidationResult dataclass"""

    def test_valid_result(self):
        """Test creating a valid result"""
        result = OrderValidationResult(is_valid=True, available_balance=Decimal("10"))

        assert result.is_valid is True
        assert result.error_message is None
        assert result.available_balance == Decimal("10")

    def test_invalid_result(self):
        """Test creating an invalid result"""
        result = OrderValidationResult(
            is_valid=False, error_message="Insufficient balance"
        )

        assert result.is_valid is False
        assert result.error_message == "Insufficient balance"
        assert result.available_balance is None


class TestOrderTypes:
    """Test different order type classes"""

    @pytest.fixture
    def base_order_data(self):
        """Base data for creating orders"""
        return {
            "order_id": str(uuid4()),
            "symbol": "BTCUSDT",
            "trade_type": TradeType.BUY,
            "quantity_ordered": Decimal("10"),
            "order_status": OrderStatus.PENDING,
            "creation_date": datetime.now(tz=timezone.utc),
        }

    def test_market_order_creation(self, base_order_data):
        """Test MarketOrder creation and methods"""
        order = MarketOrder(**base_order_data)

        assert order.get_order_type() == OrderType.MARKET
        assert order.get_reference_price() == Decimal("0")  # No fill price yet
        assert order.quantity_remaining == Decimal("10")
        assert order.is_fully_filled is False
        assert order.fill_percentage == Decimal("0")

    def test_limit_order_creation(self, base_order_data):
        """Test LimitOrder creation and methods"""
        order = LimitOrder(limit_price=Decimal("50000"), **base_order_data)

        assert order.get_order_type() == OrderType.LIMIT
        assert order.get_reference_price() == Decimal("50000")
        assert order.limit_price == Decimal("50000")

    def test_limit_order_invalid_price(self, base_order_data):
        """Test LimitOrder with invalid price"""
        with pytest.raises(ValueError, match="Limit price must be positive"):
            LimitOrder(limit_price=Decimal("0"), **base_order_data)

        with pytest.raises(ValueError, match="Limit price must be positive"):
            LimitOrder(limit_price=Decimal("-1000"), **base_order_data)

    def test_stop_order_creation(self, base_order_data):
        """Test StopOrder creation and methods"""
        order = StopOrder(stop_price=Decimal("45000"), **base_order_data)

        assert order.get_order_type() == OrderType.STOP
        assert order.get_reference_price() == Decimal("45000")
        assert order.stop_price == Decimal("45000")

    def test_stop_order_invalid_price(self, base_order_data):
        """Test StopOrder with invalid price"""
        with pytest.raises(ValueError, match="Stop price must be positive"):
            StopOrder(stop_price=Decimal("0"), **base_order_data)

    def test_order_fill_tracking(self, base_order_data):
        """Test order fill tracking properties"""
        order = MarketOrder(**base_order_data)

        # Initially unfilled
        assert order.quantity_remaining == Decimal("10")
        assert order.is_fully_filled is False
        assert order.fill_percentage == Decimal("0")

        # Partial fill
        order.quantity_filled = Decimal("3")
        order.filled_price = Decimal("50000")

        assert order.quantity_remaining == Decimal("7")
        assert order.is_fully_filled is False
        assert order.fill_percentage == Decimal("30")

        # Full fill
        order.quantity_filled = Decimal("10")

        assert order.quantity_remaining == Decimal("0")
        assert order.is_fully_filled is True
        assert order.fill_percentage == Decimal("100")


class TestOrderManager:
    """Test OrderManager functionality"""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield f"sqlite:///{db_path}"
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock PortfolioManager"""
        mock_pm = Mock(spec=PortfolioManager)
        mock_pm.calculate_available_balance.return_value = Decimal("10")
        mock_pm.get_position.return_value = Mock()
        mock_pm.get_position.return_value.current_price = Decimal("50000")
        return mock_pm

    @pytest.fixture
    def order_manager(self, mock_portfolio_manager, temp_db_path):
        """Create an OrderManager instance for testing"""
        return OrderManager(mock_portfolio_manager, db_url=temp_db_path)

    def test_order_manager_initialization(self, order_manager):
        """Test order manager initialization"""
        assert order_manager is not None
        assert order_manager.orders == {}
        assert order_manager.orders_by_symbol == {}
        assert order_manager.session is not None
        assert order_manager.engine is not None

    def test_calculate_locked_quantity_empty(self, order_manager):
        """Test locked quantity calculation with no orders"""
        locked = order_manager._calculate_locked_quantity("BTCUSDT")
        assert locked == Decimal("0")

    def test_calculate_locked_quantity_with_orders(self, order_manager):
        """Test locked quantity calculation with pending orders"""
        # Create some orders manually
        order1 = MarketOrder(
            order_id="order1",
            symbol="BTCUSDT",
            trade_type=TradeType.SELL,
            quantity_ordered=Decimal("5"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order2 = LimitOrder(
            order_id="order2",
            symbol="BTCUSDT",
            trade_type=TradeType.SELL,
            quantity_ordered=Decimal("3"),
            order_status=OrderStatus.PARTIAL,
            creation_date=datetime.now(tz=timezone.utc),
            limit_price=Decimal("55000"),
        )
        order2.quantity_filled = Decimal("1")  # 2 remaining

        order3 = MarketOrder(
            order_id="order3",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,  # Buy order shouldn't count
            quantity_ordered=Decimal("2"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        # Add orders to manager
        order_manager.orders = {"order1": order1, "order2": order2, "order3": order3}
        order_manager.orders_by_symbol = {"BTCUSDT": ["order1", "order2", "order3"]}

        locked = order_manager._calculate_locked_quantity("BTCUSDT")
        # Should be 5 (order1) + 2 (order2 remaining) = 7
        # order3 is BUY so doesn't count
        assert locked == Decimal("7")

    def test_validate_order_positive_quantity(self, order_manager):
        """Test order validation with positive quantity"""
        result = order_manager.validate_order("BTCUSDT", TradeType.BUY, Decimal("5"))
        assert result.is_valid is True
        assert result.error_message is None

    def test_validate_order_zero_quantity(self, order_manager):
        """Test order validation with zero quantity"""
        result = order_manager.validate_order("BTCUSDT", TradeType.BUY, Decimal("0"))
        assert result.is_valid is False
        assert "Quantity must be positive" in result.error_message

    def test_validate_order_negative_quantity(self, order_manager):
        """Test order validation with negative quantity"""
        result = order_manager.validate_order("BTCUSDT", TradeType.BUY, Decimal("-5"))
        assert result.is_valid is False
        assert "Quantity must be positive" in result.error_message

    def test_validate_order_sell_sufficient_balance(
        self, order_manager, mock_portfolio_manager
    ):
        """Test sell order validation with sufficient balance"""
        mock_portfolio_manager.calculate_available_balance.return_value = Decimal("10")

        result = order_manager.validate_order("BTCUSDT", TradeType.SELL, Decimal("5"))
        assert result.is_valid is True
        assert result.available_balance == Decimal("10")

    def test_validate_order_sell_insufficient_balance(
        self, order_manager, mock_portfolio_manager
    ):
        """Test sell order validation with insufficient balance"""
        mock_portfolio_manager.calculate_available_balance.return_value = Decimal("3")

        result = order_manager.validate_order("BTCUSDT", TradeType.SELL, Decimal("5"))
        assert result.is_valid is False
        assert "Insufficient balance" in result.error_message
        assert result.available_balance == Decimal("3")

    def test_validate_order_buy_no_balance_check(
        self, order_manager, mock_portfolio_manager
    ):
        """Test buy order validation (no balance check needed)"""
        # For buy orders, we don't check balance in this simple implementation
        result = order_manager.validate_order("BTCUSDT", TradeType.BUY, Decimal("100"))
        assert result.is_valid is True

    def test_place_market_order_success(self, order_manager, mock_portfolio_manager):
        """Test successful market order placement"""
        mock_portfolio_manager.calculate_available_balance.return_value = Decimal("10")

        with patch.object(
            order_manager, "_simulate_market_order_execution"
        ) as mock_execution:
            order_id = order_manager.place_market_order(
                "BTCUSDT", TradeType.SELL, Decimal("5")
            )

        assert order_id is not None
        assert order_id in order_manager.orders

        order = order_manager.orders[order_id]
        assert isinstance(order, MarketOrder)
        assert order.symbol == "BTCUSDT"
        assert order.trade_type == TradeType.SELL
        assert order.quantity_ordered == Decimal("5")
        assert order.order_status == OrderStatus.PENDING

        # Check symbol index
        assert "BTCUSDT" in order_manager.orders_by_symbol
        assert order_id in order_manager.orders_by_symbol["BTCUSDT"]

        # Verify execution was called
        mock_execution.assert_called_once()

    def test_place_market_order_validation_failed(
        self, order_manager, mock_portfolio_manager
    ):
        """Test market order placement with validation failure"""
        mock_portfolio_manager.calculate_available_balance.return_value = Decimal("2")

        order_id = order_manager.place_market_order(
            "BTCUSDT", TradeType.SELL, Decimal("5")
        )

        assert order_id is None
        assert len(order_manager.orders) == 0

    def test_place_limit_order_success(self, order_manager, mock_portfolio_manager):
        """Test successful limit order placement"""
        mock_portfolio_manager.calculate_available_balance.return_value = Decimal("10")

        order_id = order_manager.place_limit_order(
            "BTCUSDT", TradeType.SELL, Decimal("5"), Decimal("55000")
        )

        assert order_id is not None
        assert order_id in order_manager.orders

        order = order_manager.orders[order_id]
        assert isinstance(order, LimitOrder)
        assert order.symbol == "BTCUSDT"
        assert order.trade_type == TradeType.SELL
        assert order.quantity_ordered == Decimal("5")
        assert order.limit_price == Decimal("55000")
        assert order.order_status == OrderStatus.PENDING

    def test_place_limit_order_validation_failed(
        self, order_manager, mock_portfolio_manager
    ):
        """Test limit order placement with validation failure"""
        mock_portfolio_manager.calculate_available_balance.return_value = Decimal("2")

        order_id = order_manager.place_limit_order(
            "BTCUSDT", TradeType.SELL, Decimal("5"), Decimal("55000")
        )

        assert order_id is None
        assert len(order_manager.orders) == 0

    def test_simulate_market_order_execution(
        self, order_manager, mock_portfolio_manager
    ):
        """Test market order execution simulation"""
        mock_position = Mock()
        mock_position.current_price = Decimal("51000")
        mock_portfolio_manager.get_position.return_value = mock_position

        order = MarketOrder(
            order_id="test_order",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("2"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders["test_order"] = order

        with patch.object(order_manager, "_process_order_fill") as mock_fill:
            order_manager._simulate_market_order_execution(order)

        mock_fill.assert_called_once()
        call_args = mock_fill.call_args[0]
        assert call_args[0] == "test_order"  # order_id
        assert call_args[1] == Decimal("2")  # filled_quantity
        assert call_args[2] == Decimal("51000")  # fill_price

    def test_simulate_market_order_execution_no_position(
        self, order_manager, mock_portfolio_manager
    ):
        """Test market order execution simulation without existing position"""
        mock_portfolio_manager.get_position.return_value = None

        order = MarketOrder(
            order_id="test_order",
            symbol="NEWCOIN",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("2"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders["test_order"] = order

        with patch.object(order_manager, "_process_order_fill") as mock_fill:
            order_manager._simulate_market_order_execution(order)

        mock_fill.assert_called_once()
        call_args = mock_fill.call_args[0]
        assert call_args[2] == Decimal("50000")  # fallback price

    def test_process_order_fill_new_fill(self, order_manager, mock_portfolio_manager):
        """Test processing order fill for unfilled order"""
        order = MarketOrder(
            order_id="test_order",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("10"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders["test_order"] = order

        fill_time = datetime.now(tz=timezone.utc)
        order_manager._process_order_fill(
            "test_order", Decimal("3"), Decimal("50000"), fill_time
        )

        # Check order was updated
        assert order.quantity_filled == Decimal("3")
        assert order.filled_price == Decimal("50000")
        assert order.order_status == OrderStatus.PARTIAL
        assert order.last_updated == fill_time

        # Verify portfolio manager was called
        mock_portfolio_manager.process_trade.assert_called_once()

        trade_arg = mock_portfolio_manager.process_trade.call_args[0][0]
        assert trade_arg.symbol == "BTCUSDT"
        assert trade_arg.trade_type == TradeType.BUY
        assert trade_arg.quantity == Decimal("3")
        assert trade_arg.price == Decimal("50000")
        assert trade_arg.timestamp == fill_time

    def test_process_order_fill_complete_fill(
        self, order_manager, mock_portfolio_manager
    ):
        """Test processing order fill that completes the order"""
        order = MarketOrder(
            order_id="test_order",
            symbol="BTCUSDT",
            trade_type=TradeType.SELL,
            quantity_ordered=Decimal("5"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders["test_order"] = order

        fill_time = datetime.now(tz=timezone.utc)
        order_manager._process_order_fill(
            "test_order", Decimal("5"), Decimal("52000"), fill_time
        )

        # Check order was completed
        assert order.quantity_filled == Decimal("5")
        assert order.filled_price == Decimal("52000")
        assert order.order_status == OrderStatus.COMPLETED
        assert order.is_fully_filled is True

    def test_process_order_fill_weighted_average_price(
        self, order_manager, mock_portfolio_manager
    ):
        """Test weighted average price calculation for multiple fills"""
        order = MarketOrder(
            order_id="test_order",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("10"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders["test_order"] = order

        # First fill
        order_manager._process_order_fill(
            "test_order", Decimal("3"), Decimal("50000"), datetime.now(tz=timezone.utc)
        )
        assert order.filled_price == Decimal("50000")

        # Second fill
        order_manager._process_order_fill(
            "test_order", Decimal("2"), Decimal("51000"), datetime.now(tz=timezone.utc)
        )

        # Check weighted average: (3*50000 + 2*51000) / 5 = 50400
        expected_avg = (
            Decimal("3") * Decimal("50000") + Decimal("2") * Decimal("51000")
        ) / Decimal("5")
        assert order.filled_price == expected_avg
        assert order.quantity_filled == Decimal("5")
        assert order.order_status == OrderStatus.PARTIAL

    def test_process_order_fill_nonexistent_order(self, order_manager):
        """Test processing fill for non-existent order"""
        with patch.object(order_manager.logger, "error") as mock_logger:
            order_manager._process_order_fill(
                "nonexistent",
                Decimal("1"),
                Decimal("50000"),
                datetime.now(tz=timezone.utc),
            )

        mock_logger.assert_called_once()

    def test_cancel_order_success(self, order_manager):
        """Test successful order cancellation"""
        order = LimitOrder(
            order_id="test_order",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("5"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
            limit_price=Decimal("49000"),
        )

        order_manager.orders["test_order"] = order

        result = order_manager.cancel_order("test_order")

        assert result is True
        assert order.order_status == OrderStatus.CANCELED

    def test_cancel_order_nonexistent(self, order_manager):
        """Test canceling non-existent order"""
        result = order_manager.cancel_order("nonexistent")
        assert result is False

    def test_cancel_order_wrong_status(self, order_manager):
        """Test canceling order with wrong status"""
        order = MarketOrder(
            order_id="test_order",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("5"),
            order_status=OrderStatus.COMPLETED,  # Can't cancel completed order
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders["test_order"] = order

        result = order_manager.cancel_order("test_order")
        assert result is False
        assert order.order_status == OrderStatus.COMPLETED  # Unchanged

    def test_get_order(self, order_manager):
        """Test getting order by ID"""
        order = MarketOrder(
            order_id="test_order",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("5"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders["test_order"] = order

        retrieved = order_manager.get_order("test_order")
        assert retrieved is order

        # Test non-existent order
        assert order_manager.get_order("nonexistent") is None

    def test_get_orders_by_symbol(self, order_manager):
        """Test getting orders by symbol"""
        order1 = MarketOrder(
            order_id="order1",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("5"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order2 = LimitOrder(
            order_id="order2",
            symbol="BTCUSDT",
            trade_type=TradeType.SELL,
            quantity_ordered=Decimal("3"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
            limit_price=Decimal("55000"),
        )

        order3 = MarketOrder(
            order_id="order3",
            symbol="ETHUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("10"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        order_manager.orders = {"order1": order1, "order2": order2, "order3": order3}
        order_manager.orders_by_symbol = {
            "BTCUSDT": ["order1", "order2"],
            "ETHUSDT": ["order3"],
        }

        btc_orders = order_manager.get_orders_by_symbol("BTCUSDT")
        assert len(btc_orders) == 2
        assert order1 in btc_orders
        assert order2 in btc_orders

        eth_orders = order_manager.get_orders_by_symbol("ETHUSDT")
        assert len(eth_orders) == 1
        assert order3 in eth_orders

        # Test non-existent symbol
        empty_orders = order_manager.get_orders_by_symbol("NONEXISTENT")
        assert len(empty_orders) == 0

    def test_get_active_orders(self, order_manager):
        """Test getting active orders"""
        pending_order = MarketOrder(
            order_id="pending",
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("5"),
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        partial_order = LimitOrder(
            order_id="partial",
            symbol="ETHUSDT",
            trade_type=TradeType.SELL,
            quantity_ordered=Decimal("10"),
            order_status=OrderStatus.PARTIAL,
            creation_date=datetime.now(tz=timezone.utc),
            limit_price=Decimal("3000"),
        )

        completed_order = MarketOrder(
            order_id="completed",
            symbol="ADAUSDT",
            trade_type=TradeType.BUY,
            quantity_ordered=Decimal("100"),
            order_status=OrderStatus.COMPLETED,
            creation_date=datetime.now(tz=timezone.utc),
        )

        canceled_order = LimitOrder(
            order_id="canceled",
            symbol="DOTUSDT",
            trade_type=TradeType.SELL,
            quantity_ordered=Decimal("50"),
            order_status=OrderStatus.CANCELED,
            creation_date=datetime.now(tz=timezone.utc),
            limit_price=Decimal("25"),
        )

        order_manager.orders = {
            "pending": pending_order,
            "partial": partial_order,
            "completed": completed_order,
            "canceled": canceled_order,
        }

        active_orders = order_manager.get_active_orders()
        assert len(active_orders) == 2
        assert pending_order in active_orders
        assert partial_order in active_orders
        assert completed_order not in active_orders
        assert canceled_order not in active_orders

    # BUG: Do not work
    # @patch("modules.order_manager.sessionmaker")
    # @patch("modules.order_manager.create_engine")
    # def test_database_operations(
    #     self,
    #     mock_create_engine,
    #     mock_sessionmaker,
    #     mock_portfolio_manager,
    #     temp_db_path,
    # ):
    #     """Test database save operations"""
    #     mock_session = Mock()
    #     mock_sessionmaker.return_value.return_value = mock_session
    #     mock_query = Mock()
    #     mock_session.query.return_value = mock_query
    #     mock_query.filter_by.return_value.first.return_value = None
    #
    #     order_manager = OrderManager(mock_portfolio_manager, db_url=temp_db_path)
    #
    #     # Create and save an order
    #     order = MarketOrder(
    #         order_id="test_order",
    #         symbol="BTCUSDT",
    #         trade_type=TradeType.BUY,
    #         quantity_ordered=Decimal("5"),
    #         order_status=OrderStatus.PENDING,
    #         creation_date=datetime.now(tz=timezone.utc),
    #     )
    #
    #     order_manager._save_order_to_db(order)
    #
    #     # Verify database operations
    #     assert mock_session.add.called
    #     assert mock_session.commit.called

    def test_close(self, order_manager):
        """Test closing order manager"""
        mock_session = Mock()
        order_manager.session = mock_session

        order_manager.close()

        mock_session.close.assert_called_once()

    def test_validation_exception_handling(self, order_manager):
        """Test order validation with exception handling"""
        # Mock portfolio manager to raise exception
        order_manager.portfolio_manager.calculate_available_balance.side_effect = (
            Exception("Test exception")
        )

        with patch.object(order_manager.logger, "error") as mock_logger:
            result = order_manager.validate_order(
                "BTCUSDT", TradeType.SELL, Decimal("5")
            )

        assert result.is_valid is False
        assert "Validation error" in result.error_message
        mock_logger.assert_called()

    def test_process_order_fill_exception_handling(self, order_manager):
        """Test order fill processing with exception handling"""
        # Create invalid order that might cause exceptions
        order = Mock()
        order.quantity_filled = "invalid"  # This should cause issues

        order_manager.orders["test_order"] = order

        with patch.object(order_manager.logger, "error") as mock_logger:
            order_manager._process_order_fill(
                "test_order",
                Decimal("1"),
                Decimal("50000"),
                datetime.now(tz=timezone.utc),
            )

        mock_logger.assert_called()


class TestOrderEnums:
    """Test order-related enums"""

    def test_order_status_values(self):
        """Test OrderStatus enum values"""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.PARTIAL.value == "PARTIAL"
        assert OrderStatus.COMPLETED.value == "COMPLETED"
        assert OrderStatus.CANCELED.value == "CANCELED"
        assert OrderStatus.REJECTED.value == "REJECTED"

    def test_order_type_values(self):
        """Test OrderType enum values"""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"


class TestIntegration:
    """Integration tests between OrderManager and PortfolioManager"""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield f"sqlite:///{db_path}"
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def portfolio_manager(self, temp_db_path):
        """Create a real PortfolioManager for integration testing"""
        return PortfolioManager(db_url=temp_db_path)

    @pytest.fixture
    def order_manager_integrated(self, portfolio_manager, temp_db_path):
        """Create OrderManager with real PortfolioManager"""
        # Different DB path to avoid conflicts
        om_db_path = temp_db_path.replace(".db", "_orders.db")
        return OrderManager(portfolio_manager, db_url=om_db_path)

    def test_full_order_lifecycle(self, order_manager_integrated, portfolio_manager):
        """Test complete order lifecycle with real portfolio integration"""
        # First, create initial position
        initial_trade = TradeEvent(
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(initial_trade)

        # Update current price
        portfolio_manager.update_price(
            "BTCUSDT", Decimal("52000"), datetime.now(tz=timezone.utc)
        )

        # Place market sell order
        order_id = order_manager_integrated.place_market_order(
            "BTCUSDT", TradeType.SELL, Decimal("3")
        )

        assert order_id is not None

        # Check order was created
        order = order_manager_integrated.get_order(order_id)
        assert (
            order.order_status == OrderStatus.COMPLETED
        )  # Should be executed immediately
        assert order.quantity_filled == Decimal("3")

        # Check portfolio was updated
        position = portfolio_manager.get_position("BTCUSDT")
        assert position.quantity == Decimal("7")  # 10 - 3 = 7

    def test_insufficient_balance_integration(
        self, order_manager_integrated, portfolio_manager
    ):
        """Test order rejection due to insufficient balance"""
        # Create small position
        trade = TradeEvent(
            symbol="BTCUSDT",
            trade_type=TradeType.BUY,
            quantity=Decimal("2"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Try to sell more than we have
        order_id = order_manager_integrated.place_market_order(
            "BTCUSDT", TradeType.SELL, Decimal("5")
        )

        assert order_id is None  # Should be rejected

        # Position should be unchanged
        position = portfolio_manager.get_position("BTCUSDT")
        assert position.quantity == Decimal("2")


if __name__ == "__main__":
    pytest.main([__file__])
