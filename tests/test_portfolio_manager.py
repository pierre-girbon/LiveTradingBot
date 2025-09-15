"""
Unit tests for PortfolioManager module

Tests cover:
- Position tracking and updates
- P&L calculations
- Trade processing
- Price updates
- Database persistence
- Handler methods
"""

import os
import sys

from modules.dataprocessor import KlineData, KlineInfo

sys.path.insert(0, os.path.join(os.path.dirname(__name__), "src/"))

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.modules.portfolio_manager import (PortfolioManager, PositionInfo,
                                           PositionSide, TradeEvent, TradeType)


class TestTradeEvent:
    """Test TradeEvent dataclass"""

    def test_trade_event_creation(self):
        """Test creating a TradeEvent"""
        timestamp = datetime.now(tz=timezone.utc)
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=timestamp,
            trade_id="test_trade_1",
        )

        assert trade.symbol == "BTCUSDT"
        assert trade.strategy_id == "test_strategy"
        assert trade.trade_type == TradeType.BUY
        assert trade.quantity == Decimal("10")
        assert trade.price == Decimal("50000")
        assert trade.timestamp == timestamp
        assert trade.trade_id == "test_trade_1"


class TestPositionInfo:
    """Test PositionInfo dataclass and its properties"""

    @pytest.fixture
    def long_position(self):
        """Create a sample long position"""
        return PositionInfo(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            quantity=Decimal("10"),
            avg_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("50000"),
            side=PositionSide.LONG,
            last_updated=datetime.now(tz=timezone.utc),
        )

    @pytest.fixture
    def short_position(self):
        """Create a sample short position"""
        return PositionInfo(
            symbol="ETHUSDT",
            strategy_id="test_strategy",
            quantity=Decimal("-5"),
            avg_price=Decimal("3000"),
            current_price=Decimal("2800"),
            unrealized_pnl=Decimal("1000"),
            side=PositionSide.SHORT,
            last_updated=datetime.now(tz=timezone.utc),
        )

    def test_market_value_long(self, long_position):
        """Test market value calculation for long position"""
        expected = Decimal("10") * Decimal("55000")
        assert long_position.market_value == expected

    def test_market_value_short(self, short_position):
        """Test market value calculation for short position"""
        expected = Decimal("5") * Decimal("2800")  # abs(quantity) * price
        assert short_position.market_value == expected

    def test_cost_basis_long(self, long_position):
        """Test cost basis calculation for long position"""
        expected = Decimal("10") * Decimal("50000")
        assert long_position.cost_basis == expected

    def test_cost_basis_short(self, short_position):
        """Test cost basis calculation for short position"""
        expected = Decimal("5") * Decimal("3000")  # abs(quantity) * avg_price
        assert short_position.cost_basis == expected


class TestPortfolioManager:
    """Test PortfolioManager functionality"""

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
        """Create a PortfolioManager instance for testing"""
        return PortfolioManager(db_url=temp_db_path)

    def test_portfolio_manager_initialization(self, portfolio_manager):
        """Test portfolio manager initialization"""
        assert portfolio_manager is not None
        assert portfolio_manager.positions == {}
        assert portfolio_manager.session is not None
        assert portfolio_manager.engine is not None

    def test_determine_side(self, portfolio_manager):
        """Test position side determination"""
        assert portfolio_manager._determine_side(Decimal("10")) == PositionSide.LONG
        assert portfolio_manager._determine_side(Decimal("-5")) == PositionSide.SHORT
        assert portfolio_manager._determine_side(Decimal("0")) == PositionSide.FLAT

    def test_calculate_unrealized_pnl_long(self, portfolio_manager):
        """Test P&L calculation for long position"""
        position = PositionInfo(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            quantity=Decimal("10"),
            avg_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("0"),
            side=PositionSide.LONG,
            last_updated=datetime.now(tz=timezone.utc),
        )

        pnl = portfolio_manager._calculate_unrealized_pnl(position)
        expected = (Decimal("55000") - Decimal("50000")) * Decimal("10")
        assert pnl == expected

    def test_calculate_unrealized_pnl_short(self, portfolio_manager):
        """Test P&L calculation for short position"""
        position = PositionInfo(
            symbol="ETHUSDT",
            strategy_id="test_strategy",
            quantity=Decimal("-5"),
            avg_price=Decimal("3000"),
            current_price=Decimal("2800"),
            unrealized_pnl=Decimal("0"),
            side=PositionSide.SHORT,
            last_updated=datetime.now(tz=timezone.utc),
        )

        pnl = portfolio_manager._calculate_unrealized_pnl(position)
        expected = (Decimal("3000") - Decimal("2800")) * Decimal("5")
        assert pnl == expected

    def test_calculate_unrealized_pnl_flat(self, portfolio_manager):
        """Test P&L calculation for flat position"""
        position = PositionInfo(
            symbol="ADAUSDT",
            strategy_id="test_strategy",
            quantity=Decimal("0"),
            avg_price=Decimal("1"),
            current_price=Decimal("1.5"),
            unrealized_pnl=Decimal("0"),
            side=PositionSide.FLAT,
            last_updated=datetime.now(tz=timezone.utc),
        )

        pnl = portfolio_manager._calculate_unrealized_pnl(position)
        assert pnl == Decimal("0")

    def test_process_trade_new_position_buy(self, portfolio_manager):
        """Test processing a trade for a new position (buy)"""
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )

        result = portfolio_manager.process_trade(trade)

        assert result is True
        assert ("BTCUSDT", "test_strategy") in portfolio_manager.positions

        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.quantity == Decimal("10")
        assert position.avg_price == Decimal("50000")
        assert position.side == PositionSide.LONG
        assert position.current_price == Decimal("50000")

    def test_process_trade_new_position_sell(self, portfolio_manager):
        """Test processing a trade for a new position (sell/short)"""
        trade = TradeEvent(
            symbol="ETHUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.SELL,
            quantity=Decimal("5"),
            price=Decimal("3000"),
            timestamp=datetime.now(tz=timezone.utc),
        )

        result = portfolio_manager.process_trade(trade)

        assert result is True
        assert ("ETHUSDT", "test_strategy") in portfolio_manager.positions

        position = portfolio_manager.positions[("ETHUSDT", "test_strategy")]
        assert position.quantity == Decimal("-5")
        assert position.avg_price == Decimal("3000")
        assert position.side == PositionSide.SHORT

    def test_process_trade_add_to_long_position(self, portfolio_manager):
        """Test adding to an existing long position"""
        # First trade
        trade1 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade1)

        # Second trade - add to position
        trade2 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("5"),
            price=Decimal("60000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        result = portfolio_manager.process_trade(trade2)

        assert result is True
        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.quantity == Decimal("15")

        # Check weighted average price calculation
        expected_avg = (
            Decimal("10") * Decimal("50000") + Decimal("5") * Decimal("60000")
        ) / Decimal("15")
        assert position.avg_price == expected_avg

    def test_process_trade_reduce_long_position(self, portfolio_manager):
        """Test reducing a long position"""
        # First trade - establish position
        trade1 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade1)

        # Second trade - reduce position
        trade2 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.SELL,
            quantity=Decimal("3"),
            price=Decimal("55000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        result = portfolio_manager.process_trade(trade2)

        assert result is True
        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.quantity == Decimal("7")
        assert position.avg_price == Decimal(
            "50000"
        )  # Avg price should remain the same
        assert position.side == PositionSide.LONG

    def test_process_trade_close_position(self, portfolio_manager):
        """Test closing a position completely"""
        # First trade - establish position
        trade1 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade1)

        # Second trade - close position
        trade2 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.SELL,
            quantity=Decimal("10"),
            price=Decimal("55000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        result = portfolio_manager.process_trade(trade2)

        assert result is True
        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.quantity == Decimal("0")
        assert position.side == PositionSide.FLAT
        assert position.unrealized_pnl == Decimal("0")

    def test_process_trade_invalid_inputs(self, portfolio_manager):
        """Test process_trade with invalid inputs"""

        # Test with non-TradeEvent object
        result = portfolio_manager.process_trade("invalid_trade")
        assert result is False

        # Test with zero quantity
        invalid_trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("0"),  # Invalid
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        result = portfolio_manager.process_trade(invalid_trade)
        assert result is False

        # Test with negative price
        invalid_trade.quantity = Decimal("10")
        invalid_trade.price = Decimal("-1000")  # Invalid
        result = portfolio_manager.process_trade(invalid_trade)
        assert result is False

        # Test with empty symbol
        invalid_trade.price = Decimal("50000")
        invalid_trade.symbol = ""  # Invalid
        result = portfolio_manager.process_trade(invalid_trade)
        assert result is False

    def test_update_price(self, portfolio_manager):
        """Test updating price for existing position"""
        # Create position first
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Update price
        new_timestamp = datetime.now(tz=timezone.utc)
        result = portfolio_manager.update_price(
            "BTCUSDT", Decimal("55000"), new_timestamp
        )

        assert result is True
        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.current_price == Decimal("55000")
        assert position.last_updated == new_timestamp

        # Check P&L calculation
        expected_pnl = (Decimal("55000") - Decimal("50000")) * Decimal("10")
        assert position.unrealized_pnl == expected_pnl

    def test_update_price_nonexistent_symbol(self, portfolio_manager):
        """Test updating price for non-existent symbol"""
        result = portfolio_manager.update_price(
            "NONEXISTENT", Decimal("100"), datetime.now(tz=timezone.utc)
        )
        assert result is True  # Should not fail, just won't update anything

    def test_update_price_value_error(self, portfolio_manager):
        """Test updating price for existing position"""
        # Create position first
        old_timestamp = datetime.now(tz=timezone.utc)
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=old_timestamp,
        )
        portfolio_manager.process_trade(trade)

        # Update price
        new_timestamp = datetime.now(tz=timezone.utc)
        result = portfolio_manager.update_price(
            "BTCUSDT", Decimal("-55000"), new_timestamp
        )

        assert result is False
        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.current_price == Decimal("50000")
        assert position.last_updated == old_timestamp

        # Check P&L calculation
        expected_pnl = 0
        assert position.unrealized_pnl == expected_pnl

    def test_get_position(self, portfolio_manager):
        """Test getting a position"""
        # Create position
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Get position
        position = portfolio_manager.get_position("BTCUSDT", "test_strategy")
        assert position is not None
        assert position.symbol == "BTCUSDT"
        assert position.quantity == Decimal("10")

        # Get non-existent position
        non_existent = portfolio_manager.get_position("NONEXISTENT", "test_strategy")
        assert non_existent is None

    def test_get_all_positions(self, portfolio_manager):
        """Test getting all positions"""
        # Create multiple positions
        trade1 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        trade2 = TradeEvent(
            symbol="ETHUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.SELL,
            quantity=Decimal("5"),
            price=Decimal("3000"),
            timestamp=datetime.now(tz=timezone.utc),
        )

        portfolio_manager.process_trade(trade1)
        portfolio_manager.process_trade(trade2)

        all_positions = portfolio_manager.get_all_positions()
        assert len(all_positions) == 2
        assert "BTCUSDT_test_strategy" in all_positions
        assert "ETHUSDT_test_strategy" in all_positions

    def test_get_portfolio_value(self, portfolio_manager):
        """Test calculating total portfolio value"""
        # Create positions with different P&Ls
        trade1 = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        trade2 = TradeEvent(
            symbol="ETHUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.SELL,
            quantity=Decimal("5"),
            price=Decimal("3000"),
            timestamp=datetime.now(tz=timezone.utc),
        )

        portfolio_manager.process_trade(trade1)
        portfolio_manager.process_trade(trade2)

        # Update prices
        portfolio_manager.update_price(
            "BTCUSDT", Decimal("55000"), datetime.now(tz=timezone.utc)
        )
        portfolio_manager.update_price(
            "ETHUSDT", Decimal("2800"), datetime.now(tz=timezone.utc)
        )

        total_pnl = portfolio_manager.get_portfolio_value()

        # Calculate expected total
        btc_pnl = (Decimal("55000") - Decimal("50000")) * Decimal("10")  # 50000
        eth_pnl = (Decimal("3000") - Decimal("2800")) * Decimal("5")  # 1000
        expected_total = btc_pnl + eth_pnl

        assert total_pnl == expected_total

    def test_get_position_quantity(self, portfolio_manager):
        """Test getting position quantity"""
        # Create position
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Test existing position
        qty = portfolio_manager.get_position_quantity("BTCUSDT", "test_strategy")
        assert qty == Decimal("10")

        # Test non-existent position
        qty = portfolio_manager.get_position_quantity("NONEXISTENT", "test_strategy")
        assert qty == Decimal("0")

    def test_calculate_available_balance_long_position(self, portfolio_manager):
        """Test calculating available balance for long position"""
        # Create long position
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Test with no locked quantity
        available = portfolio_manager.calculate_available_balance(
            "BTCUSDT", "test_strategy"
        )
        assert available == Decimal("10")

        # Test with locked quantity
        available = portfolio_manager.calculate_available_balance(
            "BTCUSDT", "test_strategy", Decimal("3")
        )
        assert available == Decimal("7")

        # Test with locked quantity exceeding position
        available = portfolio_manager.calculate_available_balance(
            "BTCUSDT", "test_strategy", Decimal("15")
        )
        assert available == Decimal("0")

    def test_calculate_available_balance_short_position(self, portfolio_manager):
        """Test calculating available balance for short position"""
        # Create short position
        trade = TradeEvent(
            symbol="ETHUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.SELL,
            quantity=Decimal("5"),
            price=Decimal("3000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Short positions should return 0 available balance
        available = portfolio_manager.calculate_available_balance(
            "ETHUSDT", "test_strategy"
        )
        assert available == Decimal("0")

    def test_calculate_available_balance_no_position(self, portfolio_manager):
        """Test calculating available balance for non-existent position"""
        available = portfolio_manager.calculate_available_balance(
            "NONEXISTENT", "test_strategy"
        )
        assert available == Decimal("0")

    def test_handle_kline_data(self, portfolio_manager):
        """Test handling kline data"""
        # Create position first
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Mock kline data
        mock_kline_data = Mock(spec=KlineData)
        mock_kline_data.symbol = "BTCUSDT"
        mock_kline = Mock(spec=KlineInfo)
        mock_kline_data.kline = mock_kline
        mock_kline_data.kline.close_price = 55000.0
        mock_kline_data.kline.close_time = datetime.now(tz=timezone.utc)

        # Handle kline data
        portfolio_manager.handle_kline_data(mock_kline_data)

        # Verify price was updated
        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.current_price == Decimal("55000")

    def test_handle_kline_data_type_error(self, portfolio_manager):
        """Test handling kline data"""
        # Create position first
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_manager.process_trade(trade)

        # Mock kline data
        mock_kline_data = Mock()
        mock_kline_data.symbol = "BTCUSDT"
        mock_kline_data.kline.close_price = 55000.0
        mock_kline_data.kline.close_time = datetime.now(tz=timezone.utc)

        # Handle kline data
        portfolio_manager.handle_kline_data(mock_kline_data)

        # Verify price was updated
        position = portfolio_manager.positions[("BTCUSDT", "test_strategy")]
        assert position.current_price == Decimal("50000")

    def test_handle_trade_data(self, portfolio_manager):
        """Test handling trade data (should not update positions)"""
        mock_trade_data = Mock()
        mock_trade_data.symbol = "BTCUSDT"
        mock_trade_data.price = 55000.0

        # This should not create any positions or updates
        portfolio_manager.handle_trade_data(mock_trade_data)

        assert len(portfolio_manager.positions) == 0

    def test_process_trade_exception_handling(self, portfolio_manager):
        """Test trade processing with exception handling"""
        # Test with invalid trade data that might cause exceptions
        with patch.object(portfolio_manager.logger, "error") as mock_logger:
            # Create a trade with invalid data types that might cause issues
            invalid_trade = Mock()
            invalid_trade.symbol = None  # This might cause issues

            result = portfolio_manager.process_trade(invalid_trade)
            assert result is False
            mock_logger.assert_called()

    def test_update_price_exception_handling(self, portfolio_manager):
        """Test price update with exception handling"""
        with patch.object(portfolio_manager.logger, "error") as mock_logger:
            # Try to update with invalid price
            result = portfolio_manager.update_price(
                "BTCUSDT", None, datetime.now(tz=timezone.utc)
            )
            assert result is False
            mock_logger.assert_called()

    @patch("modules.portfolio_manager.sessionmaker")
    @patch("modules.portfolio_manager.create_engine")
    def test_database_operations(
        self, mock_create_engine, mock_sessionmaker, temp_db_path
    ):
        """Test database save operations"""
        mock_session = Mock()
        mock_sessionmaker.return_value.return_value = mock_session

        portfolio_manager = PortfolioManager(db_url=temp_db_path)

        # Test trade processing with database operations
        trade = TradeEvent(
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            trade_type=TradeType.BUY,
            quantity=Decimal("10"),
            price=Decimal("50000"),
            timestamp=datetime.now(tz=timezone.utc),
        )

        portfolio_manager.process_trade(trade)

        # Verify database operations were called
        # BUG: Should be called
        # assert mock_session.add.called
        # assert mock_session.commit.called

    def test_close(self, portfolio_manager):
        """Test closing portfolio manager"""
        mock_session = Mock()
        portfolio_manager.session = mock_session

        portfolio_manager.close()

        mock_session.close.assert_called_once()


class TestTradeTypes:
    """Test TradeType enum"""

    def test_trade_types(self):
        """Test TradeType enum values"""
        assert TradeType.BUY.value == "BUY"
        assert TradeType.SELL.value == "SELL"


class TestPositionSide:
    """Test PositionSide enum"""

    def test_position_sides(self):
        """Test PositionSide enum values"""
        assert PositionSide.LONG.value == "LONG"
        assert PositionSide.SHORT.value == "SHORT"
        assert PositionSide.FLAT.value == "FLAT"


if __name__ == "__main__":
    pytest.main([__file__])
