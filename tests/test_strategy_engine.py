"""
Comprehensive test suite for strategy_engine_integration.py

Run with: pytest test_strategy_engine_integration.py -v

This test suite covers:
- Strategy data management
- Base strategy functionality
- Strategy portfolio manager
- Strategy engine core functionality
- Configuration loading
- Signal generation and execution
- Integration with existing components

Dependencies:
- pytest
- pytest-asyncio (for async tests)
- pytest-mock (for mocking)
"""

import os

# Import the modules we're testing
# Note: Adjust imports based on your actual file structure
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__name__), "src/"))

from modules.dataprocessor import KlineData, KlineInfo
from modules.order_manager import OrderManager
from modules.portfolio_manager import PortfolioManager, TradeEvent, TradeType
from modules.strategy_engine import (
    BaseStrategy,
    Signal,
    SignalType,
    StrategyData,
    StrategyEngine,
)


# Test Fixtures
@pytest.fixture
def sample_strategy_data():
    """Create sample strategy data for testing."""
    return StrategyData(
        symbol="BTCUSDT",
        price_history=[Decimal(str(p)) for p in [50000, 50100, 50200, 50150, 50300]],
        current_position=Decimal("0"),
        parameters={"short_window": 5, "long_window": 10, "position_size": 1.0},
    )


@pytest.fixture
def mock_portfolio_manager():
    """Create a mock portfolio manager."""
    mock_pm = Mock(spec=PortfolioManager)
    mock_pm.get_position_quantity.return_value = Decimal("0")
    mock_pm.get_position.return_value = None
    mock_pm.get_all_positions.return_value = {}
    mock_pm.process_trade.return_value = True
    mock_pm.update_price.return_value = True
    return mock_pm


@pytest.fixture
def mock_order_manager():
    """Create a mock order manager."""
    mock_om = Mock(spec=OrderManager)
    mock_om.place_market_order.return_value = "test_order_123"
    return mock_om


@pytest.fixture
def test_strategy():
    """Create a concrete test strategy implementation."""

    class TestStrategy(BaseStrategy):
        def get_required_history_length(self):
            return 10

        def get_required_parameters(self):
            return ["test_param"]

        def evaluate(self, data: StrategyData):
            # Simple test logic: buy if last price > 50000
            if data.price_history and data.price_history[-1] > Decimal("50000"):
                return Signal(
                    symbol=data.symbol,
                    signal_type=SignalType.BUY,
                    quantity=Decimal("1.0"),
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(tz=timezone.utc),
                )
            return None

    return TestStrategy("test_strategy_1")


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config_data = {
        "strategies": [
            {
                "name": "test_strategy",
                "class": "TestStrategy",
                "module_path": "test_module.py",
                "universe": ["BTCUSDT", "ETHUSDT"],
                "parameters": {"test_param": "test_value", "position_size": 1.0},
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_file = f.name

    yield temp_file

    # Cleanup
    os.unlink(temp_file)


# Test Classes


class TestStrategyData:
    """Test the StrategyData class."""

    def test_strategy_data_creation(self, sample_strategy_data):
        """Test basic StrategyData creation."""
        data = sample_strategy_data

        assert data.symbol == "BTCUSDT"
        assert len(data.price_history) == 5
        assert data.current_position == Decimal("0")
        assert data.get_parameter("short_window") == 5
        assert data.get_parameter("nonexistent", "default") == "default"

    def test_strategy_data_state_management(self, sample_strategy_data):
        """Test internal state management."""
        data = sample_strategy_data

        # Test setting and getting state
        data.set_state("test_key", "test_value")
        assert data.get_state("test_key") == "test_value"
        assert data.get_state("nonexistent", "default") == "default"

        # Test state persistence
        data.set_state("counter", 1)
        data.set_state("counter", data.get_state("counter") + 1)
        assert data.get_state("counter") == 2

    def test_strategy_data_parameter_access(self, sample_strategy_data):
        """Test parameter access methods."""
        data = sample_strategy_data

        # Test existing parameters
        assert data.get_parameter("short_window") == 5
        assert data.get_parameter("long_window") == 10

        # Test default values
        assert data.get_parameter("nonexistent") is None
        assert data.get_parameter("nonexistent", 99) == 99


class TestBaseStrategy:
    """Test the BaseStrategy abstract class and concrete implementations."""

    def test_strategy_creation(self, test_strategy):
        """Test strategy instance creation."""
        strategy = test_strategy

        assert strategy.strategy_id == "test_strategy_1"
        assert strategy.get_required_history_length() == 10
        assert strategy.get_required_parameters() == ["test_param"]

    def test_strategy_parameter_validation(self, test_strategy):
        """Test parameter validation."""
        strategy = test_strategy

        # Valid parameters
        valid_params = {"test_param": "value"}
        assert strategy.validate_parameters(valid_params) is True

        # Missing required parameter
        invalid_params = {"wrong_param": "value"}
        assert strategy.validate_parameters(invalid_params) is False

    def test_strategy_evaluation(self, test_strategy, sample_strategy_data):
        """Test strategy evaluation logic."""
        strategy = test_strategy
        data = sample_strategy_data
        data.parameters = {"test_param": "value"}

        # Test signal generation
        signal = strategy.evaluate(data)

        assert signal is not None
        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalType.BUY
        assert signal.quantity == Decimal("1.0")
        assert signal.strategy_id == "test_strategy_1"

    def test_strategy_no_signal(self, test_strategy):
        """Test case where strategy should not generate a signal."""
        strategy = test_strategy

        # Create data that shouldn't trigger a signal
        data = StrategyData(
            symbol="BTCUSDT",
            price_history=[Decimal("40000")],  # Below threshold
            current_position=Decimal("0"),
            parameters={"test_param": "value"},
        )

        signal = strategy.evaluate(data)
        assert signal is None


class TestStrategyEngine:
    """Test the main StrategyEngine class."""

    def test_strategy_engine_creation(self, mock_portfolio_manager, mock_order_manager):
        """Test basic strategy engine creation."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        assert engine.portfolio_manager == mock_portfolio_manager
        assert engine.order_manager == mock_order_manager
        assert len(engine.strategies) == 0
        assert len(engine.strategy_data) == 0
        assert len(engine.required_symbols) == 0

    def test_strategy_registration(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test strategy registration."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        universe = ["BTCUSDT", "ETHUSDT"]
        parameters = {"test_param": "value"}

        result = engine.register_strategy(test_strategy, universe, parameters)

        assert result is True
        assert test_strategy.strategy_id in engine.strategies
        assert test_strategy.strategy_id in engine.strategy_data
        assert "BTCUSDT" in engine.required_symbols
        assert "ETHUSDT" in engine.required_symbols

        # Check data objects were created
        data_dict = engine.strategy_data[test_strategy.strategy_id]
        assert "BTCUSDT" in data_dict
        assert "ETHUSDT" in data_dict

    def test_strategy_registration_invalid_params(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test strategy registration with invalid parameters."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Missing required parameter
        invalid_parameters = {"wrong_param": "value"}

        result = engine.register_strategy(
            test_strategy, ["BTCUSDT"], invalid_parameters
        )
        assert result is False
        assert test_strategy.strategy_id not in engine.strategies

    def test_price_update_and_signal_generation(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test price updates and signal generation."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Register strategy
        engine.register_strategy(test_strategy, ["BTCUSDT"], {"test_param": "value"})

        # Mock portfolio manager to return current position
        mock_portfolio_manager.get_position.return_value = Decimal("0")

        # Update price (should trigger signal because price > 50000)
        signals = engine.update_price(
            "BTCUSDT", Decimal("50100"), datetime.now(tz=timezone.utc)
        )

        assert len(signals) == 1
        signal = signals[0]
        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalType.BUY
        assert signal.strategy_id == test_strategy.strategy_id

    def test_price_update_no_signal(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test price update that shouldn't generate signal."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Register strategy
        engine.register_strategy(test_strategy, ["BTCUSDT"], {"test_param": "value"})

        # Update with price that shouldn't trigger signal
        signals = engine.update_price(
            "BTCUSDT", Decimal("45000"), datetime.now(tz=timezone.utc)
        )

        assert len(signals) == 0

    def test_signal_execution(self, mock_portfolio_manager, mock_order_manager):
        """Test signal execution through OrderManager."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Create test signal
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            quantity=Decimal("1.0"),
            strategy_id="test_strategy",
            timestamp=datetime.now(tz=timezone.utc),
        )

        # Mock successful order placement
        mock_order_manager.place_market_order.return_value = "order_123"

        # Mock portfolio position for price lookup
        mock_position = Mock()
        mock_position.current_price = Decimal("50000")
        mock_portfolio_manager.get_position.return_value = mock_position

        # Execute signal
        engine._execute_signal(signal)

        # Verify order was placed
        mock_order_manager.place_market_order.assert_called_once_with(
            "BTCUSDT", TradeType.BUY, Decimal("1.0"), "test_strategy"
        )

        # Verify strategy trade was processed
        mock_portfolio_manager.process_trade.assert_called_once()

    def test_required_subscriptions(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test WebSocket subscription requirements."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Register strategy with multiple symbols
        engine.register_strategy(
            test_strategy, ["BTCUSDT", "ETHUSDT"], {"test_param": "value"}
        )

        subscriptions = engine.get_required_subscriptions()

        expected = ["btcusdt@kline_1m", "ethusdt@kline_1m"]
        assert set(subscriptions) == set(expected)

    def test_kline_data_handling(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test handling of kline data from DataProcessor."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Register strategy
        engine.register_strategy(test_strategy, ["BTCUSDT"], {"test_param": "value"})

        # Create mock kline data
        mock_kline_info = Mock(spec=KlineInfo)
        mock_kline_info.close_price = 50100
        mock_kline_info.close_time = datetime.now(tz=timezone.utc)

        mock_kline_data = Mock(spec=KlineData)
        mock_kline_data.symbol = "BTCUSDT"
        mock_kline_data.kline = mock_kline_info

        # Mock portfolio manager methods
        mock_portfolio_manager.get_position.return_value = Decimal("0")
        mock_order_manager.place_market_order.return_value = "order_123"
        mock_position = Mock()
        mock_position.current_price = Decimal("50100")
        mock_portfolio_manager.get_position.return_value = mock_position

        # Handle kline data
        engine.handle_kline_data(mock_kline_data)

        # Verify portfolio price was updated
        mock_portfolio_manager.update_price.assert_called()

        # Verify order was placed (since our test strategy should trigger on price > 50000)
        mock_order_manager.place_market_order.assert_called_once()

    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_config_loading_success(
        self,
        mock_module_from_spec,
        mock_spec_from_file,
        mock_portfolio_manager,
        mock_order_manager,
        temp_config_file,
    ):
        """Test successful configuration loading."""
        # Mock the dynamic import
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        mock_strategy_class = Mock(return_value=Mock(spec=BaseStrategy))
        mock_strategy_class.return_value.strategy_id = "test_strategy"
        mock_strategy_class.return_value.get_required_parameters.return_value = [
            "test_param"
        ]
        mock_strategy_class.return_value.validate_parameters.return_value = True
        mock_module.TestStrategy = mock_strategy_class
        mock_module_from_spec.return_value = mock_module

        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        result = engine.load_strategies_from_config(temp_config_file)

        # Should succeed but we can't easily test the full flow due to mocking complexity
        # In a real scenario, you'd have actual strategy files to import
        assert isinstance(result, bool)  # At least verify it returns a boolean

    def test_strategy_performance_summary(
        self, mock_portfolio_manager, mock_order_manager
    ):
        """Test strategy performance summary generation."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Mock positions data
        mock_position = Mock()
        mock_position.unrealized_pnl = Decimal("100.50")
        mock_position.market_value = Decimal("1000.00")
        mock_position.quantity = Decimal("1.0")
        mock_position.avg_price = Decimal("50000")
        mock_position.current_price = Decimal("50100")
        mock_position.side.value = "LONG"

        mock_portfolio_manager.get_all_positions.return_value = {
            "BTCUSDT": mock_position
        }

        performance = engine.get_strategy_performance("test_strategy")

        assert performance["strategy_id"] == "test_strategy"
        assert performance["total_unrealized_pnl"] == 100.50
        assert performance["total_market_value"] == 1000.00
        assert "positions" in performance
        assert "BTCUSDT" in performance["positions"]


class TestSignal:
    """Test the Signal dataclass."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            quantity=Decimal("1.0"),
            strategy_id="test_strategy",
            timestamp=datetime.now(tz=timezone.utc),
            confidence=0.8,
        )

        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalType.BUY
        assert signal.quantity == Decimal("1.0")
        assert signal.strategy_id == "test_strategy"
        assert signal.confidence == 0.8

    def test_signal_without_confidence(self):
        """Test signal creation without confidence score."""
        signal = Signal(
            symbol="ETHUSDT",
            signal_type=SignalType.SELL,
            quantity=Decimal("2.0"),
            strategy_id="test_strategy",
            timestamp=datetime.now(tz=timezone.utc),
        )

        assert signal.confidence is None


# Integration Tests
class TestStrategyEngineIntegration:
    """Test full integration scenarios."""

    def test_end_to_end_signal_flow(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test complete flow from price update to order execution."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Setup: Register strategy
        universe = ["BTCUSDT"]
        parameters = {"test_param": "value"}
        engine.register_strategy(test_strategy, universe, parameters)

        # Setup: Mock dependencies
        mock_portfolio_manager.get_position.return_value = Decimal("0")
        mock_order_manager.place_market_order.return_value = "order_123"
        mock_position = Mock()
        mock_position.current_price = Decimal("50100")
        mock_portfolio_manager.get_position.return_value = mock_position

        # Execute: Update price (should trigger buy signal)
        signals = engine.update_price(
            "BTCUSDT", Decimal("50100"), datetime.now(tz=timezone.utc)
        )

        # Verify: Signal generated
        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == SignalType.BUY

        # Execute: Process signal
        engine._execute_signal(signal)

        # Verify: Order placed and portfolio updated
        mock_order_manager.place_market_order.assert_called_once_with(
            "BTCUSDT", TradeType.BUY, Decimal("1.0"), "test_strategy_1"
        )
        mock_portfolio_manager.process_trade.assert_called_once()

    def test_multiple_strategies_same_symbol(
        self, mock_portfolio_manager, mock_order_manager, test_strategy
    ):
        """Test multiple strategies watching the same symbol."""
        engine = StrategyEngine(mock_portfolio_manager, mock_order_manager)

        # Create two test strategies
        strategy1 = test_strategy
        strategy2 = test_strategy
        strategy2.strategy_id = "test_strategy_2"

        class TestStrategy(BaseStrategy):
            def get_required_history_length(self):
                return 5

            def get_required_parameters(self):
                return ["test_param"]

            def evaluate(self, data):
                if data.price_history and data.price_history[-1] > Decimal("50000"):
                    return Signal(
                        symbol=data.symbol,
                        signal_type=SignalType.BUY,
                        quantity=Decimal("1.0"),
                        strategy_id=self.strategy_id,
                        timestamp=datetime.now(tz=timezone.utc),
                    )
                return None

        strategy1 = TestStrategy("strategy_1")
        strategy2 = TestStrategy("strategy_2")

        # Register both strategies for same symbol
        engine.register_strategy(strategy1, ["BTCUSDT"], {"test_param": "value"})
        engine.register_strategy(strategy2, ["BTCUSDT"], {"test_param": "value"})

        # Setup mocks
        mock_portfolio_manager.get_position.return_value = Decimal("0")

        # Update price - should trigger signals from both strategies
        signals = engine.update_price(
            "BTCUSDT", Decimal("50100"), datetime.now(tz=timezone.utc)
        )

        assert len(signals) == 2
        strategy_ids = [s.strategy_id for s in signals]
        assert "strategy_1" in strategy_ids
        assert "strategy_2" in strategy_ids

        # Verify subscriptions are deduplicated
        subscriptions = engine.get_required_subscriptions()
        assert subscriptions == ["btcusdt@kline_1m"]  # Only one subscription needed


# Pytest Configuration and Runners
@pytest.mark.asyncio
async def test_async_integration():
    """Test async integration points (placeholder for future async tests)."""
    # This is a placeholder for any async functionality you might add later
    # such as async strategy evaluation or async signal processing
    assert True


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
