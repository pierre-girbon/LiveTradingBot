"""
Strategy Engine Integration with Existing Trading Bot Components

This shows how the strategy engine integrates with your existing:
- PortfolioManager (extended to support strategies)
- OrderManager
- DataProcessor/WebSocketClient
- YAML configuration

## Key Components Built
- BaseStrategy - Abstract base class for all strategies
- StrategyData - Manages state/history/parameters for each strategy instance
- StrategyEngine - Plugin orchestrator that loads and manages strategies
- StrategyPortfolioManager - Extended portfolio manager with per-strategy isolation
- YAML Configuration - Clean, declarative strategy configuration
- Dynamic Loading - Loads strategy classes from Python files
"""

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import yaml

from modules.dataprocessor import KlineData
from modules.logger import get_logger
from modules.order_manager import OrderManager
from modules.portfolio_manager import (PortfolioManager, PositionInfo,
                                       TradeEvent, TradeType)


# Strategy Framework (from previous artifact, refined)
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Signal:
    symbol: str
    signal_type: SignalType
    quantity: Decimal
    strategy_id: str
    timestamp: datetime
    confidence: Optional[float] = None


@dataclass
class StrategyData:
    """Store Strategy State

    Strategy classes process the StrategyData to evaluate conditions and produce a signal
    """

    symbol: str
    """Asset symbol"""
    price_history: List[Decimal]
    """Price history"""
    current_position: Decimal
    parameters: Dict[str, Any]
    """Strategy parameters
    
    Populated with the config files
    """
    internal_state: Dict[str, Any] = field(default_factory=dict)
    """Strategy state

    Used to store data from one evaluation to the other
    """

    def get_parameter(self, key: str, default=None) -> Optional[Any]:
        return self.parameters.get(key, default)

    def set_state(self, key: str, value: Any):
        self.internal_state[key] = value

    def get_state(self, key: str, default=None) -> Optional[Any]:
        return self.internal_state.get(key, default)


class BaseStrategy(ABC):
    """Base Strategy Class

    Strategies classes must be inhireted from this abstract class
    """

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        """Strategy id as string"""
        self.logger = get_logger(__name__)

    @abstractmethod
    def get_required_history_length(self) -> int:
        pass

    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, data: StrategyData) -> Optional[Signal]:
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Valide the presence of required parameters in the parameters Dictionnary"""
        required = self.get_required_parameters()
        return all(param in parameters for param in required)


# Extended Portfolio Manager for Strategy Support
class StrategyPortfolioManager(PortfolioManager):
    """
    Extended PortfolioManager that tracks positions by strategy_id.
    Each strategy has its own isolated portfolio.
    """

    def __init__(self, db_url: Optional[str] = None):
        super().__init__(db_url)
        self.strategy_positions: Dict[str, Dict[str, PositionInfo]] = {}
        """Dictionnary of strategies positions
        - Dict[strategy_id, Dict[symbol, PositionInfo]]
        """
        # strategy_id -> symbol -> PositionInfo

    def get_strategy_position(
        self, strategy_id: str, symbol: str
    ) -> Optional[PositionInfo]:
        """Get position for a specific strategy and symbol."""
        return self.strategy_positions.get(strategy_id, {}).get(symbol)

    def get_strategy_positions(self, strategy_id: str) -> Dict[str, PositionInfo]:
        """Get all positions for a strategy."""
        return self.strategy_positions.get(strategy_id, {}).copy()

    # TODO: rewrite to avoid duplicate position update
    def process_strategy_trade(self, strategy_id: str, trade: TradeEvent) -> bool:
        """Process a trade for a specific strategy."""
        # Process trade in main portfolio (for overall tracking)
        success = self.process_trade(trade)
        if not success:
            return False

        # Track in strategy-specific portfolio
        if strategy_id not in self.strategy_positions:
            self.strategy_positions[strategy_id] = {}

        symbol = trade.symbol
        current_pos = self.strategy_positions[strategy_id].get(symbol)

        # Convert trade to signed quantity
        signed_quantity = (
            trade.quantity if trade.trade_type == TradeType.BUY else -trade.quantity
        )

        if current_pos is None:
            # New position for this strategy
            from modules.portfolio_manager import PositionSide

            self.strategy_positions[strategy_id][symbol] = PositionInfo(
                symbol=symbol,
                quantity=signed_quantity,
                avg_price=trade.price,
                current_price=trade.price,
                unrealized_pnl=Decimal("0"),
                side=self._determine_side(signed_quantity),
                last_updated=trade.timestamp,
            )
        else:
            # Update existing position (similar logic to parent class)
            new_quantity = current_pos.quantity + signed_quantity

            if new_quantity == 0:
                current_pos.quantity = Decimal("0")
                current_pos.side = self._determine_side(Decimal("0"))
                current_pos.unrealized_pnl = Decimal("0")
            elif (current_pos.quantity >= 0 and signed_quantity >= 0) or (
                current_pos.quantity <= 0 and signed_quantity <= 0
            ):
                # Adding to position - recalculate weighted average
                total_cost = (current_pos.quantity * current_pos.avg_price) + (
                    signed_quantity * trade.price
                )
                current_pos.avg_price = total_cost / new_quantity
                current_pos.quantity = new_quantity
                current_pos.side = self._determine_side(new_quantity)
            else:
                # Reducing position
                current_pos.quantity = new_quantity
                current_pos.side = self._determine_side(new_quantity)

            current_pos.last_updated = trade.timestamp

        # Update unrealized P&L
        pos = self.strategy_positions[strategy_id][symbol]
        pos.unrealized_pnl = self._calculate_unrealized_pnl(pos)

        self.logger.info(
            f"Strategy trade processed: {strategy_id} - {symbol} {trade.trade_type.value} {trade.quantity}",
            strategy_position=float(pos.quantity),
            strategy_pnl=float(pos.unrealized_pnl),
        )

        return True

    def update_strategy_price(
        self, strategy_id: str, symbol: str, price: Decimal, timestamp: datetime
    ):
        """Update price for a strategy's position."""
        if (
            strategy_id in self.strategy_positions
            and symbol in self.strategy_positions[strategy_id]
        ):
            pos = self.strategy_positions[strategy_id][symbol]
            pos.current_price = price
            pos.unrealized_pnl = self._calculate_unrealized_pnl(pos)
            pos.last_updated = timestamp

    def get_strategy_quantity(self, strategy_id: str, symbol: str) -> Decimal:
        """Get current position quantity for a strategy/symbol."""
        pos = self.get_strategy_position(strategy_id, symbol)
        return pos.quantity if pos else Decimal("0")


# Strategy Engine with Full Integration
class StrategyEngine:
    """
    Complete strategy engine that integrates with all existing components.
    """

    def __init__(
        self, portfolio_manager: StrategyPortfolioManager, order_manager: OrderManager
    ):
        self.logger = get_logger(__name__)
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager

        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        """Dictionnary of strategies
        - Dict[strategy_id, strategy class]
        """
        self.strategy_data: Dict[str, Dict[str, StrategyData]] = {}
        """StrategyData by symbol and strategy_id
        - Dict[strategy_id, Dict[symbol, StrategyData]]
        """
        self.strategy_configs: Dict[str, Dict] = {}  # Store original config
        """Store original config"""

        # Track which symbols we need for WebSocket subscriptions
        self.required_symbols: Set[str] = set()
        """Track required symbols for WebSocket subscription"""

    def load_strategies_from_config(self, config_path: str) -> bool:
        """
        Load strategies from YAML configuration file.

        **Expected format (YAML):**
        ```YAML
        strategies:
          - name: "ma_btc_strategy"
            class: "MovingAverageCrossover"
            module_path: "strategies/moving_average.py"
            universe: ["BTCUSDT"]
            parameters:
              short_window: 10
              long_window: 20
              position_size: 1.0
        ```
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            for strategy_config in config.get("strategies", []):
                success = self._load_strategy_from_config(strategy_config)
                if not success:
                    self.logger.error(
                        f"Failed to load strategy: {strategy_config.get('name', 'unknown')}"
                    )
                    return False

            self.logger.info(f"Loaded {len(self.strategies)} strategies from config")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return False

    def _load_strategy_from_config(self, config: Dict) -> bool:
        """Load a single strategy from config."""
        try:
            name = config["name"]
            class_name = config["class"]
            module_path = config.get("module_path")
            universe = config["universe"]
            parameters = config["parameters"]

            # Dynamically import strategy class
            if module_path:
                spec = importlib.util.spec_from_file_location(class_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                StrategyClass = getattr(module, class_name)
            else:
                # Try to import from a strategies package
                module = importlib.import_module(f"strategies.{class_name.lower()}")
                StrategyClass = getattr(module, class_name)

            # Create strategy instance
            strategy = StrategyClass(name)

            # Register strategy
            return self.register_strategy(strategy, universe, parameters)

        except Exception as e:
            self.logger.error(f"Failed to load strategy from config: {e}")
            return False

    def register_strategy(
        self, strategy: BaseStrategy, universe: List[str], parameters: Dict[str, Any]
    ) -> bool:
        """Register a strategy with the engine."""
        # Validate parameters
        if not strategy.validate_parameters(parameters):
            self.logger.error(f"Invalid parameters for strategy {strategy.strategy_id}")
            return False

        # Store strategy
        self.strategies[strategy.strategy_id] = strategy
        self.strategy_data[strategy.strategy_id] = {}
        self.strategy_configs[strategy.strategy_id] = {
            "universe": universe,
            "parameters": parameters,
        }

        # Add symbols to required set for WebSocket subscriptions
        self.required_symbols.update(universe)

        # Create data objects for each symbol
        for symbol in universe:
            self.strategy_data[strategy.strategy_id][symbol] = StrategyData(
                symbol=symbol,
                price_history=[],
                current_position=Decimal("0"),
                parameters=parameters.copy(),
            )

        self.logger.info(
            f"Registered strategy: {strategy.strategy_id} watching {universe}"
        )
        return True

    def get_required_subscriptions(self) -> List[str]:
        """Get WebSocket subscriptions needed for all strategies."""
        # Convert symbols to kline subscriptions (1-minute intervals)
        subscriptions = []
        for symbol in self.required_symbols:
            subscriptions.append(f"{symbol.lower()}@kline_1m")
        return subscriptions

    def handle_kline_data(self, kline_data: KlineData):
        """
        Handler for kline data from DataProcessor.
        This gets called by DataProcessor when new kline data arrives.
        """
        symbol = kline_data.symbol
        close_price = Decimal(str(kline_data.kline.close_price))
        timestamp = kline_data.kline.close_time

        # Update portfolio prices
        self.portfolio_manager.update_price(symbol, close_price, timestamp)

        # Process for all strategies
        signals = self.update_price(symbol, close_price, timestamp)

        # Execute signals
        for signal in signals:
            self._execute_signal(signal)

    def update_price(
        self, symbol: str, price: Decimal, timestamp: datetime
    ) -> List[Signal]:
        """Update price and evaluate strategies."""
        signals = []

        for strategy_id, symbol_data_dict in self.strategy_data.items():
            if symbol in symbol_data_dict:
                data = symbol_data_dict[symbol]

                # Update price history
                data.price_history.append(price)

                # Trim history to required length
                strategy = self.strategies[strategy_id]
                max_history = strategy.get_required_history_length() + 10
                if len(data.price_history) > max_history:
                    data.price_history = data.price_history[-max_history:]

                # Update current position from portfolio
                data.current_position = self.portfolio_manager.get_strategy_quantity(
                    strategy_id, symbol
                )

                # Update strategy portfolio price
                self.portfolio_manager.update_strategy_price(
                    strategy_id, symbol, price, timestamp
                )

                # Evaluate strategy
                signal = strategy.evaluate(data)
                if signal:
                    signals.append(signal)
                    self.logger.info("Received signal", signal=signal)

        return signals

    def _execute_signal(self, signal: Signal):
        """Execute a signal using OrderManager."""
        try:
            # Convert signal to TradeType
            trade_type = (
                TradeType.BUY
                if signal.signal_type == SignalType.BUY
                else TradeType.SELL
            )

            # Place market order
            order_id = self.order_manager.place_market_order(
                signal.symbol, trade_type, signal.quantity
            )

            if order_id:
                self.logger.info(
                    f"Executed signal: {signal.strategy_id} - {signal.signal_type.value} "
                    f"{signal.quantity} {signal.symbol}",
                    order_id=order_id,
                )

                # Create trade event for strategy portfolio tracking
                # Note: In real implementation, this should come from order fill confirmation
                trade = TradeEvent(
                    symbol=signal.symbol,
                    trade_type=trade_type,
                    quantity=signal.quantity,
                    price=self.portfolio_manager.get_position(
                        signal.symbol
                    ).current_price,  # Approximate
                    timestamp=signal.timestamp,
                    trade_id=f"signal_{order_id}",
                )

                # Update strategy portfolio
                self.portfolio_manager.process_strategy_trade(signal.strategy_id, trade)

            else:
                self.logger.error(f"Failed to execute signal: {signal}")

        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")

    def get_strategy_performance(self, strategy_id: str) -> Dict:
        """Get performance summary for a strategy."""
        positions = self.portfolio_manager.get_strategy_positions(strategy_id)

        total_pnl = Decimal("0")
        total_value = Decimal("0")

        for pos in positions.values():
            total_pnl += pos.unrealized_pnl
            total_value += pos.market_value

        return {
            "strategy_id": strategy_id,
            "total_unrealized_pnl": float(total_pnl),
            "total_market_value": float(total_value),
            "positions": {
                symbol: {
                    "quantity": float(pos.quantity),
                    "avg_price": float(pos.avg_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "side": pos.side.value,
                }
                for symbol, pos in positions.items()
            },
        }


# Configuration Management
# What it is used for?
@dataclass
class StrategyEngineConfig:
    """Configuration container for the strategy engine."""

    config_path: str
    strategies_dir: str = "strategies"

    @classmethod
    def from_yaml(cls, config_path: str):
        """Create config from YAML file."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Example YAML Configuration Structure
EXAMPLE_CONFIG = """
# strategies_config.yaml
strategies:
  - name: "ma_btc_strategy"
    class: "MovingAverageCrossover"
    module_path: "strategies/moving_average.py"
    universe: ["BTCUSDT"]
    parameters:
      short_window: 10
      long_window: 20
      position_size: 1.0
      
  - name: "ma_eth_strategy"  
    class: "MovingAverageCrossover"
    module_path: "strategies/moving_average.py"
    universe: ["ETHUSDT"]
    parameters:
      short_window: 5
      long_window: 15
      position_size: 2.0
      
  - name: "momentum_multi"
    class: "MomentumStrategy"
    module_path: "strategies/momentum.py"
    universe: ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    parameters:
      lookback_period: 14
      threshold: 0.05
      position_size: 0.5
"""

if __name__ == "__main__":
    print("Strategy Engine Integration Example")
    print("\nExample YAML Config:")
    print(EXAMPLE_CONFIG)
