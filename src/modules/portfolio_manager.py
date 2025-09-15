"""
Strategy-First Portfolio Manager

This design treats strategy_id as a required parameter for all operations,
eliminating complex aggregation logic and optimizing for the primary use case:
strategy-based trading with occasional reporting needs.

Key principles:
1. Every position belongs to a strategy (use "manual" for non-strategy trades)
2. No runtime aggregation - positions are kept separate by design
3. Reporting aggregation handled by dedicated methods
4. Optimized for high-frequency strategy operations
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from enum import Enum
from typing import Dict, Optional

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from modules.dataprocessor import KlineData
from modules.logger import get_logger

# load environement config
load_dotenv()

# Set decimal precision for financial calculations
getcontext().prec = 28

# Database setup
#########################################
Base = declarative_base()


class Position(Base):
    """SQLAlchemy model for position storage"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    strategy_id = Column(String(50), nullable=False)  # REQUIRED - no nullable positions
    quantity = Column(Float, nullable=False, default=0.0)
    avg_price = Column(Float, nullable=False, default=0.0)
    current_price = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime, default=datetime.now(tz=timezone.utc))

    # Unique constraint: one position per (symbol, strategy) pair
    __table_args__ = {"sqlite_autoincrement": True}


class Trade(Base):
    """SQLAlchemy model for trade history storage"""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    strategy_id = Column(String(50), nullable=False)  # REQUIRED - no nullable trades
    trade_type = Column(String(10), nullable=False)  # BUY or SELL
    quantity = Column(Float, nullable=False)  # Always positive
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    trade_id = Column(String(50), nullable=True)


# Positions Dataclass
######################
class PositionSide(Enum):
    """Position side enumeration"""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"  # No position


@dataclass
class PositionInfo:
    """Data class for position information"""

    symbol: str
    strategy_id: str  # REQUIRED - every position belongs to a strategy
    quantity: Decimal
    """Position quantity.
    - Positive for a long position
    - Negative for a short position
    - 0 for a flat position"""
    avg_price: Decimal
    """Average cost of acquisition of asset

    E.g. Buy x asset @ P1 and y asset @ P2 -> average price = (x*P1+y*P2)/(x+y)"""
    current_price: Decimal
    """Current price of asset"""
    unrealized_pnl: Decimal
    """
    - For long positions: (current_price - avg_price) * quantity
    - For short positions: (avg_price - current_price) * |quantity|
    - 0 otherwise
    """
    side: PositionSide
    """LONG/SHORT/FLAT"""
    last_updated: datetime

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of the position

        |quantity| * current_price
        """
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis of the position:

        |quantity| * average_price
        """
        return abs(self.quantity) * self.avg_price


# Trades Dataclass
#############################
class TradeType(Enum):
    """Trade type enumeration"""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradeEvent:
    """Represents a trade event (buy/sell)"""

    symbol: str
    strategy_id: str  # REQUIRED - every trade belongs to a strategy
    trade_type: TradeType  # BUY or SELL
    """BUY/SELL"""
    quantity: Decimal  # Always positive
    """Always positive"""
    price: Decimal
    timestamp: datetime
    trade_id: Optional[str] = None


# Strategy-First Portfolio Manager
##################################
class PortfolioManager:
    """
    Strategy-first portfolio manager optimized for strategy-based trading.

    Features:
    - All positions must belong to a strategy
    - Fast strategy-specific operations
    - Separate reporting methods for aggregation
    - No runtime aggregation overhead
    """

    # Default strategy for manual/non-strategy trades
    MANUAL_STRATEGY = "manual"

    def __init__(
        self,
        db_url: Optional[str] = None,
    ):
        """Portfolio init
        - Load logger
        - Setup database
        - load position from database into memory
        """
        self.logger = get_logger(__name__)

        # Database setup
        self.engine = create_engine(
            db_url if db_url else os.environ.get("DB_URL", "sqlite:///tradebot.db")
        )
        """Database Engine"""
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        """Database Session"""

        # In-memory position cache for fast access
        # Structure: {(symbol, strategy_id): PositionInfo}
        self.positions: Dict[tuple, PositionInfo] = {}
        """In memory dictionary of positions
        - Key: (symbol, strategy_id) tuple
        - Value: PositionInfo
        """

        # Load existing positions from database
        self._load_positions_from_db()

        self.logger.info("PortfolioManager initialized")

    def _load_positions_from_db(self):
        """Load existing positions from database into memory"""
        db_positions = self.session.query(Position).all()

        for pos in db_positions:
            key = (pos.symbol, pos.strategy_id)
            self.positions[key] = PositionInfo(
                symbol=pos.symbol,
                strategy_id=pos.strategy_id,
                quantity=Decimal(str(pos.quantity)),
                avg_price=Decimal(str(pos.avg_price)),
                current_price=Decimal(str(pos.current_price)),
                unrealized_pnl=Decimal("0"),  # Will be calculated
                side=self._determine_side(Decimal(str(pos.quantity))),
                last_updated=pos.last_updated,
            )

        self.logger.info(f"Loaded {len(self.positions)} positions from database")

    def _get_position_key(self, symbol: str, strategy_id: str) -> tuple:
        """Get the key for position lookup"""
        return (symbol, strategy_id)

    def _determine_side(self, quantity: Decimal) -> PositionSide:
        """Determine position side based on quantity"""
        if quantity > 0:
            return PositionSide.LONG
        elif quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT

    def _calculate_unrealized_pnl(self, position: PositionInfo) -> Decimal:
        """Calculate unrealized P&L for a position"""
        if position.side == PositionSide.FLAT:
            return Decimal("0")

        # For long positions: (current_price - avg_price) * quantity
        # For short positions: (avg_price - current_price) * |quantity|
        if position.side == PositionSide.LONG:
            return (position.current_price - position.avg_price) * position.quantity
        else:  # SHORT
            return (position.avg_price - position.current_price) * abs(
                position.quantity
            )

    def process_trade(
        self, trade: TradeEvent, strategy_id: Optional[str] = None
    ) -> bool:
        """
        Process a trade event and update position.

        **Args:**
        - trade: TradeEvent containing trade information
        - strategy_id: Optional strategy identifier (overrides trade.strategy_id if provided)

        **Returns:**
        - bool: True if trade processed successfully

        **Raises:**
        - ValueError: If trade data is invalid
        - TypeError: If trade is not a TradeEvent instance

        **Note:** Every trade must have a strategy_id. If none provided, uses trade.strategy_id or MANUAL_STRATEGY
        """
        try:
            # Input validation
            if not isinstance(trade, TradeEvent):
                raise TypeError("trade must be a TradeEvent instance")

            if trade.quantity <= 0:
                raise ValueError("Trade quantity must be positive")

            if trade.price <= 0:
                raise ValueError("Trade price must be positive")

            if not trade.symbol or not isinstance(trade.symbol, str):
                raise ValueError("Trade symbol must be a non-empty string")

            # Determine effective strategy_id
            if strategy_id is not None:
                effective_strategy_id = strategy_id
            elif trade.strategy_id:
                effective_strategy_id = trade.strategy_id
            else:
                effective_strategy_id = self.MANUAL_STRATEGY

            symbol = trade.symbol
            position_key = self._get_position_key(symbol, effective_strategy_id)
            current_pos = self.positions.get(position_key)

            self.logger.debug(
                symbol=symbol,
                strategy_id=effective_strategy_id,
                current_pos=current_pos,
            )

            # Convert trade to signed quantity based on trade type
            signed_quantity = (
                trade.quantity if trade.trade_type == TradeType.BUY else -trade.quantity
            )

            if current_pos is None:
                # New position
                self.positions[position_key] = PositionInfo(
                    symbol=symbol,
                    strategy_id=effective_strategy_id,
                    quantity=signed_quantity,
                    avg_price=trade.price,
                    current_price=trade.price,  # Will be updated by price handler
                    unrealized_pnl=Decimal("0"),
                    side=self._determine_side(signed_quantity),
                    last_updated=trade.timestamp,
                )
            else:
                # Update existing position
                new_quantity = current_pos.quantity + signed_quantity

                if new_quantity == 0:
                    # Position closed
                    self.positions[position_key].quantity = Decimal("0")
                    self.positions[position_key].side = PositionSide.FLAT
                    self.positions[position_key].unrealized_pnl = Decimal("0")
                elif (current_pos.quantity >= 0 and signed_quantity >= 0) or (
                    current_pos.quantity <= 0 and signed_quantity <= 0
                ):
                    # Adding to existing position - recalculate weighted average
                    total_cost = (current_pos.quantity * current_pos.avg_price) + (
                        signed_quantity * trade.price
                    )
                    new_avg_price = total_cost / new_quantity

                    self.positions[position_key].quantity = new_quantity
                    self.positions[position_key].avg_price = new_avg_price
                    self.positions[position_key].side = self._determine_side(
                        new_quantity
                    )
                else:
                    # Reducing/closing position - average price stays the same
                    self.positions[position_key].quantity = new_quantity
                    self.positions[position_key].side = self._determine_side(
                        new_quantity
                    )

                self.positions[position_key].last_updated = trade.timestamp

            # Recalculate P&L
            self.positions[position_key].unrealized_pnl = (
                self._calculate_unrealized_pnl(self.positions[position_key])
            )

            # Persist to database
            trade_to_save = TradeEvent(
                symbol=trade.symbol,
                strategy_id=effective_strategy_id,
                trade_type=trade.trade_type,
                quantity=trade.quantity,
                price=trade.price,
                timestamp=trade.timestamp,
                trade_id=trade.trade_id,
            )

            self._save_trade_to_db(trade_to_save)
            self._save_position_to_db(self.positions[position_key])

            self.logger.info(
                f"Processed trade: {symbol} ({effective_strategy_id}) {trade.trade_type.value} {trade.quantity} @ {trade.price}",
                position_qty=float(self.positions[position_key].quantity),
                avg_price=float(self.positions[position_key].avg_price),
                unrealized_pnl=float(self.positions[position_key].unrealized_pnl),
            )

            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid trade data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to process trade: {e}")
            return False

    def update_price(
        self,
        symbol: str,
        price: Decimal,
        timestamp: datetime,
        strategy_id: Optional[str] = None,
    ) -> bool:
        """
        Update current price for positions.

        **Args:**
        - symbol: Trading symbol
        - price: Current price
        - timestamp: Price timestamp
        - strategy_id: Optional strategy identifier. If provided, updates only that strategy.
                      If None, updates ALL positions for the symbol across all strategies.

        **Returns:**
        - bool: True if price updated successfully
        """
        try:
            # Input Validation
            if price <= 0:
                raise ValueError("Trade price must be positive")

            if not symbol or not isinstance(symbol, str):
                raise ValueError("Trade symbol must be a non-empty string")

            updated_count = 0

            if strategy_id is not None:
                # Update specific strategy position
                position_key = self._get_position_key(symbol, strategy_id)
                if position_key in self.positions:
                    self._update_position_price(
                        self.positions[position_key], price, timestamp
                    )
                    self._save_position_to_db(self.positions[position_key])
                    updated_count += 1
            else:
                # Update all positions for the symbol across all strategies
                for key, position in self.positions.items():
                    if key[0] == symbol:  # key[0] is the symbol
                        self._update_position_price(position, price, timestamp)
                        self._save_position_to_db(position)
                        updated_count += 1

            self.logger.debug(
                f"Updated price: {symbol} @ {price}, {updated_count} positions updated"
            )
            return True

        except ValueError as e:
            self.logger.error(f"Invalid data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to update price: {e}")
            return False

    def _update_position_price(
        self, position: PositionInfo, price: Decimal, timestamp: datetime
    ):
        """Helper method to update a single position's price"""
        position.current_price = price
        position.unrealized_pnl = self._calculate_unrealized_pnl(position)
        position.last_updated = timestamp

    def get_position(self, symbol: str, strategy_id: str) -> Optional[PositionInfo]:
        """
        Get current position for a specific symbol and strategy.

        **Args:**
        - symbol: Trading symbol
        - strategy_id: Strategy identifier (required)

        **Returns:**
        - PositionInfo if found, None otherwise
        """
        position_key = self._get_position_key(symbol, strategy_id)
        return self.positions.get(position_key)

    def get_all_positions(
        self, strategy_id: Optional[str] = None
    ) -> Dict[str, PositionInfo]:
        """
        Get all current positions for a strategy.

        **Args:**
        - strategy_id: Optional strategy identifier. If None, returns ALL positions grouped by strategy

        **Returns:**
        - If strategy_id provided: Dictionary mapping symbol to PositionInfo for that strategy
        - If strategy_id is None: Dictionary mapping (symbol, strategy_id) to PositionInfo
        """
        if strategy_id is not None:
            # Return positions for specific strategy, keyed by symbol
            return {
                key[0]: pos
                for key, pos in self.positions.items()
                if key[1] == strategy_id
            }
        else:
            # Return all positions, keyed by (symbol, strategy_id) string
            return {f"{key[0]}_{key[1]}": pos for key, pos in self.positions.items()}

    def get_position_quantity(self, symbol: str, strategy_id: str) -> Decimal:
        """
        Get current position quantity for a symbol and strategy.

        **Args:**
        - symbol: Trading symbol
        - strategy_id: Strategy identifier (required)

        **Returns:**
        - Position quantity
        """
        position = self.get_position(symbol, strategy_id)
        return position.quantity if position else Decimal("0")

    def calculate_available_balance(
        self, symbol: str, strategy_id: str, locked_quantity: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Calculate available balance for trading.

        **Args:**
        - symbol: Trading symbol
        - strategy_id: Strategy identifier (required)
        - locked_quantity: Quantity currently locked in pending orders

        **Returns:**
        - Available quantity that can be sold (for long positions)
        """
        position = self.get_position(symbol, strategy_id)
        if not position or position.side != PositionSide.LONG:
            return Decimal("0")

        return max(Decimal("0"), position.quantity - locked_quantity)

    def get_portfolio_value(self, strategy_id: Optional[str] = None) -> Decimal:
        """
        Calculate total portfolio unrealized P&L.

        **Args:**
        - strategy_id: Optional strategy identifier. If None, calculates across all strategies

        **Returns:**
        - Total unrealized P&L
        """
        total_pnl = Decimal("0")

        if strategy_id is not None:
            # Calculate for specific strategy
            for key, position in self.positions.items():
                if key[1] == strategy_id:
                    total_pnl += position.unrealized_pnl
        else:
            # Calculate across all positions
            for position in self.positions.values():
                total_pnl += position.unrealized_pnl

        return total_pnl

    # REPORTING METHODS (for less frequent aggregation needs)
    # =====================================================

    def get_symbol_summary(self, symbol: str) -> Dict:
        """
        Generate aggregated summary for a symbol across all strategies.

        **Usage:** For reporting and analysis only - not optimized for frequent calls.

        **Returns:**
        - Dictionary with aggregated position info and per-strategy breakdown
        """
        strategy_positions = [
            pos for key, pos in self.positions.items() if key[0] == symbol
        ]

        if not strategy_positions:
            return {
                "symbol": symbol,
                "total_quantity": Decimal("0"),
                "total_market_value": Decimal("0"),
                "total_unrealized_pnl": Decimal("0"),
                "strategies": {},
            }

        # Aggregate totals
        total_quantity = sum(pos.quantity for pos in strategy_positions)
        total_market_value = sum(pos.market_value for pos in strategy_positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in strategy_positions)

        # Calculate weighted average price
        total_cost = sum(pos.quantity * pos.avg_price for pos in strategy_positions)
        weighted_avg_price = (
            total_cost / total_quantity if total_quantity != 0 else Decimal("0")
        )

        return {
            "symbol": symbol,
            "total_quantity": total_quantity,
            "weighted_avg_price": weighted_avg_price,
            "current_price": strategy_positions[0].current_price,
            "total_market_value": total_market_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "strategies": {
                pos.strategy_id: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "side": pos.side.value,
                }
                for pos in strategy_positions
            },
        }

    def get_portfolio_summary(self) -> Dict:
        """
        Generate complete portfolio summary across all symbols and strategies.

        **Usage:** For reporting and analysis only - expensive operation.

        **Returns:**
        - Dictionary with portfolio totals and symbol breakdowns
        """
        symbols = set(key[0] for key in self.positions.keys())

        symbol_summaries = {}
        total_portfolio_pnl = Decimal("0")
        total_portfolio_value = Decimal("0")

        for symbol in symbols:
            summary = self.get_symbol_summary(symbol)
            symbol_summaries[symbol] = summary
            total_portfolio_pnl += summary["total_unrealized_pnl"]
            total_portfolio_value += summary["total_market_value"]

        return {
            "total_unrealized_pnl": total_portfolio_pnl,
            "total_market_value": total_portfolio_value,
            "symbols": symbol_summaries,
            "strategy_count": len(set(key[1] for key in self.positions.keys())),
            "position_count": len(self.positions),
        }

    def get_strategy_summary(self, strategy_id: str) -> Dict:
        """
        Generate summary for a specific strategy across all symbols.

        **Returns:**
        - Dictionary with strategy performance metrics
        """
        strategy_positions = [
            pos for key, pos in self.positions.items() if key[1] == strategy_id
        ]

        if not strategy_positions:
            return {
                "strategy_id": strategy_id,
                "total_unrealized_pnl": Decimal("0"),
                "total_market_value": Decimal("0"),
                "positions": {},
            }

        total_pnl = sum(pos.unrealized_pnl for pos in strategy_positions)
        total_value = sum(pos.market_value for pos in strategy_positions)

        return {
            "strategy_id": strategy_id,
            "total_unrealized_pnl": total_pnl,
            "total_market_value": total_value,
            "positions": {
                pos.symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "side": pos.side.value,
                }
                for pos in strategy_positions
            },
        }

    def _save_trade_to_db(self, trade: TradeEvent):
        """Save trade to database"""
        db_trade = Trade(
            symbol=trade.symbol,
            strategy_id=trade.strategy_id,
            trade_type=trade.trade_type.value,
            quantity=float(trade.quantity),
            price=float(trade.price),
            timestamp=trade.timestamp,
            trade_id=trade.trade_id,
        )

        self.session.add(db_trade)
        self.session.commit()

    def _save_position_to_db(self, position: PositionInfo):
        """Save position to database"""
        db_position = (
            self.session.query(Position)
            .filter_by(symbol=position.symbol, strategy_id=position.strategy_id)
            .first()
        )

        if db_position is None:
            db_position = Position(
                symbol=position.symbol, strategy_id=position.strategy_id
            )
            self.session.add(db_position)

        db_position.quantity = float(position.quantity)
        db_position.avg_price = float(position.avg_price)
        db_position.current_price = float(position.current_price)
        db_position.last_updated = position.last_updated

        self.session.commit()

    # Handler methods for integration with DataProcessor
    def handle_trade_data(self, trade_data):
        """
        Handler for trade data from DataProcessor.

        NOTE: This should NOT be used for position updates.
        Position updates should come from OrderManager after order confirmations.
        This handler is kept for potential future use (e.g., market data analysis).
        """
        # Market data trades are not our trades - don't update positions
        pass

    def handle_kline_data(self, kline_data):
        """Handler for kline data from DataProcessor"""
        try:
            # Input validation
            if not isinstance(kline_data, KlineData):
                raise TypeError

            symbol = kline_data.symbol
            close_price = Decimal(str(kline_data.kline.close_price))
            timestamp = kline_data.kline.close_time

            # Update all positions for this symbol across all strategies
            self.update_price(symbol, close_price, timestamp)
            self.logger.info(
                "Updated price", symbol=symbol, price=close_price, timestamp=timestamp
            )
        except TypeError as e:
            self.logger.error(f"Input is not a KlineData: {e}")
        except Exception as e:
            self.logger.error(f"Failed to handle kline data: {e}")

    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        self.logger.info("PortfolioManager closed")


# Example usage
if __name__ == "__main__":

    # Create portfolio manager
    portfolio = PortfolioManager()

    # Example trades with strategies (strategy_id is required)
    trade1 = TradeEvent(
        symbol="BTCUSDT",
        strategy_id="ma_strategy",
        trade_type=TradeType.BUY,
        quantity=Decimal("10"),
        price=Decimal("50000"),
        timestamp=datetime.now(tz=timezone.utc),
    )

    trade2 = TradeEvent(
        symbol="BTCUSDT",
        strategy_id="momentum_strategy",
        trade_type=TradeType.BUY,
        quantity=Decimal("5"),
        price=Decimal("55000"),
        timestamp=datetime.now(tz=timezone.utc),
    )

    # Process trades
    portfolio.process_trade(trade1)  # MA strategy buys 10 BTC @ 50k
    portfolio.process_trade(trade2)  # Momentum strategy buys 5 BTC @ 55k

    # Update price for all strategies
    portfolio.update_price("BTCUSDT", Decimal("58000"), datetime.now(tz=timezone.utc))

    # Fast strategy-specific operations
    ma_position = portfolio.get_position("BTCUSDT", "ma_strategy")
    if ma_position:
        print(
            f"MA Strategy Position: {ma_position.quantity} @ avg {ma_position.avg_price}"
        )
        print(f"MA Strategy P&L: {ma_position.unrealized_pnl}")

    momentum_position = portfolio.get_position("BTCUSDT", "momentum_strategy")
    if momentum_position:
        print(
            f"Momentum Position: {momentum_position.quantity} @ avg {momentum_position.avg_price}"
        )

    # Reporting (less frequent)
    print("\n=== REPORTING ===")
    btc_summary = portfolio.get_symbol_summary("BTCUSDT")
    print(f"BTC Total Quantity: {btc_summary['total_quantity']}")
    print(f"BTC Total P&L: {btc_summary['total_unrealized_pnl']}")
    print(f"Strategies: {list(btc_summary['strategies'].keys())}")

    portfolio.close()
