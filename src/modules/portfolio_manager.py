"""
Portfolio Management System for Trading Bot

This module handles position tracking, P&L calculations, and portfolio state management.
Integrates with the existing DataProcessor to receive real-time market data.
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

from modules.logger import get_logger

# load environement config
load_dotenv()

# Set decimal precision for financial calculations
getcontext().prec = 28

# Database setup
Base = declarative_base()


class PositionSide(Enum):
    """Position side enumeration"""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"  # No position


class TradeType(Enum):
    """Trade type enumeration"""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradeEvent:
    """Represents a trade event (buy/sell)"""

    symbol: str
    trade_type: TradeType  # BUY or SELL
    quantity: Decimal  # Always positive
    price: Decimal
    timestamp: datetime
    trade_id: Optional[str] = None


class Position(Base):
    """SQLAlchemy model for position storage"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    quantity = Column(Float, nullable=False, default=0.0)
    avg_price = Column(Float, nullable=False, default=0.0)
    current_price = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime, default=datetime.now(tz=timezone.utc))


class Trade(Base):
    """SQLAlchemy model for trade history storage"""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)  # BUY or SELL
    quantity = Column(Float, nullable=False)  # Always positive
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    trade_id = Column(String(50), nullable=True)


@dataclass
class PositionInfo:
    """Data class for position information"""

    symbol: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    side: PositionSide
    last_updated: datetime

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of the position"""
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis of the position"""
        return abs(self.quantity) * self.avg_price


class PortfolioManager:
    """
    Manages trading positions and portfolio state.

    Features:
    - Position tracking with weighted average cost basis
    - Real-time P&L calculation
    - Trade history persistence
    - Integration with DataProcessor handlers
    """

    def __init__(
        self, db_url: str = os.environ.get("DB_PATH", "sqlite:///portfolio.db")
    ):
        self.logger = get_logger(__name__)

        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # In-memory position cache for fast access
        self.positions: Dict[str, PositionInfo] = {}

        # Load existing positions from database
        self._load_positions_from_db()

        self.logger.info("PortfolioManager initialized")

    def _load_positions_from_db(self):
        """Load existing positions from database into memory"""
        db_positions = self.session.query(Position).all()

        for pos in db_positions:
            self.positions[pos.symbol] = PositionInfo(
                symbol=pos.symbol,
                quantity=Decimal(str(pos.quantity)),
                avg_price=Decimal(str(pos.avg_price)),
                current_price=Decimal(str(pos.current_price)),
                unrealized_pnl=Decimal("0"),  # Will be calculated
                side=self._determine_side(Decimal(str(pos.quantity))),
                last_updated=pos.last_updated,
            )

        self.logger.info(f"Loaded {len(self.positions)} positions from database")

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

    def process_trade(self, trade: TradeEvent) -> bool:
        """
        Process a trade event and update position.

        Args:
            trade: TradeEvent containing trade information

        Returns:
            bool: True if trade processed successfully
        """
        try:
            symbol = trade.symbol
            current_pos = self.positions.get(symbol)
            self.logger.debug(symbol=symbol, current_pos=current_pos)

            # Convert trade to signed quantity based on trade type
            signed_quantity = (
                trade.quantity if trade.trade_type == TradeType.BUY else -trade.quantity
            )

            if current_pos is None:
                # New position
                self.positions[symbol] = PositionInfo(
                    symbol=symbol,
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
                    self.positions[symbol].quantity = Decimal("0")
                    self.positions[symbol].side = PositionSide.FLAT
                    self.positions[symbol].unrealized_pnl = Decimal("0")
                elif (current_pos.quantity >= 0 and signed_quantity >= 0) or (
                    current_pos.quantity <= 0 and signed_quantity <= 0
                ):
                    # Adding to existing position - recalculate weighted average
                    total_cost = (current_pos.quantity * current_pos.avg_price) + (
                        signed_quantity * trade.price
                    )
                    new_avg_price = total_cost / new_quantity

                    self.positions[symbol].quantity = new_quantity
                    self.positions[symbol].avg_price = new_avg_price
                    self.positions[symbol].side = self._determine_side(new_quantity)
                else:
                    # Reducing/closing position - average price stays the same
                    self.positions[symbol].quantity = new_quantity
                    self.positions[symbol].side = self._determine_side(new_quantity)

                self.positions[symbol].last_updated = trade.timestamp

            # Recalculate P&L
            self.positions[symbol].unrealized_pnl = self._calculate_unrealized_pnl(
                self.positions[symbol]
            )

            # Persist to database
            self._save_trade_to_db(trade)
            self._save_position_to_db(self.positions[symbol])

            self.logger.info(
                f"Processed trade: {symbol} {trade.trade_type.value} {trade.quantity} @ {trade.price}",
                position_qty=float(self.positions[symbol].quantity),
                avg_price=float(self.positions[symbol].avg_price),
                unrealized_pnl=float(self.positions[symbol].unrealized_pnl),
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to process trade: {e}")
            return False

    def update_price(self, symbol: str, price: Decimal, timestamp: datetime) -> bool:
        """
        Update current price for a symbol and recalculate P&L.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp

        Returns:
            bool: True if price updated successfully
        """
        try:
            if symbol in self.positions:
                self.positions[symbol].current_price = price
                self.positions[symbol].unrealized_pnl = self._calculate_unrealized_pnl(
                    self.positions[symbol]
                )
                self.positions[symbol].last_updated = timestamp

                # Update database
                self._save_position_to_db(self.positions[symbol])

                self.logger.debug(
                    f"Updated price: {symbol} @ {price}",
                    unrealized_pnl=float(self.positions[symbol].unrealized_pnl),
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to update price: {e}")
            return False

    def _save_trade_to_db(self, trade: TradeEvent):
        """Save trade to database"""
        db_trade = Trade(
            symbol=trade.symbol,
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
            self.session.query(Position).filter_by(symbol=position.symbol).first()
        )

        if db_position is None:
            db_position = Position(symbol=position.symbol)
            self.session.add(db_position)

        db_position.quantity = float(position.quantity)
        db_position.avg_price = float(position.avg_price)
        db_position.current_price = float(position.current_price)
        db_position.last_updated = position.last_updated

        self.session.commit()

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get current position for a symbol"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """Get all current positions"""
        return self.positions.copy()

    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio unrealized P&L"""
        total_pnl = Decimal("0")
        for position in self.positions.values():
            total_pnl += position.unrealized_pnl
        return total_pnl

    def get_position_quantity(self, symbol: str) -> Decimal:
        """Get current position quantity for a symbol (for OrderManager)"""
        position = self.positions.get(symbol)
        return position.quantity if position else Decimal("0")

    def calculate_available_balance(
        self, symbol: str, locked_quantity: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Calculate available balance for trading (used by OrderManager).

        Args:
            symbol: Trading symbol
            locked_quantity: Quantity currently locked in pending orders

        Returns:
            Available quantity that can be sold (for long positions)
        """
        position = self.positions.get(symbol)
        if not position or position.side != PositionSide.LONG:
            return Decimal("0")

        return max(Decimal("0"), position.quantity - locked_quantity)

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
            symbol = kline_data.symbol
            close_price = Decimal(str(kline_data.kline.close_price))
            timestamp = kline_data.kline.close_time

            self.update_price(symbol, close_price, timestamp)

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

    # Example trades
    trade1 = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.BUY,
        quantity=Decimal("10"),
        price=Decimal("50000"),
        timestamp=datetime.now(tz=timezone.utc),
    )

    trade2 = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.BUY,
        quantity=Decimal("5"),
        price=Decimal("55000"),
        timestamp=datetime.now(tz=timezone.utc),
    )

    trade3 = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.SELL,
        quantity=Decimal("3"),
        price=Decimal("60000"),
        timestamp=datetime.now(tz=timezone.utc),
    )

    # Process trades
    portfolio.process_trade(trade1)  # Buy 10 BTC @ 50k
    portfolio.process_trade(trade2)  # Buy 5 BTC @ 55k
    portfolio.process_trade(trade3)  # Sell 3 BTC @ 60k

    # Update price
    portfolio.update_price("BTCUSDT", Decimal("58000"), datetime.now(tz=timezone.utc))

    # Check position
    position = portfolio.get_position("BTCUSDT")
    if position:
        print(f"Position: {position.quantity} @ avg {position.avg_price}")
        print(f"Current price: {position.current_price}")
        print(f"Unrealized P&L: {position.unrealized_pnl}")
        print(f"Side: {position.side.value}")

    # Test available balance calculation
    available = portfolio.calculate_available_balance("BTCUSDT", Decimal("2"))
    print(f"Available balance (with 2 locked): {available}")

    portfolio.close()
