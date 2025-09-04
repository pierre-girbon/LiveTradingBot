"""
Order Management System for Trading Bot

This module handles order validation, placement, tracking, and execution.
Acts as an intermediary between the trading bot and the exchange API.
Integrates with PortfolioManager to ensure proper position tracking.

## Components
### Databases
- OrderRecord

### Types / Dataclasses
- Orders:
    - MarketOrder
    - LimitOrder
    - StopOrder

## Example usage
```python
if __name__ == "__main__":
    import logging

    from portfolio_manager import PortfolioManager

    logging.basicConfig(level=logging.INFO)

    # Create portfolio manager and order manager
    portfolio = PortfolioManager()
    order_manager = OrderManager(portfolio)

    # Example: Place some orders
    print("=== Testing Order Management ===\n")

    # First, let's add some initial position to portfolio for testing
    initial_trade = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.BUY,
        quantity=Decimal("10"),
        price=Decimal("50000"),
        timestamp=datetime.now(tz=timezone.utc),
    )
    portfolio.process_trade(initial_trade)
    portfolio.update_price("BTCUSDT", Decimal("52000"), datetime.now(tz=timezone.utc))

    print(f"Initial position: {portfolio.get_position('BTCUSDT').quantity} BTC")

    # Test market order
    order_id_1 = order_manager.place_market_order(
        "BTCUSDT", TradeType.SELL, Decimal("3")
    )
    print(f"Placed market order: {order_id_1}")

    # Test limit order
    order_id_2 = order_manager.place_limit_order(
        "BTCUSDT", TradeType.SELL, Decimal("2"), Decimal("55000")
    )
    print(f"Placed limit order: {order_id_2}")

    # Check positions after orders
    position = portfolio.get_position("BTCUSDT")
    print(f"\nFinal position: {position.quantity} BTC @ avg {position.avg_price}")
    print(f"Unrealized P&L: {position.unrealized_pnl}")

    # Show active orders
    active_orders = order_manager.get_active_orders()
    print(f"\nActive orders: {len(active_orders)}")
    for order in active_orders:
        print(
            f"- {order.order_id}: {order.trade_type.value} {order.quantity_remaining} {order.symbol} @ {order.get_reference_price()}"
        )

    # Clean up
    order_manager.close()
    portfolio.close()
```
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from modules.logger import get_logger
from modules.portfolio_manager import PortfolioManager, TradeEvent, TradeType

# Load dotenv file
load_dotenv()

# Set decimal precision for financial calculations
getcontext().prec = 28

# Database setup
#########################""
Base = declarative_base()


class OrderRecord(Base):
    """SQLAlchemy model for order storage"""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)
    order_type = Column(String(10), nullable=False)
    quantity_ordered = Column(Float, nullable=False)
    quantity_filled = Column(Float, nullable=False, default=0.0)
    filled_price = Column(Float, nullable=False, default=0.0)
    reference_price = Column(Float, nullable=True)  # limit_price or stop_price
    order_status = Column(String(20), nullable=False)
    creation_date = Column(DateTime, nullable=False)
    last_updated = Column(DateTime, nullable=False)
    exchange_order_id = Column(String(100), nullable=True)


# Types / Dataclasses
#######################
class OrderStatus(Enum):
    """Order status enumeration"""

    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"  # Canceled by bot/user
    REJECTED = "REJECTED"  # Rejected by exchange


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class BaseOrder(ABC):
    """Abstract base class for all order types"""

    order_id: str
    symbol: str
    trade_type: TradeType
    quantity_ordered: Decimal
    order_status: OrderStatus
    creation_date: datetime

    # Fields that get updated during execution
    quantity_filled: Decimal = field(default_factory=lambda: Decimal("0"))
    filled_price: Decimal = field(default_factory=lambda: Decimal("0"))
    last_updated: datetime = field(default_factory=datetime.now)
    exchange_order_id: Optional[str] = None

    @abstractmethod
    def get_order_type(self) -> OrderType:
        """Get the order type"""
        pass

    @abstractmethod
    def get_reference_price(self) -> Decimal:
        """Get the relevant price for this order type (for validation/display)"""
        pass

    @property
    def quantity_remaining(self) -> Decimal:
        """Calculate remaining quantity to be filled"""
        return self.quantity_ordered - self.quantity_filled

    @property
    def is_fully_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.quantity_filled >= self.quantity_ordered

    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage"""
        if self.quantity_ordered == 0:
            return Decimal("0")
        return (self.quantity_filled / self.quantity_ordered) * Decimal("100")


@dataclass
class MarketOrder(BaseOrder):
    """Market order - executes immediately at best available price"""

    def get_order_type(self) -> OrderType:
        return OrderType.MARKET

    def get_reference_price(self) -> Decimal:
        """Market orders don't have a reference price until filled"""
        return self.filled_price if self.filled_price > 0 else Decimal("0")


@dataclass
class LimitOrder(BaseOrder):
    """Limit order - executes only at specified price or better"""

    limit_price: Decimal = Decimal(0)

    def get_order_type(self) -> OrderType:
        return OrderType.LIMIT

    def get_reference_price(self) -> Decimal:
        return self.limit_price

    def __post_init__(self):
        if self.limit_price <= 0:
            raise ValueError("Limit price must be positive")


@dataclass
class StopOrder(BaseOrder):
    """Stop order - triggers when market reaches stop price"""

    stop_price: Decimal = Decimal(0)

    def get_order_type(self) -> OrderType:
        return OrderType.STOP

    def get_reference_price(self) -> Decimal:
        return self.stop_price

    def __post_init__(self):
        if self.stop_price <= 0:
            raise ValueError("Stop price must be positive")


# Union type for all order types
Order = Union[MarketOrder, LimitOrder, StopOrder]


# Order Manager
#####################
@dataclass
class OrderValidationResult:
    """Result of order validation"""

    is_valid: bool
    error_message: Optional[str] = None
    available_balance: Optional[Decimal] = None


class OrderManager:
    """
    Manages order lifecycle and integrates with PortfolioManager.

    Features:
    - Order validation and placement
    - Order state tracking
    - Integration with exchange API
    - Position updates via PortfolioManager
    - Persistent order storage
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        db_url: str = os.environ.get("DB_PATH", "sqlite:///orders.db"),
    ):
        """OrderManager Init

        **Flow:**
        - Init Logger
        - Setup Database
        - Load orders from database to memory
        """
        self.logger = get_logger(__name__)
        self.portfolio_manager = portfolio_manager

        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # In-memory order tracking
        self.orders: Dict[str, Order] = {}
        """Dictionary of orders:
        - Key: UUID
        - Value: Order (Market, Limit or Stop)
        """
        self.orders_by_symbol: Dict[str, List[str]] = {}
        """Dictionary of orders by symbol
        - Key: symbol
        - Value: List[Order.order_id]
        """

        # Load existing orders from database
        self._load_orders_from_db()

        # TODO: Exchange API integration will be added later
        self.exchange_api = None

        self.logger.info("OrderManager initialized")

    def _load_orders_from_db(self):
        """Load existing orders from database into memory"""
        db_orders = self.session.query(OrderRecord).all()

        for db_order in db_orders:
            # Convert database record back to Order object
            order = self._db_record_to_order(db_order)
            if order:
                self.orders[order.order_id] = order

                # Update orders_by_symbol index
                if order.symbol not in self.orders_by_symbol:
                    self.orders_by_symbol[order.symbol] = []
                self.orders_by_symbol[order.symbol].append(order.order_id)

        self.logger.info(f"Loaded {len(self.orders)} orders from database")

    def _db_record_to_order(self, db_record: OrderRecord) -> Optional[Order]:
        """Convert database record to Order object"""
        try:
            common_fields = {
                "order_id": db_record.order_id,
                "symbol": db_record.symbol,
                "trade_type": TradeType(db_record.trade_type),
                "quantity_ordered": Decimal(str(db_record.quantity_ordered)),
                "order_status": OrderStatus(db_record.order_status),
                "creation_date": db_record.creation_date,
                "quantity_filled": Decimal(str(db_record.quantity_filled)),
                "filled_price": Decimal(str(db_record.filled_price)),
                "last_updated": db_record.last_updated,
                "exchange_order_id": db_record.exchange_order_id,
            }

            if db_record.order_type == OrderType.MARKET.value:
                return MarketOrder(**common_fields)
            elif db_record.order_type == OrderType.LIMIT.value:
                return LimitOrder(
                    **common_fields, limit_price=Decimal(str(db_record.reference_price))
                )
            elif db_record.order_type == OrderType.STOP.value:
                return StopOrder(
                    **common_fields, stop_price=Decimal(str(db_record.reference_price))
                )

        except Exception as e:
            self.logger.error(f"Failed to convert DB record to order: {e}")
            return None

    def _calculate_locked_quantity(self, symbol: str) -> Decimal:
        """Calculate total quantity locked in pending orders for a symbol"""
        locked = Decimal("0")

        order_ids = self.orders_by_symbol.get(symbol, [])
        for order_id in order_ids:
            order = self.orders.get(order_id)
            if order and order.order_status in [
                OrderStatus.PENDING,
                OrderStatus.PARTIAL,
            ]:
                if order.trade_type == TradeType.SELL:
                    locked += order.quantity_remaining

        return locked

    def validate_order(
        self, symbol: str, trade_type: TradeType, quantity: Decimal
    ) -> OrderValidationResult:
        """
        Validate if an order can be placed.

        **Args:**
        - symbol: Trading symbol
        - trade_type: BUY or SELL
        - quantity: Quantity to trade

        **Returns:**
        - OrderValidationResult with validation status
        """
        try:
            # Basic validation
            if quantity <= 0:
                return OrderValidationResult(False, "Quantity must be positive")

            # For sell orders, check if we have enough available balance
            if trade_type == TradeType.SELL:
                locked_qty = self._calculate_locked_quantity(symbol)
                available = self.portfolio_manager.calculate_available_balance(
                    symbol, locked_qty
                )

                if quantity > available:
                    return OrderValidationResult(
                        False,
                        f"Insufficient balance. Available: {available}, Requested: {quantity}",
                        available,
                    )

            # TODO: Add more validation rules (minimum order size, price bounds, etc.)

            return OrderValidationResult(
                True,
                available_balance=available if trade_type == TradeType.SELL else None,
            )

        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return OrderValidationResult(False, f"Validation error: {e}")

    def place_market_order(
        self, symbol: str, trade_type: TradeType, quantity: Decimal
    ) -> Optional[str]:
        """
        Place a market order.

        **Args:**
        - symbol: Trading symbol
        - trade_type: BUY or SELL
        - quantity: Quantity to trade

        **Returns:**
        - Order ID if successful, None if failed
        """
        # Validate order
        validation = self.validate_order(symbol, trade_type, quantity)
        if not validation.is_valid:
            self.logger.error(f"Order validation failed: {validation.error_message}")
            return None

        # Create order
        order_id = str(uuid4())
        order = MarketOrder(
            order_id=order_id,
            symbol=symbol,
            trade_type=trade_type,
            quantity_ordered=quantity,
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        # Store order
        self.orders[order_id] = order
        if symbol not in self.orders_by_symbol:
            self.orders_by_symbol[symbol] = []
        self.orders_by_symbol[symbol].append(order_id)

        # Save to database
        self._save_order_to_db(order)

        # TODO: Send order to exchange API
        # For now, we'll simulate immediate execution for market orders
        self._simulate_market_order_execution(order)

        self.logger.info(
            f"Placed market order: {trade_type.value} {quantity} {symbol}",
            order_id=order_id,
        )

        return order_id

    def place_limit_order(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: Decimal,
        limit_price: Decimal,
    ) -> Optional[str]:
        """Place a limit order"""
        validation = self.validate_order(symbol, trade_type, quantity)
        if not validation.is_valid:
            self.logger.error(f"Order validation failed: {validation.error_message}")
            return None

        order_id = str(uuid4())
        order = LimitOrder(
            order_id=order_id,
            symbol=symbol,
            trade_type=trade_type,
            quantity_ordered=quantity,
            limit_price=limit_price,
            order_status=OrderStatus.PENDING,
            creation_date=datetime.now(tz=timezone.utc),
        )

        self.orders[order_id] = order
        if symbol not in self.orders_by_symbol:
            self.orders_by_symbol[symbol] = []
        self.orders_by_symbol[symbol].append(order_id)

        self._save_order_to_db(order)

        # TODO: Send to exchange API

        self.logger.info(
            f"Placed limit order: {trade_type.value} {quantity} {symbol} @ {limit_price}",
            order_id=order_id,
        )

        return order_id

    def _simulate_market_order_execution(self, order: MarketOrder):
        """
        Simulate market order execution for testing purposes.
        TODO: Replace with real exchange API integration.
        """
        # Simulate execution at current market price
        # In reality, this would come from exchange fills
        current_position = self.portfolio_manager.get_position(order.symbol)
        if current_position:
            execution_price = current_position.current_price
        else:
            # Fallback price for new symbols
            execution_price = Decimal("50000")  # Mock price

        self._process_order_fill(
            order.order_id,
            order.quantity_ordered,
            execution_price,
            datetime.now(tz=timezone.utc),
        )

    def _process_order_fill(
        self,
        order_id: str,
        filled_quantity: Decimal,
        fill_price: Decimal,
        fill_time: datetime,
    ):
        """
        Process an order fill (partial or complete).

        Args:
            order_id: Order ID
            filled_quantity: Quantity that was filled
            fill_price: Price at which the fill occurred
            fill_time: When the fill happened
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"Order {order_id} not found for fill processing")
            return

        try:
            # Calculate new filled quantity and weighted average price
            previous_filled = order.quantity_filled
            previous_value = previous_filled * order.filled_price
            new_value = filled_quantity * fill_price

            order.quantity_filled += filled_quantity
            if order.quantity_filled > 0:
                order.filled_price = (
                    previous_value + new_value
                ) / order.quantity_filled

            # Update order status
            if order.is_fully_filled:
                order.order_status = OrderStatus.COMPLETED
            else:
                order.order_status = OrderStatus.PARTIAL

            order.last_updated = fill_time

            # Create trade event and update portfolio
            trade = TradeEvent(
                symbol=order.symbol,
                trade_type=order.trade_type,
                quantity=filled_quantity,
                price=fill_price,
                timestamp=fill_time,
                trade_id=f"{order_id}_{datetime.now(tz=timezone.utc).timestamp()}",
            )

            # Update portfolio with the trade
            self.portfolio_manager.process_trade(trade)

            # Save updated order to database
            self._save_order_to_db(order)

            self.logger.info(
                f"Processed fill: {order.symbol} {filled_quantity} @ {fill_price}",
                order_id=order_id,
                total_filled=float(order.quantity_filled),
                avg_price=float(order.filled_price),
                status=order.order_status.value,
            )

        except Exception as e:
            self.logger.error(f"Failed to process order fill: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"Order {order_id} not found")
            return False

        if order.order_status not in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
            self.logger.error(
                f"Cannot cancel order {order_id} with status {order.order_status}"
            )
            return False

        # TODO: Send cancellation to exchange API

        order.order_status = OrderStatus.CANCELED
        order.last_updated = datetime.now(tz=timezone.utc)

        self._save_order_to_db(order)

        self.logger.info(f"Canceled order {order_id}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_active_orders(self) -> List[Order]:
        """Get all active orders (PENDING or PARTIAL)"""
        return [
            order
            for order in self.orders.values()
            if order.order_status in [OrderStatus.PENDING, OrderStatus.PARTIAL]
        ]

    def _save_order_to_db(self, order: Order):
        """Save order to database"""
        db_order = (
            self.session.query(OrderRecord).filter_by(order_id=order.order_id).first()
        )

        if db_order is None:
            db_order = OrderRecord(order_id=order.order_id)
            self.session.add(db_order)

        # Update fields
        db_order.symbol = order.symbol
        db_order.trade_type = order.trade_type.value
        db_order.order_type = order.get_order_type().value
        db_order.quantity_ordered = float(order.quantity_ordered)
        db_order.quantity_filled = float(order.quantity_filled)
        db_order.filled_price = float(order.filled_price)
        db_order.reference_price = (
            float(order.get_reference_price())
            if order.get_reference_price() > 0
            else None
        )
        db_order.order_status = order.order_status.value
        db_order.creation_date = order.creation_date
        db_order.last_updated = order.last_updated
        db_order.exchange_order_id = order.exchange_order_id

        self.session.commit()

    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        self.logger.info("OrderManager closed")


# Example usage
if __name__ == "__main__":
    import logging

    from portfolio_manager import PortfolioManager

    logging.basicConfig(level=logging.INFO)

    # Create portfolio manager and order manager
    portfolio = PortfolioManager()
    order_manager = OrderManager(portfolio)

    # Example: Place some orders
    print("=== Testing Order Management ===\n")

    # First, let's add some initial position to portfolio for testing
    initial_trade = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.BUY,
        quantity=Decimal("10"),
        price=Decimal("50000"),
        timestamp=datetime.now(tz=timezone.utc),
    )
    portfolio.process_trade(initial_trade)
    portfolio.update_price("BTCUSDT", Decimal("52000"), datetime.now(tz=timezone.utc))

    print(f"Initial position: {portfolio.get_position('BTCUSDT').quantity} BTC")

    # Test market order
    order_id_1 = order_manager.place_market_order(
        "BTCUSDT", TradeType.SELL, Decimal("3")
    )
    print(f"Placed market order: {order_id_1}")

    # Test limit order
    order_id_2 = order_manager.place_limit_order(
        "BTCUSDT", TradeType.SELL, Decimal("2"), Decimal("55000")
    )
    print(f"Placed limit order: {order_id_2}")

    # Check positions after orders
    position = portfolio.get_position("BTCUSDT")
    print(f"\nFinal position: {position.quantity} BTC @ avg {position.avg_price}")
    print(f"Unrealized P&L: {position.unrealized_pnl}")

    # Show active orders
    active_orders = order_manager.get_active_orders()
    print(f"\nActive orders: {len(active_orders)}")
    for order in active_orders:
        print(
            f"- {order.order_id}: {order.trade_type.value} {order.quantity_remaining} {order.symbol} @ {order.get_reference_price()}"
        )

    # Clean up
    order_manager.close()
    portfolio.close()
