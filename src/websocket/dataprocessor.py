import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageType(Enum):
    """Supported message types."""

    TRADE = "trade"
    AGG_TRADE = "aggTrade"
    KLINE = "kline"
    TICKER = "24hrTicker"
    MINI_TICKER = "24hrMiniTicker"


# Pydantic models for each message type
class TradeData(BaseModel):
    """Pydantic model for Binance trade data validation and parsing."""

    event_type: str = Field(validation_alias="e")
    symbol: str = Field(validation_alias="s")
    trade_id: int = Field(validation_alias="t")
    price: float = Field(validation_alias="p")
    quantity: float = Field(validation_alias="q")
    trade_time: datetime = Field(validation_alias="T")
    is_buyer_market_maker: bool = Field(validation_alias="m")

    model_config = ConfigDict(extra="ignore")

    @field_validator("price", "quantity")
    def validate_positive_numbers(cls, v):
        if v <= 0:
            raise ValueError("Price and quantity must be positive")
        return v


class AggTradeData(BaseModel):
    """Pydantic model for Binance aggregate trade data."""

    event_type: str = Field(validation_alias="e")
    symbol: str = Field(validation_alias="s")
    agg_trade_id: int = Field(validation_alias="a")
    price: float = Field(validation_alias="p")
    quantity: float = Field(validation_alias="q")
    first_trade_id: int = Field(validation_alias="f")
    last_trade_id: int = Field(validation_alias="l")
    trade_time: datetime = Field(validation_alias="T")
    is_buyer_market_maker: bool = Field(validation_alias="m")

    model_config = ConfigDict(extra="ignore")

    @field_validator("price", "quantity")
    def validate_positive_numbers(cls, v):
        if v <= 0:
            raise ValueError("Price and quantity must be positive")
        return v


class KlineInfo(BaseModel):
    """Nested kline information."""

    start_time: datetime = Field(validation_alias="t")
    close_time: datetime = Field(validation_alias="T")
    symbol: str = Field(validation_alias="s")
    interval: str = Field(validation_alias="i")
    first_trade_id: int = Field(validation_alias="f")
    last_trade_id: int = Field(validation_alias="L")
    open_price: float = Field(validation_alias="o")
    close_price: float = Field(validation_alias="c")
    high_price: float = Field(validation_alias="h")
    low_price: float = Field(validation_alias="l")
    volume: float = Field(validation_alias="v")
    number_of_trades: int = Field(validation_alias="n")
    is_closed: bool = Field(validation_alias="x")
    quote_volume: float = Field(validation_alias="q")
    taker_buy_base_volume: float = Field(validation_alias="V")
    taker_buy_quote_volume: float = Field(validation_alias="Q")

    model_config = ConfigDict(extra="ignore")

    @field_validator(
        "open_price", "close_price", "high_price", "low_price", "volume", "quote_volume"
    )
    def validate_positive_numbers(cls, v):
        if v <= 0:
            raise ValueError("Price and volume values must be positive")
        return v


class KlineData(BaseModel):
    """Pydantic model for Binance kline (candlestick) data."""

    event_type: str = Field(validation_alias="e")
    symbol: str = Field(validation_alias="s")
    kline: KlineInfo = Field(validation_alias="k")

    model_config = ConfigDict(extra="ignore")


class TickerData(BaseModel):
    """Pydantic model for Binance 24hr ticker data."""

    event_type: str = Field(validation_alias="e")
    symbol: str = Field(validation_alias="s")
    price_change: float = Field(validation_alias="p")
    price_change_percent: float = Field(validation_alias="P")
    weighted_avg_price: float = Field(validation_alias="w")
    prev_close_price: float = Field(validation_alias="x")
    last_price: float = Field(validation_alias="c")
    last_quantity: float = Field(validation_alias="Q")
    best_bid_price: float = Field(validation_alias="b")
    best_bid_quantity: float = Field(validation_alias="B")
    best_ask_price: float = Field(validation_alias="a")
    best_ask_quantity: float = Field(validation_alias="A")
    open_price: float = Field(validation_alias="o")
    high_price: float = Field(validation_alias="h")
    low_price: float = Field(validation_alias="l")
    volume: float = Field(validation_alias="v")
    quote_volume: float = Field(validation_alias="q")
    open_time: datetime = Field(validation_alias="O")
    close_time: datetime = Field(validation_alias="C")
    first_trade_id: int = Field(validation_alias="F")
    last_trade_id: int = Field(validation_alias="L")
    trade_count: int = Field(validation_alias="n")

    model_config = ConfigDict(extra="ignore")

    @field_validator("volume", "quote_volume")
    def validate_positive_volumes(cls, v):
        if v < 0:
            raise ValueError("Volume values must be non-negative")
        return v


class MiniTicker(BaseModel):
    """Pydantic model for Binance 24h Mini Ticker"""

    event_type: str = Field(validation_alias="e")
    symbol: str = Field(validation_alias="s")
    close_price: float = Field(validation_alias="c")
    open_price: float = Field(validation_alias="o")
    high_price: float = Field(validation_alias="h")
    low_price: float = Field(validation_alias="l")
    volume: float = Field(validation_alias="v")
    quote_volume: float = Field(validation_alias="q")

    model_config = ConfigDict(extra="ignore")

    @field_validator("volume", "quote_volume")
    def validate_positive_volumes(cls, v):
        if v < 0:
            raise ValueError("Volume values must be non-negative")
        return v


# Union type for all possible message data
MessageData = Union[TradeData, AggTradeData, KlineData, TickerData, MiniTicker]


@dataclass
class ValidationResult:
    """Result of data validation - either success or error."""

    success: bool
    message_type: Optional[MessageType] = None
    data: Optional[MessageData] = None
    error: Optional[str] = None


class MessageRouter:
    """Routes messages to the appropriate Pydantic model based on event type."""

    # Mapping of event types to their corresponding models
    MODEL_MAP = {
        MessageType.TRADE.value: TradeData,
        MessageType.AGG_TRADE.value: AggTradeData,
        MessageType.KLINE.value: KlineData,
        MessageType.TICKER.value: TickerData,
        MessageType.MINI_TICKER.value: MiniTicker,
    }

    @classmethod
    def route_message(cls, json_data: Dict[str, Any]) -> ValidationResult:
        """
        Route a JSON message to the appropriate Pydantic model.

        Args:
            json_data: Parsed JSON data

        Returns:
            ValidationResult with validated data or error
        """
        try:
            # Get event type
            event_type = json_data.get("e")
            if not event_type:
                return ValidationResult(
                    success=False, error="Missing 'e' (event type) field"
                )

            # Find the appropriate model
            model_class = cls.MODEL_MAP.get(event_type)
            if not model_class:
                return ValidationResult(
                    success=False, error=f"Unsupported event type: {event_type}"
                )

            # Validate and create the model instance
            validated_data = model_class(**json_data)
            message_type = MessageType(event_type)

            return ValidationResult(
                success=True, message_type=message_type, data=validated_data
            )

        except Exception as e:
            return ValidationResult(success=False, error=f"Validation failed: {e}")


class DataProcessor:
    """
    Enhanced data processor that handles multiple message types.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.router = MessageRouter()

        # Event handlers for each message type
        self.on_trade: Optional[Callable[[TradeData], None]] = None
        self.on_agg_trade: Optional[Callable[[AggTradeData], None]] = None
        self.on_kline: Optional[Callable[[KlineData], None]] = None
        self.on_ticker: Optional[Callable[[TickerData], None]] = None
        self.on_validation_error: Optional[Callable[[Dict[str, Any], str], None]] = None

    def process_message(self, json_data: Dict[str, Any]) -> ValidationResult:
        """
        Process a raw WebSocket message into validated data.

        Args:
            raw_message: Raw JSON string from WebSocket

        Returns:
            ValidationResult with either valid data or error info
        """
        # Route to appropriate model and validate
        result = self.router.route_message(json_data)

        if result.success:
            # Step 3: Call the appropriate handler
            self._call_handler(result.message_type, result.data)
            self.logger.info(
                f"Processed {result.message_type.value} for {result.data.symbol}"
            )
        else:
            # Handle validation error
            if self.on_validation_error:
                self.on_validation_error(json_data, result.error)
            self.logger.error(result.error)

        return result

    def _call_handler(self, message_type: MessageType, data: MessageData):
        """Call the appropriate handler for the message type."""
        handler_map = {
            MessageType.TRADE: self.on_trade,
            MessageType.AGG_TRADE: self.on_agg_trade,
            MessageType.KLINE: self.on_kline,
            MessageType.TICKER: self.on_ticker,
        }

        handler = handler_map.get(message_type)
        if handler:
            handler(data)

    # Handler setters
    def set_trade_handler(self, handler: Callable[[TradeData], None]):
        """Set handler for trade data."""
        self.on_trade = handler

    def set_agg_trade_handler(self, handler: Callable[[AggTradeData], None]):
        """Set handler for aggregate trade data."""
        self.on_agg_trade = handler

    def set_kline_handler(self, handler: Callable[[KlineData], None]):
        """Set handler for kline data."""
        self.on_kline = handler

    def set_ticker_handler(self, handler: Callable[[TickerData], None]):
        """Set handler for ticker data."""
        self.on_ticker = handler

    def set_error_handler(self, handler: Callable[[Dict[str, Any], str], None]):
        """Set handler for validation errors."""
        self.on_validation_error = handler


# Example usage with multiple message types
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = DataProcessor()

    # Define handlers for each message type
    def handle_trade(trade: TradeData):
        print(f"🔄 TRADE: {trade.symbol} {trade.quantity} @ {trade.price}")

    def handle_agg_trade(agg_trade: AggTradeData):
        print(
            f"📊 AGG_TRADE: {agg_trade.symbol} {agg_trade.quantity} @ {agg_trade.price}"
        )

    def handle_kline(kline: KlineData):
        k = kline.kline
        print(
            f"📈 KLINE: {kline.symbol} {k.interval} O:{k.open_price} H:{k.high_price} L:{k.low_price} C:{k.close_price}"
        )

    def handle_ticker(ticker: TickerData):
        print(
            f"🎯 TICKER: {ticker.symbol} Last:{ticker.last_price} Change:{ticker.price_change_percent}%"
        )

    def handle_error(raw_msg: str, error: str):
        print(f"❌ ERROR: {error}")

    # Set all handlers
    processor.set_trade_handler(handle_trade)
    processor.set_agg_trade_handler(handle_agg_trade)
    processor.set_kline_handler(handle_kline)
    processor.set_ticker_handler(handle_ticker)
    processor.set_error_handler(handle_error)

    # Test with your example messages
    test_messages = [
        # Trade message
        """{"e": "trade", "E": 1672515782136, "s": "BNBBTC", "t": 12345, "p": "0.001", "q": "100", "T": 1672515782136, "m": true, "M": true}""",
        # AggTrade message
        """{"e": "aggTrade", "E": 1672515782136, "s": "BNBBTC", "a": 12345, "p": "0.001", "q": "100", "f": 100, "l": 105, "T": 1672515782136, "m": true, "M": true}""",
        # Kline message
        """{"e": "kline", "E": 1672515782136, "s": "BNBBTC", "k": {"t": 1672515780000, "T": 1672515839999, "s": "BNBBTC", "i": "1m", "f": 100, "L": 200, "o": "0.0010", "c": "0.0020", "h": "0.0025", "l": "0.0015", "v": "1000", "n": 100, "x": false, "q": "1.0000", "V": "500", "Q": "0.500", "B": "123456"}}""",
        # Ticker message
        """{"e": "24hrTicker", "E": 1672515782136, "s": "BNBBTC", "p": "0.0015", "P": "250.00", "w": "0.0018", "x": "0.0009", "c": "0.0025", "Q": "10", "b": "0.0024", "B": "10", "a": "0.0026", "A": "100", "o": "0.0010", "h": "0.0025", "l": "0.0010", "v": "10000", "q": "18", "O": 0, "C": 86400000, "F": 0, "L": 18150, "n": 18151}""",
    ]

    print("Testing all message types:\n")
    for msg in test_messages:
        result = processor.process_message(msg)
        if not result.success:
            print(f"❌ Failed: {result.error}")
        print()
