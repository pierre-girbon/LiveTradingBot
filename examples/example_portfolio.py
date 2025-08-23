import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from modules.dataprocessor import DataProcessor
from modules.portfolio_manager import PortfolioManager, TradeEvent, TradeType
from modules.websocketclient import RequestType, WebSocketClient


def check_position(portfolio: PortfolioManager):
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


async def main():
    # Create a live data processor
    processor = DataProcessor()

    # Create portfolio manager
    portfolio = PortfolioManager()

    # Use kline data to handle prices updates
    processor.set_kline_handler(portfolio.handle_kline_data)

    # Create and spawn a websocket client
    client = WebSocketClient("wss://stream.binance.com:9443/ws")
    client.set_message_handler(processor.process_message)
    client_task = asyncio.create_task(client.run_with_reconnect())
    await asyncio.sleep(5)  # wait for connection
    # Send subscription requests using the new API
    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@kline_1m"], id=1
    )
    print(f"Subscription result: {result.info}")
    await asyncio.sleep(5)  # wait for subscription

    # Example trades
    trade1 = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.BUY,
        quantity=Decimal(10),
        price=Decimal(50000),
        timestamp=datetime.now(tz=timezone.utc),
    )

    trade2 = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.BUY,
        quantity=Decimal(5),
        price=Decimal(55000),
        timestamp=datetime.now(tz=timezone.utc),
    )

    trade3 = TradeEvent(
        symbol="BTCUSDT",
        trade_type=TradeType.SELL,
        quantity=Decimal(3),
        price=Decimal(60000),
        timestamp=datetime.now(tz=timezone.utc),
    )

    # Process trades
    check_position(portfolio)
    await asyncio.sleep(2)
    portfolio.process_trade(trade1)  # Buy 10 BTC @ 50k
    check_position(portfolio)
    await asyncio.sleep(2)
    portfolio.process_trade(trade2)  # Buy 5 BTC @ 55k
    check_position(portfolio)
    await asyncio.sleep(2)
    portfolio.process_trade(trade3)  # Sell 3 BTC @ 60k
    check_position(portfolio)
    await asyncio.sleep(2)

    # Update price
    # portfolio.update_price("BTCUSDT", Decimal("58000"), datetime.now(tz=timezone.utc))

    portfolio.close()

    # Wait for the client task
    try:
        await client_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
