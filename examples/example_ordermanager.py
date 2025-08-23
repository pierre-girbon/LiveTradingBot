import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from modules.dataprocessor import DataProcessor
from modules.order_manager import OrderManager
from modules.portfolio_manager import PortfolioManager, TradeEvent, TradeType
from modules.websocketclient import RequestType, WebSocketClient


async def main():
    # Create portfolio manager and order manager and data processor
    portfolio = PortfolioManager()
    order_manager = OrderManager(portfolio)
    processor = DataProcessor()
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
    # portfolio.update_price("BTCUSDT", Decimal("52000"), datetime.now(tz=timezone.utc))

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


if __name__ == "__main__":
    asyncio.run(main())
