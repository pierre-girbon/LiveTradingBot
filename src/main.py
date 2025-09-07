import asyncio
import random
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
    _ = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@kline_1m"], id=1
    )
    await asyncio.sleep(5)

    _ = order_manager.place_market_order("BTCUSDT", TradeType.BUY, Decimal("10"))

    while True:

        await asyncio.sleep(5)

        choice = random.randint(0, 1)
        if choice:
            _ = order_manager.place_market_order("BTCUSDT", TradeType.BUY, Decimal("1"))
        else:
            _ = order_manager.place_market_order(
                "BTCUSDT", TradeType.SELL, Decimal("1")
            )

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

    try:
        await client_task
    except asyncio.CancelledError:
        pass

    # Clean up
    order_manager.close()
    portfolio.close()


if __name__ == "__main__":
    asyncio.run(main())
