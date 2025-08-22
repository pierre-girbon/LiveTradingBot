import asyncio
import logging
from typing import Any, Dict

from dataprocessor import DataProcessor, KlineData, TickerData, TradeData
from websocketclient import RequestType, WebSocketClient


async def main():
    logging.basicConfig(level=logging.WARNING)

    processor = DataProcessor()

    # Define Trade handler
    def trade_handler(trade: TradeData):
        print(f"{trade}")

    processor.set_trade_handler(trade_handler)

    # ticker handler
    def ticker_handler(ticker: TickerData):
        print(f"{ticker}")

    processor.set_ticker_handler(ticker_handler)

    # Create WebSocket client
    client = WebSocketClient("wss://stream.binance.com:9443/ws")

    # define handler message
    client.set_message_handler(processor.process_message)

    # Define response handler
    def response_handler(msg: Dict[str, Any]):
        pass

    client.set_response_handler(response_handler)

    # Start the client in a separate task
    client_task = asyncio.create_task(client.run_with_reconnect())
    # wait for connection
    await asyncio.sleep(5)
    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@trade"], id=1
    )
    print(f"Subscription result: {result.info}")

    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, ["btcusdt@ticker"], id=2
    )
    print(f"Subscription result: {result.info}")

    # Wait and check subscription statuses
    await asyncio.sleep(3)
    print(f"Subscription 1 status: {client.get_subscription_status(1)}")
    print(f"Subscription 2 status: {client.get_subscription_status(2)}")

    try:
        await client_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
