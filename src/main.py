"""
Updated main.py showing how to integrate the Strategy Engine
with your existing WebSocketClient, DataProcessor, PortfolioManager, and OrderManager
"""

import asyncio

from modules.dataprocessor import DataProcessor
from modules.order_manager import OrderManager
from modules.strategy_engine_integration import (StrategyEngine,
                                                 StrategyPortfolioManager)
from modules.websocketclient import RequestType, WebSocketClient


async def main():
    # Create extended portfolio manager that supports strategies
    portfolio = StrategyPortfolioManager()

    # Create order manager
    order_manager = OrderManager(portfolio)

    # Create strategy engine
    strategy_engine = StrategyEngine(portfolio, order_manager)

    # Load strategies from configuration
    config_loaded = strategy_engine.load_strategies_from_config(
        "strategies_config.yaml"
    )
    if not config_loaded:
        print("Failed to load strategy configuration")
        return

    print(f"Loaded {len(strategy_engine.strategies)} strategies")
    for strategy_id in strategy_engine.strategies:
        config = strategy_engine.strategy_configs[strategy_id]
        print(f"  - {strategy_id}: watching {config['universe']}")

    # Create data processor
    processor = DataProcessor()

    # Set up handlers - this is where the magic happens!
    # The strategy engine handles kline data and generates signals
    processor.set_kline_handler(strategy_engine.handle_kline_data)

    # Create WebSocket client
    client = WebSocketClient("wss://stream.binance.com:9443/ws")
    client.set_message_handler(processor.process_message)

    # Start WebSocket client
    client_task = asyncio.create_task(client.run_with_reconnect())
    await asyncio.sleep(5)  # Wait for connection

    # Subscribe to all symbols needed by strategies
    required_subscriptions = strategy_engine.get_required_subscriptions()
    print(f"Subscribing to: {required_subscriptions}")

    result = await client.send_subscription_request(
        RequestType.SUBSCRIBE, required_subscriptions, id=1
    )
    print(f"Subscription result: {result.info}")
    await asyncio.sleep(5)  # Wait for subscription

    print("=== Strategy Engine Running ===")
    print("Waiting for market data and strategy signals...")
    print("Press Ctrl+C to stop")

    try:
        # Let the system run and process data
        # In a real system, you might want to add:
        # - Performance monitoring
        # - Strategy statistics logging
        # - Web dashboard
        # - Risk management

        while True:
            await asyncio.sleep(10)

            # Optional: Print strategy performance every 10 seconds
            print("\n=== Strategy Performance Summary ===")
            for strategy_id in strategy_engine.strategies:
                perf = strategy_engine.get_strategy_performance(strategy_id)
                print(f"{strategy_id}:")
                print(f"  Total P&L: ${perf['total_unrealized_pnl']:.2f}")
                print(f"  Total Value: ${perf['total_market_value']:.2f}")

                if perf["positions"]:
                    print("  Positions:")
                    for symbol, pos in perf["positions"].items():
                        if pos["quantity"] != 0:
                            print(
                                f"    {symbol}: {pos['quantity']:.4f} @ ${pos['avg_price']:.2f} "
                                f"(P&L: ${pos['unrealized_pnl']:.2f})"
                            )

    except KeyboardInterrupt:
        print("\n=== Shutting Down ===")

    finally:
        # Graceful shutdown
        await client.disconnect()
        client_task.cancel()

        try:
            await client_task
        except asyncio.CancelledError:
            pass

        # Clean up
        order_manager.close()
        portfolio.close()

        print("Strategy engine stopped")


if __name__ == "__main__":
    asyncio.run(main())
