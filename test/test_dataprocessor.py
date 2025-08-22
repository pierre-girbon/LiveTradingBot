from datetime import datetime, timezone

from src.websocket.dataprocessor import (AggTradeData, MessageRouter,
                                         MessageType, MiniTicker, TickerData,
                                         TradeData)


class TestMessageRouter:

    def test_aggTrade(self):
        message = {
            "e": "aggTrade",
            "E": 1672515782136,
            "s": "BNBBTC",
            "a": 12345,
            "p": "0.001",
            "q": "100",
            "f": 100,
            "l": 105,
            "T": 1672515782136,
            "m": True,
            "M": True,
        }

        result = MessageRouter().route_message(message)

        assert result.success == True
        assert result.message_type == MessageType.AGG_TRADE
        assert type(result.data) == AggTradeData
        assert result.data.symbol == "BNBBTC"
        assert result.data.agg_trade_id == 12345
        assert result.data.price == 0.001
        assert result.data.quantity == 100
        assert result.data.first_trade_id == 100
        assert result.data.last_trade_id == 105
        assert result.data.trade_time == datetime(
            2022,
            12,
            31,
            hour=19,
            minute=43,
            second=2,
            microsecond=136000,
            tzinfo=timezone.utc,
        )
        assert result.data.is_buyer_market_maker == True

    def test_Trade(self):
        message = {
            "e": "trade",  # / Event type
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "t": 12345,  # / Trade ID
            "p": "0.001",  # / Price
            "q": "100",  # / Quantity
            "T": 1672515782136,  # / Trade time
            "m": True,  # / Is the buyer the market maker?
            "M": True,  # / Ignore
        }

        result = MessageRouter().route_message(message)

        assert result.success == True
        assert result.message_type == MessageType.TRADE
        assert type(result.data) == TradeData
        assert result.data.symbol == "BNBBTC"
        assert result.data.trade_id == 12345
        assert result.data.price == 0.001
        assert result.data.quantity == 100
        assert result.data.trade_time == datetime(
            2022,
            12,
            31,
            hour=19,
            minute=43,
            second=2,
            microsecond=136000,
            tzinfo=timezone.utc,
        )
        assert result.data.is_buyer_market_maker == True

    def test_MiniTicker(self):
        message = {
            "e": "24hrMiniTicker",  # / Event type
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "c": "0.0025",  # / Close price
            "o": "0.0010",  # / Open price
            "h": "0.0025",  # / High price
            "l": "0.0010",  # / Low price
            "v": "10000",  # / Total traded base asset volume
            "q": "18",  # / Total traded quote asset volume
        }

        result = MessageRouter().route_message(message)

        assert result.success == True
        assert result.message_type == MessageType.MINI_TICKER
        assert type(result.data) == MiniTicker
        assert result.data.symbol == "BNBBTC"
        assert result.data.close_price == 0.0025
        assert result.data.open_price == 0.0010
        assert result.data.high_price == 0.0025
        assert result.data.low_price == 0.0010
        assert result.data.volume == 10000
        assert result.data.quote_volume == 18

    def test_Ticker_data(self):
        message = {
            "e": "24hrTicker",  # / Event type
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "p": "0.0015",  # Price change
            "P": "250.00",  # Price change percent
            "w": "0.0018",  # / Weighted average price
            "x": "0.0009",  # / First trade(F)-1 price (first trade before the 24hr rolling window)
            "c": "0.0025",  # / Last price
            "Q": "10",  # / Last quantity
            "b": "0.0024",  # / Best bid price
            "B": "10",  # / Best bid quantity
            "a": "0.0026",  # / Best ask price
            "A": "100",  # / Best ask quantity
            "o": "0.0010",  # / Open price
            "h": "0.0025",  # / High price
            "l": "0.0010",  # / Low price
            "v": "10000",  # / Total traded base asset volume
            "q": "18",  # / Total traded quote asset volume
            "O": 0,  # / Statistics open time
            "C": 86400000,  # / Statistics close time
            "F": 0,  # / First trade ID
            "L": 18150,  # / Last trade Id
            "n": 18151,  # / Total number of trades
        }

        result = MessageRouter().route_message(message)

        assert result.success == True
        assert result.message_type == MessageType.TICKER
        assert type(result.data) == TickerData
        assert result.data.symbol == "BNBBTC"
        assert result.data.price_change == 0.0015
        assert result.data.price_change_percent == 250.0
        assert result.data.weighted_avg_price == 0.0018
        assert result.data.prev_close_price == 0.0009
        assert result.data.last_price == 0.0025
        assert result.data.last_quantity == 10.0
        assert result.data.open_price == 0.0010
        assert result.data.high_price == 0.0025
        assert result.data.low_price == 0.0010
        assert result.data.volume == 10000
        assert result.data.quote_volume == 18
        assert result.data.open_time == datetime(
            1970, 1, 1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        assert result.data.close_time == datetime(
            1972, 9, 27, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )

    def test_Trade_data(self):
        message = {
            "e": "trade",  # / Event type
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "t": 12345,  # / Trade ID
            "p": "-0.001",  # / Price
            "q": "100",  # / Quantity
            "T": 1672515782136,  # / Trade time
            "m": True,  # / Is the buyer the market maker?
            "M": True,  # / Ignore
        }

        result = MessageRouter().route_message(message)

        assert result.success == False

    def test_Trade_quantity(self):
        message = {
            "e": "trade",  # / Event type
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "t": 12345,  # / Trade ID
            "p": "0.001",  # / Price
            "q": "-100",  # / Quantity
            "T": 1672515782136,  # / Trade time
            "m": True,  # / Is the buyer the market maker?
            "M": True,  # / Ignore
        }

        result = MessageRouter().route_message(message)

        assert result.success == False

    def test_no_event_type(self):

        message = {
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "t": 12345,  # / Trade ID
            "p": "0.001",  # / Price
            "q": "100",  # / Quantity
            "T": 1672515782136,  # / Trade time
            "m": True,  # / Is the buyer the market maker?
            "M": True,  # / Ignore
        }

        result = MessageRouter().route_message(message)

        assert result.success == False
        assert result.error == "Missing 'e' (event type) field"

    def test_unsupported_type(self):
        message = {
            "e": "CustomType",  # / Event type
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "c": "0.0025",  # / Close price
            "o": "0.0010",  # / Open price
            "h": "0.0025",  # / High price
            "l": "0.0010",  # / Low price
            "v": "10000",  # / Total traded base asset volume
            "q": "18",  # / Total traded quote asset volume
        }

        result = MessageRouter().route_message(message)

        assert result.success == False
        assert result.error == "Unsupported event type: CustomType"

    def test_missing_field(self):
        message = {
            "e": "trade",  # / Event type
            "E": 1672515782136,  # / Event time
            "s": "BNBBTC",  # / Symbol
            "t": 12345,  # / Trade ID
            "p": 0.001,  # / Price
            "q": 100.0,  # / Quantity
            "T": 1672515782136,  # / Trade time
            "M": True,  # / Ignore
        }

        result = MessageRouter().route_message(message)

        assert result.success == False
