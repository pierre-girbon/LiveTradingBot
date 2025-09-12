"""
Example Strategy Implementation: Moving Average Crossover

Save this as: strategies/moving_average.py

This shows how to create a concrete strategy that inherits from BaseStrategy.
This strategy implements a classic moving average crossover:
- Buy when short MA crosses above long MA
- Sell when short MA crosses below long MA
"""

import os
import sys
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

# Add the parent directory to the path so we can import our base classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.strategy_engine_integration import (BaseStrategy, Signal,
                                                 SignalType, StrategyData)


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy

    Parameters required:
    - short_window: Number of periods for short moving average (e.g., 10)
    - long_window: Number of periods for long moving average (e.g., 20)
    - position_size: Size of position to take (e.g., 1.0 BTC)
    """

    def get_required_parameters(self) -> List[str]:
        """Define what parameters this strategy needs from config."""
        return ["short_window", "long_window", "position_size"]

    def validate_parameters(self, parameters: dict) -> bool:
        """Custom validation for our parameters."""
        if not super().validate_parameters(parameters):
            self.logger.error("Parameters validation error")
            return False

        # Additional validation
        short_window = parameters.get("short_window", 0)
        long_window = parameters.get("long_window", 0)
        position_size = parameters.get("position_size", 0)

        # Validate parameter types and ranges
        if not isinstance(short_window, int) or short_window <= 0:
            return False
        if not isinstance(long_window, int) or long_window <= 0:
            return False
        if short_window >= long_window:
            return False  # Short window should be shorter than long window
        if position_size <= 0:
            return False

        return True

    def get_required_history_length(self) -> int:
        """Dynamic history length based on parameters."""
        # We'll return a reasonable default here since we don't have access to parameters
        # In a more advanced implementation, you could pass parameters to this method
        return 50

    def evaluate(self, data: StrategyData) -> Optional[Signal]:
        """
        Evaluate the moving average crossover strategy.

        Logic:
        1. Calculate short and long moving averages
        2. Compare with previous values to detect crossovers
        3. Generate buy signal on bullish crossover (if not already long)
        4. Generate sell signal on bearish crossover (if currently long)
        """

        # Get parameters
        short_window = data.get_parameter("short_window")
        long_window = data.get_parameter("long_window")
        position_size = Decimal(str(data.get_parameter("position_size")))

        # Check if we have enough price history
        if len(data.price_history) < long_window:
            self.logger.info(
                "Not enough price history",
                strategy=self.strategy_id,
                price_history=len(data.price_history),
                required_history=long_window,
            )
            return None

        # Calculate current moving averages
        recent_prices = data.price_history[-long_window:]
        short_ma = sum(recent_prices[-short_window:]) / short_window
        long_ma = sum(recent_prices) / long_window

        # Get previous moving averages from internal state
        prev_short_ma = data.get_state("prev_short_ma")
        prev_long_ma = data.get_state("prev_long_ma")

        # Store current MAs for next evaluation
        data.set_state("prev_short_ma", short_ma)
        data.set_state("prev_long_ma", long_ma)

        # Need at least one previous evaluation to detect crossovers
        if prev_short_ma is None or prev_long_ma is None:
            self.logger.info("Missing Moving average", strategy=self.strategy_id)
            return None

        # Detect crossovers and generate signals
        signal = None

        self.logger.info(
            strategy=self.strategy_id,
            price=data.price_history[-1],
            short_ma=short_ma,
            long_ma=long_ma,
        )

        # Bullish crossover: short MA crosses above long MA
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            # Only buy if we don't already have a long position
            if data.current_position <= 0:
                signal = Signal(
                    symbol=data.symbol,
                    signal_type=SignalType.BUY,
                    quantity=position_size,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    confidence=self._calculate_signal_strength(
                        short_ma, long_ma, data.price_history
                    ),
                )

                # Store that we just generated a buy signal
                data.set_state("last_signal", "BUY")
                data.set_state("last_signal_price", data.price_history[-1])

        # Bearish crossover: short MA crosses below long MA
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            # Only sell if we have a long position
            if data.current_position > 0:
                signal = Signal(
                    symbol=data.symbol,
                    signal_type=SignalType.SELL,
                    quantity=data.current_position,  # Sell entire position
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    confidence=self._calculate_signal_strength(
                        short_ma, long_ma, data.price_history
                    ),
                )

                # Store that we just generated a sell signal
                data.set_state("last_signal", "SELL")
                data.set_state("last_signal_price", data.price_history[-1])

        if signal:
            self.logger.info("Sending Signal", strategy=self.strategy_id, signal=signal)

        return signal

    def _calculate_signal_strength(
        self, short_ma: Decimal, long_ma: Decimal, price_history: List[Decimal]
    ) -> float:
        """
        Calculate a confidence score for the signal (0.0 to 1.0).

        This is optional but can be useful for:
        - Risk management
        - Position sizing
        - Signal filtering
        """
        if len(price_history) < 10:
            return 0.5  # Default confidence

        # Simple confidence calculation based on:
        # 1. How strong the crossover is (larger gap = higher confidence)
        # 2. Recent price volatility (lower volatility = higher confidence)

        # 1. Crossover strength
        ma_gap = abs(short_ma - long_ma) / long_ma
        gap_confidence = min(ma_gap * 100, 1.0)  # Cap at 1.0

        # 2. Volatility factor (lower volatility = more confidence)
        recent_prices = price_history[-10:]
        price_std = Decimal(
            str(self._calculate_std_dev([float(p) for p in recent_prices]))
        )
        avg_price = sum(recent_prices) / len(recent_prices)
        volatility_ratio = price_std / avg_price
        volatility_confidence = max(0.0, 1.0 - float(volatility_ratio) * 10)

        # Combine factors
        overall_confidence = (gap_confidence + volatility_confidence) / 2
        return max(0.1, min(0.95, overall_confidence))  # Keep between 0.1 and 0.95

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Simple standard deviation calculation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5


# Example of another strategy - Momentum Strategy
class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy

    Buys when price momentum is strongly positive, sells when negative.

    Parameters required:
    - lookback_period: Number of periods to look back for momentum calculation (e.g., 14)
    - threshold: Minimum momentum threshold to trigger signals (e.g., 0.05 = 5%)
    - position_size: Size of position to take
    """

    def get_required_history_length(self) -> int:
        return 30  # Need extra history for momentum calculation

    def get_required_parameters(self) -> List[str]:
        return ["lookback_period", "threshold", "position_size"]

    def validate_parameters(self, parameters: dict) -> bool:
        if not super().validate_parameters(parameters):
            return False

        lookback = parameters.get("lookback_period", 0)
        threshold = parameters.get("threshold", 0)
        position_size = parameters.get("position_size", 0)

        return (
            isinstance(lookback, int)
            and lookback > 0
            and isinstance(threshold, (int, float))
            and 0 < threshold < 1
            and position_size > 0
        )

    def evaluate(self, data: StrategyData) -> Optional[Signal]:
        """
        Evaluate momentum strategy.

        Momentum = (Current Price - Price N periods ago) / Price N periods ago
        """
        lookback_period = data.get_parameter("lookback_period")
        threshold = data.get_parameter("threshold")
        position_size = Decimal(str(data.get_parameter("position_size")))

        if len(data.price_history) < lookback_period + 1:
            return None

        # Calculate momentum
        current_price = data.price_history[-1]
        past_price = data.price_history[-(lookback_period + 1)]
        momentum = (current_price - past_price) / past_price

        # Get previous momentum to avoid repeated signals
        prev_momentum = data.get_state("prev_momentum", 0)
        data.set_state("prev_momentum", momentum)

        signal = None

        # Strong positive momentum and we're not already long
        if (
            momentum > threshold
            and prev_momentum <= threshold
            and data.current_position <= 0
        ):
            signal = Signal(
                symbol=data.symbol,
                signal_type=SignalType.BUY,
                quantity=position_size,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                confidence=min(float(abs(momentum)), 1.0),
            )

        # Strong negative momentum and we have a position
        elif (
            momentum < -threshold
            and prev_momentum >= -threshold
            and data.current_position > 0
        ):
            signal = Signal(
                symbol=data.symbol,
                signal_type=SignalType.SELL,
                quantity=data.current_position,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                confidence=min(float(abs(momentum)), 1.0),
            )

        return signal


# Example of a more complex strategy - Mean Reversion
class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy

    Assumes that prices will revert to their moving average.
    Buys when price is significantly below MA, sells when significantly above.

    Parameters required:
    - ma_period: Period for moving average calculation
    - deviation_threshold: How many standard deviations from MA to trigger signal
    - position_size: Size of position to take
    """

    def get_required_history_length(self) -> int:
        return 60  # Need enough for MA and volatility calculations

    def get_required_parameters(self) -> List[str]:
        return ["ma_period", "deviation_threshold", "position_size"]

    def validate_parameters(self, parameters: dict) -> bool:
        if not super().validate_parameters(parameters):
            return False

        ma_period = parameters.get("ma_period", 0)
        deviation_threshold = parameters.get("deviation_threshold", 0)
        position_size = parameters.get("position_size", 0)

        return (
            isinstance(ma_period, int)
            and ma_period > 0
            and isinstance(deviation_threshold, (int, float))
            and deviation_threshold > 0
            and position_size > 0
        )

    def evaluate(self, data: StrategyData) -> Optional[Signal]:
        """
        Evaluate mean reversion strategy.
        """
        ma_period = data.get_parameter("ma_period")
        deviation_threshold = data.get_parameter("deviation_threshold")
        position_size = Decimal(str(data.get_parameter("position_size")))

        if len(data.price_history) < ma_period * 2:  # Need extra history for volatility
            return None

        # Calculate moving average
        recent_prices = data.price_history[-ma_period:]
        moving_average = sum(recent_prices) / ma_period

        # Calculate standard deviation
        variance = (
            sum((price - moving_average) ** 2 for price in recent_prices) / ma_period
        )
        std_dev = variance ** Decimal("0.5")

        if std_dev == 0:  # Avoid division by zero
            return None

        # Calculate how many standard deviations current price is from MA
        current_price = data.price_history[-1]
        z_score = (current_price - moving_average) / std_dev

        # Get previous z-score to detect crossings
        prev_z_score = data.get_state("prev_z_score", 0)
        data.set_state("prev_z_score", z_score)

        signal = None

        # Price significantly below MA (oversold) - buy signal
        if (
            z_score < -deviation_threshold
            and prev_z_score >= -deviation_threshold
            and data.current_position <= 0
        ):

            signal = Signal(
                symbol=data.symbol,
                signal_type=SignalType.BUY,
                quantity=position_size,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                confidence=min(float(abs(z_score) / deviation_threshold), 1.0),
            )

        # Price significantly above MA (overbought) - sell signal
        elif (
            z_score > deviation_threshold
            and prev_z_score <= deviation_threshold
            and data.current_position > 0
        ):

            signal = Signal(
                symbol=data.symbol,
                signal_type=SignalType.SELL,
                quantity=data.current_position,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                confidence=min(float(abs(z_score) / deviation_threshold), 1.0),
            )

        return signal


if __name__ == "__main__":
    """
    Test the strategies with sample data
    """
    print("Testing Strategy Implementations")
    print("=" * 50)

    # Test MovingAverageCrossover
    print("\n1. Testing Moving Average Crossover Strategy:")

    ma_strategy = MovingAverageCrossover("test_ma")

    # Create test data
    test_data = StrategyData(
        symbol="BTCUSDT",
        price_history=[
            Decimal(str(p))
            for p in [
                50000,
                50100,
                50200,
                50150,
                50300,
                50400,
                50350,
                50500,
                50600,
                50550,
                50700,
                50800,
                50750,
                50900,
                51000,
                50950,
                51100,
                51200,
                51150,
                51300,
                51400,
                51350,
                51500,
                51600,
                51550,
                51700,
                51800,
                51750,
                51900,
                52000,
            ]
        ],
        current_position=Decimal("0"),
        parameters={"short_window": 5, "long_window": 10, "position_size": 1.0},
    )

    # Test signal generation
    signal = ma_strategy.evaluate(test_data)
    print(f"  First evaluation: {signal}")

    # Add more data to potentially trigger a signal
    test_data.price_history.extend([Decimal(str(p)) for p in [52100, 52200, 52300]])
    signal = ma_strategy.evaluate(test_data)
    print(f"  After price increase: {signal}")

    print("\n2. Testing Momentum Strategy:")

    momentum_strategy = MomentumStrategy("test_momentum")

    momentum_data = StrategyData(
        symbol="ETHUSDT",
        price_history=[
            Decimal(str(p))
            for p in [
                3000,
                3010,
                3020,
                3015,
                3030,
                3040,
                3035,
                3050,
                3060,
                3055,
                3070,
                3080,
                3075,
                3090,
                3100,
                3095,
                3110,
                3120,
                3125,
                3140,  # Strong uptrend
            ]
        ],
        current_position=Decimal("0"),
        parameters={
            "lookback_period": 10,
            "threshold": 0.03,  # 3% threshold
            "position_size": 2.0,
        },
    )

    signal = momentum_strategy.evaluate(momentum_data)
    print(f"  Momentum signal: {signal}")

    print("\n3. Testing Mean Reversion Strategy:")

    mean_reversion_strategy = MeanReversionStrategy("test_mean_reversion")

    # Create data with a spike (good for mean reversion)
    mr_data = StrategyData(
        symbol="ADAUSDT",
        price_history=[
            Decimal(str(p))
            for p in [
                1.0,
                1.01,
                0.99,
                1.02,
                0.98,
                1.03,
                0.97,
                1.04,
                0.96,
                1.05,
                0.95,
                1.06,
                0.94,
                1.07,
                0.93,
                1.08,
                0.92,
                1.09,
                0.91,
                1.10,
                0.85,  # Significant drop - should trigger buy signal
            ]
        ],
        current_position=Decimal("0"),
        parameters={
            "ma_period": 15,
            "deviation_threshold": 1.5,  # 1.5 standard deviations
            "position_size": 100.0,
        },
    )

    signal = mean_reversion_strategy.evaluate(mr_data)
    print(f"  Mean reversion signal: {signal}")

    print("\n" + "=" * 50)
    print("Strategy testing complete!")
