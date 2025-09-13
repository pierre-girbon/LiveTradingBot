import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__name__), "src/"))
sys.path.insert(1, os.path.join(os.path.dirname(__name__), "strategies/"))

from src.modules.strategy_engine_integration import Signal, SignalType, StrategyData
from strategies.moving_average import MovingAverageCrossover


@pytest.fixture
def sample_strategy_data():
    """Create sample strategy data for testing."""
    return StrategyData(
        symbol="BTCUSDT",
        price_history=[Decimal(str(p)) for p in [50000, 50100, 50200, 50150, 50300]],
        current_position=Decimal("0"),
        parameters={"short_window": 5, "long_window": 10, "position_size": 1.0},
    )


@pytest.fixture
def mac_data_passing():
    """Strategy data passing Moving average crossover"""
    return StrategyData(
        symbol="BTCUSDT",
        price_history=[
            Decimal(str(p))
            for p in [
                107088.4296875,
                107327.703125,
                108385.5703125,
                107135.3359375,
                105698.28125,
                108859.3203125,
                109647.9765625,
                108034.3359375,
                108231.1796875,
                109232.0703125,
                108299.8515625,
                108950.2734375,
                111326.5546875,
                115987.203125,
                117516.9921875,
                117435.2265625,
                119116.1171875,
                119849.703125,
                117777.1875,
                118738.5078125,
                119289.84375,
                118003.2265625,
                117939.9765625,
                117300.7890625,
                117439.5390625,
                119995.4140625,
                118754.9609375,
                118368.0,
            ]
        ],
        current_position=Decimal("0"),
        parameters={"short_window": 5, "long_window": 10, "position_size": 1.0},
        internal_state={
            "prev_short_ma": Decimal("118286.135938"),
            "prev_long_ma": Decimal("118508.914844"),
        },
    )


@pytest.fixture
def test_moving_average_crossover():
    return MovingAverageCrossover("test_moving_average")


class TestMovingAverageCrossover:
    def test_init(self, sample_strategy_data):
        strategy = MovingAverageCrossover("test_strategy")

        assert strategy.get_required_parameters() == [
            "short_window",
            "long_window",
            "position_size",
        ]
        assert strategy.validate_parameters(sample_strategy_data.parameters) == True
        assert strategy.get_required_history_length() == 50

    def test_price_history_length(
        self, test_moving_average_crossover, sample_strategy_data
    ):
        assert test_moving_average_crossover.evaluate(sample_strategy_data) == None

    def test_data_buy(self, test_moving_average_crossover, mac_data_passing):
        _ = test_moving_average_crossover.evaluate(mac_data_passing)
        assert mac_data_passing.get_state("prev_short_ma") == 118371.740625
        assert mac_data_passing.get_state("prev_long_ma") == 118360.74453125
        assert mac_data_passing.get_state("last_signal") == "BUY"
        assert mac_data_passing.get_state("last_signal_price") == Decimal("118368.0")

    def test_crossover_buy(self, test_moving_average_crossover, mac_data_passing):
        signal = test_moving_average_crossover.evaluate(mac_data_passing)
        assert signal.symbol == "BTCUSDT"
        # assert signal.signal_type is SignalType.BUY
        assert signal.quantity == Decimal("1")
        assert signal.strategy_id == "test_moving_average"
        # assert signal.timestamp == datetime.now()
        assert signal.confidence == None
