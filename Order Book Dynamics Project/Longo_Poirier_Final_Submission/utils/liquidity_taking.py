import pandas as pd
import numpy as np
from enum import Enum
import random as rand
from scipy.optimize import fsolve
import plotnine as p9
from typing import List
import utils.data_formatting as formats

# to describe current side of pf or side recommended by strategy
class Side(Enum):
    LONG = 0
    SHORT = 1

# should only be used when order is actually executed
class Action(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

class MarketOrder:
    def __init__(self, time_placed: int, side: Side):
        self.processed = False
        self.time_placed = time_placed
        self.side = side
        self.quantity = 0.0003537925  # keeping a static quantity for initial strategy

    def execute(self, price: int, timestamp: int):
        self.processed = True
        self.price_executed = price
        self.time_executed = timestamp


class LiquidityTakingStrategy:
    def __init__(
            self,
            theos:              pd.DataFrame,
            confidence:         pd.DataFrame,
            book_data:          pd.DataFrame,
            trade_data:         pd.DataFrame,
            latency:            int,
            trading_fees:       float
    ):
        # pre-constructed variables
        self.theos = theos
        self.confidence = confidence
        self.book = book_data
        self.trade = (pd.merge_asof(trade_data, book_data[["timestamp_utc_nanoseconds"]], 
                                    on="timestamp_utc_nanoseconds", 
                                    direction="backward"))
        
        self.timestamp_idx = 0
        self.timestamp = 0
        
        self.theo = 0
        self.position = 0
        self.curr_book = None
        self.bid = None
        self.ask = None

        # hyperparameters to change strategy behavior
        self.latency = latency
        self.trading_fees = trading_fees
        
        self.action_list = []
        self.action_log = None
        self.trade_log = pd.DataFrame(columns=['timestamp', 'position', 'action'])

        self.orders: List[MarketOrder] = [] # container of all orders sent
        self.active_orders: List[MarketOrder] = [] # container of all orders sent
        self.trades: List[MarketOrder] = [] # container of all orders executed

    def get_ts(self)->int:
        return self.theos.iloc[self.timestamp_idx].name
    
    def take(self, order):
        order.execute(self.bid if order.side == Side.LONG else self.ask, self.timestamp)
        self.trades.append(order)
    
    def check_orders(self):
        """iterates through self.trades container and verifies which orders have been executed
        
        Keyword arguments:
        timestamp -- current timestamp at which to verify
        """
        idx_to_rem = []
        for idx, order in enumerate(self.active_orders):
            if order.time_sent + self.latency <= self.timestamp:
                self.take(order)
                idx_to_rem.append(idx)
        
        for idx in idx_to_rem[::-1]:
            self.active_orders.pop(idx)


    def send_order(self, side: Side, timestamp: int) -> MarketOrder:
        order = MarketOrder(timestamp, side)
        self.active_orders.append(order)
        return order

    @staticmethod
    def _utility_func(x, theo:float, confidence:float, inventory:float, position_penalty:float=10)->float:
        return -confidence * (theo - x) ** 2 - inventory/position_penalty * (x - theo)
    @staticmethod
    def _deriv_utility_func(x, theo:float, confidence:float, inventory:float, position_penalty:float=10)->float:
        return 2 * confidence * (theo - x) - inventory/position_penalty
    @staticmethod
    def _utility_opt_x(theo:float, confidence:float, inventory:float, position_penalty:float=10)->float:
        return theo - position_penalty/(2 * confidence * inventory)
    
    def assess_opportunity(self):
        utility_adj_theo = self._utility_opt_x(self.theo, self.confidence, self.position)
        if self.position == 0:
            if utility_adj_theo > self.ask:
                self.send_order(Side.LONG, self.timestamp)
            elif utility_adj_theo < self.bid:
                self.send_order(Side.SHORT, self.timestamp)
        elif self.position > 0:
            if utility_adj_theo < self.bid:
                self.send_order(Side.SHORT, self.timestamp)
        elif self.position < 0:
            if utility_adj_theo > self.ask:
                self.send_order(Side.LONG, self.timestamp)

        

    def run(self) -> None:
        for self.timestamp_idx, self.curr_book, self.theo in zip(self.book.iterrows(), self.theos):
            self.timestamp = self.curr_book.loc["timestamp_utc_nanoseconds"]
            self.bid = self.curr_book.loc["Bid1PriceMillionths"]
            self.ask = self.book.loc["Ask1PriceMillionths"]
            
            if self.orders:
                self.check_orders()
            self.assess_opportunity()

if __name__ == "__main__":
    sources = [
        ("2023-01__BTC-USD_orders.h5", "2023-01__BTC-USD_trades.h5"),
    ]
    data = formats.get_crypto_data(sources)
    sample = formats.sample_data(data, (0, 1000))

    strat = LiquidityTakingStrategy(sample["BTC-USD"]["books"], sample["BTC-USD"]["trades"], 0, 0.003)
    strat.run()