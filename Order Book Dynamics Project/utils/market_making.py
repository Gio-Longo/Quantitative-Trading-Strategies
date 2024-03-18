import pandas as pd
import numpy as np
from enum import Enum
import random as rand
from scipy.optimize import fsolve
import plotnine as p9
import tqdm


# to describe the type of ongoing order type
class Message(Enum):
    ADD = 0
    MODIFY = 1
    DELETE = 2


# to describe action to take depending on signal
class Action(Enum):
    BID_ORDER = 0  # place a bid
    ASK_ORDER = 1  # place an ask 


# to describe current side of pf or side recommended by strategy
class Side(Enum):
    BID = 0
    ASK = 1


class Order:
    def __init__(self, time_placed: int, side: Side, price: float, order_type: str):
        self.time_placed = time_placed
        self.side = side
        self.price = price
        self.type = order_type
        self.time_executed = 0
        self.quantity = 0.0003537925  # keeping a static quantity for initial strategy


def trade_fill_check(time_difference_ns: int, half_prob: float) -> bool:
    '''
    Calculates a probability of our trade filling based on how long it has been on the market.
    '''
    rand.seed(42)
    time_diff_seconds = time_difference_ns / 1_000_000_000

    probability = 1 / (1 + np.exp(-10 * (time_diff_seconds - half_prob)))

    return rand.random() <= probability


class MarketMakingStrategy:
    def __init__(
            self,
            mid_data:                         pd.DataFrame,
            mid_probabilities:                pd.DataFrame,
            book_data:                        pd.DataFrame,
            trade_data:                       pd.DataFrame,
            latency:                          int,
            theo_offset_for_price_selection:  int,
            utility_dfference_to_change_theo: int, 
            price_difference_to_change_order: int,
            seconds_to_half_prob_of_trading:  int
    ):
        # pre-constructed variables
        self.theos = mid_data
        self.confidence = mid_probabilities
        self.book = book_data
        self.trade = (pd.merge_asof(trade_data, book_data[["timestamp_utc_nanoseconds"]], 
                                    on="timestamp_utc_nanoseconds", 
                                    direction="backward"))
        
        self.inventory = 0
        self.bid = None
        self.ask = None
        self.curr_theo = 0
        self.prev_theo = 0
        self.curr_confidence = 0
        self.mid_price = 0
        self.curr_book = None
        self.pnl = 0

        # hyperparameters to change strategy behavior
        self.latency = latency
        self.theo_offset = theo_offset_for_price_selection
        self.utility_difference = utility_dfference_to_change_theo
        self.price_difference = price_difference_to_change_order
        self.half_prob = seconds_to_half_prob_of_trading
        

        # orders process before being successfully placed to deal with latency
        self.bid_order_in_process = None
        self.ask_order_in_process = None

        self.action_list = []
        self.action_log = None
        self.trade_log = pd.DataFrame(columns=['timestamp', 'bid_orders', 'ask_orders'])
        self.pnl_log = None
        self.pnl_plot = None        
    
    def __repr__(self) -> str:
        return f"Strategy with combo {self.latency, self.theo_offset,self.utility_difference,self.price_difference,self.half_prob}"

    def update_action_list(self, action):
        self.action_list.append({
            'timestamp': self.curr_book['timestamp_utc_nanoseconds'],
            'action': action,
            'inventory': self.inventory,
        })

    def create_log(self):
        self.action_log = pd.DataFrame(self.action_list)
        self.action_log.timestamp = pd.to_datetime(self.action_log.timestamp)

    def calculate_pnl_plot(self):
        if self.pnl_log.empty:
            print('No orders were filled')
            return

        pnl_df = self.pnl_log.copy()
        pnl_df.reset_index(inplace=True)
        pnl_df.rename(columns={'index': 'timestamp'}, inplace=True)
    
        min_timestamp = pnl_df['timestamp'].min()
        max_timestamp = pnl_df['timestamp'].max()

        num_breaks = 10
        breaks = pd.date_range(start=min_timestamp, end=max_timestamp, periods=num_breaks)

        self.pnl_plot = (
            p9.ggplot(pnl_df)
            + p9.aes(x='timestamp', y='cumulative_pnl') 
            + p9.geom_step(color='orange')
            + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1, size=10), 
                    axis_text_y=p9.element_text(size=10),
                    plot_title=p9.element_text(size=20), 
                    axis_title_x=p9.element_text(size=15),
                    axis_title_y=p9.element_text(size=15),
                    figure_size=(12, 8))
            + p9.labs(x='Timestamp', y='Cumulative PnL ($)', title='Cumulative PnL over Time with Steps')
            + p9.scale_x_datetime(breaks=breaks, date_labels='%H:%M:%S.%f') 
        )

    def calculate_pnl_log(self):
        bid_df = self.trade_log[['timestamp', 'bid_orders']].dropna()
        bid_df.reset_index(drop=True, inplace=True)

        ask_df = self.trade_log[['timestamp', 'ask_orders']].dropna()
        ask_df.reset_index(drop=True, inplace=True)

        self.pnl_log = pd.DataFrame({
            'timestamp': pd.concat([bid_df['timestamp'], ask_df['timestamp']], axis=1).max(axis=1),
            'bid_orders': bid_df['bid_orders']/1_000_000,
            'ask_orders': ask_df['ask_orders']/1_000_000
        })

        self.pnl_log.set_index('timestamp', inplace=True)
        self.pnl_log['pnl'] = self.pnl_log['ask_orders'] - self.pnl_log['bid_orders']
        self.pnl_log['cumulative_pnl'] = self.pnl_log['pnl'].cumsum()
        self.pnl_log.dropna(inplace=True)
        self.pnl_log.drop(columns=['bid_orders', 'ask_orders'], inplace=True)

    def place_order(self, action: Action, price: int, message: str):
        curr_time = self.curr_book['timestamp_utc_nanoseconds']
        order_side = Side.BID if action == Action.BID_ORDER else Side.ASK
        order_type = 'bid' if action == Action.BID_ORDER else 'ask'
        
        new_order = Order(
            curr_time,
            order_side,
            price,
            message
        )
        
        if action == Action.BID_ORDER:
            self.bid_order_in_process = new_order
        else:
            self.ask_order_in_process = new_order

        self.update_action_list(f'Processing {order_type} order of type {message}.')

        self.prev_theo = self.curr_theo

    def initial_positions(self):
        self.place_order(
            Action.BID_ORDER,
            self.curr_book['Bid1PriceMillionths'],
            Message.ADD
        )

        self.place_order(
            Action.ASK_ORDER,
            self.curr_book['Ask1PriceMillionths'],
            Message.ADD
        )

    def process_order(self, order_in_process, order_type):
        curr_time = self.curr_book['timestamp_utc_nanoseconds']
        if order_in_process:
            place_time = order_in_process.time_placed
            if curr_time >= place_time + self.latency:
                if order_in_process.type == Message.ADD:
                    if order_in_process.side == Side.ASK:
                        if order_in_process.price >= self.curr_book['Ask1PriceMillionths']:
                            setattr(self, order_type, order_in_process)
                            action = 'Ask order added successfully.'
                    elif order_in_process.side == Side.BID:
                        if order_in_process.price <= self.curr_book['Bid1PriceMillionths']:
                            setattr(self, order_type, order_in_process)
                            action = 'Bid order added successfully.'
                elif order_in_process.type == Message.MODIFY and getattr(self, order_type):
                    setattr(self, order_type, order_in_process)
                    action = 'Order modified successfully.'
                executed_order = getattr(self, order_type)
                if executed_order:
                    executed_order.time_executed = curr_time
                    executed_order.type = None
                else:
                    action = 'Order processing unsuccessful.'
                setattr(self, f"{order_type}_order_in_process", None)

                self.update_action_list(action)

    def process_orders(self):
        self.process_order(self.bid_order_in_process, 'bid')
        self.process_order(self.ask_order_in_process, 'ask')

    def check_fill(self):
        curr_time = self.curr_book['timestamp_utc_nanoseconds']
        possible_trade = self.trade[self.trade['timestamp_utc_nanoseconds'] == curr_time]
        
        if possible_trade.empty:
            return
        
        trade_side = possible_trade['Side'].iloc[0]
        trade_price = possible_trade['PriceMillionths'].iloc[0]
        
        is_bid = self.bid and trade_side == -1.0 and self.bid.price >= trade_price
        is_ask = self.ask and trade_side == 1.0 and self.ask.price <= trade_price
        
        if is_bid or is_ask:
            order = self.bid if is_bid else self.ask
            action_type = 'buy' if is_bid else 'sell'
            
            if trade_fill_check(curr_time - order.time_placed, self.half_prob):
                new_row = {
                    'timestamp': pd.to_datetime(order.time_executed),
                    'bid_orders': order.price if is_bid else np.nan,
                    'ask_orders': np.nan if is_bid else order.price
                }
                self.trade_log = pd.concat([self.trade_log, pd.DataFrame([new_row])], ignore_index=True)
                
                self.inventory += 1 if is_bid else -1
                setattr(self, 'bid' if is_bid else 'ask', None)

                self.update_action_list(f'Successful {action_type} execution.')

    def derivative_to_zero(self, x):
        inventory_term = (-self.inventory/2)*(self.curr_confidence - self.curr_confidence**2)
        theo_term = self.curr_confidence*(self.curr_theo/1_000_000-x)
        
        mid_price_term = (1-self.curr_confidence)*(self.mid_price/1_000_000 - x)

        return inventory_term + theo_term + mid_price_term

    def utility(self) -> float:
        prev_theo_utility = self.derivative_to_zero(self.prev_theo/1_000_000)
        
        initial_guess = self.curr_theo
        utility_theo = fsolve(self.derivative_to_zero, initial_guess)

        if abs(prev_theo_utility) >= self.utility_difference:
            return utility_theo[0] * 1_000_000

        return self.prev_theo

    def define_bid_and_ask_choice(self, theo_to_check):
        bid_multiple = np.exp(self.inventory/2) if self.inventory > 0 else 1
        ask_multiple = np.exp(-self.inventory/2) if self.inventory < 0 else 1

        bid_choice = min(self.curr_book['Bid1PriceMillionths'], theo_to_check-self.theo_offset*bid_multiple)
        ask_choice = max(self.curr_book['Ask1PriceMillionths'], theo_to_check+self.theo_offset*ask_multiple)
    
        return bid_choice, ask_choice 

    def decide_action(self):
        utility_theo = self.utility()

        if utility_theo != self.prev_theo:
            bid_choice, ask_choice = self.define_bid_and_ask_choice(utility_theo) 
        else:
            bid_choice, ask_choice = self.define_bid_and_ask_choice(self.curr_theo)

        def execute_position_action(order_type, choice, in_process, current_order):
            if not in_process:
                action = Action.BID_ORDER if order_type == "bid" else Action.ASK_ORDER

                if not current_order:
                    self.place_order(action, choice, Message.ADD)
                elif utility_theo != self.prev_theo:
                        if abs(current_order.price - choice) >= self.price_difference:
                            self.place_order(action, choice, Message.MODIFY)

        execute_position_action("bid", bid_choice, self.bid_order_in_process, self.bid)
        execute_position_action("ask", ask_choice, self.ask_order_in_process, self.ask)

    def run_strategy(self):
        for i in range(len(self.book)):
            self.curr_book = self.book.iloc[i]
            self.curr_theo = self.theos.iloc[i][0]
            self.curr_confidence = self.confidence.iloc[i][0]
            self.mid_price = self.curr_book['mid_price']
            if i == 0:
                self.initial_positions()
                continue

            # we check for fills before order processes
            self.check_fill()
            self.process_orders()

            # decide how to update positions based on utility
            self.decide_action()

        # log relevant changes such as order placements, modifications, or trades
        self.create_log()
        # create a log and plot to check pnl
        self.calculate_pnl_log()
        self.calculate_pnl_plot()
