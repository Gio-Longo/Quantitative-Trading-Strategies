import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

price_col_names = ['Bid3PriceMillionths', 'Bid2PriceMillionths', 'Bid1PriceMillionths', 'Ask1PriceMillionths', 'Ask2PriceMillionths', 'Ask3PriceMillionths']
quantity_col_names = ['Bid3SizeBillionths', 'Bid2SizeBillionths', 'Bid1SizeBillionths', 'Ask1SizeBillionths', 'Ask2SizeBillionths', 'Ask3SizeBillionths']
best_ask_price_col = "Ask1PriceMillionths"
best_ask_qty_col = "Ask1SizeBillionths"
best_bid_price_col = "Bid1PriceMillionths"
best_bid_qty_col = "Bid1SizeBillionths"

get_pair = lambda string: string.split("__")[1].split("_")[0]

def get_crypto_data(sources, path="data/"):
    data = {}
    for order, trade in sources:
        name = get_pair(order)        
        data[name] = {
            "books": pd.read_hdf("data/crypto/"+order, f"Level2/{name}", stop=10_000_000).sort_values(by="timestamp_utc_nanoseconds"), 
            "trades": pd.read_hdf("data/crypto/"+trade, f"{name}Trades/{name}Trades").sort_values(by="timestamp_utc_nanoseconds")
        }
    return data

def sample_data(data: Dict[str, Dict[str, pd.DataFrame]], range: Tuple[int])-> Dict[str, Dict[str, pd.DataFrame]]:
    """This function samples the data to be featurized.

    Args:
        data (Dict[str, Dict[str, pd.DataFrame]]): data dictionary containing the order book and trade data for each pair
        range (Tuple[int]): range (in terms of index) of the data to be featurized.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: the sampled data
    """
    sample = {}
    for pair in data.keys():
        books = data[pair]["books"].iloc[range[0]:range[1]].copy()
        start_ts = books.timestamp_utc_nanoseconds.min()
        end_ts = books.timestamp_utc_nanoseconds.max()
        sample[pair] = {
            "books": books,
            "trades": data[pair]["trades"].loc[(data[pair]["trades"].timestamp_utc_nanoseconds >= start_ts) & (data[pair]["trades"].timestamp_utc_nanoseconds <= end_ts)].copy()
        }
    return sample

def generate_y(data, delta):
    y = data["books"].copy(deep=True)
    y["timestamp_utc_nanoseconds"] = data["books"]["timestamp_utc_nanoseconds"]

    y['1sec_midprice'] = (y['Bid1PriceMillionths'] + y['Ask1PriceMillionths']) / 2

    y['timestamp_utc_nanoseconds'] = (y["timestamp_utc_nanoseconds"] + delta).astype(np.uint64) #add 1 second to the timestamp

    y = pd.merge_asof(y[['1sec_midprice', "timestamp_utc_nanoseconds"]], data["books"], on="timestamp_utc_nanoseconds", direction="backward")

    return y