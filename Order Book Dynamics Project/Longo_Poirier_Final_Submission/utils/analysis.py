import numpy as np
import pandas as pd

def theo_empirical_accuracy_ts(data, delta, delay):
    init_df = data["BTC-USD"]["books"][["theo_value", "mid_price", "timestamp_utc_nanoseconds"]].copy(deep=True)
    init_df.set_index("timestamp_utc_nanoseconds", inplace=True)
    init_df.index = init_df.index.astype(np.int64)
    theo_df = init_df[["theo_value"]]
    midprice = init_df[["mid_price"]]

    theo_df.index += delta
    theo_df.index = theo_df.index.astype(np.int64)
    midprice.index = midprice.index.astype(np.int64)

    lagged_df = pd.merge_asof(theo_df, midprice, left_index=True, right_index=True, direction="backward")
    
    crossed = (
        ((init_df["theo_value"].values >= init_df["mid_price"].values) & (lagged_df["theo_value"].values < lagged_df["mid_price"].values))|
        ((init_df["theo_value"].values <= init_df["mid_price"].values) & (lagged_df["theo_value"].values > lagged_df["mid_price"].values))
    )

    APPROX_RATE = 0.1
    predicted_pdelta_well = (
        (lagged_df["theo_value"] >= lagged_df["mid_price"] - APPROX_RATE) & (lagged_df["theo_value"] <= lagged_df["mid_price"] + APPROX_RATE)
    )
    good_pred = lambda theo, mid_price : (mid_price <= theo + APPROX_RATE) & (mid_price >= theo - APPROX_RATE)
    # Initialize a Series to store the result of the check for each row
    was_once_reached = pd.Series(False, index=init_df.index)

    for current_time, row in init_df.iterrows():
        # Define the start of the lookback window
        window_end = current_time + delta
        
        # Filter rows within the delta window
        window_df = init_df[(init_df.index >= current_time + delay) & (init_df.index < window_end)]
        
        if not window_df.empty:
            # Check if any 'mid_price' in the window equals the first 'theo_value' of the window
            first_theo_value = window_df.iloc[0]['theo_value']
            was_once_reached[current_time] = ((good_pred(first_theo_value, row["mid_price"]))).any()

    return init_df, lagged_df, was_once_reached


def rolling_mean(df: pd.DataFrame, delta: int, forward: bool = True)->pd.DataFrame:
        df = df.copy(deep=True)
        rolling_df = pd.DataFrame(index = df.index, columns = df.columns)
        for i, row in df.iterrows():
            lower_bound = row['timestamp_utc_nanoseconds'] + (- delta if not forward else 0)
            upper_bound = row['timestamp_utc_nanoseconds'] + (+ delta if forward else 0)
            
            window_data = df[(df['timestamp_utc_nanoseconds'] > lower_bound) & (df['timestamp_utc_nanoseconds'] <= upper_bound)]

            mean_value = window_data.mean()
            rolling_df.loc[i] = mean_value
        rolling_df.drop(columns=["timestamp_utc_nanoseconds"], inplace=True)
        return rolling_df