import matplotlib.pyplot as plt
from datetime import datetime, timezone
import pandas as pd
import plotnine as p9
def calculate_risk_metrics(risk_df, confidence_level=0.95):

    # Maximum Drawdown
    prev_high = 0
    max_drawdown = 0
    for i, tot_pnl in risk_df.itertuples():
        prev_high = max(prev_high, tot_pnl)
        dd = tot_pnl - prev_high
        if dd < max_drawdown:
            max_drawdown = dd
    
    # Value at Risk (VaR) - Assuming normal distribution for simplicity
    var = np.percentile(risk_df, (1 - confidence_level) * 100)
    
    # Conditional Value at Risk (CVaR) - Average of losses worse than VaR
    cvar = float(risk_df[risk_df <= var].mean())
    
    sharpe = float(risk_df.mean()/risk_df.std())
    
    avg_pnl = float(risk_df.mean())

    # Prepare the result DataFrame
    risk_metrics = pd.DataFrame({
        'Metric': ['Maximum Drawdown', 'VaR', 'CVaR', 'Sharpe Ratio', 'Avg PnL'],
        'Value': [max_drawdown, var, cvar, sharpe, avg_pnl]
    })
    
    return risk_metrics
def plot_spread(times, theos, y_val, title):
    plt.figure(figsize=(15, 5))
    # Convert nanoseconds to seconds (1 second = 1e9 nanoseconds)
    timestamp_s = times.copy(deep=True)/ 1e9
    start = timestamp_s.iloc[0]
    # Convert seconds to a datetime object
    datetime_utc = datetime.fromtimestamp(start, tz=timezone.utc)

    # Format the datetime object to a string with hours, minutes, seconds, and milliseconds
    formatted_time = datetime_utc.strftime('%H:%M:%S.%f')
    plt.plot(timestamp_s, theos, label="Theo")
    plt.plot(timestamp_s, y_val, label="Empirical Observation")
    plt.grid()
    plt.ylabel("Price (in Millionths of USD)")
    plt.xlabel(f"Nanoseconds elapsed since {formatted_time}")

    #remove upper, left and right spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    #remove bottom index
    plt.gca().xaxis.set_ticks_position('none')
    #remove bottom ticks and text
    plt.gca().xaxis.set_tick_params(size=0)
    plt.gca().xaxis.set_tick_params(width=0)
    plt.legend()
    plt.show()
def create_plot(df, timestamp_col, y_col, y_label, title, figure_size=(14, 7)):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    plot = (
        p9.ggplot(df)
        + p9.aes(x=timestamp_col, y=y_col)
        + p9.geom_step(color='orange')
        + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1, size=10),
                   axis_text_y=p9.element_text(size=10),
                   plot_title=p9.element_text(size=20),
                   axis_title_x=p9.element_text(size=15),
                   axis_title_y=p9.element_text(size=15),
                   figure_size=figure_size)
        + p9.labs(x='Timestamp', y=y_label, title=title)
        + p9.scale_x_datetime(date_breaks='2 minutes', date_labels='%H:%M:%S')
    )
    
    return plot

def plot_order_book(small_sample, recorded_transactions, symbol):
    # Plotting the main chart with Best Ask, Best Bid, and Mid Price as step plots
    plt.figure(figsize=(10, 5))
    gridsize = (4, 1)  # Defining a grid of 4 rows and 1 column
    ax_main = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

    # Sample data: replace this with your actual time series data
    # Assume 'dates' is a list of date strings and 'prices' is a list of prices
    asks = small_sample.Ask1PriceMillionths
    bids = small_sample.Bid1PriceMillionths
    theo = small_sample.theo_value

    asks_2 = small_sample.Ask2PriceMillionths.rolling(50).mean()
    bids_2 = small_sample.Bid2PriceMillionths.rolling(50).mean()


    
    # Convert date strings to pandas datetime format for better handling
    timestamps_s = (small_sample.timestamp_utc_nanoseconds - small_sample.timestamp_utc_nanoseconds.min())

    timestamp_trades = (recorded_transactions.timestamp_utc_nanoseconds - small_sample.timestamp_utc_nanoseconds.min())

    recorded_transactions.loc[:,'timestamp_utc_nanoseconds'] = timestamp_trades
    # Timestamp in UTC nanoseconds
    timestamp_ns = small_sample.timestamp_utc_nanoseconds.min()

    # Convert nanoseconds to seconds (1 second = 1e9 nanoseconds)
    timestamp_s = timestamp_ns / 1e9
    timestamp_trades = timestamp_trades #/ 1e9
    recorded_transactions.loc[:,'timestamp_trades'] = timestamp_trades
    # Convert seconds to a datetime object
    datetime_utc = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)

    # Format the datetime object to a string with hours, minutes, seconds, and milliseconds
    formatted_time = datetime_utc.strftime('%H:%M:%S.%f')

    # Create the plot
    ax_main.step(timestamps_s, asks, where='post', label='Best Ask', color="#2ca02c")
    ax_main.step(timestamps_s, bids, where='post', label='Best Bid', color='#d62728')
    ax_main.step(timestamps_s, theo, where='post', label='Theo', color='blue', linewidth=0.8)

    ax_main.plot(timestamps_s, asks_2, label='RA of Level 2 Asks', linestyle='--',color="#2ca02c", alpha=0.8, linewidth=0.8)
    ax_main.plot(timestamps_s, bids_2, label='RA of Level 2 Asks', linestyle='--',color='#d62728', alpha=0.8, linewidth=0.8)

    buy_trades = recorded_transactions[recorded_transactions.Side == 1]
    sell_trades = recorded_transactions[recorded_transactions.Side == -1]

    # For the 'buy' trades, using a brighter shade of green
    ax_main.scatter(buy_trades.timestamp_trades, buy_trades.Price, label='Buy', color='#9cff9f', marker="^", s=80, zorder=2,edgecolors='black', linewidths=0.5)  # Bright green with black outline

    # For the 'sell' trades, using a brighter shade of red
    ax_main.scatter(sell_trades.timestamp_trades, sell_trades.Price, label='Sell', color='#ff9c9c', marker="v", s=80, zorder=2,edgecolors='black', linewidths=0.5)  # Bright green with black outline


    ax_main.fill_between(timestamps_s, asks, bids, color='lightgray', step='post', alpha=0.3)

    ax_main.set_ylabel('Price')
    ax_main.set_title(f'{symbol} Order Book')
    ax_main.set_xticks([])
    # Removing spines from the main plot
    for spine_location, spine in ax_main.spines.items():
        if spine_location == 'bottom':
            # make the spine 50% transparent
            spine.set_alpha(0.5)
        else:
            spine.set_visible(False)
    # Optionally, adding grid
    plt.grid(True)

    # Show the plot
    plt.legend()


    ax_second = plt.subplot2grid(gridsize, (3, 0))


    # Small subplot for Rolling Averages
    bids_single_size = small_sample.Bid1SizeBillionths
    asks_single_size = small_sample.Ask1SizeBillionths
    bids_full_size = small_sample.Bid1SizeBillionths + small_sample.Bid2SizeBillionths
    asks_full_size = small_sample.Ask1SizeBillionths + small_sample.Ask2SizeBillionths

    book_bias = (asks_full_size - bids_full_size).rolling(50).mean()
    min_max_bias = ((book_bias - book_bias.min()) / (book_bias.max() - book_bias.min()) - 0.5) * 2
    #plt dashed line at 0
    ax_second.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax_second.set_ylabel('Book Bias')
    ax_second.fill_between(timestamps_s, 0, min_max_bias, where=(min_max_bias > 0), color='#2ca02c', alpha=0.3)
    ax_second.fill_between(timestamps_s, 0, min_max_bias, where=(min_max_bias < 0), color='#d62728', alpha=0.3)

    for spine_location, spine in ax_second.spines.items():
        if spine_location != 'bottom':
            spine.set_visible(False)
        
    # Adding labels and title
    ax_second.set_xlabel(f'Nanoseconds elapsed since {formatted_time}' )
    ax_second.grid(axis='y', linestyle='--', alpha=0.8)
    plt.tight_layout()

    plt.text(0.1, -0.3, r'[1]  $book\_bias_x= 2*\left(\frac{x - \min\{x|\forall x \in X\}}{\max\{x|\forall x \in X\} - min\{x|\forall x \in X\}} - 0.5\right)$', 
                ha='center', va='top', transform=plt.gca().transAxes,fontsize=8, color='grey')

    plt.show()



def plot_theo_lagg(lagged_df, crossed, title):
    lagged_df = lagged_df.copy(deep=True)
    lagged_df.index /= 1e+9
    start_ts = lagged_df.index[0]
    lagged_df.index -= start_ts
    theo = lagged_df["theo_value"]
    midprice = lagged_df["mid_price"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(lagged_df.index, theo, label="Theoretical Price")
    ax[0].plot(lagged_df.index, midprice, label="Mid Price", color="black", linestyle="--")
    ax[0].set_ylabel("Price")
    ax[0].grid()
    ax[0].legend()
    # remove spines
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)

    squared_errors = (theo - midprice) ** 2
    ax[1].plot(squared_errors, label="Squared Error", color="black", linewidth=0.5)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    
    # make area under the curve red
    ax[1].fill_between(squared_errors.index, squared_errors, 0, where=squared_errors > 0, facecolor='red', alpha=0.5)
    ax[1].set_xlabel("Seconds from start")
    ax[1].set_ylabel("Squared Error")
    ax[1].grid()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_butterfly_size(book_data, title):
    book_data = book_data.copy(deep=True)

    bid_sizes = book_data[['Bid1SizeBillionths', 'Bid2SizeBillionths', 'Bid3SizeBillionths']].rolling(window=100).mean()/1_000_000
    ask_sizes = book_data[['Ask1SizeBillionths', 'Ask2SizeBillionths', 'Ask3SizeBillionths']].rolling(window=100).mean()/1_000_000

    book_data.received_utc_microseconds = (book_data.received_utc_microseconds - book_data.received_utc_microseconds.iat[0])/1e+6
    start_ts = book_data.received_utc_microseconds.iat[0]


    plt.figure(figsize=(10, 5))
    plt.title(title)
    # draw line at 0
    plt.axhline(0, color='black', lw=2)

    plt.fill_between(book_data.received_utc_microseconds, 0, bid_sizes.Bid1SizeBillionths, color="green", alpha=0.5, label="Bid1")
    plt.fill_between(book_data.received_utc_microseconds, bid_sizes.Bid1SizeBillionths+bid_sizes.Bid2SizeBillionths, color="green", alpha=0.3, label="Bid2")
    plt.fill_between(book_data.received_utc_microseconds, bid_sizes.Bid1SizeBillionths+bid_sizes.Bid2SizeBillionths,  bid_sizes.sum(axis=1), color="green", alpha=0.1, label="Bid3")

    # do asks
    plt.fill_between(book_data.received_utc_microseconds, 0, -ask_sizes.Ask1SizeBillionths, color="red", alpha=0.5, label="Ask1")
    plt.fill_between(book_data.received_utc_microseconds, -ask_sizes.Ask1SizeBillionths, -ask_sizes.Ask1SizeBillionths-ask_sizes.Ask2SizeBillionths, color="red", alpha=0.3, label="Ask2")
    plt.fill_between(book_data.received_utc_microseconds, -ask_sizes.Ask1SizeBillionths-ask_sizes.Ask2SizeBillionths,  -ask_sizes.sum(axis=1), color="red", alpha=0.1, label="Ask3")

    # format y-axis by applying absolute value
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(abs(int(x)))))
    
    plt.xlabel("Time (in seconds since start)")
    plt.ylabel("Quantity of limit orders")
    plt.legend()
    plt.grid()
    plt.show()