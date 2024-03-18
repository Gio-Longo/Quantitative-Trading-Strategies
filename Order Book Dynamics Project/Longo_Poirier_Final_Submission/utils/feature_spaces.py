from scipy.optimize import fsolve
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import List, Dict, Callable

class FeatureSpace(BaseEstimator, TransformerMixin):
    features = {}

    def _collide_trades_n_books(self, trades: pd.DataFrame, books: pd.DataFrame)->pd.DataFrame:
        """This method consolidates the trade and book data into a single dataframe.

        Args:
            trades (pd.DataFrame): The trade data
            books (pd.DataFrame): The book data

        Returns:
            pd.DataFrame: A dataframe containing the consolidated data
        """
        return pd.merge_asof(trades, books, on="timestamp_utc_nanoseconds", direction="backward")

    def fit(self, X, y=None):
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
    
    def info(self):
        info_df = pd.DataFrame(self.features, index=["output column name"])
        info_df.columns.name = "method name"
        return info_df
    

class FirstFeatureSpace(FeatureSpace):
    features = {
        "midprice": "midprice",
        "vwap": "vwap",
        "bid_ask_spread": "bid_ask_spread",
        "delta_eigvals": "delta_eigval_<ith_eigval>", 
    }
    PRICE_COLS = ['Bid3PriceMillionths', 'Bid2PriceMillionths', 'Bid1PriceMillionths', 'Ask1PriceMillionths', 'Ask2PriceMillionths', 'Ask3PriceMillionths']
    BID_PRICE_COLS = PRICE_COLS[:3]
    ASK_PRICE_COLS = PRICE_COLS[3:]
    BEST_BID_PRICE_COL = "Bid1PriceMillionths"
    BEST_ASK_PRICE_COL = "Ask1PriceMillionths"

    QUANTITY_COLS = ['Bid3SizeBillionths', 'Bid2SizeBillionths', 'Bid1SizeBillionths', 'Ask1SizeBillionths', 'Ask2SizeBillionths', 'Ask3SizeBillionths']
    BID_QUANTITY_COLS = QUANTITY_COLS[:3]
    ASK_QUANTITY_COLS = QUANTITY_COLS[3:]
    BEST_BID_QUANTITY_COL = "Bid1SizeBillionths"
    BEST_ASK_QUANTITY_COL = "Ask1SizeBillionths"

    def __init__(
            self, 
            *, 
            eigvals:                List[int] = [1,2], 
            eigval_smooth_window:   int =       10, 
            ar_windows:             List[int] = [1e+6, 5e+7, 6e+10], 
            future_vol_window:      int =       5e+7, 
            ar_model:               Callable =  LinearRegression, 
            **ar_params:            Dict,
        ) -> None:
        """Innitiate the InitialFeatureSpace object.

        Args:
            eigvals (List[int], optional): The different eigenvalues from the delta matrix we want to extract as features. Defaults to [1,2].
            ar_windows (List[int], optional): The different timescales we want to evaluate in nanoseconds. Defaults to [1e+6, 5e+7, 6e+10].
            future_vol_window (int, optional): The window size for the future volatility. Defaults to 5e+7.
        """
        super().__init__()
        self.eigvals = eigvals
        self.eigval_smooth_window = eigval_smooth_window
        self.ar_windows = ar_windows
        self.future_vol_window = future_vol_window
        self.ar_model = ar_model(**ar_params)
        self.imputer = SimpleImputer(strategy="mean")

    ### Other methods ###
    def _compute_deltas(self, books: pd.DataFrame)->np.ndarray:
        """This method makes an estimation on the value of incoming limit orders, and computes their distance to the midprice.

        Args:
            books (pd.DataFrame): A book dataframe that contains the necessary columns to compute the deltas.

        Returns:
            np.ndarray: 1-to-1 mapping of the deltas to the books dataframe.
        """
        qty_diffs = books[self.QUANTITY_COLS].diff()
        qty_diffs.iloc[0] = 0
        midprice = self._midprice(
            best_ask_col = books[self.BEST_ASK_PRICE_COL], 
            best_bid_col = books[self.BEST_BID_PRICE_COL], 
            weights = pd.Series(0.5, index=books.index))
        deltas = (qty_diffs.abs().values * books[self.PRICE_COLS].subtract(midprice,axis=0).abs().values).sum(axis=1) / qty_diffs.abs().sum(axis=1).values

        return deltas

    @staticmethod
    def rolling_quadratic_difference_avg(df: pd.DataFrame, delta: int, forward: bool = True)->pd.DataFrame:
        df = df.copy(deep=True)
        rolling_df = pd.DataFrame(index = df.index, columns = df.columns)
        for i, row in df.iterrows():
            lower_bound = row['timestamp_utc_nanoseconds'] + (- delta if not forward else 0)
            upper_bound = row['timestamp_utc_nanoseconds'] + (+ delta if forward else 0)
            
            window_data = df[(df['timestamp_utc_nanoseconds'] > lower_bound) & (df['timestamp_utc_nanoseconds'] <= upper_bound)]

            mean_value = (window_data.diff()**2).sum()/np.sqrt(delta)
            rolling_df.loc[i] = mean_value
        return rolling_df

    #####################

    ## Feature methods ##
    def _midprice(self, best_ask_col: np.ndarray, best_bid_col: np.ndarray, weights: np.ndarray)->np.ndarray:
        return best_ask_col*(weights) + best_bid_col*(1-weights)
    def _vwap(self, price_mat: np.ndarray, quantity_mat: np.ndarray)->np.ndarray:
        return np.sum(price_mat/1_000_000 * quantity_mat/1_000_000_000, axis=1) / np.sum(quantity_mat/1_000_000_000, axis=1)*1_000_000
    def _bid_ask_spread(self, best_ask_col: np.ndarray, best_bid_col: np.ndarray)->np.ndarray:
        return best_ask_col - best_bid_col
    def _delta_eigvals(self, books: np.ndarray, window: int)->np.ndarray:
        deltas = self._compute_deltas(books)
        eigvals = np.zeros((window, window))      
        for i in tqdm(range(window, deltas.shape[0]), desc="Computing delta eigenvalues"):
            moves = deltas[i-window:i]
            # replace nan with mean observed so far
            moves = np.nan_to_num(moves, nan=np.nanmean(deltas[:i])).reshape(1, -1)
            covmat = moves.T @ moves
            eigs = np.linalg.eigvals(covmat)
            eigvals = np.vstack((eigvals, eigs))
        return eigvals
    def _rolling_ar_vol(self, returns: np.ndarray, window: int)->np.ndarray:
        colname = returns.columns[0] if returns.columns[0] != "timestamp_utc_nanoseconds" else returns.columns[1]
        _temp_df = pd.DataFrame(index = returns.index, columns = ["timestamp_utc_nanoseconds"] + [f"rolling_ar_vol_{_window}ns" for _window in self.ar_windows])
        _temp_df["timestamp_utc_nanoseconds"] = returns["timestamp_utc_nanoseconds"]
        for _window in self.ar_windows:
            _temp_df[f"rolling_vol_{window}ns"] = self.rolling_quadratic_difference_avg(returns, _window, forward=False)[colname]
        
        _temp_df["forward_vol"] = self.rolling_quadratic_difference_avg(returns, self.future_vol_window, forward=True)[colname]
        _temp_df["const"] = 1

        regressor_cols = [f"rolling_ar_vol_{_window}ns" for _window in self.ar_windows] + ["const"]
        regressors = self.imputer.fit_transform(_temp_df[regressor_cols])
        self.ar_model.fit(regressors, _temp_df["forward_vol"])
        return self.ar_model.predict(regressors)
    #####################

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        assert "books" in X, "The input dataframe must contain a 'books' column"
        assert "trades" in X, "The input dataframe must contain a 'trades' column"
        assert "timestamp_utc_nanoseconds" in X["books"].columns, "The input dataframe must contain a 'timestamp_utc_nanoseconds' column"
        assert "timestamp_utc_nanoseconds" in X["trades"].columns, "The input dataframe must contain a 'timestamp_utc_nanoseconds' column"

        assert set(self.PRICE_COLS).issubset(X["books"].columns), f"The input books dataframe must contain the following columns: {self.PRICE_COLS}"
        assert set(self.QUANTITY_COLS).issubset(X["books"].columns), f"The input books dataframe must contain the following columns: {self.QUANTITY_COLS}"

        _X = pd.DataFrame(index = X["books"].index)
        _X["timestamp_utc_nanoseconds"] = X["books"]["timestamp_utc_nanoseconds"]
        # Consolidating the price and quantity matrices
        
        # Applying features
        #1. Midprice
        weights = X["books"][self.BEST_ASK_QUANTITY_COL] / (X["books"][self.BEST_ASK_QUANTITY_COL] + X["books"][self.BEST_BID_QUANTITY_COL])

        _X["midprice"] = self._midprice(best_ask_col = X["books"][self.BEST_ASK_PRICE_COL], best_bid_col = X["books"][self.BEST_BID_PRICE_COL], weights = weights)

        #2. VWAP
        price_mat = X["books"][self.PRICE_COLS].values
        quantity_mat = X["books"][self.QUANTITY_COLS].values
        _X["vwap"] = self._vwap(price_mat = price_mat, quantity_mat = quantity_mat)

        #3. Bid-ask spread
        _X["bid_ask_spread"] = self._bid_ask_spread(best_ask_col = X["books"][self.BEST_ASK_PRICE_COL], best_bid_col = X["books"][self.BEST_BID_PRICE_COL])

        #4. Delta eigenvalues
        eigs = self._delta_eigvals(books = X["books"], window = 5)
        for i in self.eigvals:
            _X[f"delta_eigval_{i}"] = eigs[:, i]
            _X[f"delta_eigval_{i}"] = _X[f"delta_eigval_{i}"].astype(float)

        #5. Rolling AR volatility
        #_X["rolling_ar_vol"] = self._rolling_ar_vol(returns = _X[["vwap","timestamp_utc_nanoseconds"]], window = 5)
        _X.drop(columns=["timestamp_utc_nanoseconds"], inplace=True)
        return _X

class SecondFeatureSpace(FirstFeatureSpace):
    features = {
        "midprice": "midprice",
        "vwap": "vwap",
        "bid_ask_spread": "bid_ask_spread",
        "delta_eigvals": "delta_eigval_<ith_eigval>", 
        "poisson_map": "poisson_map"
    }
    PRICE_COLS = ['Bid3PriceMillionths', 'Bid2PriceMillionths', 'Bid1PriceMillionths', 'Ask1PriceMillionths', 'Ask2PriceMillionths', 'Ask3PriceMillionths']
    BID_PRICE_COLS = PRICE_COLS[:3]
    ASK_PRICE_COLS = PRICE_COLS[3:]
    BEST_BID_PRICE_COL = "Bid1PriceMillionths"
    BEST_ASK_PRICE_COL = "Ask1PriceMillionths"

    QUANTITY_COLS = ['Bid3SizeBillionths', 'Bid2SizeBillionths', 'Bid1SizeBillionths', 'Ask1SizeBillionths', 'Ask2SizeBillionths', 'Ask3SizeBillionths']
    BID_QUANTITY_COLS = QUANTITY_COLS[:3]
    ASK_QUANTITY_COLS = QUANTITY_COLS[3:]
    BEST_BID_QUANTITY_COL = "Bid1SizeBillionths"
    BEST_ASK_QUANTITY_COL = "Ask1SizeBillionths"

    def __init__(
            self, 
            *, 
            eigvals:                List[int] = [1,2], 
            eigval_smooth_window:   int =       10
        ) -> None:
        """Innitiate the InitialFeatureSpace object.

        Args:
            eigvals (List[int], optional): The different eigenvalues from the delta matrix we want to extract as features. Defaults to [1,2].

        Examples:

        ```python
        # Example 1
        feature_space = SecondFeatureSpace(eigvals=[1,2], eigval_smooth_window=10)
        ```

        """
        super().__init__()
        self.eigvals = eigvals
        self.eigval_smooth_window = eigval_smooth_window

    def _poisson_map(self):
        return None

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        assert "books" in X, "The input dataframe must contain a 'books' column"
        assert "trades" in X, "The input dataframe must contain a 'trades' column"
        assert "timestamp_utc_nanoseconds" in X["books"].columns, "The input dataframe must contain a 'timestamp_utc_nanoseconds' column"
        assert "timestamp_utc_nanoseconds" in X["trades"].columns, "The input dataframe must contain a 'timestamp_utc_nanoseconds' column"

        assert set(self.PRICE_COLS).issubset(X["books"].columns), f"The input books dataframe must contain the following columns: {self.PRICE_COLS}"
        assert set(self.QUANTITY_COLS).issubset(X["books"].columns), f"The input books dataframe must contain the following columns: {self.QUANTITY_COLS}"

        _X = pd.DataFrame(index = X["books"].index)
        _X["timestamp_utc_nanoseconds"] = X["books"]["timestamp_utc_nanoseconds"]
        # Consolidating the price and quantity matrices
        
        # Applying features
        #1. Midprice
        weights = X["books"][self.BEST_ASK_QUANTITY_COL] / (X["books"][self.BEST_ASK_QUANTITY_COL] + X["books"][self.BEST_BID_QUANTITY_COL])

        _X["midprice"] = self._midprice(best_ask_col = X["books"][self.BEST_ASK_PRICE_COL], best_bid_col = X["books"][self.BEST_BID_PRICE_COL], weights = weights)

        #2. VWAP
        price_mat = X["books"][self.PRICE_COLS].values
        quantity_mat = X["books"][self.QUANTITY_COLS].values
        _X["vwap"] = self._vwap(price_mat = price_mat, quantity_mat = quantity_mat)

        #3. Bid-ask spread
        _X["bid_ask_spread"] = self._bid_ask_spread(best_ask_col = X["books"][self.BEST_ASK_PRICE_COL], best_bid_col = X["books"][self.BEST_BID_PRICE_COL])

        #4. Delta eigenvalues
        eigs = self._delta_eigvals(books = X["books"], window = 5)
        for i in self.eigvals:
            _X[f"delta_eigval_{i}"] = eigs[:, i]
            _X[f"delta_eigval_{i}"] = _X[f"delta_eigval_{i}"].astype(float)

        #5. Poisson MAP
        
        
        return _X


SecondFeatureSpace()
