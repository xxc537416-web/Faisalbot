import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import ta # Technical Analysis library
import mplfinance as mpf
import time

# --- 1. CONFIGURATION & DARK MODE CSS INJECTION ---

# Set wide layout and page title
st.set_page_config(layout="wide", page_title="Premium Algo Dashboard")

# Inject professional Dark Mode CSS for Chrome/Streamlit viewing
st.markdown("""
<style>
/* 1. Sets the background for the entire application to a deep charcoal/black */
.stApp {
    background-color: #0e1117; 
    color: #FAFAFA; /* Light text color */
}
/* 2. Styles for the sidebar and main content areas for consistency */
.main, .css-1d3f8gq, .css-18e3th9 {
    background-color: #0e1117;
}
/* 3. Styles for interactive elements like buttons and tabs */
.stTabs [data-baseweb="tab-list"] button, .stButton>button {
    background-color: #262730; 
    color: #FAFAFA !important; 
    border-radius: 5px;
}
/* 4. Ensures text within tables, headers, and general content is white */
.dataframe, h1, h2, h3, p, .css-1qxtsq9 {
    color: #FAFAFA !important; 
}
</style>
""", unsafe_allow_html=True)


# --- 2. CORE CLASSES (DataFetcher, StrategyLogic, Backtester) ---

class DataFetcher:
    """Fetches historical OHLCV data with robust error handling."""
    def __init__(self, ticker: str, timeframe: str = '1h'):
        self.ticker = ticker
        self.timeframe = timeframe

    @st.cache_data
    def fetch_data(_self, period='6mo') -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            data = yf.download(_self.ticker, period=period, interval=_self.timeframe, progress=False)
            if data.empty:
                st.error(f"ERROR: No data found for {_self.ticker}")
                return pd.DataFrame()
            data.columns = [c.capitalize() for c in data.columns]
            return data
        except Exception as e:
            st.error(f"[DataFetcher ERROR] {e}")
            return pd.DataFrame()

class StrategyLogic:
    """Calculates CORRECT indicators and generates BUY/SELL signals."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.signals = pd.DataFrame(index=self.df.index)
        self.rsi_period = 14
        self.sma_period = 50
        
    def compute_indicators(self):
        """Compute CORRECT RSI(14) using 'ta' library and SMA(50)."""
        self.df['RSI'] = ta.momentum.RSIIndicator(close=self.df['Close'], window=self.rsi_period, fillna=False).rsi()
        self.df['SMA50'] = self.df['Close'].rolling(self.sma_period).mean()

    def generate_signals(self):
        """Generate BUY/SELL signals based on the strategy rules."""
        self.compute_indicators()
        df_clean = self.df.dropna()
        
        # BUY Logic: RSI crosses above 60 AND Close > SMA50
        buy_condition = ((df_clean['RSI'] > 60) & 
                         (df_clean['RSI'].shift(1) <= 60) &
                         (df_clean['Close'] > df_clean['SMA50'])).astype(int)
                         
        # SELL/EXIT Logic: RSI crosses below 40
        sell_condition = ((df_clean['RSI'] < 40) & 
                          (df_clean['RSI'].shift(1) >= 40)).astype(int)
                          
        self.signals = pd.DataFrame(index=df_clean.index)
        self.signals['BUY'] = buy_condition
        self.signals['SELL'] = sell_condition
        
        return self.signals

class Backtester:
    """Performs realistic backtesting using High/Low SL/TP checks."""
    def __init__(self, df: pd.DataFrame, signals: pd.DataFrame, sl_pct: float, tp_pct: float):
        self.df = df.loc[signals.index]
        self.signals = signals
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.trades = []

    def run_backtest(self):
        """Simulates trades with priority given to SL/TP over strategy exit."""
        position = None
        
        for idx in range(len(self.df)):
            current_candle = self.df.iloc[idx]
            current_signals = self.signals.iloc[idx]
            
            # --- 1. Position Management (Realistic Exit) ---
            if position:
                # Check High/Low for realistic SL/TP hits within the candle
                hit_sl = current_candle['Low'] <= position['sl']
                hit_tp = current_candle['High'] >= position['tp']
                exit_signal = current_signals['SELL']
                
                exit_price = None
                exit_reason = None
                
                # Priority 1: SL/TP
                if hit_sl:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif hit_tp:
                    exit_price = position['tp']
                    exit_reason = 'TP'
                
                # Priority 2: Strategy Exit
                elif exit_signal:
                    exit_price = current_candle['Close']
                    exit_reason = 'Strategy'

                if exit_price:
                    trade_return = (exit_price - position['entry_price']) / position['entry_price']
                    self.trades.append({
                        'entry_date': self.df.index[position['entry_idx']],
                        'exit_date': self.df.index[idx],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return': trade_return,
                        'reason': exit_reason,
                    })
                    position = None 

            # --- 2. Enter Buy ---
            if current_signals['BUY'] and position is None:
                entry_price = current_candle['Close']
                sl = entry_price * (1 - self.sl_pct)
                tp = entry_price * (1 + self.tp_pct)
                position = {'entry_idx': idx, 'entry_price': entry_price, 'sl': sl, 'tp': tp}

        return pd.DataFrame(self.trades)

    def calculate_metrics(self, trades_df: pd.DataFrame):
        """Compute CAGR, Sharpe Ratio, Max Drawdown, and Success Rate."""
        if trades_df.empty: 
             return {'Error': 'No trades executed.', 'CAGR': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Total Trades': 0, 'Success Rate (TP vs SL)': 0}

        trades_df['cumulative'] = (1 + trades_df['return']).cumprod()
        total_years = (self.df.index[-1] - self.df.index[0]).days / 365
        cagr = trades_df['cumulative'].iloc[-1] ** (1 / total_years) - 1

        annualized_return = trades_df['return'].mean() * 252 
        risk_free_rate = 0.03
        annualized_volatility = trades_df['return'].std() * np.sqrt(252)
        sharpe = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility else 0

        cumulative_max = trades_df['cumulative'].cummax()
        max_dd = (trades_df['cumulative'] / cumulative_max - 1).min()

        tp_hits = sum(trades_df['reason'] == 'TP')
        sl_hits = sum(trades_df['reason'] == 'SL')
        total_risk_trades = tp_hits + sl_hits
        success_rate = tp_hits / total_risk_trades * 100 if total_risk_trades > 0 else 0

        return {
            'CAGR': f"{cagr * 100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd * 100:.2f}%",
            'Total Trades': len(trades_df),
            'Success Rate (TP vs SL)': f"{success_rate:.2f}%"
        }

# --- 3. TRADE EXECUTOR (Placeholder for Live API) ---
class TradeExecutor:
    """Placeholder for live trading (Needs real API integration)."""
    def place_buy_order(self, price: float, quantity: int):
        st.info(f"LIVE EXECUTION SIMULATED: Placed BUY order at ${price:.2f} for {quantity} units.")


# --- 4. STREAMLIT FRONT-END LOGIC ---

def run_strategy_and_backtest(ticker, period, timeframe, sl_pct, tp_pct):
    """Main function to run the strategy and return results for Streamlit."""
    
    # 1. Data Fetching
    fetcher = DataFetcher(ticker, timeframe)
    data = fetcher.fetch_data(period=period)
    
    if data.empty:
        return None, None, None

    # 2. Strategy & Signals
    strategy = StrategyLogic(data)
    signals = strategy.generate_signals()
    
    # 3. Backtest
    backtester = Backtester(data, signals, sl_pct, tp_pct)
    trades_df = backtester.run_backtest()
    metrics = backtester.calculate_metrics(trades_df)
    
    return data, trades_df, metrics

def display_dashboard():
    """Renders the Streamlit application."""
    st.title("🛡️ PREMIUM Algo-Trading Dashboard")
    st.subheader("RSI/SMA Strategy Analysis (with Realistic H/L SL/TP)")

    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("⚙️ Settings")
        ticker = st.text_input("Ticker Symbol", "BTC-USD")
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
        period = st.selectbox("Historical Data Period", ["6mo", "1y", "2y"], index=0)
        
        st.subheader("Risk Management")
        # Use slider for better user experience
        sl_pct_input = st.slider("Stop Loss (%)", 1.0, 5.0, 1.5, step=0.1)
        tp_pct_input = st.slider("Take Profit (%)", 1.0, 10.0, 3.0, step=0.1)
        
        # Convert percentages to decimals for the Backtester
        sl_pct = sl_pct_input / 100
        tp_pct = tp_pct_input / 100
        
        if st.button("Run Strategy & Backtest", use_container_width=True):
            st.session_state.run_test = True
            
    # --- Main Display ---
    if st.session_state.get('run_test', False):
        st.markdown("---")
        
        data, trades_df, metrics = run_strategy_and_backtest(ticker, period, timeframe, sl_pct, tp_pct)
        
        if data is None:
            return

        # 1. Metrics Display
        st.header("📈 Key Metrics (Premium Analysis)")
        cols = st.columns(5)
        
        # Displaying key metrics using colored indicators
        cols[0].metric("CAGR (Annual Return)", metrics.get('CAGR', 'N/A'))
        cols[1].metric("Sharpe Ratio", metrics.get('Sharpe Ratio', 'N/A'))
        cols[2].metric("Max Drawdown (MDD)", metrics.get('Max Drawdown', 'N/A'))
        cols[3].metric("Total Trades", metrics.get('Total Trades', 'N/A'))
        cols[4].metric("Success Rate (TP vs SL)", metrics.get('Success Rate (TP vs SL)', 'N/A'))

        st.markdown("---")

        # 2. Visualization
        st.header("📊 Chart & Signal Visualization")
        
        try:
            ap = [
                mpf.make_addplot(data['SMA50'].dropna(), color='#007bff', panel=0, type='line'),
                mpf.make_addplot(data['RSI'].dropna(), color='purple', panel=1, ylabel='RSI', 
                                 ylim=(0, 100), hlines={'hlines':[60, 40], 'colors':['g','r'], 'linewidths':0.5}),
            ]
            
            # --- Trade Annotations (Arrows) ---
            if not trades_df.empty:
                buy_points = pd.Series(np.nan, index=data.index)
                sell_points = pd.Series(np.nan, index=data.index)
                
                for idx, trade in trades_df.iterrows():
                    # Place entry arrow slightly below the candle low
                    buy_points.loc[trade['entry_date']] = data['Low'].loc[trade['entry_date']] * 0.995 
                    # Place exit arrow slightly above the candle high
                    sell_points.loc[trade['exit_date']] = data['High'].loc[trade['exit_date']] * 1.005 

                ap.append(mpf.make_addplot(buy_points.dropna(), type='scatter', marker='^', markersize=200, color='green', panel=0))
                ap.append(mpf.make_addplot(sell_points.dropna(), type='scatter', marker='v', markersize=200, color='red', panel=0))

            fig, axlist = mpf.plot(data, type='candle', addplot=ap, style='yahoo', figsize=(18, 10),
