# =========================================
# Ultimate AI Investment Bot - Single File Version
# =========================================

# ------------------- IMPORTS -------------------
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from flask import Flask
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import os

# ------------------- MOCK API -------------------
def fetch_ohlcv(ticker):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=180)
    close = np.cumsum(np.random.randn(180)) + 100
    open_ = close + np.random.randn(180)
    high = np.maximum(open_, close) + np.random.rand(180)
    low = np.minimum(open_, close) - np.random.rand(180)
    volume = np.random.randint(1e5, 1e6, 180)
    df = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)
    return df

def fetch_fundamentals(ticker):
    return {'PE': np.random.rand()*30, 'ROE': np.random.rand()*20, 'DebtEquity': np.random.rand()*2}

def fetch_sentiment(ticker):
    return np.random.rand()

# ------------------- DATA PROCESSOR -------------------
class DataProcessor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.fundamentals = None
        self.sentiment_score = None

    def fetch_data(self):
        self.data = fetch_ohlcv(self.ticker)
        self.fundamentals = fetch_fundamentals(self.ticker)
        self.sentiment_score = fetch_sentiment(self.ticker)
        return self.data, self.fundamentals, self.sentiment_score

    def feature_engineering(self):
        df = self.data.copy()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df.fillna(method='bfill', inplace=True)
        self.data = df
        return df

    def get_sentiment(self):
        score = self.sentiment_score
        label = 'Bullish' if score > 0.6 else 'Bearish' if score < 0.4 else 'Neutral'
        return score, label

# ------------------- AI MODEL -------------------
class AIModule:
    def __init__(self):
        self.model = None
        self.scaler = None

    def prepare_data(self, df, feature_cols=None):
        if feature_cols is None:
            feature_cols = ['Close','RSI','MACD','MACD_Signal','BB_High','BB_Low']
        data = df[feature_cols]
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(30,len(scaled)):
            X.append(scaled[i-30:i])
            y.append(scaled[i,0])
        return np.array(X), np.array(y)

    def build_lstm(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def train(self, X, y, epochs=10, batch_size=16):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict_future(self, recent_data):
        feature_cols = ['Close','RSI','MACD','MACD_Signal','BB_High','BB_Low']
        scaled = self.scaler.transform(recent_data[feature_cols])
        X_input = np.expand_dims(scaled[-30:], axis=0)
        preds_scaled = self.model.predict(X_input)
        preds = self.scaler.inverse_transform(np.concatenate([preds_scaled, np.zeros((preds_scaled.shape[0],5))], axis=1))[:,0]
        expected_change = (preds[-1]-recent_data['Close'].iloc[-1])/recent_data['Close'].iloc[-1]*100
        return expected_change

    def calculate_var(self, returns, confidence=0.95):
        var = np.percentile(returns,(1-confidence)*100)
        if var > -0.02:
            return 'Low'
        elif var > -0.05:
            return 'Medium'
        else:
            return 'High'

    def expected_gain(self, trade_size, predicted_change_pct, confidence, loss_down):
        gain = trade_size*(predicted_change_pct/100)
        expected = confidence*gain - (1-confidence)*loss_down
        return expected

    def up_down_signal(self, predicted_change):
        return 'Up' if predicted_change>0 else 'Down'

# ------------------- DASH LAYOUT -------------------
def create_layout():
    layout = html.Div([
        html.H1('Ultimate AI Investment Bot', style={'textAlign':'center'}),
        html.Div([
            dcc.Input(id='ticker-input', type='text', placeholder='Enter Stock Ticker', style={'width':'50%'}),
            dcc.Input(id='trade-size', type='number', placeholder='Trade Size', style={'width':'20%'}),
            html.Button('Analyse', id='analyse-btn', n_clicks=0)
        ], style={'textAlign':'center','margin':'20px'}),
        dcc.Interval(id='interval-update', interval=60000, n_intervals=0),
        html.Div([dcc.Graph(id='candlestick-chart')]),
        html.Div([
            html.Div(id='predicted-change', className='card', style={'display':'inline-block','width':'24%'}),
            html.Div(id='sentiment-score', className='card', style={'display':'inline-block','width':'24%'}),
            html.Div(id='risk-level', className='card', style={'display':'inline-block','width':'24%'}),
            html.Div(id='expected-gain', className='card', style={'display':'inline-block','width':'24%'}),
            html.Div(id='confidence-score', className='card', style={'display':'inline-block','width':'24%'})
        ], style={'textAlign':'center','margin':'20px'})
    ])
    return layout

# ------------------- FLASK + DASH -------------------
server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/')
app.layout = create_layout()

@app.callback(
    Output('candlestick-chart','figure'),
    Output('predicted-change','children'),
    Output('sentiment-score','children'),
    Output('risk-level','children'),
    Output('expected-gain','children'),
    Output('confidence-score','children'),
    Input('analyse-btn','n_clicks'),
    Input('interval-update','n_intervals'),
    State('ticker-input','value'),
    State('trade-size','value')
)
def update_dashboard(n_clicks, n_intervals, ticker, trade_size):
    if not ticker or not trade_size:
        return go.Figure(), '', '', '', '', ''

    dp = DataProcessor(ticker)
    df, fundamentals, sentiment_score = dp.fetch_data()
    df = dp.feature_engineering()
    sentiment_val, sentiment_label = dp.get_sentiment()

    ai = AIModule()
    X, y = ai.prepare_data(df)
    ai.build_lstm(X.shape[1:])
    ai.train(X, y, epochs=5)
    predicted_change = ai.predict_future(df)
    signal = ai.up_down_signal(predicted_change)

    returns = df['Close'].pct_change().dropna().values
    risk_level = ai.calculate_var(returns)

    confidence = 0.8
    loss_down = trade_size*3
    expected_gain = ai.expected_gain(trade_size, predicted_change, confidence, loss_down)

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='green', decreasing_line_color='red')])
    fig.update_layout(x