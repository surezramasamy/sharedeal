import os
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import requests
import yfinance as yf
from textblob import TextBlob
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import threading
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, text, func, Date, Text, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager


# -------------------------------------------------
# CONFIG & LOGGING
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StockApp")


DATABASE_URL = "sqlite:///./databases/stock_recommender.db"
NSE_CSV_PATH = "nse.csv"


# API Keys
NEWSAPI_KEY = "f44ef3442436422983ac6a1c353e5f21"
MARKETAUX_KEY = "yspwIQtwoqi3MgTAHYBwK9XKjkGjPboN4ad4TfNq"
FINNHUB_KEY = "d4vam39r01qnm7pqkmcgd4vam39r01qnm7pqkmd0"
NEWSDATA_KEY = "pub_bdc5fdeb66494d93b9468dbf758fd615"


# -------------------------------------------------
# DATABASE SETUP
# -------------------------------------------------
if not os.path.exists("./databases"):
    os.makedirs("./databases")


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class StockModel(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True)
    company_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class StockScoreModel(Base):
    __tablename__ = "stock_scores"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    company_name = Column(String)
    current_price = Column(Float, default=0.0)  # NEW: Store current price
    sentiment_score = Column(Float)
    technical_score = Column(Float)
    final_score = Column(Float)
    recommendation = Column(String)
    sentiment = Column(String)
    technical = Column(String)
    rsi = Column(Float)
    cci = Column(Float)
    macd_signal = Column(String)
    eps = Column(Float, default=0.0)
    pe = Column(Float, default=0.0)
    sma_signal = Column(String)
    bb_signal = Column(String)
    volume_signal = Column(String)
    news_count = Column(Integer)
    data_source = Column(String)
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)


class StockPredictionModel(Base):
    __tablename__ = "stock_predictions"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    prediction_date = Column(Date, default=datetime.utcnow().date, index=True)
    current_price = Column(Float)
    forecast_json = Column(String)
    chart_base64 = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint('ticker', 'prediction_date', name='unique_ticker_date'),)


Base.metadata.create_all(bind=engine)


with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS stock_predictions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            prediction_date DATE,
            current_price REAL,
            forecast_json TEXT,
            chart_base64 TEXT,
            created_at DATETIME,
            UNIQUE(ticker, prediction_date)
        )
    """))
    conn.commit()


# -------------------------------------------------
# DB UTILS
# -------------------------------------------------
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------
# APP & TEMPLATES
# -------------------------------------------------
templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# -------------------------------------------------
# LOAD NSE STOCKS LIST
# -------------------------------------------------
def load_stock_pool() -> List[Dict]:
    if not os.path.exists(NSE_CSV_PATH): return []
    try:
        df = pd.read_csv(NSE_CSV_PATH, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if "SYMBOL" in df.columns:
            df = df.rename(columns={"SYMBOL": "ticker", "NAME OF COMPANY": "name"})
        elif "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "ticker", "Company Name": "name"})
        df["ticker"] = df["ticker"].str.strip() + ".NS"
        df["name"] = df["name"].str.strip()
        return df.drop_duplicates(subset=["ticker"]).to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return []
stock_pool = load_stock_pool()


# -------------------------------------------------
# NEWS & SENTIMENT
# -------------------------------------------------
def fetch_news(company_name: str) -> List[str]:
    articles = []
    if not MARKETAUX_KEY:
        return []
    try:
        params = {
            "api_token": MARKETAUX_KEY,
            "search": company_name,
            "language": "en",
            "limit": 5
        }
        r = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            articles = [a["title"] for a in data["data"] if "title" in a][:5]
    except Exception as e:
        logger.error(f"News fetch failed for {company_name}: {e}")
    return articles


def analyze_sentiment(articles):
    if not articles: return 0.5, "Neutral", 0
    score_sum = sum(TextBlob(title).sentiment.polarity for title in articles)
    avg_score = score_sum / len(articles)
    final_sent = (avg_score + 1) / 2
    label = "Positive" if final_sent > 0.6 else "Negative" if final_sent < 0.4 else "Neutral"
    return round(final_sent, 2), label, len(articles)


# -------------------------------------------------
# PRICE FETCHER (NEW)
# -------------------------------------------------
def get_current_price(ticker: str) -> Tuple[float, datetime]:
    """Fetch current/last close price from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return 0.0, datetime.utcnow()
        
        last_price = float(hist['Close'].iloc[-1])
        last_date = hist.index[-1]
        return last_price, last_date
    except Exception as e:
        logger.error(f"Price fetch failed for {ticker}: {e}")
        return 0.0, datetime.utcnow()


# -------------------------------------------------
# TECHNICAL ANALYSIS (ROBUST)
# -------------------------------------------------
def get_technical_score(ticker: str):
    # Default response on any failure
    response = {
        "score": 0.5, "trend": "Neutral", "rsi": 50.0, "cci": 0.0,
        "macd": "N/A (0.00 / 0.00)", "sma": "N/A (Removed)", "bb": "N/A (H: 0 / L: 0)",
        "eps": 0.0, "pe": 0.0, "vol": "Normal", "arrow": "→", "label": "Neutral"
    }


    try:
        # Direct yfinance call (no dependency on get_yfinance_data_with_retry)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        info = stock.info


        if hist.empty or len(hist) < 40:
            logger.warning(f"{ticker}: Not enough data for indicators (<40 days)")
            return response


        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        current_price = float(close.iloc[-1])


        # Fundamentals
        eps = float(info.get('trailingEps') or info.get('forwardEps') or 0.0)
        pe_raw = info.get('trailingPE') or info.get('forwardPE')
        pe = float(pe_raw) if pe_raw is not None and pe_raw > 0 else (current_price / eps if eps > 0 else 25.0)


        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0


        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci_series = (tp - sma_tp) / (0.015 * mad.replace(0, 1e-6))
        cci_val = float(cci_series.iloc[-1]) if not pd.isna(cci_series.iloc[-1]) else 0.0


        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        m_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        s_val = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
        macd_str = f"{m_val:.2f} / {s_val:.2f}"
        macd_full = f"Bullish ({macd_str})" if m_val > s_val else f"Bearish ({macd_str})"


        # Bollinger Bands
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        upper = bb_sma + 2 * bb_std
        lower = bb_sma - 2 * bb_std
        u_val = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else current_price
        l_val = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else current_price
        bb_display = f"(H: {round(u_val)} / L: {round(l_val)})"
        bb_full = f"Overbought {bb_display}" if current_price > u_val else f"Oversold {bb_display}" if current_price < l_val else f"Range {bb_display}"


        # Volume
        vol_avg = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
        vol_curr = float(volume.iloc[-1])
        vol_str = "High" if vol_curr > vol_avg * 1.5 else "Low" if vol_curr < vol_avg * 0.5 else "Normal"


        # Scoring
        rsi_score = 0.4 if rsi_val < 30 else -0.4 if rsi_val > 70 else 0
        cci_score = 0.2 if cci_val < -100 else -0.2 if cci_val > 100 else 0
        macd_score = 0.35 if m_val > s_val else -0.35
        bb_score = 0.15 if "Oversold" in bb_full else -0.15 if "Overbought" in bb_full else 0


        total_score = round(max(0.0, min(1.0, 0.5 + rsi_score + cci_score + macd_score + bb_score)), 2)


        trend_arrow = "↑" if total_score >= 0.6 else "↓" if total_score <= 0.4 else "→"
        trend_label = "Bullish" if total_score >= 0.6 else "Bearish" if total_score <= 0.4 else "Neutral"


        return {
            "score": total_score,
            "trend": trend_label,
            "rsi": round(rsi_val, 2),
            "cci": round(cci_val, 2),
            "macd": macd_full,
            "sma": "N/A (Removed)",
            "bb": bb_full,
            "eps": round(eps, 2),
            "pe": round(pe, 2),
            "vol": vol_str,
            "arrow": trend_arrow,
            "label": trend_label
        }


    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {e}")
        return response
# -------------------------------------------------
# MAIN WRAPPER
# -------------------------------------------------
executor = ThreadPoolExecutor(max_workers=5)


def analyze_stock_full(ticker, company_name):
    news_future = executor.submit(fetch_news, company_name)
    tech_future = executor.submit(get_technical_score, ticker)
    price_future = executor.submit(get_current_price, ticker)  # NEW
    
    articles = news_future.result()
    tech_data = tech_future.result()
    current_price, price_date = price_future.result()  # NEW
    
    sent_score, sent_label, news_count = analyze_sentiment(articles)
    final_score = round((tech_data["score"] * 0.6) + (sent_score * 0.4), 2)
    recommendation = "BUY" if final_score >= 0.6 else "SELL" if final_score <= 0.4 else "HOLD"


    return {
        "ticker": ticker,
        "company_name": company_name,
        "current_price": current_price,  # NEW
        "sentiment_score": sent_score,
        "sentiment": sent_label,
        "technical_score": tech_data["score"],
        "technical": tech_data["trend"],
        "rsi": tech_data["rsi"],
        "cci": tech_data["cci"],
        "macd_signal": tech_data["macd"],
        "sma_signal": tech_data["sma"],
        "bb_signal": tech_data["bb"],
        "eps": tech_data["eps"],
        "pe": tech_data["pe"],
        "volume_signal": tech_data["vol"],
        "final_score": final_score,
        "recommendation": recommendation,
        "news_count": news_count,
        "trend_arrow": tech_data["arrow"]
    }


# -------------------------------------------------
# GLOBAL REFRESH STATUS
# -------------------------------------------------
REFRESH_JOB_STATUS = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "current_ticker": ""
}


# -------------------------------------------------
# BACKGROUND REFRESH LOGIC
# -------------------------------------------------
def refresh_all_stocks_async():
    global REFRESH_JOB_STATUS
    
    with get_db() as db:
        stocks_to_refresh = db.query(StockModel).all()


    if not stocks_to_refresh:
        REFRESH_JOB_STATUS = {"status": "complete", "progress": 100, "total": 0, "current_ticker": "No stocks"}
        return


    total_stocks = len(stocks_to_refresh)
    REFRESH_JOB_STATUS.update({
        "status": "running",
        "progress": 0,
        "total": total_stocks,
        "current_ticker": ""
    })
    
    logger.info(f"Starting full refresh for {total_stocks} stocks.")
    
    try:
        for i, stock in enumerate(stocks_to_refresh):
            ticker = stock.ticker
            company_name = stock.company_name
            
            REFRESH_JOB_STATUS.update({
                "progress": int(((i + 1) / total_stocks) * 100),
                "current_ticker": ticker
            })
            
            data = analyze_stock_full(ticker, company_name)
            
            with get_db() as db:
                db.query(StockScoreModel).filter(StockScoreModel.ticker == ticker).delete()
                new_score = StockScoreModel(
                    ticker=ticker, company_name=company_name,
                    current_price=data['current_price'],  # NEW
                    sentiment_score=data['sentiment_score'],
                    technical_score=data['technical_score'],
                    final_score=data['final_score'],
                    recommendation=data['recommendation'],
                    sentiment=data['sentiment'],
                    technical=data['technical'],
                    rsi=data['rsi'],
                    cci=data['cci'],
                    macd_signal=data['macd_signal'],
                    sma_signal=data['sma_signal'],
                    bb_signal=data['bb_signal'],
                    eps=data['eps'],
                    pe=data['pe'],
                    volume_signal=data['volume_signal'],
                    news_count=data['news_count'],
                    data_source="YFinance"
                )
                db.add(new_score)
                db.commit()


            logger.info(f"Refreshed {ticker} ({i+1}/{total_stocks}). Score: {data['final_score']}")


        REFRESH_JOB_STATUS["status"] = "complete"
        logger.info("Full refresh job completed successfully.")


    except Exception as e:
        REFRESH_JOB_STATUS["status"] = "error"
        logger.error(f"Error during full refresh: {e}")
    finally:
        REFRESH_JOB_STATUS["progress"] = 100


# -------------------------------------------------
# LSTM PREDICTION WITH ADVANCED CHART & METRICS
# -------------------------------------------------
def run_lstm_prediction(ticker: str) -> Optional[Dict]:
    """Updated LSTM prediction with improvements:
    - Uses only 1 year of data (to avoid old fluctuations)
    - Adds NIFTY50 Close as feature for market correlation
    - Improved architecture (3 layers)
    - Reduced epochs to 400 for faster training (~10-15s per stock)
    - Longer sequence length (60 days)
    - Detailed logging
    - Reuses backtest for metrics (real values always)
    """
    logger.info(f"Starting improved LSTM prediction for {ticker}")


    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")  # Only recent 1 year data
        if hist.empty or len(hist) < 150:
            logger.warning(f"{ticker}: Insufficient data (<150 days)")
            return None


        logger.info(f"{ticker}: Fetched {len(hist)} days of data")


        df = hist[['Close', 'Volume', 'High', 'Low']].copy()


        # Add NIFTY50 as key market feature
        try:
            nifty = yf.Ticker("^NSEI").history(period="1y")["Close"]
            df['NIFTY_Close'] = nifty.reindex(df.index).ffill().bfill()
            df['NIFTY_Return'] = df['NIFTY_Close'].pct_change().fillna(0)
            logger.info("Added NIFTY50 features successfully")
        except Exception as e:
            logger.warning(f"Could not fetch NIFTY: {e}")
            df['NIFTY_Close'] = df['Close']  # Fallback
            df['NIFTY_Return'] = 0.0


        # Initialize indicators
        df['RSI'] = 50.0
        df['MACD'] = 0.0
        df['MACD_Signal'] = 0.0
        df['BB_High'] = df['Close']
        df['BB_Low'] = df['Close']
        df['CCI'] = 0.0
        df['PE'] = 25.0


        # Calculate indicators
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].ffill().bfill().fillna(50.0)


            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()


            bb_sma = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['BB_High'] = bb_sma + 2 * bb_std
            df['BB_Low'] = bb_sma - 2 * bb_std
            df[['BB_High', 'BB_Low']] = df[['BB_High', 'BB_Low']].ffill().bfill().fillna(df['Close'])


            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            df['CCI'] = (tp - sma_tp) / (0.015 * mad.replace(0, 1e-6))
            df['CCI'] = df['CCI'].ffill().bfill().fillna(0.0)


            info = stock.info
            pe = info.get('trailingPE') or info.get('forwardPE') or 25.0
            df['PE'] = pe
        except Exception as e:
            logger.warning(f"Indicator calculation issue: {e}")


        # Features including NIFTY
        features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'CCI', 'PE', 'NIFTY_Close', 'NIFTY_Return']
        df_model = df[features].copy()
        df_model = df_model.ffill().bfill().fillna(0)


        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_model)


        seq_length = 60  # Better context
        if len(scaled) < seq_length + 20:
            logger.warning("Not enough sequences")
            return None


        X = np.array([scaled[i:i + seq_length] for i in range(len(scaled) - seq_length)])


        # Improved 3-layer LSTM
        class ImprovedLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(len(features), 128, batch_first=True, dropout=0.3)
                self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.3)
                self.lstm3 = nn.LSTM(64, 32, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.4)


            def forward(self, x):
                out, _ = self.lstm1(x)
                out, _ = self.lstm2(out)
                out, _ = self.lstm3(out)
                out = self.dropout(out[:, -1, :])
                return self.fc(out)


        model = ImprovedLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)


        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(scaled[seq_length:, 0], dtype=torch.float32).unsqueeze(1)


        logger.info(f"Training improved model with {len(X)} sequences...")
        model.train()
        for epoch in range(400):  # Reduced for speed (~12-18s per stock)
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        logger.info("Training completed")


        # Forecast 6 days
        model.eval()
        predictions = []
        current_seq = scaled[-seq_length:].copy()
        with torch.no_grad():
            for _ in range(6):
                input_tensor = torch.tensor(current_seq.reshape(1, seq_length, -1), dtype=torch.float32)
                pred_scaled = model(input_tensor)
                pred_val = pred_scaled.item()
                predictions.append(pred_val)
                new_row = current_seq[-1].copy()
                new_row[0] = pred_val
                current_seq = np.vstack((current_seq[1:], new_row))


        pred_prices = scaler.inverse_transform(np.hstack([np.array(predictions).reshape(-1,1), np.zeros((6, len(features)-1))]))[:, 0]


        last_date = df.index[-1]
        dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=6)


        forecast = []
        prev_price = df['Close'].iloc[-1]
        for d, p in zip(dates, pred_prices):
            change = p - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            forecast.append({
                "date": d.strftime("%Y-%m-%d (%a)"),
                "price": round(p, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2)
            })
            prev_price = p


        # Backtest & Metrics (reused for chart)
        logger.info("Running backtest for efficiency metrics...")
        backtest_predictions = []
        try:
            actual_prices_15 = df['Close'][-15:].values
            actual_dates_15 = df.index[-15:]


            for i in range(15):
                start_idx = -(seq_length + 15 + i)
                end_idx = -(15 + i)
                seq_scaled = scaler.transform(df_model.iloc[start_idx:end_idx])
                input_tensor = torch.tensor(seq_scaled[-seq_length:].reshape(1, seq_length, -1), dtype=torch.float32)
                with torch.no_grad():
                    pred_scaled = model(input_tensor)
                pred_val = pred_scaled.item()
                dummy = np.zeros((1, len(features)))
                dummy[0, 0] = pred_val
                pred_price = scaler.inverse_transform(dummy)[0, 0]
                backtest_predictions.append(pred_price)


            errors = np.abs(actual_prices_15 - np.array(backtest_predictions))
            mae = np.mean(errors)
            mape = np.mean(errors / actual_prices_15) * 100 if np.all(actual_prices_15 > 0) else 0.0
            model_accuracy = round(100 - mape, 2)
            model_precision = round(100 - (np.std(errors / actual_prices_15) * 100), 2)
            model_mae = round(mae, 2)


            logger.info(f"Efficiency: Accuracy {model_accuracy}%, Precision {model_precision}%, MAE ₹{model_mae}")
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            model_accuracy = model_precision = model_mae = None
            backtest_predictions = []


        # Chart
        chart_base64 = None
        if backtest_predictions:
            try:
                today_pred = pred_prices[0]
                today_date = last_date + timedelta(days=1) if datetime.now().weekday() >= 5 else last_date


                fig, ax = plt.subplots(figsize=(14, 8))
                ax.plot(actual_dates_15, actual_prices_15, label='Actual (15 Days)', color='blue', linewidth=3, marker='o')
                ax.plot(actual_dates_15, backtest_predictions, label='Backtested', color='green', linewidth=3, linestyle='--', marker='s')
                ax.plot([today_date], [today_pred], label="Today's Pred", color='purple', marker='D', markersize=12)
                ax.plot(dates, pred_prices, label='6-Day Forecast', color='orange', linewidth=3, linestyle='--', marker='o')


                ax.axvline(x=last_date, color='gray', linestyle=':')
                ax.set_title(f'{ticker} - Efficiency & Forecast')
                ax.set_ylabel('Price (₹)')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()


                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Chart failed: {e}")


        logger.info(f"Prediction completed for {ticker}")


        return {
            "ticker": ticker,
            "current_price": round(df['Close'].iloc[-1], 2),
            "forecast": forecast,
            "chart_base64": chart_base64,
            "model_accuracy": model_accuracy,
            "model_precision": model_precision,
            "model_mae": model_mae
        }


    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        return None
    
# -------------------------------------------------
# PREDICTION ROUTE & BACKGROUND SAVE
# -------------------------------------------------
@app.get("/predict/{ticker}", response_class=HTMLResponse)
async def predict_stock(ticker: str, request: Request, background_tasks: BackgroundTasks):
    today = datetime.utcnow().date()
    
    with get_db() as db:
        pred = db.query(StockPredictionModel).filter(
            StockPredictionModel.ticker == ticker,
            func.date(StockPredictionModel.prediction_date) == today
        ).first()
        
        if pred:
            data = {
                "ticker": ticker,
                "current_price": pred.current_price,
                "forecast": json.loads(pred.forecast_json),
                "chart_base64": pred.chart_base64,
                "model_accuracy": None,  # Not saved
                "model_precision": None,
                "model_mae": None,
                "status": "complete"
            }
        else:
            background_tasks.add_task(run_and_save_prediction, ticker, today)
            data = {"ticker": ticker, "status": "running"}
    
    return templates.TemplateResponse("components/prediction_modal.html", {"request": request, "data": data})


def run_and_save_prediction(ticker: str, today: date):
    try:
        result = run_lstm_prediction(ticker)
        if not result:
            return
        
        with get_db() as db:
            db.query(StockPredictionModel).filter(
                StockPredictionModel.ticker == ticker,
                func.date(StockPredictionModel.prediction_date) == today
            ).delete()
            
            pred = StockPredictionModel(
                ticker=ticker,
                prediction_date=today,
                current_price=result['current_price'],
                forecast_json=json.dumps(result['forecast']),
                chart_base64=result.get('chart_base64')
            )
            db.add(pred)
            db.commit()
    except Exception as e:
        logger.error(f"Background save failed: {e}")


# -------------------------------------------------
# LIVE PRICE ENDPOINT (NEW)
# -------------------------------------------------
@app.get("/price/{ticker}")
async def get_live_price(ticker: str):
    """Fetch and return current price in JSON format"""
    current_price, price_date = get_current_price(ticker)
    return {
        "ticker": ticker,
        "price": round(current_price, 2),
        "currency": "₹",
        "last_updated": price_date.isoformat()
    }


# -------------------------------------------------
# OTHER ROUTES (UNCHANGED)
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(): return RedirectResponse("/index")


@app.get("/stock-select", response_class=HTMLResponse)
async def stock_select(request: Request):
    return templates.TemplateResponse("stock_select.html", {"request": request})


@app.get("/search-stocks", response_class=HTMLResponse)
async def search_stocks(query: str = ""):
    if not query: return HTMLResponse("")
    q = query.lower()
    filtered = [s for s in stock_pool if q in s["ticker"].lower() or q in s["name"].lower()][:50]
    html = ""
    for s in filtered:
        html += f"""
        <div class="form-check">
            <input class="form-check-input" type="checkbox" name="selectedTickers" value="{s['ticker']}" id="{s['ticker']}">
            <label class="form-check-label" for="{s['ticker']}">
                <strong>{s['ticker']}</strong> - {s['name']}
            </label>
        </div>
        """
    return HTMLResponse(html if html else "<p>No matches</p>")


@app.post("/add-selected")
async def add_selected(request: Request, selectedTickers: List[str] = Form(...)):
    with get_db() as db:
        for t in selectedTickers:
            name = next((s["name"] for s in stock_pool if s["ticker"] == t), t)
            
            if not db.query(StockModel).filter(StockModel.ticker == t).first():
                db.add(StockModel(ticker=t, company_name=name))
                db.commit()


            data = analyze_stock_full(t, name)
            
            db.query(StockScoreModel).filter(StockScoreModel.ticker == t).delete()
            db.add(StockScoreModel(
                ticker=t, company_name=name,
                current_price=data['current_price'],  # NEW
                sentiment_score=data['sentiment_score'],
                technical_score=data['technical_score'],
                final_score=data['final_score'],
                recommendation=data['recommendation'],
                sentiment=data['sentiment'],
                technical=data['technical'],
                rsi=data['rsi'],
                cci=data['cci'],
                macd_signal=data['macd_signal'],
                sma_signal=data['sma_signal'],
                bb_signal=data['bb_signal'],
                eps=data['eps'],
                pe=data['pe'],
                volume_signal=data['volume_signal'],
                news_count=data['news_count'],
                data_source="YFinance"
            ))
        db.commit()
    
    return templates.TemplateResponse("add_success.html", {"request": request})


@app.get("/index")
async def index(request: Request):
    with get_db() as db:
        stocks = db.query(StockScoreModel).order_by(StockScoreModel.final_score.desc()).all()
    return templates.TemplateResponse("index.html", {"request": request, "stocks": stocks})


@app.post("/refresh-all-start")
async def refresh_all_start(request: Request):
    global REFRESH_JOB_STATUS
    if REFRESH_JOB_STATUS["status"] in ["running"]:
        return templates.TemplateResponse("components/refresh_status.html", {"request": request, "status": REFRESH_JOB_STATUS})
    
    REFRESH_JOB_STATUS = {"status": "running", "progress": 0, "total": 0, "current_ticker": ""}
    threading.Thread(target=refresh_all_stocks_async, daemon=True).start()
    
    return templates.TemplateResponse("components/refresh_status.html", {"request": request, "status": REFRESH_JOB_STATUS})


@app.get("/refresh-all-status", response_class=HTMLResponse)
async def refresh_all_status(request: Request):
    global REFRESH_JOB_STATUS
    return templates.TemplateResponse("components/refresh_status.html", {"request": request, "status": REFRESH_JOB_STATUS})


@app.post("/refresh-single/{ticker}")
async def refresh_single(ticker: str, request: Request):
    with get_db() as db:
        stock = db.query(StockModel).filter(StockModel.ticker == ticker).first()
        if not stock: 
            return HTMLResponse(f"<tr><td colspan='17'>Stock {ticker} not found.</td></tr>")
        
        data = analyze_stock_full(ticker, stock.company_name)
        
        db.query(StockScoreModel).filter(StockScoreModel.ticker == ticker).delete()
        new_score = StockScoreModel(
            ticker=ticker, company_name=stock.company_name,
            current_price=data['current_price'],  # NEW
            sentiment_score=data['sentiment_score'],
            technical_score=data['technical_score'],
            final_score=data['final_score'],
            recommendation=data['recommendation'],
            sentiment=data['sentiment'],
            technical=data['technical'],
            rsi=data['rsi'],
            cci=data['cci'],
            macd_signal=data['macd_signal'],
            sma_signal=data['sma_signal'],
            bb_signal=data['bb_signal'],
            eps=data['eps'],
            pe=data['pe'],
            volume_signal=data['volume_signal'],
            news_count=data['news_count'],
            data_source="YFinance"
        )
        db.add(new_score)
        db.commit()


    return templates.TemplateResponse("components/stock_row.html", {"request": request, "stock": new_score})


@app.delete("/delete-stock/{ticker}")
async def delete_stock(ticker: str):
    with get_db() as db:
        db.query(StockModel).filter(StockModel.ticker == ticker).delete()
        db.query(StockScoreModel).filter(StockScoreModel.ticker == ticker).delete()
        db.commit()
    return Response(status_code=200)


@app.get("/news/{ticker}")
async def get_news_html(ticker: str):
    with get_db() as db:
        s = db.query(StockModel).filter(StockModel.ticker == ticker).first()
        name = s.company_name if s else ticker
    articles = fetch_news(name)
    html = f"<h5>Latest News for {name}</h5><ul>"
    for a in articles: html += f"<li>{a}</li>"
    html += "</ul>"
    return HTMLResponse(html)


@app.get("/run-daily-predictions")
async def run_daily_predictions_endpoint():
    run_daily_predictions_async()
    return {"status": "Daily predictions started"}


def run_daily_predictions_async():
    today = datetime.utcnow().date()
    
    with get_db() as db:
        stocks = db.query(StockModel).filter(StockModel.is_active == True).all()
    
    for stock in stocks:
        ticker = stock.ticker
        
        with get_db() as db:
            existing = db.query(StockPredictionModel).filter(
                StockPredictionModel.ticker == ticker,
                func.date(StockPredictionModel.prediction_date) == today
            ).first()
            if existing:
                continue
        
        run_and_save_prediction(ticker, today)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)