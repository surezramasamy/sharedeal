import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import time
import json
import matplotlib
import pandas as pd
import requests
import yfinance as yf
from textblob import TextBlob
from fastapi import Request
# --- FIX 1: Add 'func' to imports for robust SQLAlchemy usage ---
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, text, func, Date, Text, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response 
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
# --- FIX 2: Removed lru_cache import as it's removed from fetch_news ---
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import threading # Added missing import
from fastapi import BackgroundTasks
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from fastapi import BackgroundTasks


from datetime import datetime, timedelta, date
from datetime import date
import matplotlib.pyplot as plt
from io import BytesIO
import base64
matplotlib.use('Agg')

# -------------------------------------------------
# CONFIG & LOGGING
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StockApp")

DATABASE_URL = "sqlite:///./databases/stock_recommender.db"
NSE_CSV_PATH = "nse.csv"

# API Keys (Replace with your actual keys)
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


Base.metadata.create_all(bind=engine)

# Create prediction table if not exists
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
# DB MIGRATION (Auto-adds columns if missing)
# -------------------------------------------------
def migrate_db():
    required_cols = {
        "eps": "REAL", "pe": "REAL", "macd_signal": "TEXT", 
        "data_source": "TEXT", "cci": "REAL", "sma_signal": "TEXT", 
        "bb_signal": "TEXT", "volume_signal": "TEXT"
    }
    with engine.connect() as conn:
        res = conn.execute(text("PRAGMA table_info(stock_scores)"))
        existing = {row[1] for row in res}
        for col, typ in required_cols.items():
            if col not in existing:
                try:
                    conn.execute(text(f"ALTER TABLE stock_scores ADD COLUMN {col} {typ}"))
                except: 
                    logger.warning(f"Could not add column {col}. Database is likely up-to-date.")
        conn.commit()
migrate_db()

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
# CRITICAL DATA RETRIEVAL FIX (RETRY LOGIC)
# -------------------------------------------------
def get_yfinance_data_with_retry(ticker: str) -> Tuple[pd.DataFrame, Dict]:
    """Tries to fetch history (2y) and info with retries."""
    max_retries = 3
    history = pd.DataFrame()
    info = {}
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            # Fetch History (Still fetching 2y, just in case)
            history = stock.history(period="2y")
            
            # Fetch Info
            info = stock.info
            
            if not history.empty and info:
                logger.info(f"{ticker}: Data successfully retrieved on attempt {attempt + 1}.")
                return history, info
            
            logger.warning(f"{ticker}: Partial or empty data on attempt {attempt + 1}. Retrying...")
            time.sleep(1 + attempt) # Wait longer on subsequent failures
            
        except requests.exceptions.ReadTimeout:
            logger.warning(f"{ticker}: Request timed out on attempt {attempt + 1}. Retrying...")
            time.sleep(2 + attempt)
        except Exception as e:
            logger.warning(f"{ticker}: General failure on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(1 + attempt)
            
    logger.error(f"{ticker}: Failed to retrieve required data after {max_retries} attempts.")
    return pd.DataFrame(), {}


# -------------------------------------------------
# TECHNICAL ANALYSIS (SMA REMOVED, ROBUST ERROR HANDLING)
# -------------------------------------------------
def get_technical_score(ticker: str):
    # Default response dictionary, returned on any failure
    response = {
        "score": 0.5, "trend": "Neutral", "rsi": 50.0, "cci": 0.0, 
        "macd": "N/A (0.00 / 0.00)", 
        "sma": "N/A (Removed)", # REMOVED SMA
        "bb": "N/A (H: 0 / L: 0)", 
        "eps": 0.0, "pe": 0.0, 
        "vol": "Normal", "arrow": "→", "label": "Neutral"
    }

    try:
        hist, info = get_yfinance_data_with_retry(ticker)

        # Need at least 26 days for MACD
        if hist.empty or len(hist) < 26: 
            logger.warning(f"{ticker}: Insufficient history or failed retrieval.")
            return response

        # --- Fundamentals ---
        eps = info.get('trailingEps') or info.get('forwardEps') or 0.0
        pe = info.get('trailingPE')
        current_price = hist['Close'].iloc[-1]
        
        if (pe is None or pe == "None" or pe <= 0) and eps and eps > 0:
            pe = current_price / eps
        if pe is None or pe <= 0: pe = 0.0

        # --- Data Prep ---
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        
        # --- 1. RSI (14) ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        rsi = round(rsi, 2) if not pd.isna(rsi) else 50.0
        
        rsi_score = 0.0
        if rsi < 30: rsi_score = 0.4
        elif rsi > 70: rsi_score = -0.4

        # --- 2. CCI (14) ---
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(14).mean()
        md_series = (tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))
        
        if pd.isna(md_series.iloc[-1]) or md_series.iloc[-1] == 0:
             cci_val = 0.0
        else:
            cci_series = (tp - sma_tp) / (0.015 * md_series.replace(0, 1e-6))
            cci_val = cci_series.iloc[-1]
        
        cci_val = round(cci_val, 2) if not pd.isna(cci_val) else 0.0
        
        cci_score = 0.0
        if cci_val > 100: cci_score = -0.2  # Overbought
        elif cci_val < -100: cci_score = 0.2 # Oversold
        
        # --- 3. MACD (12, 26, 9) ---
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        m_val = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0
        s_val = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0.0
        
        macd_str = f"{m_val:.2f} / {s_val:.2f}"
        macd_score = 0.35 if m_val > s_val else -0.35
        macd_full_str = f"Bullish ({macd_str})" if m_val > s_val else f"Bearish ({macd_str})"
        if abs(m_val - s_val) < 1e-4: macd_full_str = f"Neutral ({macd_str})"

        # --- 4. SMA (REMOVED LOGIC) ---
        sma_score = 0.0
        sma_full_str = "N/A (Removed)"


        # --- 5. Bollinger Bands (20, 2) ---
        bb_sma = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        upper = bb_sma + 2 * bb_std
        lower = bb_sma - 2 * bb_std
        
        u_val = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else 0.0
        l_val = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else 0.0
        curr = close.iloc[-1] if not pd.isna(close.iloc[-1]) else 0.0
        
        bb_score = 0.0
        bb_full_str = f"N/A (H: {u_val:.0f} / L: {l_val:.0f})"
        
        if u_val > 0 and l_val > 0:
            bb_display = f"(H: {u_val:.0f} / L: {l_val:.0f})"
            if curr > u_val:
                bb_full_str = f"Overbought {bb_display}"
                bb_score = -0.15
            elif curr < l_val:
                bb_full_str = f"Oversold {bb_display}"
                bb_score = 0.15
            else:
                bb_full_str = f"Range {bb_display}"
                bb_score = 0.0

        # --- 6. Volume ---
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        vol_curr = hist['Volume'].iloc[-1]
        
        vol_str = "Normal"
        if not pd.isna(vol_avg) and not pd.isna(vol_curr) and vol_avg > 0:
            if vol_curr > (vol_avg * 1.5): vol_str = "High"
            elif vol_curr < (vol_avg * 0.5): vol_str = "Low"

        # --- Final Scoring ---
        # SMA is removed from scoring
        total_tech_score = 0.5 + rsi_score + cci_score + macd_score + bb_score
        total_tech_score = max(0.0, min(1.0, total_tech_score))
        total_tech_score = round(total_tech_score, 2)
        
        trend_arrow = "↑" if total_tech_score >= 0.6 else "↓" if total_tech_score <= 0.4 else "→"
        trend_label = "Bullish" if total_tech_score >= 0.6 else "Bearish" if total_tech_score <= 0.4 else "Neutral"

        return {
            "score": total_tech_score,
            "trend": trend_label,
            "rsi": rsi,
            "cci": cci_val,
            "macd": macd_full_str, 
            "sma": sma_full_str,   # Returns "N/A (Removed)"
            "bb": bb_full_str,     
            "eps": round(eps, 2),
            "pe": round(pe, 2),
            "vol": vol_str,
            "arrow": trend_arrow,
            "label": trend_label
        }
    except Exception as e:
        logger.error(f"Critical Error in Technical Analysis for {ticker}: {e}")
        # Return the robust default dictionary on error
        return response


# -------------------------------------------------
# NEWS & SENTIMENT
# -------------------------------------------------
# --- FIX 2: REMOVED @lru_cache and increased timeout ---
def fetch_news(company_name: str) -> List[str]:
    """
    Fetches the top news article titles for a given company name using the MarketAux API.
    """
    articles = []
    
    # 1. Check if the API key is available
    if not MARKETAUX_KEY:
        logger.warning("MARKETAUX_KEY environment variable is not set. Cannot fetch news.")
        return []

    try:
        # MarketAux endpoint for general news
        base_url = "https://api.marketaux.com/v1/news/all"
        
        # Parameters to filter by keyword/company name, limit results, and set language
        params = {
            "api_token": MARKETAUX_KEY,
            "search": company_name,
            "language": "en",
            "limit": 5, 
            # Using filter_entities is helpful to ensure relevance if possible
            "filter_entities": "true" 
        }

        # Increased timeout to 10 seconds for robustness
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status() # Raises an HTTPError for 4XX/5XX responses
        
        data = r.json()
        
        if "data" in data:
            # MarketAux returns article list under the 'data' key
            # Extract the first 5 titles
            articles.extend([a["title"] for a in data["data"] if "title" in a][:5])
            
    except requests.exceptions.HTTPError as e:
        # Catch specific HTTP errors (like 401 Unauthorized or 429 Rate Limit)
        logger.error(f"MarketAux API HTTP error for {company_name}: {r.status_code} - {e}")
        
    except requests.exceptions.RequestException as e:
        # Catch network or timeout errors
        logger.error(f"MarketAux API connection failed for {company_name}: {e}")
        
    except Exception as e:
        # Catch other errors (like JSON parsing)
        logger.error(f"Error processing MarketAux data for {company_name}: {e}")
        
    # Use set to ensure uniqueness, then convert back to list
    return list(set(articles))

def analyze_sentiment(articles):
    if not articles: return 0.5, "Neutral", 0
    score_sum = sum(TextBlob(title).sentiment.polarity for title in articles)
    avg_score = score_sum / len(articles)
    final_sent = (avg_score + 1) / 2 # Normalize (-1 to 1) -> (0 to 1)
    
    if final_sent > 0.6: label = "Positive"
    elif final_sent < 0.4: label = "Negative"
    else: label = "Neutral"
    return round(final_sent, 2), label, len(articles)

# -------------------------------------------------
# MAIN WRAPPER
# -------------------------------------------------
executor = ThreadPoolExecutor(max_workers=5)

def analyze_stock_full(ticker, company_name):
    news_future = executor.submit(fetch_news, company_name)
    tech_future = executor.submit(get_technical_score, ticker)
    
    articles = news_future.result()
    tech_data = tech_future.result()
    
    sent_score, sent_label, news_count = analyze_sentiment(articles)
    # Give technicals a slight edge
    final_score = round((tech_data["score"] * 0.6) + (sent_score * 0.4), 2)
    recommendation = "BUY" if final_score >= 0.6 else "SELL" if final_score <= 0.4 else "HOLD"

    # This dictionary returns the full keys needed by the database model
    return {
        "ticker": ticker,
        "company_name": company_name,
        "sentiment_score": sent_score,
        "sentiment": sent_label,
        "technical_score": tech_data["score"],
        "technical": tech_data["trend"],
        "rsi": tech_data["rsi"],
        "cci": tech_data["cci"],
        # Map the short keys from tech_data to the full keys for the DB
        "macd_signal": tech_data["macd"],
        "sma_signal": tech_data["sma"],  # Now contains "N/A (Removed)"
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
# GLOBAL REFRESH STATUS (NEW)
# -------------------------------------------------
REFRESH_JOB_STATUS = {
    "status": "idle", # idle, running, complete, error
    "progress": 0,    # 0 to 100
    "total": 0,
    "current_ticker": ""
}

# -------------------------------------------------
# BACKGROUND REFRESH LOGIC (NEW)
# -------------------------------------------------
def refresh_all_stocks_async():
    """Performs the analysis for all stocks in the background."""
    global REFRESH_JOB_STATUS
    
    with get_db() as db:
        stocks_to_refresh = db.query(StockModel).all()

    if not stocks_to_refresh:
        REFRESH_JOB_STATUS = {"status": "complete", "progress": 100, "total": 0, "current_ticker": "No stocks to refresh."}
        return

    total_stocks = len(stocks_to_refresh)
    REFRESH_JOB_STATUS.update({
        "status": "running",
        "progress": 0,
        "total": total_stocks,
        "start_time": datetime.now()
    })
    
    logger.info(f"Starting full refresh for {total_stocks} stocks.")
    
    try:
        for i, stock in enumerate(stocks_to_refresh):
            ticker = stock.ticker
            company_name = stock.company_name
            
            # Update job status for polling
            REFRESH_JOB_STATUS.update({
                "progress": int(((i + 1) / total_stocks) * 100),
                "current_ticker": ticker
            })
            
            # --- 1. Perform Analysis ---
            data = analyze_stock_full(ticker, company_name)
            
            # --- 2. Save New Score ---
            with get_db() as db:
                db.query(StockScoreModel).filter(StockScoreModel.ticker == ticker).delete()
                new_score = StockScoreModel(
                    ticker=ticker, company_name=company_name,
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
        # Final progress set to 100
        REFRESH_JOB_STATUS["progress"] = 100
class StockPredictionModel(Base):
    __tablename__ = "stock_predictions"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    prediction_date = Column(Date, default=datetime.utcnow().date, index=True)  # Date only
    current_price = Column(Float)
    forecast_json = Column(String)  # JSON string of forecast list
    chart_base64 = Column(Text, nullable=True)  # Base64 image
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint('ticker', 'prediction_date', name='unique_ticker_date'),)
# -------------------------------------------------
# ROUTES (Updated with new endpoints)
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
    
    # Now 'request' is available
    return templates.TemplateResponse("add_success.html", {"request": request})

@app.get("/index")
async def index(request: Request):
    with get_db() as db:
        # Note: Since your refresh/add logic deletes old scores, this simple query is correct.
        # If you ever switch back to keeping historical scores, you must use 
        # func.max(StockScoreModel.id) as shown in the previous fix.
        stocks = db.query(StockScoreModel).order_by(StockScoreModel.final_score.desc()).all()
    return templates.TemplateResponse("index.html", {"request": request, "stocks": stocks})

# NEW ROUTE: Start the Refresh All Job
@app.post("/refresh-all-start")
async def refresh_all_start(request: Request):
    global REFRESH_JOB_STATUS
    if REFRESH_JOB_STATUS["status"] in ["running"]:
        # Return status update if already running
        return templates.TemplateResponse("components/refresh_status.html", {"request": request, "status": REFRESH_JOB_STATUS})
    
    # Reset status and start the job in the background thread
    REFRESH_JOB_STATUS = {"status": "running", "progress": 0, "total": 0, "current_ticker": ""}
    # Switched to threading.Thread to ensure it runs even in a non-async Executor environment
    threading.Thread(target=refresh_all_stocks_async, daemon=True).start()
    
    # Return the initial status component for HTMX to start polling
    return templates.TemplateResponse("components/refresh_status.html", {"request": request, "status": REFRESH_JOB_STATUS})

# NEW ROUTE: Get the Refresh All Status (for HTMX Polling)
@app.get("/refresh-all-status", response_class=HTMLResponse)
async def refresh_all_status(request: Request):
    global REFRESH_JOB_STATUS
    # This endpoint returns the status fragment, which HTMX will check
    return templates.TemplateResponse("components/refresh_status.html", {"request": request, "status": REFRESH_JOB_STATUS})

@app.post("/refresh-single/{ticker}")
async def refresh_single(ticker: str, request: Request):
    with get_db() as db:
        stock = db.query(StockModel).filter(StockModel.ticker == ticker).first()
        if not stock: 
            return HTMLResponse(f"<tr><td colspan='17'>Stock {ticker} not found.</td></tr>")
        
        data = analyze_stock_full(ticker, stock.company_name)
        
        # Delete old score & add new
        db.query(StockScoreModel).filter(StockScoreModel.ticker == ticker).delete()
        new_score = StockScoreModel(
            ticker=ticker, company_name=stock.company_name,
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

    # Need to return the complete table row (TR) for HTMX to swap correctly
    return templates.TemplateResponse("components/stock_row.html", {"request": request, "stock": new_score})

@app.delete("/delete-stock/{ticker}")
async def delete_stock(ticker: str):
    with get_db() as db:
        db.query(StockModel).filter(StockModel.ticker == ticker).delete()
        db.query(StockScoreModel).filter(StockScoreModel.ticker == ticker).delete()
        db.commit()
    # Return an empty Response with a successful status code (200)
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

# -------------------------------------------------
# LSTM PREDICTION FOR SINGLE STOCK (NEW)
# -------------------------------------------------
import torch
from sklearn.preprocessing import MinMaxScaler

def run_lstm_prediction(ticker: str) -> Optional[Dict]:
    """Runs the LSTM forecast for a single ticker and returns results with chart."""
    try:
        # Fetch more historical data for better training
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")  # Increased to 1 year for stability
        if hist.empty or len(hist) < 100:
            logger.warning(f"{ticker}: Insufficient data (<100 days)")
            return None

        df = hist[['Close', 'Volume', 'High', 'Low']].copy()

        # Add technical indicators
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        bb_sma = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_High'] = bb_sma + 2 * bb_std
        df['BB_Low'] = bb_sma - 2 * bb_std

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)

        info = stock.info
        eps = info.get('trailingEps', 0.0)
        pe = info.get('trailingPE', 0.0)
        current_price = df['Close'].iloc[-1]
        if pe == 0 and eps > 0:
            pe = current_price / eps
        df['PE'] = pe

        features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'CCI', 'PE']
        df_model = df[features].dropna()
        if len(df_model) < 60:
            return None

        # Scale
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_model)

        # Create sequences
        seq_length = 30
        X = []
        for i in range(len(scaled) - seq_length):
            X.append(scaled[i:i + seq_length])
        X = np.array(X)
        if len(X) < 10:
            return None

        # Model with dropout for stability
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(len(features), 64, num_layers=2, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(64, 1)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out)

        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(scaled[seq_length:, 0], dtype=torch.float32).unsqueeze(1)  # All targets

        # Train longer for better convergence
        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # --- Stable Recursive Forecast ---
        model.eval()
        predictions = []
        current_seq = scaled[-seq_length:].copy()

        with torch.no_grad():
            for _ in range(6):
                input_tensor = torch.tensor(current_seq.reshape(1, seq_length, -1), dtype=torch.float32)
                pred_scaled = model(input_tensor)
                pred_val = pred_scaled.item()
                predictions.append(pred_val)

                # Shift and update only Close price
                new_row = current_seq[-1].copy()
                new_row[0] = pred_val
                current_seq = np.vstack((current_seq[1:], new_row))

        # Inverse transform
        pred_scaled_array = np.array(predictions).reshape(-1, 1)
        dummy = np.zeros((6, len(features)))
        dummy[:, 0] = pred_scaled_array.flatten()
        pred_prices = scaler.inverse_transform(dummy)[:, 0]

        # Dates
        last_date = df.index[-1]
        dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=6)

        # Forecast list
        forecast = []
        prev_price = df['Close'].iloc[-1]
        for date, price in zip(dates, pred_prices):
            change = price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            forecast.append({
                "date": date.strftime("%Y-%m-%d (%a)"),
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2)
            })
            prev_price = price

        # --- Generate Chart ---
                # --- Model Efficiency Chart: Last 7 Actual + Last 7 Backtested Predicted + Next 6 Forecast ---
        {% if data.chart_base64 %}
            <div class="chart-container mb-5">
            <div class="alert alert-info text-center small mb-3">
                <strong>Advanced Efficiency:</strong> Blue = Actual (last 7 days) | Green dashed = Backtested predictions (last 7 days) | Orange dashed = Forecast (next 6 days)
            </div>
            <img src="data:image/png;base64,{{ data.chart_base64 }}" class="img-fluid rounded shadow">
            </div>
            {% else %}
            <div class="alert alert-warning text-center mb-5">
            Advanced chart unavailable (data issue). Table forecast below.
            </div>
        {% endif %}

        # Forecast list (same as before)
        forecast = []
        prev_price = df['Close'].iloc[-1]
        for date, price in zip(dates, pred_prices):
            change = price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            forecast.append({
                "date": date.strftime("%Y-%m-%d (%a)"),
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2)
            })
            prev_price = price

        return {
            "ticker": ticker,
            "current_price": round(df['Close'].iloc[-1], 2),
            "forecast": forecast,
            "chart_base64": chart_base64  # Can be None
        }

    except Exception as e:
        logger.error(f"LSTM Prediction failed for {ticker}: {e}")
        return None
    
@app.get("/predict-page/{ticker}", response_class=HTMLResponse)
async def predict_page(ticker: str, request: Request):
    today = datetime.utcnow().date()
    
    with get_db() as db:
        pred = db.query(StockPredictionModel).filter(
            StockPredictionModel.ticker == ticker,
            func.date(StockPredictionModel.prediction_date) == today
        ).first()
    
    if not pred:
        return HTMLResponse(f"""
        <div class="container py-5 text-center">
          <h3>No prediction available for {ticker} today</h3>
          <p>Daily predictions run at 1:00 AM UTC. Check back later or try tomorrow.</p>
          <a href="/index" class="btn btn-primary mt-3">← Back to Dashboard</a>
        </div>
        """)
    
    data = {
        "ticker": ticker,
        "current_price": pred.current_price,
        "forecast": json.loads(pred.forecast_json),
        "chart_base64": pred.chart_base64
    }
    
    return templates.TemplateResponse("predict_page.html", {"request": request, "data": data})


def run_daily_predictions_async():
    today = datetime.utcnow().date()
    
    with get_db() as db:
        stocks = db.query(StockModel).filter(StockModel.is_active == True).all()
    
    for stock in stocks:
        ticker = stock.ticker
        company_name = stock.company_name
        
        # Skip if already done today
        with get_db() as db:
            existing = db.query(StockPredictionModel).filter(
                StockPredictionModel.ticker == ticker,
                func.date(StockPredictionModel.prediction_date) == today
            ).first()
            if existing:
                continue
        
        result = run_lstm_prediction(ticker)
        if result:
            with get_db() as db:
                pred = StockPredictionModel(
                    ticker=ticker,
                    prediction_date=today,
                    current_price=result['current_price'],
                    forecast_json=json.dumps(result['forecast']),
                    chart_base64=result.get('chart_base64')
                )
                db.add(pred)
                db.commit()
            logger.info(f"Daily prediction saved for {ticker}")




@app.get("/predict/{ticker}", response_class=HTMLResponse)
async def predict_stock(ticker: str, request: Request, background_tasks: BackgroundTasks):
    today = datetime.utcnow().date()
    
    with get_db() as db:
        pred = db.query(StockPredictionModel).filter(
            StockPredictionModel.ticker == ticker,
            func.date(StockPredictionModel.prediction_date) == today
        ).first()
    
    if pred:
        # Instant result from DB
        data = {
            "ticker": ticker,
            "current_price": pred.current_price,
            "forecast": json.loads(pred.forecast_json),
            "chart_base64": pred.chart_base64,
            "status": "complete"
        }
    else:
        # Start background prediction
        background_tasks.add_task(run_and_save_prediction, ticker, today)
        data = {
            "ticker": ticker,
            "status": "running"
        }
    
    # Always use the same template
    return templates.TemplateResponse("components/prediction_modal.html", {"request": request, "data": data})

def run_and_save_prediction(ticker: str, today: date):
    try:
        result = run_lstm_prediction(ticker)
        if not result:
            logger.warning(f"Background prediction failed for {ticker}")
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
        logger.info(f"Background prediction completed and saved for {ticker}")
    except Exception as e:
        logger.error(f"Error in background prediction for {ticker}: {e}")

@app.get("/run-daily-predictions")
async def run_daily_predictions_endpoint():
    """Endpoint called by Railway cron job"""
    run_daily_predictions_async()  # Reuse your existing function
    return {"status": "Daily predictions completed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)