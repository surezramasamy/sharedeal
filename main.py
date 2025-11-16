# main.py
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import time

import pandas as pd
import requests
import yfinance as yf
from textblob import TextBlob
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, func, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATABASE_URL = "sqlite:///./databases/stock_recommender.db"
NEWSAPI_KEY = "f44ef3442436422983ac6a1c353e5f21"
NSE_CSV_PATH = "nse.csv"

# -------------------------------------------------
# DB MODELS
# -------------------------------------------------
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
    macd_signal = Column(String)
    news_count = Column(Integer)
    data_source = Column(String)
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)


Base.metadata.create_all(bind=engine)


# -------------------------------------------------
# MIGRATION: Add columns and fill NULLs
# -------------------------------------------------
def migrate_db():
    required_cols = {
        "technical_score": "REAL",
        "technical": "TEXT",
        "rsi": "REAL",
        "macd_signal": "TEXT",
        "data_source": "TEXT"
    }
    with engine.connect() as conn:
        res = conn.execute(text("PRAGMA table_info(stock_scores)"))
        existing = {row[1] for row in res}
        for col, typ in required_cols.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE stock_scores ADD COLUMN {col} {typ}"))
                logger.info(f"Added missing column: {col}")

        # Fill NULLs
        fills = {
            "rsi": "50.0",
            "macd_signal": "'N/A'",
            "data_source": "'N/A'",
            "technical_score": "0.5",
            "technical": "'Neutral'"
        }
        for col, default in fills.items():
            conn.execute(text(f"UPDATE stock_scores SET {col} = {default} WHERE {col} IS NULL"))
            logger.info(f"Filled NULLs in {col}")

        conn.commit()


migrate_db()


# -------------------------------------------------
# CLEAN COMMA-JOINED TICKERS
# -------------------------------------------------
def clean_comma_joined_tickers():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT id, ticker, company_name FROM stocks WHERE ticker LIKE '%,%'"))
        bad = res.fetchall()
        if not bad:
            return
        logger.info(f"Cleaning {len(bad)} malformed rows...")
        for row_id, t_str, c_str in bad:
            tickers = [t.strip() for t in t_str.split(",") if t.strip()]
            companies = [c.strip() for c in c_str.split(",") if c.strip()]
            for i, t in enumerate(tickers):
                name = companies[i] if i < len(companies) else t
                conn.execute(text("INSERT OR IGNORE INTO stocks (ticker, company_name) VALUES (:t, :n)"), {"t": t, "n": name})
            conn.execute(text("DELETE FROM stocks WHERE id = :id"), {"id": row_id})
        conn.commit()
        logger.info("Cleaned comma-joined tickers.")


clean_comma_joined_tickers()


# -------------------------------------------------
# DB SESSION
# -------------------------------------------------
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------
# TEMPLATES & APP
# -------------------------------------------------
templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------
def load_stock_pool() -> List[Dict]:
    if not os.path.exists(NSE_CSV_PATH):
        logger.error(f"CSV NOT FOUND: {NSE_CSV_PATH}")
        return []
    try:
        df = pd.read_csv(NSE_CSV_PATH, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        required = ["SYMBOL", "NAME OF COMPANY"]
        if not all(col in df.columns for col in required):
            logger.error(f"CSV missing columns: {required}")
            return []
        df = df[required].rename(columns={"SYMBOL": "ticker", "NAME OF COMPANY": "name"})
        df["ticker"] = df["ticker"].str.strip() + ".NS"
        df["name"] = df["name"].str.strip()
        pool = df.drop_duplicates(subset=["ticker"]).to_dict(orient="records")
        logger.info(f"Loaded {len(pool)} stocks from CSV")
        return pool
    except Exception as e:
        logger.error(f"CSV load failed: {e}")
        return []


stock_pool = load_stock_pool()


# -------------------------------------------------
# TECHNICAL ANALYSIS
# -------------------------------------------------
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

def get_technical_score(ticker: str) -> tuple[float, str, float, str, str]:
    """
    Returns: (score, label, rsi, signal_text, source)
    - RSI only (MACD removed)
    - Real RSI from yfinance (same as test_rsi_fixed.py)
    - Fallback: 0.5, "Neutral (error)", 50.0, "N/A", "N/A"
    """
    end = datetime.utcnow()
    start = end - timedelta(days=90)  # 90 days = safe for RSI

    for attempt in range(3):
        try:
            logger.info(f"[{ticker}] Fetching from yfinance (attempt {attempt + 1})")
            symbol = ticker if ticker.endswith('.NS') else ticker + '.NS'
            
            # FIX: Use list to avoid MultiIndex
            df = yf.download([symbol], start=start, end=end, progress=False, auto_adjust=True)

            if df.empty:
                logger.warning(f"[{ticker}] Empty data")
                if attempt == 2:
                    return 0.5, "Neutral (error)", 50.0, "N/A", "N/A"
                continue

            # Extract Close safely
            close = df['Close'][symbol] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            volume = df['Volume'][symbol] if isinstance(df.columns, pd.MultiIndex) else df['Volume']
            open_p = df['Open'][symbol] if isinstance(df.columns, pd.MultiIndex) else df['Open']
            high = df['High'][symbol] if isinstance(df.columns, pd.MultiIndex) else df['High']
            low = df['Low'][symbol] if isinstance(df.columns, pd.MultiIndex) else df['Low']

            df_clean = pd.DataFrame({
                'Open': open_p, 'High': high, 'Low': low,
                'Close': close, 'Volume': volume
            }).dropna()

            if len(df_clean) < 15:
                logger.warning(f"[{ticker}] Not enough data: {len(df_clean)} rows")
                if attempt == 2:
                    return 0.5, "Neutral (error)", 50.0, "N/A", "N/A"
                continue

            # === RSI (from test_rsi_fixed.py) ===
            delta = df_clean['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-6)
            rsi_series = 100 - (100 / (1 + rs))
            current_rsi = rsi_series.iloc[-1]

            if pd.isna(current_rsi):
                current_rsi = 50.0
            else:
                current_rsi = round(float(current_rsi), 1)

            # === Candlestick Patterns (last 3 days) ===
            df_clean['up'] = df_clean['Close'] > df_clean['Open']
            df_clean['body_ratio'] = abs(df_clean['Close'] - df_clean['Open']) / (df_clean['High'] - df_clean['Low'] + 1e-6)
            recent = df_clean.tail(3)

            hammer = (recent['body_ratio'] < 0.3) & (recent['Low'] == df_clean['Low'].tail(3).min()) & recent['up']
            engulfing_bull = (recent['up'].iloc[-1]) and (recent['Close'].iloc[-1] > recent['Open'].iloc[-2]) and (recent['Open'].iloc[-1] < recent['Close'].iloc[-2])
            shooting_star = (recent['body_ratio'] < 0.3) & (recent['High'] == df_clean['High'].tail(3).max()) & ~recent['up']
            engulfing_bear = (~recent['up'].iloc[-1]) and (recent['Close'].iloc[-1] < recent['Open'].iloc[-2]) and (recent['Open'].iloc[-1] > recent['Close'].iloc[-2])
            vol_surge = df_clean['Volume'].iloc[-1] > df_clean['Volume'].rolling(10).mean().iloc[-1] * 1.5

            # === Scoring (NO MACD) ===
            rsi_score = 0.4 if current_rsi < 30 else -0.4 if current_rsi > 70 else 0.15 if 45 <= current_rsi <= 55 else 0
            pattern_score = 0.25 if (hammer.any() or engulfing_bull) else -0.25 if (shooting_star.any() or engulfing_bear) else 0
            volume_score = 0.1 if vol_surge else 0

            raw = 0.5 + rsi_score + pattern_score + volume_score
            score = round(max(0.0, min(1.0, raw)), 2)
            label = "Strong Up" if score >= 0.8 else "Up" if score >= 0.6 else "Strong Down" if score <= 0.2 else "Down" if score <= 0.4 else "Neutral"

            signal_text = f"RSI: {current_rsi}"
            if current_rsi < 30: signal_text += " (Oversold)"
            elif current_rsi > 70: signal_text += " (Overbought)"
            elif 45 <= current_rsi <= 55: signal_text += " (Stable)"

            logger.info(f"[{ticker}] RSI: {current_rsi} | Tech: {score} | Label: {label}")
            return score, label, current_rsi, signal_text, "yfinance"

        except Exception as e:
            logger.error(f"[{ticker}] Error (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                logger.error(f"[{ticker}] All attempts failed. Returning fallback.")
                return 0.5, "Neutral (error)", 50.0, "N/A", "N/A"


# -------------------------------------------------
# SENTIMENT + FINAL SCORE
# -------------------------------------------------
def fetch_news(company: str):
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": company, "language": "en", "pageSize": 10, "apiKey": NEWSAPI_KEY},
            timeout=10
        )
        return r.json().get("articles", []) if r.status_code == 200 else []
    except Exception as e:
        logger.warning(f"News failed: {e}")
        return []


def compute_sentiment(articles):
    if not articles:
        return 0.5, "Neutral (no news)", 0
    pol = []
    for a in articles:
        txt = (a.get("title") or "") + " " + (a.get("description") or "")
        if txt.strip():
            pol.append(TextBlob(txt).sentiment.polarity)
    if not pol:
        return 0.5, "Neutral (no text)", 0
    avg = sum(pol) / len(pol)
    score = round((avg + 1) / 2, 2)
    label = "Positive" if score >= 0.60 else "Negative" if score <= 0.40 else "Neutral"
    return score, label, len(pol)


def decide(final_score):
    return "BUY" if final_score >= 0.60 else "SELL" if final_score <= 0.40 else "HOLD"


def analyze_stock(ticker: str, company: str) -> Dict:
    articles = fetch_news(company)
    sent_score, sent_label, news_cnt = compute_sentiment(articles)
    tech_score, tech_label, rsi_val, macd_sig, data_src = get_technical_score(ticker)
    final = round(0.7 * sent_score + 0.3 * tech_score, 2)

    return {
        "ticker": ticker,
        "company_name": company,
        "sentiment_score": sent_score,
        "technical_score": tech_score,
        "final_score": final,
        "recommendation": decide(final),
        "sentiment": sent_label,
        "technical": tech_label,
        "rsi": rsi_val,
        "macd_signal": macd_sig,
        "news_count": news_cnt,
        "data_source": data_src,
        "analyzed_at": datetime.utcnow(),
    }


# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse("/stock-select")


@app.get("/stock-select", response_class=HTMLResponse)
async def stock_select(request: Request):
    return templates.TemplateResponse("stock_select.html", {"request": request})


@app.get("/search-stocks", response_class=HTMLResponse)
async def search_stocks(query: str = ""):
    filtered = [s for s in stock_pool if query.lower() in s["ticker"].lower() or query.lower() in s["name"].lower()][:100]
    html = "".join(
        f'<label class="list-group-item d-flex justify-content-between align-items-center">'
        f'<div><input type="checkbox" name="selectedTickers" value="{s["ticker"]}" class="form-check-input me-2">'
        f'<strong>{s["ticker"]}</strong> - {s["name"]}</div></label>'
        for s in filtered
    ) or '<div class="list-group-item text-center text-muted">No stocks found.</div>'
    return HTMLResponse(html)


@app.post("/add-selected", response_class=HTMLResponse)
async def add_selected(selectedTickers: List[str] = Form(...)):
    if not selectedTickers:
        return HTMLResponse('<div class="alert alert-danger">Select at least one stock.</div>'
                            '<div class="mt-3 text-center"><a href="/stock-select" class="btn btn-primary">Back</a></div>')

    added = []
    skipped = []
    with get_db() as db:
        for t in selectedTickers:
            if db.query(StockModel).filter(StockModel.ticker == t).first():
                skipped.append(t)
                continue
            name = next((s["name"] for s in stock_pool if s["ticker"] == t), t)
            db.add(StockModel(ticker=t, company_name=name))
            added.append(t)
            result = analyze_stock(t, name)
            db.add(StockScoreModel(**result))
            logger.info(f"Added {t} | RSI: {result['rsi']} | MACD: {result['macd_signal']} | Final: {result['final_score']}")
        db.commit()

    msg = f'<div class="alert alert-success">Added {len(added)} stock(s)</div>'
    if skipped: msg += f'<div class="alert alert-info">Skipped {len(skipped)}</div>'
    msg += '<div class="mt-3 text-center"><a href="/index" class="btn btn-success btn-lg">View Results</a></div>'
    return HTMLResponse(msg)


@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    with get_db() as db:
        subq = db.query(StockScoreModel.ticker, func.max(StockScoreModel.analyzed_at).label("max_dt")).group_by(StockScoreModel.ticker).subquery()
        latest = db.query(StockScoreModel).join(subq, (StockScoreModel.ticker == subq.c.ticker) & (StockScoreModel.analyzed_at == subq.c.max_dt)).all()
        latest = sorted(latest, key=lambda x: x.final_score, reverse=True)
    return templates.TemplateResponse("index.html", {"request": request, "stocks": latest})

# -------------------------------------------------
# REFRESH
# -------------------------------------------------

# Keep this if you want a simple sync version (for <10 stocks)
@app.post("/refresh-all", response_class=HTMLResponse)
async def refresh_all(request: Request):
    with get_db() as db:
        stocks = db.query(StockModel).filter(StockModel.is_active == True).all()
        refreshed = failed = 0
        for stock in stocks:
            try:
                result = analyze_stock(stock.ticker, stock.company_name)
                db.add(StockScoreModel(**result))
                refreshed += 1
            except Exception as e:
                logger.error(f"Failed {stock.ticker}: {e}")
                failed += 1
        db.commit()
    failed_msg = f'<p class="text-warning"><strong>{failed}</strong> failed</p>' if failed else ''
    msg = f"""
    <div class="alert alert-success text-center">
        <h4>Refresh Complete!</h4>
        <p><strong>{refreshed}</strong> stocks updated</p>
        {failed_msg}
        <a href="/index" class="btn btn-primary mt-2">Back</a>
    </div>
    """
    return HTMLResponse(msg)
    
# === INDIVIDUAL REFRESH ===
@app.post("/refresh-single/{ticker}")
async def refresh_single(ticker: str):
    with get_db() as db:
        stock = db.query(StockModel).filter(StockModel.ticker == ticker).first()
        if not stock:
            return {"status": "error", "message": "Not found"}
        try:
            result = analyze_stock(stock.ticker, stock.company_name)
            db.add(StockScoreModel(**result))
            db.commit()
            return {"status": "success", "ticker": ticker}
        except Exception as e:
            logger.error(f"Refresh failed {ticker}: {e}")
            return {"status": "error", "ticker": ticker}

# === REFRESH ALL (One-by-one, non-blocking) ===

@app.post("/refresh-all-start")
async def refresh_all_start():
    with get_db() as db:
        stocks = db.query(StockModel).filter(StockModel.is_active == True).all()
        total = len(stocks)
        refreshed = failed = 0
        for stock in stocks:
            try:
                result = analyze_stock(stock.ticker, stock.company_name)
                db.add(StockScoreModel(**result))
                refreshed += 1
            except Exception as e:
                logger.error(f"Failed {stock.ticker}: {e}")
                failed += 1
            db.commit()  # Commit per stock = safe
        return {
            "total": total,
            "refreshed": refreshed,
            "failed": failed,
            "message": f"{refreshed}/{total} updated"
        }


# === DELETE STOCK ===
@app.delete("/delete-stock/{ticker}")
async def delete_stock(ticker: str):
    """
    DELETE /delete-stock/RELIANCE.NS
    → Removes stock + all its scores from DB
    """
    with get_db() as db:
        # Find stock
        stock = db.query(StockModel).filter(StockModel.ticker == ticker).first()
        if not stock:
            return {"status": "error", "message": "Stock not found"}

        # Delete all scores
        deleted_scores = db.query(StockScoreModel).filter(StockScoreModel.ticker == ticker).delete()
        
        # Delete stock
        db.delete(stock)
        db.commit()

        logger.info(f"Deleted {ticker} | Removed {deleted_scores} score(s)")
        return {"status": "success", "ticker": ticker}
        
@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/index")
# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)