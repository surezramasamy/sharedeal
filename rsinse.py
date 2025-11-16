# test_rsi_fixed.py - FINAL WORKING VERSION
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rsi_yfinance(symbol: str, period: int = 14):
    logger.info(f"\n=== TEST: {symbol} ===")
    try:
        ticker = symbol if symbol.endswith('.NS') else symbol + '.NS'
        logger.info(f"Step 1: Downloading {ticker}")
        
        # FIX: Pass as list to avoid multi-index issues
        df = yf.download([ticker], period="3mo", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty:
            logger.error("Step 2: Empty DataFrame!")
            return 50.0

        # FIX: Extract single ticker column
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'][ticker]
        else:
            close = df['Close']

        logger.info(f"Step 2: Fetched {len(close)} rows")
        logger.info(f"Step 3: Tail Close prices:\n{close.tail()}")

        if len(close) < period + 1:
            logger.warning("Step 4: Not enough data for RSI")
            return 50.0

        # RSI Calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        rsi_series = 100 - (100 / (1 + rs))
        
        current_rsi = rsi_series.iloc[-1]
        logger.info(f"Step 4: RSI = {current_rsi}")

        if pd.isna(current_rsi):
            logger.error("Step 5: RSI is NaN")
            return 50.0

        rsi_value = round(float(current_rsi), 1)  # ← CRITICAL: Convert to float
        logger.info(f"Final RSI({period}) = {rsi_value}")

        if rsi_value < 30:
            logger.info("Oversold → Strong BUY")
        elif rsi_value > 70:
            logger.info("Overbought → SELL")
        elif 45 <= rsi_value <= 55:
            logger.info("Stable → Mild BUY")
        else:
            logger.info("Neutral")

        return rsi_value

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 50.0

# TEST
if __name__ == "__main__":
    stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    for stock in stocks:
        rsi = test_rsi_yfinance(stock)
        print(f"{stock}: RSI = {rsi}")