# stock_analysis_script.py - Step-by-Step Stock Analysis from DB

import sqlite3
from datetime import datetime
import requests
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
DATABASE_PATH = "./databases/stock_recommender.db"
NEWSAPI_KEY = "f44ef3442436422983ac6a1c353e5f21"

# Step 1: Connect to DB and list stocks
def list_stocks_from_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, company_name FROM stocks")
    stocks = cursor.fetchall()
    conn.close()
    if not stocks:
        logger.info("No stocks available in the database.")
        return []
    logger.info(f"Found {len(stocks)} stocks in DB.")
    return stocks

# Step 2: Fetch news for a company
def fetch_news(company: str):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": company,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": NEWSAPI_KEY,
            "pageSize": 10,
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            logger.info(f"Fetched {len(articles)} news articles for {company}.")
            return articles
        else:
            logger.warning(f"NewsAPI error for {company}: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching news for {company}: {e}")
        return []

# Step 3: Analyze sentiment from news
def analyze_sentiment(articles):
    if not articles:
        return 0.5, "Neutral (no news)"

    scores = []
    for art in articles:
        text = (art.get("title") or "") + " " + (art.get("description") or "")
        if text.strip():
            polarity = TextBlob(text).sentiment.polarity
            scores.append(polarity)

    avg_polarity = sum(scores) / len(scores) if scores else 0
    sentiment_score = (avg_polarity + 1) / 2  # Normalize to 0-1
    sentiment_label = "Positive" if sentiment_score > 0.6 else "Negative" if sentiment_score < 0.4 else "Neutral"
    logger.info(f"Sentiment score: {sentiment_score:.2f} ({sentiment_label})")
    return sentiment_score, sentiment_label

# Step 4: Make buy/hold decision
def make_decision(sentiment_score: float):
    if sentiment_score >= 0.6:
        return "BUY"
    elif sentiment_score <= 0.4:
        return "SELL"
    else:
        return "HOLD"

# Main function: Step-by-step analysis
def run_analysis():
    stocks = list_stocks_from_db()
    if not stocks:
        print("No stocks in DB. Add some stocks first.")
        return

    results = []
    for ticker, company in stocks:
        print(f"\nAnalyzing {ticker} - {company}:")
        
        # Step 2: News
        articles = fetch_news(company)
        has_news = "Yes" if articles else "No"
        print(f" - Is there news related to this share? {has_news}")
        
        # Step 3: Sentiment
        sentiment_score, sentiment_label = analyze_sentiment(articles)
        print(f" - Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")
        
        # Step 4: Decision
        decision = make_decision(sentiment_score)
        print(f" - Decision: {decision}")
        
        results.append({
            "ticker": ticker,
            "company": company,
            "has_news": has_news,
            "sentiment": sentiment_label,
            "score": sentiment_score,
            "decision": decision
        })
    
    # Optional: Save or display results
    print("\nSummary Table:")
    print("{:<15} {:<30} {:<10} {:<15} {:<10} {:<10}".format(
        "Ticker", "Company", "Has News", "Sentiment", "Score", "Decision"
    ))
    for res in results:
        print("{:<15} {:<30} {:<10} {:<15} {:<10.2f} {:<10}".format(
            res["ticker"], res["company"][:27] + "..." if len(res["company"]) > 30 else res["company"],
            res["has_news"], res["sentiment"], res["score"], res["decision"]
        ))

if __name__ == "__main__":
    run_analysis()