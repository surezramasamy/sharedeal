# clean_db.py
import sqlite3

conn = sqlite3.connect("./databases/stock_recommender.db")
cur = conn.cursor()

# Find bad rows
cur.execute("SELECT id, ticker FROM stocks WHERE ticker LIKE '%,%'")
bad = cur.fetchall()

print(f"Found {len(bad)} bad entries:")

for id_, ticker in bad:
    print(f"  ID {id_}: {ticker}")
    # Split and re-insert clean
    tickers = [t.strip() for t in ticker.split(",") if t.strip()]
    cur.execute("DELETE FROM stocks WHERE id = ?", (id_,))
    for t in tickers:
        # Get company name from stock_pool or use ticker
        cur.execute("INSERT OR IGNORE INTO stocks (ticker, company_name) VALUES (?, ?)", (t, t))

conn.commit()
conn.close()
print("Database cleaned.")