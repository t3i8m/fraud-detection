import io
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CHUNKSIZE = 500_000

conn = psycopg2.connect(host="localhost",port=os.environ["POSTGRES_PORT"],dbname=os.environ["POSTGRES_DB"],user=os.environ["POSTGRES_USER"],password=os.environ["POSTGRES_PASSWORD"],)
cur = conn.cursor()

fraud_labels = pd.read_json("data/raw/train_fraud_labels.json")
fraud_labels = fraud_labels.reset_index().rename(columns={"index": "id", "target": "target"})
fraud_labels["id"] = fraud_labels["id"].astype(int)

for chunk in pd.read_csv("data/raw/transactions_data.csv", parse_dates=["date"], chunksize=CHUNKSIZE):
    chunk["amount"] = chunk["amount"].astype(str).str.replace("$", "", regex=False).astype(float)
    chunk = chunk.merge(fraud_labels, on="id", how="inner")

    buffer = io.StringIO()
    chunk.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    cur.copy_expert("COPY transactions FROM STDIN WITH (FORMAT csv, NULL '')", buffer)
    conn.commit()
    print("loaded chunk:", len(chunk))

dollar_cols = {"users": ["per_capita_income", "yearly_income", "total_debt"],"cards": ["credit_limit"],}

for table, path in [("users", "data/raw/users_data.csv"), ("cards", "data/raw/cards_data.csv")]:
    df = pd.read_csv(path)
    for col in dollar_cols[table]:
        df[col] = df[col].astype(str).str.replace("$", "", regex=False).astype(float)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, NULL '')", buffer)
    conn.commit()
    print("loaded:", table)

cur.close()
conn.close()
