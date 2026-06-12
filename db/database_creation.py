import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

df = pd.read_csv("../data/processed/cleaned_data.csv")
df.columns = df.columns.str.lower()


engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
)

df.to_sql("sales", engine, if_exists="replace", index=False)

print("DATA LOADED SUCCESSFULLY")