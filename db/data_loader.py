import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Laden von Umgebungsvariablen
load_dotenv()

user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

# Verbindung zu PostgreSQL herstellen
engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
)

def load_data():
    df = pd.read_sql("SELECT * FROM sales", engine)
    return df
