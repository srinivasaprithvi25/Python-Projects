import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

def get_db_engine():
    db_type = os.getenv("DB_TYPE")
    user = quote_plus(os.getenv("DB_USER"))
    password = quote_plus(os.getenv("DB_PASSWORD"))
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")

    if db_type == "postgresql":
        uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    elif db_type == "mysql":
        uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
    elif db_type == "mssql":
        driver = quote_plus("ODBC Driver 17 for SQL Server")
        uri = f"mssql+pyodbc://{user}:{password}@{host}/{dbname}?driver={driver}"
    elif db_type == "sqlite":
        uri = f"sqlite:///{os.getenv('DB_PATH')}"
    else:
        raise ValueError("Unsupported DB type")

    return create_engine(uri)
