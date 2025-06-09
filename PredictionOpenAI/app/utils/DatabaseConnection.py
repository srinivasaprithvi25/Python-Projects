import os
import json
from sqlalchemy import create_engine, text, inspect
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


    engine = create_engine(uri)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Connected to database")
        list_schemas_and_tables(engine)

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

    return engine
def list_schemas_and_tables(engine):
    """Print available schemas, tables, and columns. Save as JSON for the LLM."""
    inspector = inspect(engine)
    schema_info = {}
    try:
        print("\n\U0001F4C2 Schemas and Tables in the Database:")
        for schema in inspector.get_schema_names():
            print(f"\n\U0001F4C1 Schema: {schema}")
            schema_info[schema] = {}
            for table in inspector.get_table_names(schema=schema):
                print(f"  \U0001F4C4 {table}")
                columns = inspector.get_columns(table, schema=schema)
                schema_info[schema][table] = {
                    col["name"]: str(col["type"]) for col in columns
                }
                for col in columns:
                    print(f"    - {col['name']} ({col['type']})")
    except Exception as e:
        print(f"❌ Failed to list schemas/tables: {e}")
    # persist schema info for prompting
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/schema_info.json", "w") as f:
            json.dump(schema_info, f, indent=2)
    except Exception:
        pass

