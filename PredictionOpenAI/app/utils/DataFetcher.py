import os
import pandas as pd
from sqlalchemy import text, inspect
from utils.DatabaseConnection import get_db_engine

def _fetch_mongo(query, db_name, collection):
    from pymongo import MongoClient
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", 27017))
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    if user and password:
        uri = f"mongodb://{user}:{password}@{host}:{port}"
    else:
        uri = f"mongodb://{host}:{port}"
    client = MongoClient(uri)
    coll = client[db_name][collection]
    if isinstance(query, list):
        cursor = coll.aggregate(query)
    else:
        cursor = coll.find(query)
    return pd.DataFrame(list(cursor))

def fetch_data(query_info, extra_columns=None):
    db_type = os.getenv("DB_TYPE")
    date_col = query_info["date_column"]

    if "query" in query_info:
        if db_type == "mongodb":
            df = _fetch_mongo(query_info["query"], os.getenv("DB_NAME"), query_info["table"])
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            return df.sort_values(by=date_col)

        engine = get_db_engine()
        sql = text(query_info["query"])
        print("🔍 Executing SQL:", sql.text)
        df = pd.read_sql_query(sql, engine, parse_dates=[date_col])
        return df.sort_values(by=date_col)

    table = query_info.get('table')
    engine = get_db_engine()
    target_col = query_info['target_column']
    schema = query_info.get('schema')
    filters = query_info.get('filters', '')
    tables = query_info.get('tables')
    joins = query_info.get('joins')

    columns = query_info.get('columns', [date_col, target_col])
    if extra_columns:
        columns.extend(extra_columns)

    col_str = ', '.join(columns)

    if tables:
        # Build FROM clause using provided tables and joins
        if isinstance(tables, str):
            tables = [tables]
        from_clause = tables[0]
        if joins:
            if isinstance(joins, list):
                for j in joins:
                    if isinstance(j, dict):
                        join_type = j.get('type', 'JOIN')
                        from_clause += f" {join_type} {j['table']} ON {j['on']}"
                    else:
                        from_clause += f" {j}"
            else:
                from_clause += f" {joins}"
        elif len(tables) > 1:
            # simple comma separated tables if no joins provided
            from_clause = ', '.join(tables)
        sql = text(f"SELECT {col_str} FROM {from_clause}")
    else:
        inspector = inspect(engine)
        table_name = table.split('.')[-1] if '.' in table else table
        schema_name = schema or (table.split('.')[0] if '.' in table else None)
        table_columns = [c['name'] for c in inspector.get_columns(table_name, schema=schema_name)]
        for col in columns:
            if col not in table_columns:
                raise ValueError(f"Column '{col}' does not exist in table '{table}'")

        full_table = f"{schema_name}.{table_name}" if schema_name else table_name
        sql = text(f"SELECT {col_str} FROM {full_table}")
    if filters:
        sql = text(f"{sql.text} WHERE {filters}")

    print("🔍 Executing SQL:", sql.text)

    df = pd.read_sql_query(sql, engine, parse_dates=[date_col])
    return df.sort_values(by=date_col)
