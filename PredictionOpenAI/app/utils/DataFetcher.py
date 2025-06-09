import os
import pandas as pd
import sqlparse
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
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

def validate_sql(sql):
    """Basic SQL syntax validation using sqlparse."""
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            raise ValueError("No SQL statement found")
    except Exception as exc:
        raise ValueError(f"Invalid SQL syntax: {exc}") from exc

def fetch_data(query_info, extra_columns=None):
    db_type = os.getenv("DB_TYPE")
    date_col = query_info["date_column"]

    if "query" in query_info:
        if db_type == "mongodb":
            df = _fetch_mongo(query_info["query"], os.getenv("DB_NAME"), query_info["table"])
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col)
        else:
            engine = get_db_engine()
            validate_sql(query_info["query"])
            sql = text(query_info["query"])
            print("🔍 Executing SQL:", sql.text)
            try:
                df = pd.read_sql_query(sql, engine, parse_dates=[date_col])
            except SQLAlchemyError as exc:
                raise ValueError(f"Failed to execute SQL: {sql.text}\n{exc}") from exc
            df = df.sort_values(by=date_col)

        required_cols = [date_col]
        target_col = query_info.get("target_column", [])
        if isinstance(target_col, list):
            required_cols.extend(target_col)
        else:
            required_cols.append(target_col)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Query result missing required columns: {missing}")
        return df


    table = query_info.get('table')
    engine = get_db_engine()
    target_col = query_info['target_column']
    if isinstance(target_col, list):
        target_col = target_col[0]
    schema = query_info.get('schema')
    filters = query_info.get('filters', '')
    join_clause = query_info.get('join_clause')
    order_by = query_info.get('order_by')
    limit = query_info.get('limit')

    columns = query_info.get('columns', [date_col, target_col])
    if extra_columns:
        columns.extend(extra_columns)

    col_str = ', '.join(columns)

    inspector = inspect(engine)
    table_name = table.split('.')[-1] if '.' in table else table
    schema_name = schema or (table.split('.')[0] if '.' in table else None)
    if not join_clause:
        table_columns = [c['name'] for c in inspector.get_columns(table_name, schema=schema_name)]
        for col in columns:
            if col not in table_columns:
                raise ValueError(f"Column '{col}' does not exist in table '{table}'")

    full_table = f"{schema_name}.{table_name}" if schema_name else table_name
    sql_parts = [f"SELECT {col_str} FROM {full_table}"]
    if join_clause:
        sql_parts.append(join_clause)
    if filters:
        sql_parts.append(f"WHERE {filters}")
    if order_by:
        sql_parts.append(f"ORDER BY {order_by}")
    if limit:
        sql_parts.append(f"LIMIT {limit}")
    sql_str = ' '.join(sql_parts)
    validate_sql(sql_str)
    sql = text(sql_str)

    print("🔍 Executing SQL:", sql.text)

    try:
        df = pd.read_sql_query(sql, engine, parse_dates=[date_col])
    except SQLAlchemyError as exc:
        raise ValueError(f"Failed to execute SQL: {sql.text}\n{exc}") from exc
    df = df.sort_values(by=date_col)
    required_cols = [date_col]
    if isinstance(query_info['target_column'], list):
        required_cols.extend(query_info['target_column'])
    else:
        required_cols.append(query_info['target_column'])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Query result missing required columns: {missing}")
    return df
