import pandas as pd
from sqlalchemy import text, inspect
from utils.DatabaseConnection import get_db_engine

def fetch_data(query_info, extra_columns=None):
    engine = get_db_engine()
    date_col = query_info['date_column']
    target_col = query_info['target_column']
    table = query_info['table']
    filters = query_info.get('filters', '')

    columns = query_info.get('columns', [date_col, target_col])
    if extra_columns:
        columns.extend(extra_columns)

    inspector = inspect(engine)
    table_columns = [c['name'] for c in inspector.get_columns(table)]
    for col in columns:
        if col not in table_columns:
            raise ValueError(f"Column '{col}' does not exist in table '{table}'")

    col_str = ', '.join(columns)
    sql = text(f"SELECT {col_str} FROM {table}")
    if filters:
        sql = text(f"{sql.text} WHERE {filters}")

    print("🔍 Executing SQL:", sql.text)


    df = pd.read_sql_query(sql, engine, parse_dates=[date_col])
    return df.sort_values(by=date_col)
