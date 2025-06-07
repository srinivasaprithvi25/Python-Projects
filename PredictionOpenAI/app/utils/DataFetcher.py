import pandas as pd
from utils.DatabaseConnection import get_db_engine

def fetch_data(query_info, extra_columns=None):
    engine = get_db_engine()
    date_col = query_info['date_column']
    target_col = query_info['target_column']
    table = query_info['table']
    filters = query_info.get('filters', '')

    columns = [date_col, target_col, 'Customercode']
    if extra_columns:
        columns.extend(extra_columns)

    col_str = ', '.join(columns)
    sql = f"SELECT {col_str} FROM {table}"
    if filters:
        sql += f" WHERE {filters}"

    df = pd.read_sql(sql, engine, parse_dates=[date_col])
    return df.sort_values(by=date_col)
