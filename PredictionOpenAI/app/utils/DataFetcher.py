import pandas as pd
from utils.DatabaseConnection import get_db_engine

def fetch_data(query_info):
    engine = get_db_engine()
    date_col = query_info['date_column']
    target_col = query_info['target_column']
    table = query_info['table']
    filters = query_info.get('filters', '')

    sql = f"SELECT {date_col}, {target_col}, Customercode FROM {table}"
    if filters:
        sql += f" WHERE {filters}"

    df = pd.read_sql(sql, engine, parse_dates=[date_col])
    return df.sort_values(by=date_col)
