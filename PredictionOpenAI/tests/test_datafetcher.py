import os
import sys
import pandas as pd
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sqlalchemy import text
from PredictionOpenAI.app.utils.DatabaseConnection import get_db_engine
from PredictionOpenAI.app.utils.DataFetcher import fetch_data


DB_FILE = 'test.db'

def setup_module(module):
    os.environ['DB_TYPE'] = 'sqlite'
    os.environ['DB_PATH'] = DB_FILE
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE sales (date TEXT, sales INTEGER, category TEXT)'))
        conn.execute(text("INSERT INTO sales VALUES ('2023-01-01', 100, 'A')"))
        conn.execute(text('CREATE TABLE category_lookup (category TEXT, description TEXT)'))
        conn.execute(text("INSERT INTO category_lookup VALUES ('A', 'Category A')"))


def teardown_module(module):
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


def test_fetch_data_valid_columns():
    query_info = {
        'table': 'sales',
        'date_column': 'date',
        'target_column': 'sales',
        'columns': ['date', 'sales', 'category']
    }
    df = fetch_data(query_info)
    assert list(df.columns) == ['date', 'sales', 'category']
    assert not df.empty


def test_fetch_data_invalid_column():
    query_info = {
        'table': 'sales',
        'date_column': 'date',
        'target_column': 'sales',
        'columns': ['date', 'sales', 'missing']
    }
    with pytest.raises(ValueError):
        fetch_data(query_info)



def test_fetch_data_direct_query():
    query_info = {
        'query': 'SELECT date, sales FROM sales',
        'table': 'sales',
        'date_column': 'date',
        'target_column': 'sales'
    }
    df = fetch_data(query_info)
    assert list(df.columns) == ['date', 'sales']
    assert not df.empty

def test_fetch_data_mongodb(monkeypatch):
    os.environ['DB_TYPE'] = 'mongodb'

    def dummy_mongo(query, db_name, collection):
        return pd.DataFrame({'date': ['2023-01-01'], 'sales': [100]})

    monkeypatch.setattr('utils.DataFetcher._fetch_mongo', dummy_mongo)
    query_info = {
        'query': {'date': '2023-01-01'},
        'table': 'sales',
        'date_column': 'date',
        'target_column': 'sales'
    }
    df = fetch_data(query_info)
    assert not df.empty
    os.environ['DB_TYPE'] = 'sqlite'


def test_fetch_data_join_query():
    query_info = {
        'tables': ['sales s', 'category_lookup c'],
        'joins': 'JOIN category_lookup c ON s.category = c.category',
        'date_column': 's.date',
        'target_column': 's.sales',
        'columns': ['s.date', 's.sales', 'c.description']
    }
    df = fetch_data(query_info)
    assert list(df.columns) == ['date', 'sales', 'description']
    assert df['description'].iloc[0] == 'Category A'

