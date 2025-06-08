import json

import os
import sys
import types
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from PredictionOpenAI.app.utils import QueryProcessor


class DummyResponse:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]

def dummy_create(*args, **kwargs):
    return DummyResponse('{invalid}')


def dummy_create_valid(*args, **kwargs):
    content = json.dumps({
        "table": "sales",
        "date_column": "date",
        "target_column": "sales",
        "query": "SELECT * FROM sales"
    })
    return DummyResponse(content)


def dummy_create_join(*args, **kwargs):
    content = json.dumps({
        "tables": ["sales s", "category_lookup c"],
        "joins": "JOIN category_lookup c ON s.category = c.category",
        "date_column": "s.date",
        "target_column": "s.sales"
    })
    return DummyResponse(content)



def test_parse_query_invalid_json(monkeypatch):
    monkeypatch.setattr(QueryProcessor.client.chat.completions, 'create', dummy_create)
    with pytest.raises(ValueError):
        QueryProcessor.parse_query('test query')


def test_parse_query_returns_query(monkeypatch):
    monkeypatch.setattr(QueryProcessor.client.chat.completions, 'create', dummy_create_valid)
    result = QueryProcessor.parse_query('test query')
    assert result['query'] == 'SELECT * FROM sales'


def test_parse_query_with_joins(monkeypatch):
    monkeypatch.setattr(QueryProcessor.client.chat.completions, 'create', dummy_create_join)
    result = QueryProcessor.parse_query('test query')
    assert result['tables'][0] == 'sales s'
    assert 'joins' in result
