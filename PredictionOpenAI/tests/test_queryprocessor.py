import json
import types
import pytest

from utils import QueryProcessor

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


def test_parse_query_invalid_json(monkeypatch):
    monkeypatch.setattr(QueryProcessor.client.chat.completions, 'create', dummy_create)
    with pytest.raises(ValueError):
        QueryProcessor.parse_query('test query')


def test_parse_query_returns_query(monkeypatch):
    monkeypatch.setattr(QueryProcessor.client.chat.completions, 'create', dummy_create_valid)
    result = QueryProcessor.parse_query('test query')
    assert result['query'] == 'SELECT * FROM sales'
