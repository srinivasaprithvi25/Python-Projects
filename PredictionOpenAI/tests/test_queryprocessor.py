import json
import types
import pytest

from utils import QueryProcessor

class DummyResponse:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]

def dummy_create(*args, **kwargs):
    return DummyResponse('{invalid}')


def test_parse_query_invalid_json(monkeypatch):
    monkeypatch.setattr(QueryProcessor.client.chat.completions, 'create', dummy_create)
    with pytest.raises(ValueError):
        QueryProcessor.parse_query('test query')
