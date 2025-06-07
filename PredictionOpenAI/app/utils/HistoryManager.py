import json
import os

HISTORY_FILE = "data/history.json"

def save_query_history(query, metadata):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)

    history.append({"query": query, "metadata": metadata})
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
