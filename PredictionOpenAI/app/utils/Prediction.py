import os
from utils.QueryProcessor import parse_query
from utils.DataFetcher import fetch_data
from utils.Forecaster import run_forecasting_models
from utils.Visualizer import plot_and_save
from utils.HistoryManager import save_query_history

def main():
    print("🧠 Welcome to ForecastAI")
    query = input("Ask your business question: ")

    try:
        parsed = parse_query(query)
        print("✅ Parsed Query:", parsed)

        df = fetch_data(parsed)
        predictions = run_forecasting_models(df, parsed)
        plot_and_save(df, predictions, parsed)

        save_query_history(query, parsed)
    except Exception as e:
        print(f"❌ Error: {e}")
