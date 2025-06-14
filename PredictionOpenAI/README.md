# Forecast AI

An LLM-integrated time series forecasting tool that uses OpenAI to interpret business questions, generate SQL queries, retrieve data, and run models like SARIMA, LSTM, XGBoost, and ARIMA to forecast future values.

## Features

- Natural language interface for business forecasting
- Smart SQL query generation using LLM
- Multi-model ensemble: SARIMA, LSTM, XGBoost, ARIMA
- Dockerized for deployment
- Logs and prompt history support

## Usage

```bash
docker-compose up --build
```

The container expects a `.env` file in this directory containing your
database connection details and OpenAI key:

```bash
OPENAI_API_KEY=your-openai-key
DB_TYPE=postgresql  # or mysql, mssql, sqlite
DB_USER=username
DB_PASSWORD=password
DB_HOST=hostname
DB_PORT=5432
DB_NAME=database_name
# If using SQLite set DB_PATH instead of the above connection options
```

### Running with Docker Compose

1. Build and start the container:
   ```bash
   docker-compose up --build
   ```
2. The application will launch an interactive CLI where you can type a
   business question to generate a forecast.

### Running Locally

If you prefer to run outside of Docker, create a virtual environment and
install the dependencies manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ensure the same `.env` file is present (or set the environment variables in
your shell) and then run:

```bash
python app/main.py
```

Forecast plots are saved under `logs/` and query history under
`data/history.json`.

