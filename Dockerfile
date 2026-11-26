FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEPSE_DATA_DIR=/app/data \
    NEPSE_REPORTS_DIR=/app/reports

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY data/fixtures ./data/fixtures

RUN pip install .

# Build the fixture database and reports into the image so both services start
# without a separate init step.
RUN python scripts/fetch_data.py --source fixture && python scripts/train.py

RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8501 8000

CMD ["streamlit", "run", "src/nepse_lens/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
