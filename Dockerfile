FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agentic_security/ agentic_security/
COPY web/ web/
RUN mkdir -p data

EXPOSE 8000
# Listen on Railway's injected $PORT (falls back to 8000 for local runs)
CMD ["sh", "-c", "python -m uvicorn agentic_security.api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
