FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agentic_security/ agentic_security/
COPY web/ web/

EXPOSE 8000

CMD ["python", "-m", "agentic_security", "serve", "--host", "0.0.0.0"]
