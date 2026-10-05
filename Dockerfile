FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY web ./web
ENV PYTHONUNBUFFERED=1
# NEBIUS_API_KEY comes from the service environment; only the Run it button needs it.
CMD ["sh", "-c", "uvicorn --app-dir backend main:app --host 0.0.0.0 --port ${PORT:-8080}"]
