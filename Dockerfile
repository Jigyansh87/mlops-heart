FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies only
COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

# Copy application code and model artifacts
COPY src/app.py /app/app.py
COPY artifacts /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
