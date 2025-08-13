# ---- Base (linux/amd64-friendly) ----
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code & artifacts
COPY app ./app
COPY car_price_model_retail.pkl car_price_model_private.pkl car_price_model_tradein.pkl CFXCLEANEDDATA081225.csv ./

# Runtime
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
