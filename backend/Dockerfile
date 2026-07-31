FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for build andlibpcap
RUN apt-get update && apt-get install -y \
    build-essential \
    libpcap-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Initialize model and database
RUN python synthetic_data.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
