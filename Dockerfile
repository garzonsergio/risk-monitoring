FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8005

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8005"]