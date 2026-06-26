FROM python:3.11-slim

WORKDIR /app

# Install git (needed for repo cloning)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 cortex

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

RUN mkdir -p data && chown -R cortex:cortex /app

USER cortex

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
