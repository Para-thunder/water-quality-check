# Dockerfile for MLOps project
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt ./
RUN pip install --no-cache-dir -r requirements_docker.txt

COPY . .

# Expose port for Flask
EXPOSE 8000

# Default command to run the Flask API
CMD ["python", "src/predict_api_flask.py"]
