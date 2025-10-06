FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directory for DuckDB database
RUN mkdir -p /app/datasets

# Initialize the database
RUN python init_db.py

# Expose the port
EXPOSE 8000

# Start the application
CMD ["python", "main.py"] 