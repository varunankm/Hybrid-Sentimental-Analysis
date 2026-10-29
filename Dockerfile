FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader vader_lexicon punkt wordnet averaged_perceptron_tagger omw-1.4 punkt_tab

# Copy application code
COPY . .

# Expose port
EXPOSE 10000

# Start gunicorn
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 180
