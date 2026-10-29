FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all deps with Python 3.11 wheels
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2+cpu && \
    pip install safetensors==0.4.2 && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader vader_lexicon punkt wordnet \
    averaged_perceptron_tagger omw-1.4 punkt_tab

COPY . .

EXPOSE 10000

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 180
