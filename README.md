# Hybrid Sentiment Analysis

A Flask web application for analysing app reviews and user feedback using multiple NLP algorithms.

## Features

- **Google Play Store** — paste any app URL and analyse live reviews
- **File Upload** — upload CSV, Excel, TXT, JSON or PDF files of feedback
- **Four Algorithms** — VADER, BERT, LSTM, and a Hybrid ensemble (BERT 50% + VADER 30% + LSTM 20%)
- **Aspect-level analysis** — usability, performance, design, features, reliability and more
- **Interactive charts** — sentiment pie, aspect bar chart, radar chart (Plotly)
- **PDF report** — downloadable analysis report

## Tech Stack

- **Backend** — Python, Flask, gunicorn
- **NLP** — NLTK (VADER), Hugging Face Transformers (DistilBERT), TextBlob, spaCy
- **Charts** — Plotly
- **PDF** — ReportLab, PyPDF2

## Local Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon punkt
python app.py
```

Then open `http://localhost:5000`

## Deployment

Deployed on [Render](https://render.com) — see `render.yaml` for build config.
