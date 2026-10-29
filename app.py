from flask import Flask, render_template, request, send_file, jsonify
from google_play_scraper import app as gp_app, reviews, reviews_all, Sort
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
from collections import defaultdict
from transformers import pipeline
import torch
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
import os
from functools import lru_cache
import time
import pandas as pd
import io as _io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize BERT sentiment pipeline
try:
    bert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"Warning: BERT pipeline initialization failed: {e}")
    bert_pipeline = None

app = Flask(__name__)

ASPECTS = {
    'usability': ['use', 'user-friendly', 'intuitive', 'easy', 'difficult', 'complicated', 'interface', 'navigation', 'simple', 'hard'],
    'performance': ['fast', 'slow', 'speed', 'crash', 'bug', 'glitch', 'freeze', 'responsive', 'performance', 'loading', 'lag'],
    'design': ['design', 'layout', 'look', 'ui', 'ux', 'beautiful', 'ugly', 'clean', 'modern', 'interface', 'theme', 'style'],
    'features': ['feature', 'functionality', 'option', 'capability', 'tool', 'function', 'add', 'suggestion', 'request'],
    'reliability': ['reliable', 'stable', 'consistent', 'crash', 'bug', 'error', 'issue', 'problem', 'work', 'broken'],
    'support': ['support', 'help', 'contact', 'customer service', 'response', 'assistance', 'email'],
    'privacy': ['privacy', 'data', 'security', 'permission', 'safe', 'secure', 'trust'],
    'price': ['price', 'cost', 'free', 'paid', 'subscription', 'purchase', 'expensive', 'cheap'],
    'updates': ['update', 'version', 'latest', 'new', 'old', 'frequent', 'improvement']
}

# OPTIMIZATION 1: Cache spaCy NLP processing
@lru_cache(maxsize=5000)
def cached_nlp_process(text):
    """Cache spaCy NLP results to avoid reprocessing"""
    return nlp(text.lower())

def get_playstore_reviews(app_id, analysis_type='quick'):
    try:
        if analysis_type == 'full':
            all_reviews = reviews_all(
                app_id,
                sleep_milliseconds=0,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
            )
            return [{'content': review['content'], 'score': review['score']}
                   for review in all_reviews if review.get('content')]
        else:
            result, _ = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
                count=500  # Fetch 500 reviews for quick analysis
            )
            return [{'content': review['content'], 'score': review['score']} 
                   for review in result if review.get('content')]

    except Exception as e:
        print(f"Error: {e}")
        return None

# OPTIMIZATION 3: Use cached NLP processing
def identify_aspects(text):
    doc = cached_nlp_process(text)  # Use cached version instead of nlp()
    found_aspects = defaultdict(float)
    
    # Get all words and lemmatize them
    text_words = {token.lemma_ for token in doc}
    
    # Check each aspect
    for aspect, keywords in ASPECTS.items():
        for keyword in keywords:
            if keyword in text_words:
                found_aspects[aspect] += 1
                
    return dict(found_aspects)

def analyze_sentiment(reviews, algorithm='vader'):
    if not reviews:
        return {}, {}, [], {}, {}
    
    if algorithm == 'vader':
        return analyze_sentiment_vader(reviews)
    elif algorithm == 'bert':
        return analyze_sentiment_bert(reviews)
    elif algorithm == 'lstm':
        return analyze_sentiment_lstm(reviews)
    elif algorithm == 'hybrid':
        return analyze_sentiment_hybrid(reviews)
    else:
        return analyze_sentiment_vader(reviews)

# OPTIMIZATION 4: Optimize VADER analysis - reuse SIA instance
def analyze_sentiment_vader(reviews):
    sia = SentimentIntensityAnalyzer()  # Create once, reuse for all reviews
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results = []
    aspect_scores = defaultdict(list)
    
    for review in reviews:
        text = review['content']
        score = sia.polarity_scores(text)['compound']
        
        # Overall sentiment
        sentiment = 'neutral'
        if score >= 0.05:
            sentiment = 'positive'
            counts['positive'] += 1
        elif score <= -0.05:
            sentiment = 'negative'
            counts['negative'] += 1
        else:
            counts['neutral'] += 1
            
        # Aspect-based analysis
        aspects = identify_aspects(text)
        review_aspects = {}
        
        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_scores[aspect].append(score)
                
                if score >= 0.05:
                    aspect_sentiments[aspect]['positive'] += 1
                elif score <= -0.05:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1
                    
                review_aspects[aspect] = score

        detailed_results.append({
            'text': text,
            'sentiment': sentiment,
            'aspects': review_aspects
        })

    total = len(reviews)
    counts['total'] = total
    percentages = {k: round((v/total)*100, 2) for k, v in counts.items() if k != 'total'} if total > 0 else {}
    
    # Calculate average sentiment scores for each aspect
    aspect_averages = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            aspect_averages[aspect] = sum(scores) / len(scores)
    
    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages

def analyze_sentiment_bert(reviews):
    if bert_pipeline is None:
        return analyze_sentiment_vader(reviews)
    
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results = []
    aspect_scores = defaultdict(list)
    
    for review in reviews:
        text = review['content']
        # Truncate text to avoid token limit (BERT has 512 token limit)
        truncated_text = text[:512] if len(text) > 512 else text
        
        try:
            # Get BERT prediction
            result = bert_pipeline(truncated_text)[0]
            label = result['label'].upper()  # BERT returns 'POSITIVE' or 'NEGATIVE'
            confidence = result['score']  # Range: 0.5 to 1.0
            
            # Convert BERT confidence to sentiment score range [-1, 1]
            # If confidence is low (close to 0.5), the sentiment is less certain
            if label == 'POSITIVE':
                # Map confidence [0.5, 1.0] to score [0, 1]
                score = (confidence - 0.5) * 2.0
                
                # Determine sentiment based on confidence threshold
                if confidence >= 0.85:
                    sentiment = 'positive'
                    counts['positive'] += 1
                elif confidence >= 0.65:
                    sentiment = 'positive'
                    counts['positive'] += 1
                else:
                    # Low confidence positive = neutral
                    sentiment = 'neutral'
                    counts['neutral'] += 1
                    score = 0.0
                    
            elif label == 'NEGATIVE':
                # Map confidence [0.5, 1.0] to score [-1, 0]
                score = -((confidence - 0.5) * 2.0)
                
                # Determine sentiment based on confidence threshold
                if confidence >= 0.85:
                    sentiment = 'negative'
                    counts['negative'] += 1
                elif confidence >= 0.65:
                    sentiment = 'negative'
                    counts['negative'] += 1
                else:
                    # Low confidence negative = neutral
                    sentiment = 'neutral'
                    counts['neutral'] += 1
                    score = 0.0
            else:
                sentiment = 'neutral'
                counts['neutral'] += 1
                score = 0.0
                
        except Exception as e:
            print(f"BERT analysis error: {e}")
            sentiment = 'neutral'
            counts['neutral'] += 1
            score = 0.0
            
        # Aspect-based analysis
        aspects = identify_aspects(text)
        review_aspects = {}
        
        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_scores[aspect].append(score)
                
                if score >= 0.2:
                    aspect_sentiments[aspect]['positive'] += 1
                elif score <= -0.2:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1
                    
                review_aspects[aspect] = score

        detailed_results.append({
            'text': text,
            'sentiment': sentiment,
            'aspects': review_aspects
        })

    total = len(reviews)
    counts['total'] = total
    percentages = {k: round((v/total)*100, 2) for k, v in counts.items() if k != 'total'} if total > 0 else {}
    
    # Calculate average sentiment scores for each aspect
    aspect_averages = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            aspect_averages[aspect] = sum(scores) / len(scores)
    
    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages

def analyze_sentiment_lstm(reviews):
    # LSTM implementation using a simple approach with TextBlob for now
    # For a production system, you would load a trained LSTM model
    from textblob import TextBlob
    
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results = []
    aspect_scores = defaultdict(list)
    
    for review in reviews:
        text = review['content']
        
        try:
            # Use TextBlob polarity as a proxy for LSTM sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # Range: -1 to 1
            
            # Convert to sentiment
            sentiment = 'neutral'
            if polarity >= 0.1:
                sentiment = 'positive'
                counts['positive'] += 1
            elif polarity <= -0.1:
                sentiment = 'negative'
                counts['negative'] += 1
            else:
                counts['neutral'] += 1
        except Exception as e:
            print(f"LSTM analysis error: {e}")
            sentiment = 'neutral'
            polarity = 0
            counts['neutral'] += 1
            
        # Aspect-based analysis
        aspects = identify_aspects(text)
        review_aspects = {}
        
        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_scores[aspect].append(polarity)
                
                if polarity >= 0.1:
                    aspect_sentiments[aspect]['positive'] += 1
                elif polarity <= -0.1:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1
                    
                review_aspects[aspect] = polarity

        detailed_results.append({
            'text': text,
            'sentiment': sentiment,
            'aspects': review_aspects
        })

    total = len(reviews)
    counts['total'] = total
    percentages = {k: round((v/total)*100, 2) for k, v in counts.items() if k != 'total'} if total > 0 else {}
    
    # Calculate average sentiment scores for each aspect
    aspect_averages = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            aspect_averages[aspect] = sum(scores) / len(scores)
    
    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages

def analyze_sentiment_hybrid(reviews):
    """
    Hybrid model: combines VADER, BERT, and TextBlob (LSTM proxy) scores
    using a weighted ensemble:
        BERT   → 50%  (deep contextual understanding)
        VADER  → 30%  (fast lexicon-based, good on short text)
        TextBlob → 20% (morphological / polarity baseline)
    Each review gets a confidence score reflecting model agreement.
    """
    from textblob import TextBlob

    sia = SentimentIntensityAnalyzer()

    # Weights for each model
    W_BERT  = 0.50
    W_VADER = 0.30
    W_LSTM  = 0.20

    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results  = []
    aspect_scores     = defaultdict(list)

    for review in reviews:
        text = review['content']

        # ── 1. VADER score  (range -1 … +1) ──────────────────────────────
        vader_score = sia.polarity_scores(text)['compound']

        # ── 2. BERT score  (range -1 … +1) ───────────────────────────────
        bert_score = 0.0
        if bert_pipeline is not None:
            try:
                truncated = text[:512]
                result    = bert_pipeline(truncated)[0]
                label     = result['label'].upper()
                conf      = result['score']
                # map [0.5, 1.0] confidence → [-1, +1] signed score
                raw = (conf - 0.5) * 2.0
                bert_score = raw if label == 'POSITIVE' else -raw
            except Exception:
                bert_score = vader_score   # fallback

        # ── 3. TextBlob score  (range -1 … +1) ───────────────────────────
        try:
            lstm_score = TextBlob(text).sentiment.polarity
        except Exception:
            lstm_score = 0.0

        # ── 4. Weighted ensemble ──────────────────────────────────────────
        hybrid_score = (W_BERT  * bert_score +
                        W_VADER * vader_score +
                        W_LSTM  * lstm_score)

        # ── 5. Confidence  (1 – normalised std-dev of the three scores) ───
        scores_list = [vader_score, bert_score, lstm_score]
        mean   = sum(scores_list) / 3
        std    = (sum((s - mean) ** 2 for s in scores_list) / 3) ** 0.5
        # std is in [0, ~1.15]; normalise to [0, 1] then invert → confidence
        confidence = round(max(0.0, 1.0 - (std / 1.155)), 2)

        # ── 6. Final label ────────────────────────────────────────────────
        if hybrid_score >= 0.05:
            sentiment = 'positive'
            counts['positive'] += 1
        elif hybrid_score <= -0.05:
            sentiment = 'negative'
            counts['negative'] += 1
        else:
            sentiment = 'neutral'
            counts['neutral'] += 1

        # ── 7. Aspect analysis ────────────────────────────────────────────
        aspects        = identify_aspects(text)
        review_aspects = {}

        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_scores[aspect].append(hybrid_score)

                if hybrid_score >= 0.05:
                    aspect_sentiments[aspect]['positive'] += 1
                elif hybrid_score <= -0.05:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1

                review_aspects[aspect] = hybrid_score

        detailed_results.append({
            'text':       text,
            'sentiment':  sentiment,
            'aspects':    review_aspects,
            # extra hybrid metadata
            'hybrid_score':    round(hybrid_score, 4),
            'confidence':      confidence,
            'model_scores':    {
                'vader': round(vader_score, 4),
                'bert':  round(bert_score,  4),
                'lstm':  round(lstm_score,  4),
            }
        })

    total = len(reviews)
    counts['total'] = total
    percentages = (
        {k: round((v / total) * 100, 2) for k, v in counts.items() if k != 'total'}
        if total > 0 else {}
    )

    aspect_averages = {
        aspect: sum(sc) / len(sc)
        for aspect, sc in aspect_scores.items()
        if sc
    }

    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages


def create_visualizations(results, aspect_sentiments, aspect_averages):
    # Overall sentiment pie chart
    fig1 = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[results['positive'], results['neutral'], results['negative']],
        marker=dict(colors=['#43e97b', '#ffd700', '#ff6f61'])
    )])
    fig1.update_layout(title='Overall Sentiment Distribution')
    
    # Aspect-based sentiment bar chart
    aspects = list(aspect_sentiments.keys())
    positive_vals = [aspect_sentiments[aspect]['positive'] for aspect in aspects]
    neutral_vals = [aspect_sentiments[aspect]['neutral'] for aspect in aspects]
    negative_vals = [aspect_sentiments[aspect]['negative'] for aspect in aspects]

    fig2 = go.Figure(data=[
        go.Bar(name='Positive', x=aspects, y=positive_vals, marker_color='#43e97b'),
        go.Bar(name='Neutral', x=aspects, y=neutral_vals, marker_color='#ffd700'),
        go.Bar(name='Negative', x=aspects, y=negative_vals, marker_color='#ff6f61')
    ])
    fig2.update_layout(
        title='Aspect-based Sentiment Analysis',
        barmode='stack'
    )
    
    # Radar chart for aspect sentiment scores
    fig3 = go.Figure(data=go.Scatterpolar(
        r=[aspect_averages.get(aspect, 0) for aspect in aspects],
        theta=aspects,
        fill='toself',
        marker_color='#ff6f61'
    ))
    fig3.update_layout(
        title='Aspect Sentiment Scores',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )
        )
    )
    
    return {
        'sentiment_pie': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'aspect_bar': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'aspect_radar': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    }

def generate_pdf_report(app_title, results, counts, detailed_reviews, aspect_sentiments, algorithm_used):
    """Generate a professional PDF report"""
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.3*inch, bottomMargin=0.3*inch, leftMargin=0.4*inch, rightMargin=0.4*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles with enhanced typography
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#6366f1'),
            spaceAfter=4,
            alignment=1,  # Center
            fontName='Helvetica-Bold',
            letterSpacing=0.5
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=6,
            spaceBefore=6,
            fontName='Helvetica-Bold',
            borderBottomColor=colors.HexColor('#6366f1'),
            borderBottomWidth=1.5
        )
        
        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=8,
            fontName='Helvetica-Bold',
            textTransform='uppercase',
            letterSpacing=0.5
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=9,
            spaceAfter=4,
            fontName='Helvetica',
            leading=11,
            textColor=colors.HexColor('#334155')
        )
        
        metadata_style = ParagraphStyle(
            'Metadata',
            parent=styles['BodyText'],
            fontSize=9,
            spaceAfter=4,
            fontName='Helvetica',
            textColor=colors.HexColor('#64748b')
        )
        
        # Title
        story.append(Paragraph("📊 Sentiment Analysis Report", title_style))
        story.append(Spacer(1, 0.08*inch))
        
        # App Name - Large and prominent below heading
        app_name_style = ParagraphStyle(
            'AppName',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=4,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph(app_title, app_name_style))
        story.append(Spacer(1, 0.08*inch))
        
        # Metadata section
        analysis_date = datetime.now().strftime('%B %d, %Y • %H:%M:%S')
        metadata_text = f"<b>Analysis Date:</b> {analysis_date} | <b>Algorithm:</b> {algorithm_used.upper()}"
        story.append(Paragraph(metadata_text, metadata_style))
        story.append(Spacer(1, 0.12*inch))
        
        # Summary Section
        story.append(Paragraph("📈 Executive Summary", heading_style))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Reviews Analyzed', str(counts.get('total', 0))],
            ['😊 Positive Sentiment', f"{results.get('positive', 0)}% ({counts.get('positive', 0)} reviews)"],
            ['😐 Neutral Sentiment', f"{results.get('neutral', 0)}% ({counts.get('neutral', 0)} reviews)"],
            ['😞 Negative Sentiment', f"{results.get('negative', 0)}% ({counts.get('negative', 0)} reviews)"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9ff'), colors.HexColor('#ffffff')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#6366f1')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
            ('LEFTPADDING', (0, 1), (-1, -1), 6),
            ('RIGHTPADDING', (0, 1), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.15*inch))
        
        # Aspect Analysis Section
        story.append(Paragraph("🎯 Aspect-Based Sentiment Analysis", heading_style))
        aspect_data = [['Aspect', '😊 Positive', '😐 Neutral', '😞 Negative']]
        for aspect, sentiments in aspect_sentiments.items():
            aspect_data.append([
                aspect.capitalize(),
                str(sentiments.get('positive', 0)),
                str(sentiments.get('neutral', 0)),
                str(sentiments.get('negative', 0))
            ])
        
        aspect_table = Table(aspect_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        
        # Enhanced aspect table styling with beautiful colors
        aspect_style_commands = [
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('ALIGNMENT', (0, 0), (-1, 0), 'CENTER'),
            
            # Row coloring - alternating rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9ff'), colors.HexColor('#ffffff')]),
            
            # Cell styling
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
            ('LEFTPADDING', (0, 1), (-1, -1), 5),
            ('RIGHTPADDING', (0, 1), (-1, -1), 5),
            
            # Grid lines
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#6366f1')),
            
            # Column-specific styling for sentiment counts
            ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#dbeafe')),  # Positive - light blue
            ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#fef3c7')),  # Neutral - light yellow
            ('BACKGROUND', (3, 1), (3, -1), colors.HexColor('#fee2e2')),  # Negative - light red
            
            # Font weight for numbers
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica-Bold'),
            
            # Borders for columns
            ('RIGHTPADDING', (0, 1), (0, -1), 8),
        ]
        
        aspect_table.setStyle(TableStyle(aspect_style_commands))
        story.append(aspect_table)
        story.append(Spacer(1, 0.1*inch))
        
        # Professional footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['BodyText'],
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=1,  # Center
            borderTopColor=colors.HexColor('#e2e8f0'),
            borderTopWidth=1,
            spaceAfter=2
        )
        
        footer_text = f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Sentiment Analysis Report"
        story.append(Paragraph(footer_text, footer_style))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

def reset_analysis():
    """Reset all analysis data and form fields"""
    return render_template(
        'index.html',
        results=None,
        counts=None,
        detailed_reviews=None,
        error=None,
        app_title=None,
        charts=None,
        aspect_sentiments=None,
        aspect_averages=None,
        algorithm_used='vader'
    )


@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    counts = None
    detailed_reviews = None
    error = None
    app_title = None
    charts = None
    aspect_sentiments = None
    aspect_averages = None
    algorithm_used = 'vader'
    analysis_time = None

    if request.method == 'POST':
        app_url_or_id = request.form.get('app_url')
        analysis_type = request.form.get('analysis_type', 'quick')
        algorithm = request.form.get('algorithm', 'vader')
        algorithm_used = algorithm

        if app_url_or_id:
            try:
                app_id = None
                match = re.search(r'id=([^&]+)', app_url_or_id)
                if match:
                    app_id = match.group(1)
                elif '.' in app_url_or_id and ' ' not in app_url_or_id:
                    app_id = app_url_or_id

                if not app_id:
                    error = "Invalid input. Please enter a valid Google Play Store URL or an App ID."
                else:
                    analysis_start_time = time.time()
                    app_details = gp_app(app_id)
                    app_title = app_details['title']

                    fetched_reviews = get_playstore_reviews(app_id, analysis_type)
                    if fetched_reviews is not None:
                        results, counts, detailed_reviews, aspect_sentiments, aspect_averages = analyze_sentiment(fetched_reviews, algorithm)
                        charts = create_visualizations(results, aspect_sentiments, aspect_averages)
                        analysis_time = round(time.time() - analysis_start_time, 2)
                    else:
                        error = "Unable to fetch reviews. The app may not exist or has no reviews."
            except Exception as e:
                error = f"An error occurred: Please check the app URL or ID. Details: {e}"

    return render_template(
        'index.html',
        results=results,
        counts=counts,
        detailed_reviews=detailed_reviews,
        error=error,
        app_title=app_title,
        charts=charts,
        aspect_sentiments=aspect_sentiments,
        aspect_averages=aspect_averages,
        algorithm_used=algorithm_used,
        analysis_time=analysis_time,
        input_mode='playstore',
    )

@app.route('/download_report/<report_format>', methods=['POST'])
def download_report(report_format):
    """Download sentiment analysis report as PDF"""
    try:
        # Get data from request
        data = request.get_json()
        app_title = data.get('app_title', 'Unknown App')
        results = data.get('results', {})
        counts = data.get('counts', {})
        detailed_reviews = data.get('detailed_reviews', [])
        aspect_sentiments = data.get('aspect_sentiments', {})
        algorithm_used = data.get('algorithm_used', 'vader')
        
        if report_format.lower() == 'pdf':
            pdf_buffer = generate_pdf_report(app_title, results, counts, detailed_reviews, aspect_sentiments, algorithm_used)
            if pdf_buffer:
                filename = f"Sentiment_Report_{app_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                return send_file(
                    pdf_buffer,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=filename
                )
        
        return {'error': 'Unable to generate report'}, 400
    
    except Exception as e:
        print(f"Download error: {e}")
        return {'error': str(e)}, 500

# ── File upload helpers ────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt', 'json', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Noise patterns used by the content cleaner ─────────────────────────────
import string as _string

# Column / field names that look like headers, not reviews
_HEADER_WORDS = {
    'review', 'reviews', 'text', 'content', 'feedback', 'comment', 'comments',
    'body', 'message', 'description', 'opinion', 'rating', 'score', 'date',
    'title', 'name', 'user', 'username', 'author', 'id', 'index', 'no',
    'number', 'sr', 'sl', 'timestamp', 'time', 'created', 'updated',
    'product', 'item', 'category', 'type', 'status', 'source', 'platform',
    'response', 'reply', 'subject', 'email', 'phone', 'page', 'url', 'link',
}

# Patterns that indicate a line is NOT a real sentence / review
_NOISE_RE = re.compile(
    r'^[\d\s\-_=|/\\.,;:!?#@%^&*()\[\]{}<>]+$'   # purely numeric / symbol
    r'|^[A-Z][A-Z\s\-_/]{0,40}$'                  # ALL-CAPS short labels ("FEEDBACK FORM")
    r'|^https?://'                                  # URLs
    r'|^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z.]+$'  # email addresses
)


def _is_review_line(text: str) -> bool:
    """
    Return True if `text` looks like a real review / feedback sentence.
    Filters out:
      - Blank / whitespace-only strings
      - Pure numbers, symbols, dashes (table cells, IDs)
      - Very short tokens (< 4 chars) — likely column labels
      - Known header/field-name words
      - ALL-CAPS short labels ("TITLE", "FEEDBACK FORM")
      - URLs and e-mail addresses
      - Lines with fewer than 3 words (probably a label, not a sentence)
    """
    t = text.strip()
    if not t:
        return False

    # Too short to be meaningful
    if len(t) < 4:
        return False

    # Matches a noise pattern
    if _NOISE_RE.match(t):
        return False

    # Single bare word that is a known header
    words = t.split()
    if len(words) == 1 and t.lower().strip(_string.punctuation) in _HEADER_WORDS:
        return False

    # Fewer than 3 words — likely a label, not a sentence
    if len(words) < 3:
        return False

    # Must contain at least one letter (not purely numeric / symbolic)
    if not any(c.isalpha() for c in t):
        return False

    return True


def _clean_review_text(text: str) -> str:
    """Light normalisation: collapse whitespace, strip leading/trailing noise."""
    # Remove leading bullets / numbers (e.g. "1. ", "• ", "- ")
    text = re.sub(r'^[\s\-•*#\d.]+', '', text).strip()
    # Collapse internal whitespace runs
    text = re.sub(r'\s+', ' ', text)
    return text


def _extract_text_from_pdf(raw_bytes: bytes) -> list:
    """
    Extract every text block from a PDF (page by page) and return raw lines.
    Uses PyPDF2 (already installed as a reportlab dependency).
    """
    import PyPDF2
    lines = []
    try:
        reader = PyPDF2.PdfReader(_io.BytesIO(raw_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ''
            for line in page_text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")
    return lines


def parse_uploaded_file(file_storage):
    """
    Parse an uploaded file into a list of {'content': str, 'score': int} dicts.

    Supported formats
    -----------------
    CSV / Excel  – detects the review/text column automatically; skips header rows
                   and any cell that looks like a label or noise value.
    TXT          – one review per non-empty line; noisy / header lines are skipped.
    JSON         – list of strings OR list of objects with a text/review/content key.
    PDF          – text is extracted page-by-page; only sentence-like blocks are kept.

    Content cleaning applied to ALL formats
    ---------------------------------------
    • Strips leading bullets, numbering, and whitespace
    • Drops lines shorter than 3 words
    • Drops pure-numeric / symbol-only lines
    • Drops known header/column-name words
    • Drops ALL-CAPS short labels, URLs, and e-mail addresses
    """
    filename  = file_storage.filename.lower()
    ext       = filename.rsplit('.', 1)[-1] if '.' in filename else ''
    raw_bytes = file_storage.read()

    reviews_out = []

    def _add(text):
        cleaned = _clean_review_text(text)
        if _is_review_line(cleaned):
            reviews_out.append({'content': cleaned, 'score': 0})

    # ── PDF ────────────────────────────────────────────────────────────────────
    if ext == 'pdf':
        lines = _extract_text_from_pdf(raw_bytes)
        for line in lines:
            _add(line)
        return reviews_out

    # ── TXT ────────────────────────────────────────────────────────────────────
    if ext == 'txt':
        text = raw_bytes.decode('utf-8', errors='ignore')
        for line in text.splitlines():
            _add(line)
        return reviews_out

    # ── JSON ───────────────────────────────────────────────────────────────────
    if ext == 'json':
        try:
            data = json.loads(raw_bytes.decode('utf-8', errors='ignore'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

        candidates = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict):
                    for key in ('review', 'text', 'content', 'feedback',
                                'comment', 'body', 'message', 'description'):
                        if key in item and isinstance(item[key], str):
                            candidates.append(item[key])
                            break
        elif isinstance(data, dict):
            # Might be {reviews: [...]} or a flat key→value map
            for v in data.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            candidates.append(item)
                        elif isinstance(item, dict):
                            for key in ('review', 'text', 'content', 'feedback',
                                        'comment', 'body', 'message', 'description'):
                                if key in item and isinstance(item[key], str):
                                    candidates.append(item[key])
                                    break
                    break
                elif isinstance(v, str):
                    candidates.append(v)

        for c in candidates:
            _add(c)
        return reviews_out

    # ── CSV / Excel ────────────────────────────────────────────────────────────
    try:
        if ext in ('xlsx', 'xls'):
            df = pd.read_excel(_io.BytesIO(raw_bytes))
        else:
            for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
                try:
                    df = pd.read_csv(_io.BytesIO(raw_bytes), encoding=enc,
                                     on_bad_lines='skip')
                    break
                except Exception:
                    continue
            else:
                raise ValueError("Could not decode the CSV file. Try saving it as UTF-8.")
    except Exception as e:
        raise ValueError(f"Could not read the file: {e}")

    # ── Find the best text column ──────────────────────────────────────────────
    # Priority 1 – known review/text column names
    TEXT_COLS = ['review', 'reviews', 'text', 'content', 'feedback', 'comment',
                 'comments', 'body', 'message', 'description', 'opinion']
    col = None
    for candidate in TEXT_COLS:
        for c in df.columns:
            if c.strip().lower() == candidate:
                col = c
                break
        if col:
            break

    # Priority 2 – the string column with the longest average value
    # (real review text is long; label/category columns are short)
    if col is None:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if str_cols:
            col = max(
                str_cols,
                key=lambda c: df[c].dropna().apply(lambda v: len(str(v))).mean()
            )

    if col is None:
        raise ValueError(
            "Could not find a text column. Make sure your file has a column "
            "named 'review', 'text', 'content', 'feedback', or 'comment'."
        )

    for val in df[col].dropna():
        _add(str(val))

    return reviews_out


@app.route('/analyze_upload', methods=['POST'])
def analyze_upload():
    """Analyse reviews from an uploaded file (CSV, Excel, TXT, JSON, PDF)."""
    algorithm         = request.form.get('algorithm', 'vader')
    error             = None
    results           = None
    counts            = None
    detailed_reviews  = None
    charts            = None
    aspect_sentiments = None
    aspect_averages   = None
    app_title         = None
    analysis_time     = None
    algorithm_used    = algorithm
    input_mode        = 'upload'

    uploaded_file = request.files.get('feedback_file')

    if not uploaded_file or uploaded_file.filename == '':
        error = "No file selected. Please choose a CSV, Excel, TXT, JSON or PDF file."
    elif not allowed_file(uploaded_file.filename):
        error = "Unsupported file type. Please upload a .csv, .xlsx, .xls, .txt, .json or .pdf file."
    else:
        try:
            t0     = time.time()
            parsed = parse_uploaded_file(uploaded_file)

            if not parsed:
                error = (
                    "The file was read but no reviews were found. "
                    "Check that your file has a column named 'review', 'text', "
                    "'content', 'feedback', or 'comment'."
                )
            else:
                if len(parsed) > 2000:
                    parsed = parsed[:2000]

                app_title = f"Uploaded file · {uploaded_file.filename}"
                results, counts, detailed_reviews, aspect_sentiments, aspect_averages = \
                    analyze_sentiment(parsed, algorithm)
                charts        = create_visualizations(results, aspect_sentiments, aspect_averages)
                analysis_time = round(time.time() - t0, 2)

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"An unexpected error occurred: {e}"

    return render_template(
        'index.html',
        results=results,
        counts=counts,
        detailed_reviews=detailed_reviews,
        error=error,
        app_title=app_title,
        charts=charts,
        aspect_sentiments=aspect_sentiments,
        aspect_averages=aspect_averages,
        algorithm_used=algorithm_used,
        analysis_time=analysis_time,
        input_mode=input_mode,
    )


if __name__ == '__main__':
    app.run(debug=False)
