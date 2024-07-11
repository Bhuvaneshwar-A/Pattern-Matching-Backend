from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime, date

app = Flask(__name__)

def fetch_historical_data(ticker: str, start_date: date, end_date: date):
    """Fetches historical price and volume data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close', 'Volume']]

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def normalize_diff(data):
    """Normalize data by calculating percentage change"""
    return (data.pct_change().fillna(0) + 1).values[1:]  # Remove the first element (NaN)

def custom_cosine_similarity(a, b):
    """Calculate cosine similarity between two arrays."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def find_consecutive_patterns(sample_data, sample_ema, historical_data, historical_ema,
                              sample_volume=None, historical_volume=None, include_volume=False):
    """Finds similar consecutive segments in historical data using price, EMA, and optionally volume."""
    input_size = len(sample_data)
    similarities = []

    sample_price_norm = normalize_diff(pd.Series(sample_data))
    sample_ema_norm = normalize_diff(pd.Series(sample_ema))

    if include_volume and sample_volume is not None:
        sample_volume_norm = normalize_diff(pd.Series(sample_volume))

    for i in range(len(historical_data) - input_size):
        historical_segment = historical_data.iloc[i:i + input_size]
        historical_ema_segment = historical_ema.iloc[i:i + input_size]

        price_similarity = custom_cosine_similarity(sample_price_norm, normalize_diff(historical_segment))
        ema_similarity = custom_cosine_similarity(sample_ema_norm, normalize_diff(historical_ema_segment))

        if include_volume and historical_volume is not None:
            historical_volume_segment = historical_volume.iloc[i:i + input_size]
            volume_similarity = custom_cosine_similarity(sample_volume_norm, normalize_diff(historical_volume_segment))
            average_similarity = np.mean([price_similarity, ema_similarity, volume_similarity])
        else:
            average_similarity = np.mean([price_similarity, ema_similarity])

        similarities.append((i, i + input_size - 1, average_similarity))

    similarities.sort(key=lambda x: -x[2])
    return similarities[:5]  # Return top 5 matches

@app.route('/search_data', methods=['POST'])
def search_data():
    try:
        data = request.json
        ticker = data['ticker']
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()
        sample_data = data['sample_data']
        ema_period = data.get('ema_period', 14)
        include_volume = data.get('include_volume', False)

        # Fetch historical data from yfinance
        historical_data = fetch_historical_data(ticker, start_date, end_date)

        if historical_data.empty:
            return jsonify({"error": "No data found for this ticker!"}), 404

        # Calculate EMA for historical data
        historical_ema = calculate_ema(historical_data['Close'], ema_period)

        # Prepare sample data
        sample_data = pd.Series(sample_data)
        sample_ema = calculate_ema(sample_data, ema_period)

        # Prepare volume data if included
        sample_volume = None
        historical_volume = None
        if include_volume:
            # For this example, we'll use random data for sample volume
            # In a real scenario, you'd want to provide this from the frontend
            sample_volume = pd.Series(np.random.randn(len(sample_data)))
            historical_volume = historical_data['Volume']

        top_matches = find_consecutive_patterns(sample_data, sample_ema,
                                                historical_data['Close'], historical_ema,
                                                sample_volume, historical_volume,
                                                include_volume)

        results = []
        for start, end, similarity in top_matches:
            match_dates = historical_data.index[start:end + 1]
            results.append({
                "start_date": match_dates[0].isoformat(),
                "end_date": match_dates[-1].isoformat(),
                "similarity": float(similarity)
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
