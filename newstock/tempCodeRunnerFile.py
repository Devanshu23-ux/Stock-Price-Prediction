from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import os
import subprocess
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

# Import your stock predictor
from app import NSEStockPredictor

app = Flask(__name__)

def get_next_trading_day():
    # Get NSE calendar
    nse = mcal.get_calendar('NSE')
    
    # Get today's date
    today = datetime.now().date()
    
    # Get the next 5 trading days (for buffer)
    try:
        schedule = nse.schedule(start_date=today, end_date=today + timedelta(days=10))
        
        # Find the first valid trading day after today
        for idx, row in schedule.iterrows():
            trading_date = idx.date()
            if trading_date > today:
                return trading_date.isoformat()
    except Exception as e:
        print(f"Error getting market calendar: {e}")
    
    # Fallback logic for weekends if calendar fails
    next_day = today + timedelta(days=1)
    while next_day.weekday() > 4:  # Saturday = 5, Sunday = 6
        next_day += timedelta(days=1)
    return next_day.isoformat()

@app.route('/')
def index():
    return render_template('search.html')

# Route for rendering the prediction UI with company and symbol
@app.route('/predict', methods=['GET'])
def show_prediction_page():
    company = request.args.get('company')
    symbol = request.args.get('symbol')
    return render_template('index.html', company=company, symbol=symbol)

# Enhanced POST route for model prediction logic
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    tickers = data.get('tickers', ['RELIANCE', 'TCS'])

    # Check if models exist for all requested tickers
    missing_tickers = []
    for ticker in tickers:
        model_path = os.path.join('models', f"{ticker}_model.pth")
        scaler_path = os.path.join('models', f"{ticker}_scalers.pkl")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            missing_tickers.append(ticker)
    
    # If any tickers are missing models, run the standalone script to create them
    if missing_tickers:
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Run the script for missing tickers
            cmd = ['python', 'app.py', '--tickers'] + missing_tickers
            subprocess.run(cmd, check=True)
            print(f"Created models for {missing_tickers}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating models: {e}")
            return jsonify({
                'error': f"Failed to create prediction models for {missing_tickers}",
                'details': str(e)
            }), 500

    # Now proceed with predictions
    predictor = NSEStockPredictor(
        tickers=tickers,
        window_size=60,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )

    # Fetch data
    predictor.fetch_data(period='2y')

    # Load models (they should exist now)
    predictor.load_saved_models()

    # Make predictions
    predictions = predictor.predict_tomorrow()

    # Convert NumPy data to Python native types
    def convert_to_python(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(i) for i in obj]
        return obj

    predictions = convert_to_python(predictions)
    
    # Get the next trading day
    next_trading_date = get_next_trading_day()
    
    # Return both predictions and next trading date
    return jsonify({
        'predictions': predictions,
        'next_trading_date': next_trading_date
    })

if __name__ == '__main__':
    app.run(debug=True)
    