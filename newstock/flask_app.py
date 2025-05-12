from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import os
import subprocess
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import sys

# Add the parent directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import your stock predictor
# Adjust this import based on where your NSEStockPredictor class is located
try:
    from app import NSEStockPredictor
except ImportError:
    # Try alternative import paths
    try:
        sys.path.append(os.path.join(current_dir, 'newstock'))
        from app import NSEStockPredictor
    except ImportError:
        print("Error: Could not import NSEStockPredictor. Check the path to app.py")
        raise

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
    return render_template('index.html')

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
            
            # Find the correct path to app.py
            app_path = find_app_py()
            if not app_path:
                raise FileNotFoundError("Could not find app.py in any of the expected locations")
            
            # Run the script for missing tickers with the correct path
            cmd = ['python', app_path, '--tickers'] + missing_tickers
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Created models for {missing_tickers}")
        except Exception as e:
            print(f"Error creating models: {e}")
            return jsonify({
                'error': f"Failed to create prediction models for {missing_tickers}",
                'details': str(e)
            }), 500

    # Now proceed with predictions
    try:
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
    except Exception as e:
        print(f"Error making predictions: {e}")
        return jsonify({
            'error': f"Failed to make predictions",
            'details': str(e)
        }), 500

def find_app_py():
    """Try to find app.py in various possible locations"""
    # List of potential locations relative to the current file
    possible_locations = [
        os.path.join(os.path.dirname(__file__), 'app.py'),                 # Same directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py'), # Parent directory
        os.path.join(os.path.dirname(__file__), 'newstock', 'app.py'),      # newstock subdirectory
        os.path.join(os.path.dirname(__file__), '..', 'app.py'),           # Parent using relative path
        os.path.abspath('app.py'),                                         # Current working directory
        os.path.join(os.getcwd(), 'app.py')                                # Explicit current working directory
    ]
    
    # Check each location
    for location in possible_locations:
        if os.path.isfile(location):
            print(f"Found app.py at: {location}")
            return location
            
    # If not found in expected locations, look recursively up to 2 directories deep
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for root, dirs, files in os.walk(base_dir):
        if 'app.py' in files:
            location = os.path.join(root, 'app.py')
            print(f"Found app.py at: {location}")
            return location
            
    # If still not found, return None
    print("Could not find app.py in any expected location")
    return None

if __name__ == '__main__':
    app.run(debug=True)