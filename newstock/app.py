import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import argparse
import datetime
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StockDataset(Dataset):
    """PyTorch Dataset for stock data"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model for stock price prediction"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class NSEStockPredictor:
    def __init__(self, tickers, window_size=60, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Initialize the NSE Stock Predictor
        
        Args:
            tickers (list): List of NSE stock symbols
            window_size (int): Number of previous days to use for prediction
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = {}
        self.scalers = {}
        self.data = {}
        
    def fetch_data(self, period='2y'):
        """
        Fetch data for all tickers from yfinance
        
        Args:
            period (str): Period to fetch data for ('1y', '2y', 'max', etc.)
        """
        for ticker in self.tickers:
            # For NSE stocks in yfinance, append ".NS" to the ticker
            yf_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
            
            print(f"Fetching data for {ticker} using yfinance...")
            try:
                # Get stock data from yfinance
                stock = yf.Ticker(yf_ticker)
                stock_data = stock.history(period=period)
                
                if not stock_data.empty:
                    # Add additional technical indicators as features
                    stock_data['Pct_Change'] = stock_data['Close'].pct_change()
                    stock_data['Return'] = stock_data['Close'] / stock_data['Close'].shift(1) - 1
                    stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
                    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
                    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
                    stock_data['Std_20'] = stock_data['Close'].rolling(window=20).std()
                    
                    # Feature for price range volatility
                    stock_data['Volatility'] = (stock_data['High'] - stock_data['Low']) / stock_data['Open']
                    
                    # MACD (Moving Average Convergence Divergence)
                    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
                    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
                    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
                    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
                    
                    # RSI (Relative Strength Index)
                    delta = stock_data['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    stock_data['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Fill NaN values
                    stock_data = stock_data.fillna(method='bfill')
                    stock_data = stock_data.fillna(method='ffill')  # In case bfill doesn't handle all NaNs
                    
                    self.data[ticker] = stock_data
                    print(f"Downloaded {len(stock_data)} days of trading data for {ticker}")
                else:
                    print(f"No data found for {ticker}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
    
    def prepare_data(self, ticker, feature_columns=None, target_column='High'):
        """
        Prepare data for a specific ticker
        
        Args:
            ticker (str): Stock ticker symbol
            feature_columns (list): List of columns to use as features
            target_column (str): Column to predict
            
        Returns:
            tuple: Training and testing data
        """
        data = self.data[ticker]
        
        # Default feature columns if none specified
        if feature_columns is None:
            feature_columns = ['Close', 'Volume', 'Pct_Change', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD']
        
        # Check if all requested columns exist
        existing_columns = [col for col in feature_columns if col in data.columns]
        if len(existing_columns) < len(feature_columns):
            missing = set(feature_columns) - set(existing_columns)
            print(f"Warning: Some requested feature columns do not exist: {missing}")
            feature_columns = existing_columns
        
        # Extract features and target
        feature_data = data[feature_columns].values
        target_data = data[target_column].values.reshape(-1, 1)
        
        # Normalize the data
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        feature_data_scaled = scaler_X.fit_transform(feature_data)
        target_data_scaled = scaler_y.fit_transform(target_data)
        
        # Create windowed dataset
        X, y = [], []
        for i in range(self.window_size, len(feature_data_scaled)):
            # For each window, include all features
            X.append(feature_data_scaled[i-self.window_size:i])
            y.append(target_data_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split data into training and testing sets (80-20 split)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Save scalers
        self.scalers[ticker] = {
            'feature': scaler_X,
            'target': scaler_y,
            'feature_columns': feature_columns
        }
        
        return X_train, y_train, X_test, y_test
    
    def train_models(self, feature_columns=None, target_column='High', save_dir='models'):
        """
        Train PyTorch LSTM models for all tickers
        
        Args:
            feature_columns (list): List of columns to use as features
            target_column (str): Column to predict
            save_dir (str): Directory to save models
        """
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for ticker in self.tickers:
            if ticker not in self.data or self.data[ticker].empty:
                print(f"No data available for {ticker}, skipping...")
                continue
                
            print(f"\n===== Training model for {ticker} =====")
            
            # Prepare data
            X_train, y_train, X_test, y_test = self.prepare_data(
                ticker, feature_columns, target_column
            )
            
            # Get number of features
            num_features = X_train.shape[2]
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Testing data shape: {X_test.shape}")
            print(f"Number of features: {num_features}")
            
            # Create PyTorch datasets and dataloaders
            train_dataset = StockDataset(X_train, y_train)
            test_dataset = StockDataset(X_test, y_test)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
            
            # Initialize model
            model = LSTMModel(
                input_size=num_features, 
                hidden_size=50, 
                num_layers=2, 
                dropout=0.2
            ).to(device)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience = 10  # Early stopping patience
            counter = 0
            best_model = None
            
            print(f"Training model for {ticker}...")
            for epoch in range(self.epochs):
                # Training
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = model(X_batch)
                        loss = criterion(outputs.squeeze(), y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(test_loader)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # Save best model
                    best_model = model.state_dict().copy()
                else:
                    counter += 1
                
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch+1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Load best model
            if best_model is not None:
                model.load_state_dict(best_model)
            
            # Evaluate model
            model.eval()
            with torch.no_grad():
                # Convert test data to tensors
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
                
                # Make predictions
                test_predictions = model(X_test_tensor).cpu().numpy()
                
                # Inverse transform the predictions and actual values
                test_predictions_reshaped = test_predictions.reshape(-1, 1)
                test_predictions_actual = self.scalers[ticker]['target'].inverse_transform(test_predictions_reshaped)
                
                y_test_reshaped = y_test.reshape(-1, 1)
                y_test_actual = self.scalers[ticker]['target'].inverse_transform(y_test_reshaped)
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean(np.square(y_test_actual - test_predictions_actual)))
                print(f"RMSE for {ticker}: ₹{rmse:.2f}")
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test_actual - test_predictions_actual) / y_test_actual)) * 100
                print(f"MAPE for {ticker}: {mape:.2f}%")
            
            # Store model
            self.models[ticker] = model
            
            # Save model and scalers
            model_path = os.path.join(save_dir, f"{ticker}_model.pth")
            scaler_path = os.path.join(save_dir, f"{ticker}_scalers.pkl")
            
            torch.save(model.state_dict(), model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[ticker], f)
            
            print(f"Saved model and scalers for {ticker}")
            
            # Plot training history
            plt.figure(figsize=(12, 6))
            plt.plot(train_losses)
            plt.plot(val_losses)
            plt.title(f'Model Loss During Training ({ticker})')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(os.path.join(save_dir, f"{ticker}_training_history.png"))
            plt.close()
            
            # Plot predictions
            plt.figure(figsize=(16, 8))
            plt.plot(y_test_actual, label='Actual')
            plt.plot(test_predictions_actual, label='Predicted')
            plt.title(f'High Price: Actual vs Predicted ({ticker})')
            plt.xlabel('Time')
            plt.ylabel('Price (₹)')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{ticker}_predictions.png"))
            plt.close()
    
    def load_saved_models(self, save_dir='models'):
        """Load saved models and scalers"""
        for ticker in self.tickers:
            model_path = os.path.join(save_dir, f"{ticker}_model.pth")
            scaler_path = os.path.join(save_dir, f"{ticker}_scalers.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                # Load scalers first to get feature dimensions
                with open(scaler_path, 'rb') as f:
                    self.scalers[ticker] = pickle.load(f)
                
                # Get number of features
                num_features = len(self.scalers[ticker]['feature_columns'])
                
                # Initialize model
                model = LSTMModel(
                    input_size=num_features, 
                    hidden_size=50, 
                    num_layers=2, 
                    dropout=0.2
                ).to(device)
                
                # Load model weights
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()  # Set to evaluation mode
                
                # Store model
                self.models[ticker] = model
                
                print(f"Loaded saved model and scalers for {ticker}")
            else:
                print(f"No saved model found for {ticker}")
    
    def predict_tomorrow(self, feature_columns=None):
        """
        Predict tomorrow's high price for all tickers
        
        Args:
            feature_columns (list): List of columns to use as features
            
        Returns:
            dict: Predicted high prices for each ticker
        """
        predictions = {}
        
        for ticker in self.tickers:
            if ticker not in self.models or ticker not in self.data:
                print(f"No model or data available for {ticker}")
                continue
            
            data = self.data[ticker]
            
            # Get feature columns used during training
            if feature_columns is None and ticker in self.scalers:
                feature_columns = self.scalers[ticker]['feature_columns']
            elif feature_columns is None:
                feature_columns = ['Close', 'Volume', 'Pct_Change', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD']
            
            # Check if all columns exist
            existing_columns = [col for col in feature_columns if col in data.columns]
            if len(existing_columns) < len(feature_columns):
                missing = set(feature_columns) - set(existing_columns)
                print(f"Warning: Some feature columns missing for prediction: {missing}")
                feature_columns = existing_columns
            
            # Get the last window_size days
            feature_data = data[feature_columns].values[-self.window_size:]
            
            # Reshape for normalization
            feature_data_reshaped = feature_data.reshape(-1, len(feature_columns))
            feature_data_scaled = self.scalers[ticker]['feature'].transform(feature_data_reshaped)
            
            # Reshape back to match LSTM input requirements [batch_size, seq_length, num_features]
            feature_data_scaled = feature_data_scaled.reshape(1, self.window_size, len(feature_columns))
            
            # Convert to PyTorch tensor
            X_pred = torch.tensor(feature_data_scaled, dtype=torch.float32).to(device)
            
            # Get model
            model = self.models[ticker]
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_pred).cpu().numpy()
            
            # Inverse transform to get actual price
            pred = self.scalers[ticker]['target'].inverse_transform(pred_scaled.reshape(-1, 1))
            predictions[ticker] = pred[0][0]
            
            # Get last few actual high prices for comparison
            last_week_high = data['High'].iloc[-5:].values
            print(f"\n===== Prediction for {ticker} =====")
            print(f"Predicted High Price for Tomorrow: ₹{predictions[ticker]:.2f}")
            print("Last 5 trading days' high prices:")
            for i, price in enumerate(last_week_high):
                print(f"  {data.index[-5+i].date()}: ₹{price:.2f}")
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description="NSE Stock price prediction using PyTorch LSTM with yfinance data")
    parser.add_argument("--tickers", nargs="+", default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
                        help="List of NSE stock tickers to analyze (without .NS extension)")
    parser.add_argument("--window", type=int, default=60,
                        help="Window size (number of previous days to use)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--period", type=str, default="2y",
                        help="Period to fetch data for ('1y', '2y', 'max', etc.)")
    parser.add_argument("--load", action="store_true",
                        help="Load existing models instead of training new ones")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NSEStockPredictor(
        tickers=args.tickers,
        window_size=args.window,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr
    )
    
    # Fetch data
    predictor.fetch_data(period=args.period)
    
    if args.load:
        # Load saved models
        predictor.load_saved_models()
    else:
        # Train models
        predictor.train_models()
    
    # Predict tomorrow's high prices
    predictions = predictor.predict_tomorrow()
    
    # Print summary of predictions
    print("\n===== Tomorrow's Predicted High Prices =====")
    for ticker, price in predictions.items():
        print(f"{ticker}: ₹{price:.2f}")

if __name__ == "__main__":
    main()