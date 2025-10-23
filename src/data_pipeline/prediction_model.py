import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

from src.data_pipeline.constants import *

# Directory for data and performance files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

# Configure PyTorch for optimal GPU performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --------------------------------------------------
# 1. Data Loading and Preprocessing
# --------------------------------------------------

def get_ticker_dir(ticker):
    ticker_dir = TICKER_PATH(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)

    return ticker_dir

def load_historical_prices(ticker, date_str=None):
    ticker_dir = get_ticker_dir(ticker)
    if date_str:
        filename = f"historical_prices_{date_str}.json"
        filepath = ticker_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No historical price data for {ticker} on {date_str}")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # Find latest file
    prefix = "historical_prices_"
    candidates = []
    for fname in os.listdir(ticker_dir):
        if fname.startswith(prefix) and fname.endswith('.json'):
            try:
                date_part = fname[len(prefix):-5]
                datetime.strptime(date_part, '%Y-%m-%d')
                candidates.append((date_part, fname))
            except Exception:
                continue
    
    if not candidates:
        raise FileNotFoundError(f"No historical price data for {ticker}")
    
    latest_date, latest_fname = max(candidates, key=lambda x: x[0])
    filepath = os.path.join(ticker_dir, latest_fname)
    with open(filepath, 'r') as f:
        return json.load(f)

def detect_outliers(data, contamination=0.1):
    """Detect outliers using Isolation Forest"""
    if len(data) < 10:
        return np.zeros(len(data), dtype=bool)
    
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = isolation_forest.fit_predict(data.reshape(-1, 1))
    return outliers == -1

def preprocess_data(prices, sequence_length=60, prediction_horizon=1, remove_outliers=True):
    """Advanced data preprocessing with outlier handling"""
    # Extract relevant features
    features = []
    for price in prices:
        features.append([
            price['Open'], price['High'], price['Low'], price['Close'], 
            price.get('Volume', 0)
        ])
    
    data = np.array(features, dtype=np.float32)
    
    # Handle missing values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Outlier detection and handling
    if remove_outliers and len(data) > 10:
        close_prices = data[:, 3]  # Close prices
        outlier_mask = detect_outliers(close_prices)
        
        # Replace outliers with median of surrounding values
        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier:
                start_idx = max(0, i-5)
                end_idx = min(len(data), i+6)
                median_val = np.median(data[start_idx:end_idx, 3])
                data[i, 3] = median_val
    
    # Robust scaling
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i+prediction_horizon-1, 3])  # Predict close price
    
    return np.array(X), np.array(y), scaler

# --------------------------------------------------
# 2. Base Model Classes
# --------------------------------------------------

class BasePredictionModel(ABC):
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
        self.scaler = None
        self.trained = False
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def save_model(self, path):
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
                'scaler': self.scaler,
                'trained': self.trained
            }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get('model_state_dict') and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint.get('scaler')
        self.trained = checkpoint.get('trained', False)

# --------------------------------------------------
# 3. WEAK MODE: Enhanced Traditional Models
# --------------------------------------------------

class WeakPredictionModel(BasePredictionModel):
    """Enhanced traditional ML models with GPU acceleration where possible"""
    
    def __init__(self):
        super().__init__()
        self.models = {
            'poly2': None,
            'poly3': None, 
            'linear': None,
            'robust_linear': None
        }
        self.performance = {}
    
    def train(self, prices, prediction_horizon=1):
        # Extract close prices
        close_prices = np.array([p['Close'] for p in prices])
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices
        
        # Polynomial models
        for degree in [2, 3]:
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            self.models[f'poly{degree}'] = (model, poly)
        
        # Linear model
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        self.models['linear'] = linear_model
        
        # Robust linear model (less sensitive to outliers)
        robust_model = HuberRegressor(epsilon=1.35)
        robust_model.fit(X, y)
        self.models['robust_linear'] = robust_model
        
        self.trained = True
        return self
    
    def predict(self, prices, days_ahead=1):
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        n_days = len(prices)
        X_pred = np.array([[n_days + days_ahead - 1]])
        
        predictions = {}
        
        # Polynomial predictions
        for degree in [2, 3]:
            if self.models[f'poly{degree}']:
                model, poly = self.models[f'poly{degree}']
                X_pred_poly = poly.transform(X_pred)
                pred = model.predict(X_pred_poly)[0]
                predictions[f'poly{degree}'] = pred
        
        # Linear predictions
        if self.models['linear']:
            pred = self.models['linear'].predict(X_pred)[0]
            predictions['linear'] = pred
        
        if self.models['robust_linear']:
            pred = self.models['robust_linear'].predict(X_pred)[0]
            predictions['robust_linear'] = pred
        
        # Return ensemble average with robustness weighting
        weights = {'poly2': 0.2, 'poly3': 0.2, 'linear': 0.3, 'robust_linear': 0.3}
        ensemble_pred = sum(predictions[k] * weights[k] for k in predictions if k in weights)
        
        return ensemble_pred, predictions

# --------------------------------------------------
# 4. MEDIUM MODE: LSTM with Ensemble
# --------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last time step
        out = self.fc(out)
        return out

class MediumPredictionModel(BasePredictionModel):
    """LSTM-based ensemble with outlier robustness"""
    
    def __init__(self, ensemble_size=3):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = []
        self.optimizers = []
        self.criterion = nn.MSELoss()
        self.scaler = None
        
        # Create ensemble of LSTM models with different architectures
        for i in range(ensemble_size):
            hidden_sizes = [64, 128, 256]
            num_layers = [1, 2, 3]
            model = LSTMModel(
                hidden_size=hidden_sizes[i % len(hidden_sizes)],
                num_layers=num_layers[i % len(num_layers)]
            ).to(self.device)
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            
            self.models.append(model)
            self.optimizers.append(optimizer)
    
    def train(self, prices, prediction_horizon=1, epochs=100, batch_size=32):
        # Preprocess data with outlier handling
        X, y, self.scaler = preprocess_data(prices, remove_outliers=True)
        
        if len(X) < 10:
            raise ValueError("Insufficient data for training")
        
        # Keep tensors on CPU for DataLoader
        X_tensor = torch.FloatTensor(X)  # Remove .to(self.device)
        y_tensor = torch.FloatTensor(y)  # Remove .to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        # Train ensemble
        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    # Move to GPU here
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        outputs = model(batch_X).squeeze()
                        loss = self.criterion(outputs, batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"Model {model_idx+1}, Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.trained = True
        return self

    
    def predict(self, prices, days_ahead=1):
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        # Preprocess recent data
        X, _, _ = preprocess_data(prices, remove_outliers=True)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Use last sequence for prediction
        X_pred = torch.FloatTensor(X[-1:]).to(self.device)
        
        # Ensemble predictions
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_pred).cpu().numpy()[0, 0]
                predictions.append(pred)
        
        # Ensemble average with outlier removal
        predictions = np.array(predictions)
        
        # Remove outlier predictions
        q75, q25 = np.percentile(predictions, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        filtered_preds = predictions[(predictions >= lower_bound) & (predictions <= upper_bound)]
        ensemble_pred = np.mean(filtered_preds) if len(filtered_preds) > 0 else np.mean(predictions)
        
        # Inverse transform
        if self.scaler:
            # Create dummy array for inverse transform
            dummy = np.zeros((1, 5))
            dummy[0, 3] = ensemble_pred  # Close price index
            original_scale = self.scaler.inverse_transform(dummy)[0, 3]
            return original_scale, {'ensemble_mean': ensemble_pred, 'individual': predictions.tolist()}
        
        return ensemble_pred, {'ensemble_mean': ensemble_pred, 'individual': predictions.tolist()}

# --------------------------------------------------
# 5. STRONG MODE: Transformer-based Model
# --------------------------------------------------

class TransformerModel(nn.Module):
    def __init__(self, input_size=5, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.output_projection(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class StrongPredictionModel(BasePredictionModel):
    """Advanced Transformer-based model with mixed precision training"""
    
    def __init__(self, ensemble_size=5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers
        self.scaler_amp = GradScaler()  # For mixed precision
        
        # Create diverse ensemble
        for i in range(ensemble_size):
            d_models = [128, 256, 512]
            nheads = [4, 8, 16]
            num_layers = [3, 4, 6]
            
            model = TransformerModel(
                d_model=d_models[i % len(d_models)],
                nhead=nheads[i % len(nheads)],
                num_layers=num_layers[i % len(num_layers)]
            ).to(self.device)
            
            optimizer = optim.AdamW(
                model.parameters(), lr=0.0001, weight_decay=1e-5, 
                betas=(0.9, 0.999), eps=1e-8
            )
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
            
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
    
    def train(self, prices, prediction_horizon=1, epochs=200, batch_size=16):
        # Advanced preprocessing
        X, y, self.scaler = preprocess_data(
            prices, sequence_length=120, prediction_horizon=prediction_horizon, 
            remove_outliers=True
        )
        
        if len(X) < 20:
            raise ValueError("Insufficient data for training")
        
        # Keep tensors on CPU for DataLoader, move to GPU during training
        X_tensor = torch.FloatTensor(X)  # Remove .to(self.device)
        y_tensor = torch.FloatTensor(y)  # Remove .to(self.device)
        
        # Create data loader with pin_memory for efficient GPU transfer
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        # Train ensemble with mixed precision
        for model_idx, (model, optimizer, scheduler) in enumerate(zip(self.models, self.optimizers, self.schedulers)):
            model.train()
            best_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            for epoch in range(epochs):
                total_loss = 0
                
                for batch_X, batch_y in dataloader:
                    # Move tensors to GPU here, after DataLoader
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    # Mixed precision training
                    with autocast():
                        outputs = model(batch_X).squeeze()
                        loss = self.criterion(outputs, batch_y)
                    
                    # Backward pass with gradient scaling
                    self.scaler_amp.scale(loss).backward()
                    self.scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                    
                    total_loss += loss.item()
                
                scheduler.step()
                avg_loss = total_loss / len(dataloader)
                
                # Early stopping logic...
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 25 == 0:
                    print(f"Model {model_idx+1}, Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                
                if patience_counter >= patience:
                    print(f"Early stopping for model {model_idx+1} at epoch {epoch}")
                    break
        
        self.trained = True
        return self
    
    def predict(self, prices, days_ahead=1):
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        # Preprocess data
        X, _, _ = preprocess_data(prices, sequence_length=120, remove_outliers=True)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")
        
        X_pred = torch.FloatTensor(X[-1:]).to(self.device)
        
        # Ensemble predictions with uncertainty estimation
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                with autocast():
                    pred = model(X_pred).cpu().numpy()[0, 0]
                    predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Robust ensemble with uncertainty
        ensemble_mean = np.mean(predictions)
        ensemble_std = np.std(predictions)
        ensemble_median = np.median(predictions)
        
        # Weighted average favoring median for robustness
        final_pred = 0.7 * ensemble_median + 0.3 * ensemble_mean
        
        # Inverse transform
        if self.scaler:
            dummy = np.zeros((1, 5))
            dummy[0, 3] = final_pred
            original_scale = self.scaler.inverse_transform(dummy)[0, 3]
            
            return original_scale, {
                'ensemble_mean': ensemble_mean,
                'ensemble_median': ensemble_median,
                'ensemble_std': ensemble_std,
                'final_prediction': final_pred,
                'individual': predictions.tolist(),
                'confidence_interval': (
                    original_scale - 1.96 * ensemble_std,
                    original_scale + 1.96 * ensemble_std
                )
            }
        
        return final_pred, {
            'ensemble_mean': ensemble_mean,
            'ensemble_median': ensemble_median,
            'ensemble_std': ensemble_std,
            'individual': predictions.tolist()
        }

# --------------------------------------------------
# 6. Model Factory and Main Interface
# --------------------------------------------------

class PredictionModelFactory:
    @staticmethod
    def create_model(mode: str):
        """Create prediction model based on mode"""
        if mode.lower() == 'weak':
            return WeakPredictionModel()
        elif mode.lower() == 'medium':
            return MediumPredictionModel()
        elif mode.lower() == 'strong':
            return StrongPredictionModel()
        else:
            raise ValueError(f"Unknown prediction mode: {mode}. Use 'weak', 'medium', or 'strong'")

class PredictionManager:
    @staticmethod
    def run_prediction_for_ticker(ticker, days_ahead=1, mode='medium', date_str=None):
        """
        Main function to run prediction for a ticker
        
        Args:
            ticker: Stock ticker symbol
            days_ahead: Number of days to predict ahead
            mode: Prediction mode ('weak', 'medium', 'strong')
            date_str: Optional date string for specific data
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load historical prices
            prices = load_historical_prices(ticker, date_str)
            if not prices:
                raise ValueError(f"No price data for {ticker}")
            
            current_price = prices[-1]['Close']
            
            # Create and train model
            print(f"Training {mode.upper()} model for {ticker}...")
            model = PredictionModelFactory.create_model(mode)
            
            # Clear GPU memory before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model.train(prices, prediction_horizon=days_ahead)
            
            # Make prediction
            predicted_price, model_details = model.predict(prices, days_ahead)
            
            # Calculate prediction score
            prediction_score = (predicted_price - current_price) / current_price
            
            # Prepare results
            prediction_data = {
                'ticker': ticker,
                'prediction_mode': mode,
                'expected_price': float(predicted_price),
                'current_price': float(current_price),
                'prediction_score': float(prediction_score),
                'days_ahead': days_ahead,
                'prediction_date': (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d'),
                'model_details': model_details,
            }
            
            # Save prediction
            PredictionManager.save_prediction_to_json(ticker, prediction_data, mode)
            
            # Clear GPU memory after prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return prediction_data
            
        except Exception as e:
            print(f"Error in prediction for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'prediction_mode': mode,
                'days_ahead': days_ahead
            }

    @staticmethod
    def save_prediction_to_json(ticker, prediction_data, mode):
        """Save prediction results to JSON file"""
        ticker_dir = get_ticker_dir(ticker)
        
        # Create predictions directory
        predictions_dir = os.path.join(ticker_dir, 'predictions')
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        
        # Generate filename
        current_date = datetime.now().strftime('%Y-%m-%d')
        filename = f"predictions_{mode}_{current_date}.json"
        filepath = os.path.join(predictions_dir, filename)
        
        # Prepare save data
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_data,
        }
        
        # Load existing data
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(save_data)
            else:
                existing_data = [existing_data, save_data]
        else:
            existing_data = [save_data]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)
        
        return filepath

    # GPU memory management utilities
    @staticmethod
    def get_gpu_memory_info():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
        return None

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory and cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
