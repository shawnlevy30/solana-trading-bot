"""
Machine Learning Model for Solana Memecoin Trading Bot

This module implements machine learning models for predicting price movements
and identifying profitable trading opportunities for Solana memecoins.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

# ML libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Import from other modules
from .collector import DataCollector
from .data_pipeline import DataPipeline
from .security_audit import SecurityAuditSystem
from .api_config import TIMEFRAMES, MARKET_CAP_TIERS, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_model")

class MLModelManager:
    """
    Manages machine learning models for predicting price movements
    and identifying profitable trading opportunities.
    """
    
    def __init__(self, data_collector: DataCollector, data_pipeline: DataPipeline, security_audit: SecurityAuditSystem):
        self.data_collector = data_collector
        self.data_pipeline = data_pipeline
        self.security_audit = security_audit
        self.models_dir = os.path.join(os.getcwd(), "ml_models")
        self.data_dir = os.path.join(os.getcwd(), "ml_data")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize model containers
        self.price_direction_model = None
        self.price_magnitude_model = None
        self.entry_point_model = None
        self.exit_point_model = None
        self.scalers = {}
        
        logger.info("MLModelManager initialized")
    
    def prepare_training_data(self, token_addresses: List[str], chain_id: str = "solana") -> Dict:
        """
        Prepare training data from multiple tokens
        Returns processed datasets ready for model training
        """
        logger.info(f"Preparing training data for {len(token_addresses)} tokens")
        
        all_features = []
        all_price_directions = []
        all_price_changes = []
        all_entry_points = []
        all_exit_points = []
        
        # Process each token
        for token_address in token_addresses:
            try:
                # Get processed data
                processed_data = self.data_pipeline.process_token_data(token_address, chain_id)
                
                # Extract historical data
                historical_data = processed_data.get("historical_data", {}).get("data", [])
                
                if not historical_data or len(historical_data) < 20:
                    logger.warning(f"Insufficient historical data for {token_address}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(historical_data)
                
                # Convert string columns to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate features
                features_df = self._calculate_features(df)
                
                # Calculate target variables
                targets = self._calculate_targets(df)
                
                # Add to collections
                all_features.append(features_df)
                all_price_directions.extend(targets['price_direction'])
                all_price_changes.extend(targets['price_change'])
                all_entry_points.extend(targets['entry_points'])
                all_exit_points.extend(targets['exit_points'])
                
                logger.info(f"Processed training data for {token_address}")
            except Exception as e:
                logger.warning(f"Error processing training data for {token_address}: {e}")
        
        # Combine all data
        if not all_features:
            logger.error("No valid training data collected")
            return {}
        
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Handle missing values
        combined_features = combined_features.fillna(0)
        
        # Create final datasets
        datasets = {
            'features': combined_features,
            'price_direction': np.array(all_price_directions),
            'price_change': np.array(all_price_changes),
            'entry_points': np.array(all_entry_points),
            'exit_points': np.array(all_exit_points)
        }
        
        # Save datasets for future use
        self._save_datasets(datasets)
        
        logger.info(f"Training data preparation complete: {len(combined_features)} samples")
        return datasets
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features from historical price data
        Returns DataFrame with calculated features
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['price_range'] = (df['high'] - df['low']) / df['low']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(df) >= window:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ma_dist_{window}'] = (df['close'] - df[f'ma_{window}']) / df[f'ma_{window}']
                df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_ma_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
        
        # Volatility indicators
        for window in [5, 10, 20]:
            if len(df) >= window:
                df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std()
        
        # RSI
        for window in [7, 14]:
            if len(df) >= window:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(df) >= 26:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if len(df) >= 20:
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Lagged features (previous candle values)
        for lag in range(1, 6):
            if len(df) > lag:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Drop rows with NaN values (typically the first few rows due to lagging indicators)
        df = df.dropna()
        
        # Select only feature columns (exclude raw price/volume data)
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        return df[feature_cols]
    
    def _calculate_targets(self, df: pd.DataFrame) -> Dict:
        """
        Calculate target variables for model training
        Returns dictionary with different target variables
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Future price changes (for different horizons)
        horizons = [1, 3, 6, 12, 24]  # Future candles to look ahead
        
        for horizon in horizons:
            if len(df) > horizon:
                # Future price change
                df[f'future_price_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Price direction (1 if price goes up, 0 if down)
        price_direction = []
        price_change = []
        entry_points = []
        exit_points = []
        
        # Use 6-candle horizon as default
        horizon = 6
        future_col = f'future_price_{horizon}'
        
        if future_col in df.columns:
            # Price direction (classification)
            price_direction = (df[future_col] > 0.02).astype(int).tolist()  # 2% threshold for significant move
            
            # Price change percentage (regression)
            price_change = df[future_col].tolist()
            
            # Entry points (good buying opportunities)
            # Define as: price will increase by at least 5% within horizon, and current RSI < 40
            if 'rsi_14' in df.columns:
                entry_points = ((df[future_col] > 0.05) & (df['rsi_14'] < 40)).astype(int).tolist()
            else:
                entry_points = (df[future_col] > 0.05).astype(int).tolist()
            
            # Exit points (good selling opportunities)
            # Define as: price will decrease by at least 3% within horizon, and current RSI > 70
            if 'rsi_14' in df.columns:
                exit_points = ((df[future_col] < -0.03) & (df['rsi_14'] > 70)).astype(int).tolist()
            else:
                exit_points = (df[future_col] < -0.03).astype(int).tolist()
        
        return {
            'price_direction': price_direction,
            'price_change': price_change,
            'entry_points': entry_points,
            'exit_points': exit_points
        }
    
    def _save_datasets(self, datasets: Dict) -> None:
        """Save prepared datasets to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.data_dir, f"training_data_{timestamp}.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump(datasets, f)
        
        logger.info(f"Datasets saved to {filepath}")
    
    def train_models(self, datasets: Optional[Dict] = None) -> Dict:
        """
        Train machine learning models using prepared datasets
        Returns training results and metrics
        """
        logger.info("Starting model training")
        
        # Load datasets if not provided
        if not datasets:
            datasets = self._load_latest_datasets()
            
            if not datasets:
                logger.error("No datasets available for training")
                return {"success": False, "error": "No datasets available"}
        
        # Extract data
        features = datasets['features']
        price_direction = datasets['price_direction']
        price_change = datasets['price_change']
        entry_points = datasets['entry_points']
        exit_points = datasets['exit_points']
        
        # Check if we have enough data
        if len(features) < 100:
            logger.warning(f"Limited training data: only {len(features)} samples")
        
        # Train price direction model (classification)
        direction_results = self._train_price_direction_model(features, price_direction)
        
        # Train price magnitude model (regression)
        magnitude_results = self._train_price_magnitude_model(features, price_change)
        
        # Train entry point model (classification)
        entry_results = self._train_entry_point_model(features, entry_points)
        
        # Train exit point model (classification)
        exit_results = self._train_exit_point_model(features, exit_points)
        
        # Save all models
        self._save_models()
        
        # Compile results
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "samples_count": len(features),
            "direction_model": direction_results,
            "magnitude_model": magnitude_results,
            "entry_model": entry_results,
            "exit_model": exit_results
        }
        
        logger.info("Model training completed successfully")
        return results
    
    def _load_latest_datasets(self) -> Optional[Dict]:
        """Load the most recent saved datasets"""
        data_files = [f for f in os.listdir(self.data_dir) if f.startswith("training_data_") and f.endswith(".pkl")]
        
        if not data_files:
            return None
        
        # Sort by timestamp (newest first)
        latest_file = sorted(data_files, reverse=True)[0]
        filepath = os.path.join(self.data_dir, latest_file)
        
        with open(filepath, 'rb') as f:
            datasets = pickle.load(f)
        
        logger.info(f"Loaded datasets from {filepath}")
        return datasets
    
    def _train_price_direction_model(self, features: pd.DataFrame, price_direction: np.ndarray) -> Dict:
        """
        Train model to predict price direction (up/down)
        Returns training results and metrics
        """
        logger.info("Training price direction model")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, price_direction, test_size=0.2, random_state=42
            )
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=42))
            ])
            
            # Define hyperparameters for grid search
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10]
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1
            )
            
            # Train model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Save model and scaler
            self.price_direction_model = best_model
            self.scalers['price_direction'] = best_model.named_steps['scaler']
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test,
(Content truncated due to size limit. Use line ranges to read in chunks)