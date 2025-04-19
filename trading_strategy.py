"""
Trading Strategy Module for Solana Memecoin Trading Bot

This module implements trading strategies for Solana memecoins, integrating
data from the data pipeline, security audit system, and machine learning models
to make informed trading decisions.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

# Import from other modules
from .collector import DataCollector
from .data_pipeline import DataPipeline
from .security_audit import SecurityAuditSystem
from .ml_model import MLModelManager
from .custom_filters import CustomFilters
from .api_config import TIMEFRAMES, MARKET_CAP_TIERS, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_strategy")

class TradingStrategy:
    """
    Base class for trading strategies
    """
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {name} strategy")
    
    def generate_signal(self, token_data: Dict) -> Dict:
        """
        Generate trading signal based on token data
        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement generate_signal method")
    
    def calculate_position_size(self, token_data: Dict, portfolio_value: float, risk_level: str) -> float:
        """
        Calculate position size based on token data and portfolio value
        Returns position size in USD
        """
        # Default implementation - override in subclasses for specific logic
        
        # Base position size as percentage of portfolio
        base_percentages = {
            "LOW": 0.05,      # 5% of portfolio for low risk
            "MODERATE": 0.03, # 3% of portfolio for moderate risk
            "HIGH": 0.01,     # 1% of portfolio for high risk
            "VERY HIGH": 0.005, # 0.5% of portfolio for very high risk
            "EXTREME": 0      # 0% for extreme risk (don't trade)
        }
        
        # Get risk level from security audit
        security = token_data.get("security", {})
        token_risk = security.get("risk_level", "EXTREME")
        
        # Get base percentage based on risk level
        base_percentage = base_percentages.get(token_risk, 0)
        
        # Adjust based on strategy-specific risk level
        risk_multipliers = {
            "LOW": 1.5,       # Increase position for low risk strategy
            "MODERATE": 1.0,  # Standard position for moderate risk
            "HIGH": 0.5       # Reduce position for high risk strategy
        }
        
        risk_multiplier = risk_multipliers.get(risk_level, 0.5)
        
        # Calculate position size
        position_size = portfolio_value * base_percentage * risk_multiplier
        
        # Cap position size
        max_position = portfolio_value * 0.1  # Maximum 10% of portfolio
        position_size = min(position_size, max_position)
        
        return position_size
    
    def calculate_stop_loss(self, token_data: Dict, entry_price: float) -> float:
        """
        Calculate stop loss price based on token data and entry price
        Returns stop loss price
        """
        # Default implementation - override in subclasses for specific logic
        
        # Default to 10% below entry price
        return entry_price * 0.9
    
    def calculate_take_profit(self, token_data: Dict, entry_price: float) -> float:
        """
        Calculate take profit price based on token data and entry price
        Returns take profit price
        """
        # Default implementation - override in subclasses for specific logic
        
        # Default to 20% above entry price
        return entry_price * 1.2

class MLBasedStrategy(TradingStrategy):
    """
    Trading strategy based on machine learning predictions
    """
    
    def __init__(self, ml_model_manager: MLModelManager, confidence_threshold: float = 0.7):
        super().__init__("ML-Based Strategy")
        self.ml_model_manager = ml_model_manager
        self.confidence_threshold = confidence_threshold
    
    def generate_signal(self, token_data: Dict) -> Dict:
        """
        Generate trading signal based on ML predictions
        Returns signal with action, confidence, and reasoning
        """
        # Extract predictions
        predictions = token_data.get("predictions", {})
        price_direction = predictions.get("price_direction", {})
        price_change = predictions.get("price_change", {})
        entry_point = predictions.get("entry_point", {})
        exit_point = predictions.get("exit_point", {})
        
        # Extract security info
        security = token_data.get("security", {})
        risk_level = security.get("risk_level", "EXTREME")
        red_flags = security.get("red_flags", [])
        
        # Default signal
        signal = {
            "action": "HOLD",
            "confidence": 0.0,
            "reasoning": ["Insufficient data for decision"]
        }
        
        # Check if we have predictions
        if not predictions:
            return signal
        
        # Initialize reasoning
        reasoning = []
        
        # Check security risk level
        if risk_level in ["EXTREME", "VERY HIGH"]:
            signal["action"] = "AVOID"
            signal["confidence"] = 0.9
            reasoning.append(f"Security risk level is {risk_level}")
            if red_flags:
                reasoning.append(f"Red flags: {', '.join(red_flags[:3])}")
            signal["reasoning"] = reasoning
            return signal
        
        # Check entry point prediction
        entry_prediction = entry_point.get("prediction", 0)
        entry_probability = entry_point.get("probability", 0.0)
        
        # Check exit point prediction
        exit_prediction = exit_point.get("prediction", 0)
        exit_probability = exit_point.get("probability", 0.0)
        
        # Check price direction prediction
        direction_prediction = price_direction.get("prediction", 0)
        direction_probability = price_direction.get("probability", 0.0)
        
        # Check price change prediction
        change_prediction = price_change.get("prediction", 0.0)
        
        # Determine action based on predictions
        if exit_prediction == 1 and exit_probability >= self.confidence_threshold:
            signal["action"] = "SELL"
            signal["confidence"] = exit_probability
            reasoning.append(f"Exit point detected with {exit_probability:.2f} confidence")
            
            if direction_prediction == 0 and direction_probability >= 0.6:
                reasoning.append(f"Price expected to go down with {direction_probability:.2f} confidence")
                signal["confidence"] = (exit_probability + direction_probability) / 2
            
            if change_prediction < -0.05:
                reasoning.append(f"Price expected to drop by {abs(change_prediction)*100:.2f}%")
        
        elif entry_prediction == 1 and entry_probability >= self.confidence_threshold:
            signal["action"] = "BUY"
            signal["confidence"] = entry_probability
            reasoning.append(f"Entry point detected with {entry_probability:.2f} confidence")
            
            if direction_prediction == 1 and direction_probability >= 0.6:
                reasoning.append(f"Price expected to go up with {direction_probability:.2f} confidence")
                signal["confidence"] = (entry_probability + direction_probability) / 2
            
            if change_prediction > 0.05:
                reasoning.append(f"Price expected to rise by {change_prediction*100:.2f}%")
        
        elif direction_prediction == 1 and direction_probability >= self.confidence_threshold:
            signal["action"] = "BUY"
            signal["confidence"] = direction_probability
            reasoning.append(f"Price expected to go up with {direction_probability:.2f} confidence")
            
            if change_prediction > 0.05:
                reasoning.append(f"Price expected to rise by {change_prediction*100:.2f}%")
        
        elif direction_prediction == 0 and direction_probability >= self.confidence_threshold:
            signal["action"] = "SELL"
            signal["confidence"] = direction_probability
            reasoning.append(f"Price expected to go down with {direction_probability:.2f} confidence")
            
            if change_prediction < -0.05:
                reasoning.append(f"Price expected to drop by {abs(change_prediction)*100:.2f}%")
        
        else:
            signal["action"] = "HOLD"
            signal["confidence"] = max(0.5, 1 - max(direction_probability, entry_probability, exit_probability))
            reasoning.append("No strong signals detected")
        
        # Add security considerations
        if risk_level == "HIGH":
            reasoning.append(f"Security risk level is {risk_level} - exercise caution")
            if signal["action"] == "BUY":
                signal["confidence"] *= 0.8  # Reduce confidence for buys on high risk tokens
        
        # Add any red flags
        if red_flags and (signal["action"] == "BUY" or signal["action"] == "HOLD"):
            reasoning.append(f"Note: {red_flags[0]}")
        
        signal["reasoning"] = reasoning
        return signal
    
    def calculate_position_size(self, token_data: Dict, portfolio_value: float, risk_level: str) -> float:
        """
        Calculate position size based on ML predictions and risk level
        Returns position size in USD
        """
        # Get base position size from parent class
        base_position = super().calculate_position_size(token_data, portfolio_value, risk_level)
        
        # Extract predictions
        predictions = token_data.get("predictions", {})
        price_direction = predictions.get("price_direction", {})
        direction_probability = price_direction.get("probability", 0.5)
        
        # Adjust position size based on prediction confidence
        confidence_multiplier = (direction_probability - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
        confidence_multiplier = max(0.1, min(1.5, confidence_multiplier))  # Limit to 0.1-1.5 range
        
        adjusted_position = base_position * confidence_multiplier
        
        return adjusted_position
    
    def calculate_stop_loss(self, token_data: Dict, entry_price: float) -> float:
        """
        Calculate stop loss based on ML predictions and technical indicators
        Returns stop loss price
        """
        # Extract technical indicators
        technical_indicators = token_data.get("technical_indicators", {})
        support_levels = technical_indicators.get("support_levels", [])
        
        # Extract predictions
        predictions = token_data.get("predictions", {})
        price_change = predictions.get("price_change", {})
        change_prediction = price_change.get("prediction", 0.0)
        
        # Find nearest support level below entry price
        nearest_support = None
        for level in sorted(support_levels, reverse=True):
            if level < entry_price:
                nearest_support = level
                break
        
        # If we have a support level, use it with a small buffer
        if nearest_support is not None:
            buffer = (entry_price - nearest_support) * 0.1  # 10% of the distance
            return nearest_support - buffer
        
        # Otherwise, use expected price change to inform stop loss
        if change_prediction > 0:
            # For positive expected change, use tighter stop loss
            return entry_price * 0.92  # 8% below entry
        else:
            # For negative expected change, use wider stop loss
            return entry_price * 0.85  # 15% below entry
    
    def calculate_take_profit(self, token_data: Dict, entry_price: float) -> float:
        """
        Calculate take profit based on ML predictions and technical indicators
        Returns take profit price
        """
        # Extract technical indicators
        technical_indicators = token_data.get("technical_indicators", {})
        resistance_levels = technical_indicators.get("resistance_levels", [])
        
        # Extract predictions
        predictions = token_data.get("predictions", {})
        price_change = predictions.get("price_change", {})
        change_prediction = price_change.get("prediction", 0.0)
        
        # Find nearest resistance level above entry price
        nearest_resistance = None
        for level in sorted(resistance_levels):
            if level > entry_price:
                nearest_resistance = level
                break
        
        # If we have a resistance level, use it with a small buffer
        if nearest_resistance is not None:
            buffer = (nearest_resistance - entry_price) * 0.1  # 10% of the distance
            return nearest_resistance + buffer
        
        # Otherwise, use expected price change to inform take profit
        if change_prediction > 0.1:
            # For strong positive expected change, use higher take profit
            return entry_price * (1 + max(0.2, change_prediction * 1.5))
        else:
            # For modest expected change, use standard take profit
            return entry_price * 1.2  # 20% above entry

class OrderblockStrategy(TradingStrategy):
    """
    Trading strategy based on orderblock analysis
    """
    
    def __init__(self):
        super().__init__("Orderblock Strategy")
    
    def generate_signal(self, token_data: Dict) -> Dict:
        """
        Generate trading signal based on orderblock analysis
        Returns signal with action, confidence, and reasoning
        """
        # Extract orderblock analysis
        orderblock_analysis = token_data.get("orderblock_analysis", {})
        current_in_orderblock = orderblock_analysis.get("current_in_orderblock", False)
        orderblocks = orderblock_analysis.get("orderblocks", [])
        
        # Extract technical indicators
        technical_indicators = token_data.get("technical_indicators", {})
        rsi_14 = technical_indicators.get("rsi_14")
        
        # Extract security info
        security = token_data.get("security", {})
        risk_level = security.get("risk_level", "EXTREME")
        
        # Default signal
        signal = {
            "action": "HOLD",
            "confidence": 0.0,
            "reasoning": ["Insufficient data for decision"]
        }
        
        # Initialize reasoning
        reasoning = []
        
        # Check security risk level
        if risk_level in ["EXTREME", "VERY HIGH"]:
            signal["action"] = "AVOID"
            signal["confidence"] = 0.9
            reasoning.append(f"Security risk level is {risk_level}")
            signal["reasoning"] = reasoning
            return signal
        
        # Check if price is in an orderblock
        if current_in_orderblock:
            signal["action"] = "BUY"
            signal["confidence"] = 0.8
            reasoning.append("Price is currently in an identified orderblock (support zone)")
            
            # Check RSI for confirmation
            if rsi_14 is not None:
                if rsi_14 < 30:
                    signal["confidence"] = 0.9
                    reasoning.append(f"RSI confirms oversold condition: {rsi_14:.2f}")
                elif rsi_14 > 70:
                    signal["confidence"] = 0.6
                    reasoning.append(f"RSI indicates overbought despite orderblock: {rsi_14:.2f}")
        else:
            # Not in orderblock
            if orderblocks:
                # We have orderblocks but 
(Content truncated due to size limit. Use line ranges to read in chunks)