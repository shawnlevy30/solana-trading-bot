"""
Timeframe Adaptability Module for Solana Memecoin Trading Bot

This module enhances the timeframe adaptability features already implemented
in the data collection module, providing more sophisticated adaptability
based on market conditions and trading patterns.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

# Import from data collection module
from .collector import DataCollector
from .api_config import TIMEFRAMES, MARKET_CAP_TIERS, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("timeframe_adaptability.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("timeframe_adaptability")

class TimeframeAdaptabilityManager:
    """
    Manages timeframe adaptability based on coin characteristics and market conditions.
    Extends the basic adaptability implemented in the data collection module.
    """
    
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        logger.info("TimeframeAdaptabilityManager initialized")
    
    def detect_volatility(self, historical_data: List[Dict]) -> float:
        """
        Calculate volatility from historical price data
        Returns volatility as a percentage
        """
        if not historical_data or len(historical_data) < 2:
            return 0.0
        
        # Extract close prices
        try:
            prices = [float(candle.get('close', 0)) for candle in historical_data]
            prices = [p for p in prices if p > 0]  # Filter out zeros
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns) * 100
            
            return volatility
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.0
    
    def detect_trend_strength(self, historical_data: List[Dict]) -> float:
        """
        Calculate trend strength from historical price data
        Returns a value between -1 (strong downtrend) and 1 (strong uptrend)
        """
        if not historical_data or len(historical_data) < 10:
            return 0.0
        
        try:
            # Extract close prices
            prices = [float(candle.get('close', 0)) for candle in historical_data]
            prices = [p for p in prices if p > 0]  # Filter out zeros
            
            if len(prices) < 10:
                return 0.0
            
            # Calculate simple moving averages
            short_window = 5
            long_window = 20
            
            # Ensure we have enough data
            if len(prices) < long_window:
                long_window = len(prices) // 2
                short_window = long_window // 2
            
            short_ma = np.mean(prices[-short_window:])
            long_ma = np.mean(prices[-long_window:])
            
            # Calculate trend strength
            trend_strength = (short_ma / long_ma - 1) * 10  # Scale for better readability
            
            # Clamp between -1 and 1
            return max(min(trend_strength, 1.0), -1.0)
        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 0.0
    
    def detect_volume_profile(self, historical_data: List[Dict]) -> Dict:
        """
        Analyze volume profile from historical data
        Returns volume characteristics
        """
        if not historical_data or len(historical_data) < 5:
            return {"volume_trend": 0, "volume_spikes": 0, "avg_volume": 0}
        
        try:
            # Extract volumes
            volumes = [float(candle.get('volume', 0)) for candle in historical_data]
            volumes = [v for v in volumes if v > 0]  # Filter out zeros
            
            if len(volumes) < 5:
                return {"volume_trend": 0, "volume_spikes": 0, "avg_volume": 0}
            
            # Calculate average volume
            avg_volume = np.mean(volumes)
            
            # Calculate volume trend (positive means increasing volume)
            recent_avg = np.mean(volumes[-3:])
            older_avg = np.mean(volumes[:-3])
            volume_trend = (recent_avg / older_avg - 1) if older_avg > 0 else 0
            
            # Detect volume spikes (how many standard deviations above mean)
            std_volume = np.std(volumes)
            if std_volume > 0 and avg_volume > 0:
                recent_max = max(volumes[-3:])
                volume_spikes = (recent_max - avg_volume) / std_volume
            else:
                volume_spikes = 0
            
            return {
                "volume_trend": volume_trend,
                "volume_spikes": volume_spikes,
                "avg_volume": avg_volume
            }
        except Exception as e:
            logger.warning(f"Error analyzing volume profile: {e}")
            return {"volume_trend": 0, "volume_spikes": 0, "avg_volume": 0}
    
    def recommend_timeframes(self, token_address: str, chain_id: str = "solana") -> Dict:
        """
        Recommend optimal timeframes based on comprehensive analysis
        Returns primary and secondary timeframes with rationale
        """
        logger.info(f"Recommending timeframes for {token_address}")
        
        # Get token data
        token_data = self.data_collector.get_token_data(token_address, chain_id)
        
        # Extract metadata and historical data
        metadata = token_data.get("metadata", {})
        historical_data = token_data.get("historical_data", {}).get("data", [])
        
        # Get basic timeframe recommendation based on coin age
        coin_age_category = metadata.get("coin_age_category", "new")
        market_cap_tier = metadata.get("market_cap_tier", "micro")
        
        # Get default recommendations from config
        age_config = DATA_CONFIG["age_timeframe_mapping"].get(coin_age_category, {})
        primary_timeframe = age_config.get("primary_timeframe", "5m")
        secondary_timeframes = age_config.get("secondary_timeframes", ["15m", "1h"])
        
        # Calculate advanced metrics
        volatility = self.detect_volatility(historical_data)
        trend_strength = self.detect_trend_strength(historical_data)
        volume_profile = self.detect_volume_profile(historical_data)
        
        # Adjust timeframes based on advanced metrics
        adjusted_timeframes = self._adjust_timeframes_based_on_metrics(
            primary_timeframe,
            secondary_timeframes,
            volatility,
            trend_strength,
            volume_profile,
            coin_age_category,
            market_cap_tier
        )
        
        # Prepare detailed response
        recommendation = {
            "token_address": token_address,
            "chain_id": chain_id,
            "coin_age_category": coin_age_category,
            "market_cap_tier": market_cap_tier,
            "metrics": {
                "volatility": volatility,
                "trend_strength": trend_strength,
                "volume_profile": volume_profile
            },
            "timeframes": adjusted_timeframes,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Timeframe recommendation for {token_address}: {adjusted_timeframes['primary']}")
        return recommendation
    
    def _adjust_timeframes_based_on_metrics(
        self,
        primary_timeframe: str,
        secondary_timeframes: List[str],
        volatility: float,
        trend_strength: float,
        volume_profile: Dict,
        coin_age_category: str,
        market_cap_tier: str
    ) -> Dict:
        """
        Adjust timeframes based on calculated metrics
        Returns adjusted primary and secondary timeframes with rationale
        """
        # Define timeframe options from shortest to longest
        timeframe_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        # Find index of current primary timeframe
        try:
            primary_index = timeframe_options.index(primary_timeframe)
        except ValueError:
            primary_index = 1  # Default to 5m if not found
        
        # Initialize adjustment logic
        adjustment = 0
        rationale = []
        
        # Adjust for volatility
        if volatility > 5.0:  # High volatility
            adjustment -= 1
            rationale.append(f"High volatility ({volatility:.2f}%) suggests shorter timeframe")
        elif volatility < 1.0:  # Low volatility
            adjustment += 1
            rationale.append(f"Low volatility ({volatility:.2f}%) allows longer timeframe")
        
        # Adjust for trend strength
        if abs(trend_strength) > 0.7:  # Strong trend
            adjustment += 1
            direction = "uptrend" if trend_strength > 0 else "downtrend"
            rationale.append(f"Strong {direction} (strength: {trend_strength:.2f}) allows longer timeframe")
        elif abs(trend_strength) < 0.2:  # Weak or no trend
            adjustment -= 1
            rationale.append(f"Weak trend (strength: {trend_strength:.2f}) suggests shorter timeframe")
        
        # Adjust for volume profile
        if volume_profile["volume_spikes"] > 2.0:  # Significant volume spikes
            adjustment -= 1
            rationale.append(f"Volume spikes detected ({volume_profile['volume_spikes']:.2f} std) suggest shorter timeframe")
        
        if volume_profile["volume_trend"] > 0.5:  # Increasing volume
            adjustment -= 1
            rationale.append(f"Increasing volume trend ({volume_profile['volume_trend']:.2f}) suggests shorter timeframe")
        
        # Special case for new coins
        if coin_age_category == "new":
            adjustment = min(adjustment, 0)  # Never go longer than default for new coins
            rationale.append("New coin status caps maximum timeframe")
        
        # Special case for micro cap
        if market_cap_tier == "micro":
            adjustment = min(adjustment, 0)  # Never go longer than default for micro caps
            rationale.append("Micro cap status caps maximum timeframe")
        
        # Calculate new primary timeframe index with bounds checking
        new_primary_index = max(0, min(len(timeframe_options) - 1, primary_index + adjustment))
        new_primary_timeframe = timeframe_options[new_primary_index]
        
        # Calculate new secondary timeframes (one shorter, one longer if possible)
        new_secondary_timeframes = []
        if new_primary_index > 0:
            new_secondary_timeframes.append(timeframe_options[new_primary_index - 1])
        if new_primary_index < len(timeframe_options) - 1:
            new_secondary_timeframes.append(timeframe_options[new_primary_index + 1])
        
        # Ensure we have at least one secondary timeframe
        if not new_secondary_timeframes:
            new_secondary_timeframes = [primary_timeframe]
        
        return {
            "primary": new_primary_timeframe,
            "secondary": new_secondary_timeframes,
            "adjustment": adjustment,
            "rationale": rationale
        }
    
    def get_optimal_history_length(self, token_address: str, timeframe: str, chain_id: str = "solana") -> int:
        """
        Determine optimal history length based on token characteristics and timeframe
        Returns number of candles to fetch
        """
        # Get token data
        token_data = self.data_collector.get_token_data(token_address, chain_id)
        
        # Extract metadata
        metadata = token_data.get("metadata", {})
        market_cap_tier = metadata.get("market_cap_tier", "micro")
        
        # Get base history length from config
        base_length = DATA_CONFIG["market_cap_history_mapping"].get(market_cap_tier, 100)
        
        # Adjust based on timeframe (longer timeframes need fewer candles)
        timeframe_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        try:
            timeframe_index = timeframe_options.index(timeframe)
            # Scale down as timeframe gets longer
            adjustment_factor = 1.0 - (timeframe_index * 0.1)
            adjusted_length = int(base_length * adjustment_factor)
            return max(50, adjusted_length)  # Ensure at least 50 candles
        except ValueError:
            return base_length  # Return base length if timeframe not recognized
    
    def adapt_to_market_conditions(self, token_address: str, chain_id: str = "solana") -> Dict:
        """
        Comprehensive adaptation to current market conditions
        Returns optimal parameters for data collection and analysis
        """
        # Get timeframe recommendations
        timeframe_rec = self.recommend_timeframes(token_address, chain_id)
        
        # Get optimal history length for primary timeframe
        primary_timeframe = timeframe_rec["timeframes"]["primary"]
        history_length = self.get_optimal_history_length(token_address, primary_timeframe, chain_id)
        
        # Prepare comprehensive adaptation parameters
        adaptation = {
            "token_address": token_address,
            "chain_id": chain_id,
            "timeframe_recommendation": timeframe_rec,
            "data_collection_params": {
                "primary_timeframe": primary_timeframe,
                "secondary_timeframes": timeframe_rec["timeframes"]["secondary"],
                "history_length": history_length
            },
            "analysis_params": {
                "volatility_level": "high" if timeframe_rec["metrics"]["volatility"] > 3.0 else "medium" if timeframe_rec["metrics"]["volatility"] > 1.0 else "low",
                "trend_direction": "up" if timeframe_rec["metrics"]["trend_strength"] > 0.3 else "down" if timeframe_rec["metrics"]["trend_strength"] < -0.3 else "sideways",
                "volume_profile": "spiking" if timeframe_rec["metrics"]["volume_profile"]["volume_spikes"] > 2.0 else "increasing" if timeframe_rec["metrics"]["volume_profile"]["volume_trend"] > 0.3 else "stable"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Market adaptation complete for {token_address}")
        return adaptation
