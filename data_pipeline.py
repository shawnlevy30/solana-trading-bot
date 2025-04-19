"""
Data Pipeline Module for Solana Memecoin Trading Bot

This module processes collected data to extract critical metrics needed for analysis
and trading decisions. It implements the data pipeline for transforming raw data
into actionable insights.
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
from .timeframe_adaptability import TimeframeAdaptabilityManager
from .api_config import TIMEFRAMES, MARKET_CAP_TIERS, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_pipeline")

class DataPipeline:
    """
    Processes collected data to extract critical metrics needed for analysis
    and trading decisions.
    """
    
    def __init__(self, data_collector: DataCollector, adaptability_manager: TimeframeAdaptabilityManager):
        self.data_collector = data_collector
        self.adaptability_manager = adaptability_manager
        self.output_dir = os.path.join(os.getcwd(), "processed_data")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("DataPipeline initialized")
    
    def process_token_data(self, token_address: str, chain_id: str = "solana") -> Dict:
        """
        Process token data to extract critical metrics
        Returns processed data with all critical metrics
        """
        logger.info(f"Processing data for token {token_address}")
        
        # Get token data
        token_data = self.data_collector.get_token_data(token_address, chain_id)
        
        # Get optimal timeframe adaptation
        adaptation = self.adaptability_manager.adapt_to_market_conditions(token_address, chain_id)
        
        # Extract key data
        dexscreener_data = token_data.get("dexscreener_data", {})
        birdeye_price = token_data.get("birdeye_price", {})
        historical_data = token_data.get("historical_data", {}).get("data", [])
        metadata = token_data.get("metadata", {})
        
        # Extract pairs data from DexScreener
        pairs = dexscreener_data.get("pairs", [])
        pair = pairs[0] if pairs else {}
        
        # Process and calculate critical metrics
        processed_data = {
            "token_address": token_address,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat(),
            "basic_info": self._extract_basic_info(pair, birdeye_price),
            "market_metrics": self._calculate_market_metrics(pair, birdeye_price),
            "technical_indicators": self._calculate_technical_indicators(historical_data),
            "holder_metrics": self._extract_holder_metrics(pair),
            "liquidity_metrics": self._calculate_liquidity_metrics(pair),
            "volume_metrics": self._calculate_volume_metrics(pair, historical_data),
            "social_metrics": self._extract_social_metrics(pair),
            "security_metrics": self._calculate_security_metrics(pair),
            "adaptability_metrics": adaptation,
            "orderblock_analysis": self._perform_orderblock_analysis(historical_data)
        }
        
        # Save processed data
        self._save_processed_data(processed_data, token_address)
        
        logger.info(f"Data processing completed for {token_address}")
        return processed_data
    
    def _extract_basic_info(self, pair: Dict, birdeye_price: Dict) -> Dict:
        """Extract basic token information"""
        base_token = pair.get("baseToken", {})
        quote_token = pair.get("quoteToken", {})
        
        return {
            "name": base_token.get("name", ""),
            "symbol": base_token.get("symbol", ""),
            "address": base_token.get("address", ""),
            "quote_token": quote_token.get("symbol", ""),
            "dex_id": pair.get("dexId", ""),
            "pair_address": pair.get("pairAddress", ""),
            "pair_created_at": pair.get("pairCreatedAt", 0),
            "age_in_days": (datetime.now().timestamp() - pair.get("pairCreatedAt", 0)) / 86400,
            "price_usd": pair.get("priceUsd", 0),
            "price_native": pair.get("priceNative", 0),
            "birdeye_price": birdeye_price.get("data", {}).get("value", 0)
        }
    
    def _calculate_market_metrics(self, pair: Dict, birdeye_price: Dict) -> Dict:
        """Calculate market-related metrics"""
        price_change = pair.get("priceChange", {})
        
        return {
            "market_cap": pair.get("marketCap", 0),
            "fully_diluted_valuation": pair.get("fdv", 0),
            "market_cap_fdv_ratio": pair.get("marketCap", 0) / pair.get("fdv", 1) if pair.get("fdv", 0) > 0 else 0,
            "price_change_5m": price_change.get("m5", 0),
            "price_change_1h": price_change.get("h1", 0),
            "price_change_6h": price_change.get("h6", 0),
            "price_change_24h": price_change.get("h24", 0),
            "price_change_7d": price_change.get("d7", 0),
            "all_time_high": birdeye_price.get("data", {}).get("ath", 0),
            "all_time_low": birdeye_price.get("data", {}).get("atl", 0),
            "is_at_ath": self._is_at_all_time_high(pair, birdeye_price)
        }
    
    def _is_at_all_time_high(self, pair: Dict, birdeye_price: Dict) -> bool:
        """Check if token is at all-time high"""
        current_price = pair.get("priceUsd", 0)
        ath = birdeye_price.get("data", {}).get("ath", 0)
        
        if ath <= 0 or current_price <= 0:
            return False
        
        # Consider it at ATH if within 5% of ATH
        return current_price >= ath * 0.95
    
    def _calculate_technical_indicators(self, historical_data: List[Dict]) -> Dict:
        """Calculate technical indicators from historical data"""
        if not historical_data or len(historical_data) < 20:
            return {
                "rsi_14": None,
                "macd": None,
                "macd_signal": None,
                "macd_histogram": None,
                "bollinger_upper": None,
                "bollinger_middle": None,
                "bollinger_lower": None,
                "support_levels": [],
                "resistance_levels": []
            }
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # Calculate Bollinger Bands
            middle_band = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            upper_band = middle_band + (std_dev * 2)
            lower_band = middle_band - (std_dev * 2)
            
            # Identify support and resistance levels
            support_levels = self._identify_support_levels(df)
            resistance_levels = self._identify_resistance_levels(df)
            
            return {
                "rsi_14": rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
                "macd": macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else None,
                "macd_signal": signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else None,
                "macd_histogram": histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else None,
                "bollinger_upper": upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else None,
                "bollinger_middle": middle_band.iloc[-1] if not pd.isna(middle_band.iloc[-1]) else None,
                "bollinger_lower": lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else None,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            }
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {
                "rsi_14": None,
                "macd": None,
                "macd_signal": None,
                "macd_histogram": None,
                "bollinger_upper": None,
                "bollinger_middle": None,
                "bollinger_lower": None,
                "support_levels": [],
                "resistance_levels": []
            }
    
    def _identify_support_levels(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Identify support levels using local minima"""
        try:
            # Find local minima
            local_min = []
            for i in range(window, len(df) - window):
                if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                    local_min.append(df['low'].iloc[i])
            
            # Group close levels (within 2% of each other)
            if not local_min:
                return []
            
            local_min.sort()
            grouped_levels = []
            current_group = [local_min[0]]
            
            for i in range(1, len(local_min)):
                if local_min[i] <= current_group[-1] * 1.02:  # Within 2%
                    current_group.append(local_min[i])
                else:
                    grouped_levels.append(sum(current_group) / len(current_group))
                    current_group = [local_min[i]]
            
            if current_group:
                grouped_levels.append(sum(current_group) / len(current_group))
            
            # Return top 3 most significant levels
            return sorted(grouped_levels)[:3]
        except Exception as e:
            logger.warning(f"Error identifying support levels: {e}")
            return []
    
    def _identify_resistance_levels(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Identify resistance levels using local maxima"""
        try:
            # Find local maxima
            local_max = []
            for i in range(window, len(df) - window):
                if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                    local_max.append(df['high'].iloc[i])
            
            # Group close levels (within 2% of each other)
            if not local_max:
                return []
            
            local_max.sort()
            grouped_levels = []
            current_group = [local_max[0]]
            
            for i in range(1, len(local_max)):
                if local_max[i] <= current_group[-1] * 1.02:  # Within 2%
                    current_group.append(local_max[i])
                else:
                    grouped_levels.append(sum(current_group) / len(current_group))
                    current_group = [local_max[i]]
            
            if current_group:
                grouped_levels.append(sum(current_group) / len(current_group))
            
            # Return top 3 most significant levels
            return sorted(grouped_levels)[:3]
        except Exception as e:
            logger.warning(f"Error identifying resistance levels: {e}")
            return []
    
    def _extract_holder_metrics(self, pair: Dict) -> Dict:
        """Extract holder-related metrics"""
        # Note: This is placeholder as DexScreener doesn't provide holder info directly
        # In a real implementation, this would fetch data from Solana explorers or other sources
        return {
            "total_holders": 0,  # Placeholder
            "top_10_concentration": 0,  # Placeholder
            "holder_growth_24h": 0,  # Placeholder
            "average_holding_time": 0  # Placeholder
        }
    
    def _calculate_liquidity_metrics(self, pair: Dict) -> Dict:
        """Calculate liquidity-related metrics"""
        liquidity = pair.get("liquidity", {})
        
        return {
            "liquidity_usd": liquidity.get("usd", 0),
            "liquidity_base": liquidity.get("base", 0),
            "liquidity_quote": liquidity.get("quote", 0),
            "liquidity_market_cap_ratio": liquidity.get("usd", 0) / pair.get("marketCap", 1) if pair.get("marketCap", 0) > 0 else 0
        }
    
    def _calculate_volume_metrics(self, pair: Dict, historical_data: List[Dict]) -> Dict:
        """Calculate volume-related metrics"""
        volume = pair.get("volume", {})
        txns = pair.get("txns", {})
        
        # Calculate volume spike (current vs average)
        avg_volume = 0
        if historical_data and len(historical_data) > 1:
            try:
                volumes = [float(candle.get('volume', 0)) for candle in historical_data[:-1]]  # Exclude current
                volumes = [v for v in volumes if v > 0]  # Filter out zeros
                if volumes:
                    avg_volume = sum(volumes) / len(volumes)
            except Exception as e:
                logger.warning(f"Error calculating average volume: {e}")
        
        current_volume = volume.get("h24", 0)
        volume_spike = (current_volume / avg_volume - 1) * 100 if avg_volume > 0 else 0
        
        # Calculate buy/sell ratio
        h24_buys = txns.get("h24", {}).get("buys", 0)
        h24_sells = txns.get("h24", {}).get("sells", 0)
        buy_sell_ratio = h24_buys / h24_sells if h24_sells > 0 else float('inf')
        
        return {
            "volume_24h": volume.get("h24", 0),
            "volume_6h": volume.get("h6", 0),
            "volume_1h": volume.get("h1", 0),
            "volume_market_cap_ratio": volume.get("h24", 0) / pair.get("marketCap", 1) if pair.get("marketCap", 0) > 0 else 0,
            "volume_liquidity_ratio": volume.get("h24", 0) / pair.get("liquidity", {}).get("usd", 1) if pair.get("liquidity", {}).get("usd", 0) > 0 else 0,
            "volume_spike_percentage": volume_spike,
            "buy_transactions_24h": h24_buys,
            "sell_transactions_24h": h24_sells,
            "buy_sell_ratio": buy_sell_ratio
        }
    
    def _extract_social_metrics(self, pair: Dict) -> Dict:
        """Extract social media metrics"""
        # Note: This is placeholder as DexScreener doesn't provide social metrics directly
        # In a real implementation, this would fetch data from Twitter API or other sources
        return {
            "twitter_mentions_24h": 0,  # Placeholder
            "twitter_sentiment": 0,  # Placeholder
            "influencer_mentions": 0,  # Placeholder
            "social_volume_change": 0  # Placeholder
        }
    
    def _calculate_security_metrics(self, pair: Dict) -> Dict:
        """Calculate security-related metrics"""
        # Note: Some of these are placeholders and would require additional data sources
        return {
            "is_pump_fun": self._is_pump_fun_token(pair),
            "bundle_concentration": 0,  # Placeholder
            "sniper_concentration": 0,  # Placeholder
     
(Content truncated due to size limit. Use line ranges to read in chunks)