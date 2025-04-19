"""
Custom Filters Module for Solana Memecoin Trading Bot

This module implements custom filters for coin selection based on various criteria
including technical indicators, market metrics, and security checks.
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
from .data_pipeline import DataPipeline
from .api_config import TIMEFRAMES, MARKET_CAP_TIERS, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("custom_filters.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("custom_filters")

class CustomFilters:
    """
    Implements custom filters for coin selection based on various criteria.
    """
    
    def __init__(self, data_collector: DataCollector, adaptability_manager: TimeframeAdaptabilityManager, data_pipeline: DataPipeline):
        self.data_collector = data_collector
        self.adaptability_manager = adaptability_manager
        self.data_pipeline = data_pipeline
        self.output_dir = os.path.join(os.getcwd(), "filtered_coins")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("CustomFilters initialized")
    
    def apply_all_filters(self, token_address: str, chain_id: str = "solana") -> Dict:
        """
        Apply all filters to a token and return the results
        Returns a dictionary with filter results and an overall pass/fail status
        """
        logger.info(f"Applying all filters to token {token_address}")
        
        # Process token data
        processed_data = self.data_pipeline.process_token_data(token_address, chain_id)
        
        # Extract key components
        basic_info = processed_data.get("basic_info", {})
        market_metrics = processed_data.get("market_metrics", {})
        technical_indicators = processed_data.get("technical_indicators", {})
        liquidity_metrics = processed_data.get("liquidity_metrics", {})
        volume_metrics = processed_data.get("volume_metrics", {})
        security_metrics = processed_data.get("security_metrics", {})
        orderblock_analysis = processed_data.get("orderblock_analysis", {})
        
        # Apply individual filters
        filter_results = {
            "market_cap_filter": self._apply_market_cap_filter(market_metrics),
            "liquidity_filter": self._apply_liquidity_filter(liquidity_metrics, market_metrics),
            "volume_filter": self._apply_volume_filter(volume_metrics, market_metrics),
            "technical_filter": self._apply_technical_filter(technical_indicators),
            "security_filter": self._apply_security_filter(security_metrics, basic_info),
            "orderblock_filter": self._apply_orderblock_filter(orderblock_analysis),
            "age_filter": self._apply_age_filter(basic_info),
            "price_action_filter": self._apply_price_action_filter(market_metrics)
        }
        
        # Determine overall result
        critical_filters = ["security_filter", "liquidity_filter"]
        passed_critical = all(filter_results[f]["passed"] for f in critical_filters)
        passed_count = sum(1 for f in filter_results.values() if f["passed"])
        total_count = len(filter_results)
        
        # Token must pass all critical filters and at least 60% of all filters
        overall_passed = passed_critical and (passed_count / total_count >= 0.6)
        
        # Calculate score based on filter results
        score = self._calculate_filter_score(filter_results)
        
        result = {
            "token_address": token_address,
            "chain_id": chain_id,
            "token_name": basic_info.get("name", ""),
            "token_symbol": basic_info.get("symbol", ""),
            "filter_results": filter_results,
            "passed_filters": passed_count,
            "total_filters": total_count,
            "passed_critical": passed_critical,
            "overall_passed": overall_passed,
            "filter_score": score,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save filter results
        self._save_filter_results(result, token_address)
        
        logger.info(f"Filter application completed for {token_address}: {'PASSED' if overall_passed else 'FAILED'}")
        return result
    
    def _apply_market_cap_filter(self, market_metrics: Dict) -> Dict:
        """
        Filter based on market capitalization
        Returns filter result with pass/fail status and reason
        """
        market_cap = market_metrics.get("market_cap", 0)
        fdv = market_metrics.get("fully_diluted_valuation", 0)
        
        # Define thresholds based on requirements
        min_market_cap = 50000  # $50K minimum
        max_market_cap = 100000000  # $100M maximum for new coins
        max_fdv_ratio = 3  # Maximum FDV to MC ratio
        
        # Check conditions
        passed = True
        reasons = []
        
        if market_cap < min_market_cap:
            passed = False
            reasons.append(f"Market cap too low: ${market_cap:,.2f} < ${min_market_cap:,.2f}")
        
        if market_cap > max_market_cap:
            passed = False
            reasons.append(f"Market cap too high: ${market_cap:,.2f} > ${max_market_cap:,.2f}")
        
        if fdv > 0 and market_cap > 0:
            fdv_ratio = fdv / market_cap
            if fdv_ratio > max_fdv_ratio:
                passed = False
                reasons.append(f"FDV to MC ratio too high: {fdv_ratio:.2f} > {max_fdv_ratio}")
        
        if passed and not reasons:
            reasons.append(f"Market cap in acceptable range: ${market_cap:,.2f}")
        
        return {
            "name": "Market Cap Filter",
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "market_cap": market_cap,
                "fully_diluted_valuation": fdv,
                "fdv_ratio": fdv / market_cap if market_cap > 0 and fdv > 0 else 0
            }
        }
    
    def _apply_liquidity_filter(self, liquidity_metrics: Dict, market_metrics: Dict) -> Dict:
        """
        Filter based on liquidity metrics
        Returns filter result with pass/fail status and reason
        """
        liquidity_usd = liquidity_metrics.get("liquidity_usd", 0)
        market_cap = market_metrics.get("market_cap", 0)
        liquidity_ratio = liquidity_metrics.get("liquidity_market_cap_ratio", 0)
        
        # Define thresholds based on requirements
        min_liquidity = 10000  # $10K minimum
        min_liquidity_ratio = 0.05  # 5% of market cap minimum
        
        # Check conditions
        passed = True
        reasons = []
        
        if liquidity_usd < min_liquidity:
            passed = False
            reasons.append(f"Liquidity too low: ${liquidity_usd:,.2f} < ${min_liquidity:,.2f}")
        
        if liquidity_ratio < min_liquidity_ratio:
            passed = False
            reasons.append(f"Liquidity to MC ratio too low: {liquidity_ratio:.2%} < {min_liquidity_ratio:.2%}")
        
        if passed and not reasons:
            reasons.append(f"Liquidity acceptable: ${liquidity_usd:,.2f} ({liquidity_ratio:.2%} of MC)")
        
        return {
            "name": "Liquidity Filter",
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "liquidity_usd": liquidity_usd,
                "market_cap": market_cap,
                "liquidity_ratio": liquidity_ratio
            }
        }
    
    def _apply_volume_filter(self, volume_metrics: Dict, market_metrics: Dict) -> Dict:
        """
        Filter based on volume metrics
        Returns filter result with pass/fail status and reason
        """
        volume_24h = volume_metrics.get("volume_24h", 0)
        volume_ratio = volume_metrics.get("volume_market_cap_ratio", 0)
        buy_sell_ratio = volume_metrics.get("buy_sell_ratio", 1)
        
        # Define thresholds based on requirements
        min_volume = 5000  # $5K minimum 24h volume
        min_volume_ratio = 0.03  # 3% of market cap minimum
        min_buy_sell_ratio = 0.5  # At least 1 buy for every 2 sells
        max_buy_sell_ratio = 5.0  # At most 5 buys for every sell
        
        # Check conditions
        passed = True
        reasons = []
        
        if volume_24h < min_volume:
            passed = False
            reasons.append(f"24h volume too low: ${volume_24h:,.2f} < ${min_volume:,.2f}")
        
        if volume_ratio < min_volume_ratio:
            passed = False
            reasons.append(f"Volume to MC ratio too low: {volume_ratio:.2%} < {min_volume_ratio:.2%}")
        
        if buy_sell_ratio < min_buy_sell_ratio:
            passed = False
            reasons.append(f"Buy/Sell ratio too low: {buy_sell_ratio:.2f} < {min_buy_sell_ratio}")
        
        if buy_sell_ratio > max_buy_sell_ratio:
            passed = False
            reasons.append(f"Buy/Sell ratio too high: {buy_sell_ratio:.2f} > {max_buy_sell_ratio}")
        
        if passed and not reasons:
            reasons.append(f"Volume metrics acceptable: ${volume_24h:,.2f} with {buy_sell_ratio:.2f} buy/sell ratio")
        
        return {
            "name": "Volume Filter",
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "volume_24h": volume_24h,
                "volume_ratio": volume_ratio,
                "buy_sell_ratio": buy_sell_ratio
            }
        }
    
    def _apply_technical_filter(self, technical_indicators: Dict) -> Dict:
        """
        Filter based on technical indicators
        Returns filter result with pass/fail status and reason
        """
        rsi_14 = technical_indicators.get("rsi_14")
        macd = technical_indicators.get("macd")
        macd_signal = technical_indicators.get("macd_signal")
        macd_histogram = technical_indicators.get("macd_histogram")
        
        # Define thresholds based on requirements
        min_rsi = 30  # Minimum RSI (oversold)
        max_rsi = 70  # Maximum RSI (overbought)
        
        # Check conditions
        passed = True
        reasons = []
        
        # Skip technical checks if data is missing
        if rsi_14 is None:
            reasons.append("Insufficient data for RSI calculation")
            return {
                "name": "Technical Filter",
                "passed": True,  # Pass by default if data is missing
                "reasons": reasons,
                "metrics": {
                    "rsi_14": None,
                    "macd": None,
                    "macd_signal": None,
                    "macd_histogram": None
                }
            }
        
        # Check RSI
        if rsi_14 > max_rsi:
            passed = False
            reasons.append(f"RSI overbought: {rsi_14:.2f} > {max_rsi}")
        
        # Check MACD (if available)
        if macd is not None and macd_signal is not None:
            # MACD crossing above signal line is bullish
            if macd > macd_signal and macd_histogram > 0:
                reasons.append(f"MACD bullish crossover: {macd:.6f} > {macd_signal:.6f}")
            # MACD crossing below signal line is bearish
            elif macd < macd_signal and macd_histogram < 0:
                reasons.append(f"MACD bearish crossover: {macd:.6f} < {macd_signal:.6f}")
        
        if passed and not reasons:
            reasons.append(f"Technical indicators acceptable: RSI {rsi_14:.2f}")
        
        return {
            "name": "Technical Filter",
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "rsi_14": rsi_14,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram
            }
        }
    
    def _apply_security_filter(self, security_metrics: Dict, basic_info: Dict) -> Dict:
        """
        Filter based on security metrics
        Returns filter result with pass/fail status and reason
        """
        is_pump_fun = security_metrics.get("is_pump_fun", False)
        is_at_ath = security_metrics.get("is_at_ath", False)
        age_in_days = basic_info.get("age_in_days", 0)
        
        # Define thresholds based on requirements
        min_age_for_ath_buy = 3  # Minimum age in days to buy at ATH
        
        # Check conditions
        passed = True
        reasons = []
        
        # Check if token is from pump.fun (high risk)
        if is_pump_fun:
            passed = False
            reasons.append("Token is from pump.fun platform (high risk)")
        
        # Check if token is at ATH and too new
        if is_at_ath and age_in_days < min_age_for_ath_buy:
            passed = False
            reasons.append(f"Token is at ATH and too new: {age_in_days:.1f} days < {min_age_for_ath_buy} days")
        
        if passed and not reasons:
            reasons.append("No security red flags detected")
        
        return {
            "name": "Security Filter",
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "is_pump_fun": is_pump_fun,
                "is_at_ath": is_at_ath,
                "age_in_days": age_in_days
            }
        }
    
    def _apply_orderblock_filter(self, orderblock_analysis: Dict) -> Dict:
        """
        Filter based on orderblock analysis
        Returns filter result with pass/fail status and reason
        """
        current_in_orderblock = orderblock_analysis.get("current_in_orderblock", False)
        orderblock_count = orderblock_analysis.get("orderblock_count", 0)
        
        # Define thresholds based on requirements
        # No specific thresholds, but being in an orderblock is positive
        
        # Check conditions
        passed = True
        reasons = []
        
        if current_in_orderblock:
            reasons.append("Price is currently in an identified orderblock (support zone)")
        else:
            # Not being in an orderblock is not a failure condition
            reasons.append("Price is not currently in an identified orderblock")
        
        return {
            "name": "Orderblock Filter",
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "current_in_orderblock": current_in_orderblock,
                "orderblock_count": orderblock_count
            }
        }
    
    def _apply_age_filter(self, basic_info: Dict) -> Dict:
        """
        Filter based on token age
        Returns filter result with pass/fail status and reason
        """
        age_in_days = basic_info.get("age_in_days", 0)
        
        # Define thresholds based on requirements
        min_age = 0.1  # Minimum age in days (2.4 hours)
        max_age = 90  # Maximum age in days for new coin consideration
        
        # Check conditions
        passed = True
        reasons = []
        
        if age_in_days < min_age:
            passed = False
            reasons.append(f"Token too new: {age_in_days:.2f} days < {min_age} days")
        
        if age_in_days > max_age:
            passed = False
            reasons.append(f"Token too old for new coin strategy: {age_in_days:.2f} days > {max_age} days")
        
        if passed and not reasons:
            reasons.append(f"Token age acceptable: {age_in_days:.2f} days")
        
        return {
            "name": "Age Filter",
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "age_in_days": age_in_days
            }
        }
    
    def _apply_price_
(Content truncated due to size limit. Use line ranges to read in chunks)