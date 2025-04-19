"""
Security Audit System for Solana Memecoin Trading Bot

This module implements a comprehensive security audit system to identify potential
risks and red flags in memecoin investments, helping to protect users from scams,
rug pulls, and other security threats.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

# Import from other modules
from .collector import DataCollector
from .data_pipeline import DataPipeline
from .custom_filters import CustomFilters
from .api_config import TIMEFRAMES, MARKET_CAP_TIERS, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("security_audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("security_audit")

class SecurityAuditSystem:
    """
    Implements a comprehensive security audit system to identify potential
    risks and red flags in memecoin investments.
    """
    
    def __init__(self, data_collector: DataCollector, data_pipeline: DataPipeline, custom_filters: CustomFilters):
        self.data_collector = data_collector
        self.data_pipeline = data_pipeline
        self.custom_filters = custom_filters
        self.output_dir = os.path.join(os.getcwd(), "security_audits")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("SecurityAuditSystem initialized")
    
    def perform_security_audit(self, token_address: str, chain_id: str = "solana") -> Dict:
        """
        Perform a comprehensive security audit on a token
        Returns detailed audit results with risk assessment
        """
        logger.info(f"Performing security audit for token {token_address}")
        
        # Process token data
        processed_data = self.data_pipeline.process_token_data(token_address, chain_id)
        
        # Extract key components
        basic_info = processed_data.get("basic_info", {})
        market_metrics = processed_data.get("market_metrics", {})
        liquidity_metrics = processed_data.get("liquidity_metrics", {})
        volume_metrics = processed_data.get("volume_metrics", {})
        holder_metrics = processed_data.get("holder_metrics", {})
        
        # Perform specific security checks
        audit_results = {
            "token_info": self._audit_token_info(basic_info),
            "liquidity_analysis": self._audit_liquidity(liquidity_metrics, market_metrics),
            "holder_analysis": self._audit_holders(holder_metrics),
            "transaction_analysis": self._audit_transactions(volume_metrics),
            "contract_analysis": self._audit_contract(token_address, chain_id),
            "market_behavior": self._audit_market_behavior(market_metrics, volume_metrics),
            "social_signals": self._audit_social_signals(token_address)
        }
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(audit_results)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(audit_results, risk_level)
        
        # Compile final audit report
        audit_report = {
            "token_address": token_address,
            "chain_id": chain_id,
            "token_name": basic_info.get("name", ""),
            "token_symbol": basic_info.get("symbol", ""),
            "audit_timestamp": datetime.now().isoformat(),
            "audit_results": audit_results,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "red_flags": self._identify_red_flags(audit_results)
        }
        
        # Save audit report
        self._save_audit_report(audit_report, token_address)
        
        logger.info(f"Security audit completed for {token_address} with risk level: {risk_level}")
        return audit_report
    
    def _audit_token_info(self, basic_info: Dict) -> Dict:
        """
        Audit basic token information for suspicious patterns
        Returns audit results for token info
        """
        name = basic_info.get("name", "")
        symbol = basic_info.get("symbol", "")
        age_in_days = basic_info.get("age_in_days", 0)
        
        # Check for suspicious patterns in name/symbol
        suspicious_terms = ["safe", "moon", "elon", "doge", "shib", "inu", "pepe", "cum", "pump", "lambo", "gem"]
        suspicious_count = sum(1 for term in suspicious_terms if term.lower() in name.lower() or term.lower() in symbol.lower())
        
        # Check for excessive use of emojis
        emoji_count = 0
        for char in name + symbol:
            if ord(char) > 127:  # Simple check for non-ASCII chars (including emojis)
                emoji_count += 1
        
        # Determine risk factors
        risk_factors = []
        
        if suspicious_count >= 2:
            risk_factors.append(f"Name/symbol contains multiple suspicious terms ({suspicious_count})")
        
        if emoji_count >= 2:
            risk_factors.append(f"Name/symbol contains multiple emoji-like characters ({emoji_count})")
        
        if age_in_days < 1:
            risk_factors.append(f"Token is extremely new ({age_in_days:.2f} days old)")
        
        # Calculate risk score for this category (0-100)
        risk_score = min(100, suspicious_count * 15 + emoji_count * 10 + (1 / max(1, age_in_days)) * 30)
        
        return {
            "name": name,
            "symbol": symbol,
            "age_in_days": age_in_days,
            "suspicious_term_count": suspicious_count,
            "emoji_count": emoji_count,
            "risk_factors": risk_factors,
            "risk_score": risk_score
        }
    
    def _audit_liquidity(self, liquidity_metrics: Dict, market_metrics: Dict) -> Dict:
        """
        Audit liquidity metrics for potential rug pull risks
        Returns audit results for liquidity
        """
        liquidity_usd = liquidity_metrics.get("liquidity_usd", 0)
        liquidity_ratio = liquidity_metrics.get("liquidity_market_cap_ratio", 0)
        market_cap = market_metrics.get("market_cap", 0)
        
        # Determine risk factors
        risk_factors = []
        
        if liquidity_usd < 5000:
            risk_factors.append(f"Very low liquidity: ${liquidity_usd:,.2f}")
        
        if liquidity_ratio < 0.03:
            risk_factors.append(f"Dangerously low liquidity ratio: {liquidity_ratio:.2%}")
        elif liquidity_ratio < 0.05:
            risk_factors.append(f"Low liquidity ratio: {liquidity_ratio:.2%}")
        
        if market_cap > 1000000 and liquidity_usd < 50000:
            risk_factors.append(f"Insufficient liquidity (${liquidity_usd:,.2f}) for market cap (${market_cap:,.2f})")
        
        # Calculate risk score for this category (0-100)
        # Higher score = higher risk
        if liquidity_usd == 0:
            liquidity_risk = 100
        else:
            liquidity_risk = min(100, max(0, 100 - (liquidity_usd / 1000)))
        
        ratio_risk = min(100, max(0, (0.1 - liquidity_ratio) / 0.1 * 100))
        
        risk_score = (liquidity_risk * 0.6) + (ratio_risk * 0.4)
        
        return {
            "liquidity_usd": liquidity_usd,
            "liquidity_ratio": liquidity_ratio,
            "market_cap": market_cap,
            "risk_factors": risk_factors,
            "risk_score": risk_score
        }
    
    def _audit_holders(self, holder_metrics: Dict) -> Dict:
        """
        Audit holder distribution for concentration risks
        Returns audit results for holder analysis
        """
        # Note: In a real implementation, this would fetch actual holder data
        # from blockchain explorers. Using placeholder data for now.
        total_holders = holder_metrics.get("total_holders", 0)
        top_10_concentration = holder_metrics.get("top_10_concentration", 0)
        
        # Placeholder for demonstration
        if total_holders == 0:
            total_holders = 100  # Placeholder
            top_10_concentration = 0.85  # Placeholder (85% held by top 10)
        
        # Determine risk factors
        risk_factors = []
        
        if total_holders < 50:
            risk_factors.append(f"Very few holders: {total_holders}")
        
        if top_10_concentration > 0.9:
            risk_factors.append(f"Extreme holder concentration: {top_10_concentration:.2%} held by top 10")
        elif top_10_concentration > 0.7:
            risk_factors.append(f"High holder concentration: {top_10_concentration:.2%} held by top 10")
        
        # Calculate risk score for this category (0-100)
        holder_count_risk = min(100, max(0, (200 - total_holders) / 2))
        concentration_risk = min(100, top_10_concentration * 100)
        
        risk_score = (holder_count_risk * 0.3) + (concentration_risk * 0.7)
        
        return {
            "total_holders": total_holders,
            "top_10_concentration": top_10_concentration,
            "risk_factors": risk_factors,
            "risk_score": risk_score
        }
    
    def _audit_transactions(self, volume_metrics: Dict) -> Dict:
        """
        Audit transaction patterns for suspicious activity
        Returns audit results for transaction analysis
        """
        volume_24h = volume_metrics.get("volume_24h", 0)
        buy_transactions = volume_metrics.get("buy_transactions_24h", 0)
        sell_transactions = volume_metrics.get("sell_transactions_24h", 0)
        buy_sell_ratio = volume_metrics.get("buy_sell_ratio", 1)
        
        # Determine risk factors
        risk_factors = []
        
        if buy_transactions + sell_transactions < 10:
            risk_factors.append(f"Very low transaction count: {buy_transactions + sell_transactions} in 24h")
        
        if buy_sell_ratio > 10:
            risk_factors.append(f"Extremely high buy/sell ratio: {buy_sell_ratio:.2f}")
        elif buy_sell_ratio < 0.1:
            risk_factors.append(f"Extremely low buy/sell ratio: {buy_sell_ratio:.2f}")
        
        if volume_24h < 1000:
            risk_factors.append(f"Negligible 24h volume: ${volume_24h:,.2f}")
        
        # Calculate risk score for this category (0-100)
        transaction_count_risk = min(100, max(0, (50 - (buy_transactions + sell_transactions)) * 2))
        
        # Both extremely high and extremely low buy/sell ratios are risky
        ratio_risk = 0
        if buy_sell_ratio > 1:
            ratio_risk = min(100, (buy_sell_ratio - 1) * 10)
        else:
            ratio_risk = min(100, (1 - buy_sell_ratio) * 100)
        
        volume_risk = min(100, max(0, (10000 - volume_24h) / 100))
        
        risk_score = (transaction_count_risk * 0.3) + (ratio_risk * 0.4) + (volume_risk * 0.3)
        
        return {
            "volume_24h": volume_24h,
            "buy_transactions": buy_transactions,
            "sell_transactions": sell_transactions,
            "buy_sell_ratio": buy_sell_ratio,
            "risk_factors": risk_factors,
            "risk_score": risk_score
        }
    
    def _audit_contract(self, token_address: str, chain_id: str) -> Dict:
        """
        Audit contract code for security vulnerabilities
        Returns audit results for contract analysis
        """
        # Note: In a real implementation, this would analyze the actual contract code
        # or use a service like Solana Explorer API. Using placeholder data for now.
        
        # Placeholder for demonstration
        is_verified = True  # Placeholder
        has_mint_function = False  # Placeholder
        has_blacklist_function = False  # Placeholder
        has_fee_change_function = False  # Placeholder
        
        # Determine risk factors
        risk_factors = []
        
        if not is_verified:
            risk_factors.append("Contract code is not verified")
        
        if has_mint_function:
            risk_factors.append("Contract contains mint function (potential unlimited supply)")
        
        if has_blacklist_function:
            risk_factors.append("Contract contains blacklist function (potential trading restrictions)")
        
        if has_fee_change_function:
            risk_factors.append("Contract allows fee changes (potential for extreme fees)")
        
        # Calculate risk score for this category (0-100)
        risk_score = 0
        
        if not is_verified:
            risk_score += 70
        
        if has_mint_function:
            risk_score += 50
        
        if has_blacklist_function:
            risk_score += 30
        
        if has_fee_change_function:
            risk_score += 30
        
        risk_score = min(100, risk_score)
        
        return {
            "is_verified": is_verified,
            "has_mint_function": has_mint_function,
            "has_blacklist_function": has_blacklist_function,
            "has_fee_change_function": has_fee_change_function,
            "risk_factors": risk_factors,
            "risk_score": risk_score
        }
    
    def _audit_market_behavior(self, market_metrics: Dict, volume_metrics: Dict) -> Dict:
        """
        Audit market behavior for manipulation patterns
        Returns audit results for market behavior
        """
        price_change_5m = market_metrics.get("price_change_5m", 0)
        price_change_1h = market_metrics.get("price_change_1h", 0)
        price_change_24h = market_metrics.get("price_change_24h", 0)
        volume_spike = volume_metrics.get("volume_spike_percentage", 0)
        
        # Determine risk factors
        risk_factors = []
        
        if price_change_5m > 20:
            risk_factors.append(f"Extreme 5m price increase: {price_change_5m:.2f}%")
        
        if price_change_1h > 50:
            risk_factors.append(f"Extreme 1h price increase: {price_change_1h:.2f}%")
        
        if price_change_24h > 200:
            risk_factors.append(f"Extreme 24h price increase: {price_change_24h:.2f}%")
        
        if volume_spike > 500:
            risk_factors.append(f"Extreme volume spike: {volume_spike:.2f}%")
        
        # Check for pump and dump pattern
        if price_change_1h > 30 and price_change_5m < -5:
            risk_factors.append(f"Potential pump and dump pattern: 1h {price_change_1h:.2f}%, 5m {price_change_5m:.2f}%")
        
        # Calculate risk score for this category (0-100)
        price_5m_risk = min(100, max(0, price_change_5m * 2))
        price_1h_risk = min(100, max(0, price_change_1h))
        price_24h_risk = min(100, max(0, price_change_24h / 3))
        volume_risk = min(100, max(0, volume_spike / 10))
        
        # Pump and dump pattern gets high risk
        pump_dump_risk = 0
        if price_change_1h > 30 and price_change_5m < -5:
            pump_dump_risk = 90
        
        risk_score = (
            price_5m_risk * 0.2 +
            price_1h_risk * 0.2 +
            price_24h_risk * 0.2 +
            volume_risk * 0.2 +
            pump_dump_risk * 0.2
        )
        
        return {
            "price_change_5m": price_change_5m,
            "price_change_1h": price_change_1h,
            "price_change_24h": price_change_24h,
            "volume_spike": volume_spike,
            "risk_factors": risk_factors,
            "risk_score": risk_score
        }
    
    def _audit_social_signals(self, token_address: str) -> Dict:
        """
        Audit social media signals for potential scam indicators
        Returns audit results for social signals
        """
        # Note: In a real implementation, this would analyze social media data
        # from Twitter, Telegram, etc. Using placeholder data for now.
     
(Content truncated due to size limit. Use line ranges to read in chunks)