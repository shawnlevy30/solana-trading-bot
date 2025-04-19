"""
Data Collection Module for Solana Memecoin Trading Bot

This module handles the collection of data from various APIs for Solana memecoins.
It includes functions for fetching historical and real-time data, with adaptability
based on coin age and market cap.
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Import API configuration
from .api_config import (
    DEXSCREENER_BASE_URL, DEXSCREENER_ENDPOINTS, DEXSCREENER_RATE_LIMITS,
    BIRDEYE_BASE_URL, BIRDEYE_ENDPOINTS, BIRDEYE_CHAIN_ID,
    GECKOTERMINAL_BASE_URL, GECKOTERMINAL_ENDPOINTS, GECKOTERMINAL_NETWORK,
    TIMEFRAMES, MARKET_CAP_TIERS, DATA_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_collection")

class RateLimiter:
    """Rate limiter to prevent exceeding API rate limits"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.call_timestamps = []
    
    def wait_if_needed(self):
        """Wait if we're about to exceed the rate limit"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.call_timestamps = [ts for ts in self.call_timestamps 
                               if current_time - ts < 60]
        
        # If we've reached the limit, wait until we can make another call
        if len(self.call_timestamps) >= self.calls_per_minute:
            oldest_timestamp = min(self.call_timestamps)
            sleep_time = 60 - (current_time - oldest_timestamp)
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add current timestamp to the list
        self.call_timestamps.append(time.time())

class APIClient:
    """Base API client with common functionality"""
    
    def __init__(self):
        self.session = requests.Session()
        self.cache = {}
        self.rate_limiters = {}
    
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, rate_limiter_key: str = "default") -> Dict:
        """Make an API request with rate limiting and retries"""
        
        # Initialize rate limiter if it doesn't exist
        if rate_limiter_key not in self.rate_limiters:
            # Default to 60 requests per minute if not specified
            rate_limit = DEXSCREENER_RATE_LIMITS.get(rate_limiter_key, 60)
            self.rate_limiters[rate_limiter_key] = RateLimiter(rate_limit)
        
        # Apply rate limiting
        self.rate_limiters[rate_limiter_key].wait_if_needed()
        
        # Try the request with retries
        for attempt in range(DATA_CONFIG["retry_attempts"]):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=headers,
                    timeout=DATA_CONFIG["timeout"]
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{DATA_CONFIG['retry_attempts']}): {e}")
                if attempt < DATA_CONFIG["retry_attempts"] - 1:
                    time.sleep(DATA_CONFIG["retry_delay"])
                else:
                    logger.error(f"All retry attempts failed for URL: {url}")
                    raise
    
    def _get_cached_or_fetch(self, cache_key: str, fetch_func, *args, **kwargs) -> Dict:
        """Get data from cache or fetch it if not available or expired"""
        current_time = time.time()
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < DATA_CONFIG["cache_expiry"]:
                logger.debug(f"Using cached data for {cache_key}")
                return cached_data
        
        # Fetch fresh data
        data = fetch_func(*args, **kwargs)
        self.cache[cache_key] = (data, current_time)
        return data

class DexScreenerAPI(APIClient):
    """Client for interacting with DexScreener API"""
    
    def __init__(self):
        super().__init__()
        self.base_url = DEXSCREENER_BASE_URL
    
    def get_token_profiles(self) -> Dict:
        """Get the latest token profiles"""
        url = f"{self.base_url}{DEXSCREENER_ENDPOINTS['token_profiles']}"
        return self._make_request(url, rate_limiter_key="token_profiles")
    
    def get_token_boosts_latest(self) -> Dict:
        """Get the latest boosted tokens"""
        url = f"{self.base_url}{DEXSCREENER_ENDPOINTS['token_boosts_latest']}"
        return self._make_request(url, rate_limiter_key="token_boosts")
    
    def get_token_boosts_top(self) -> Dict:
        """Get tokens with most active boosts"""
        url = f"{self.base_url}{DEXSCREENER_ENDPOINTS['token_boosts_top']}"
        return self._make_request(url, rate_limiter_key="token_boosts")
    
    def get_token_orders(self, chain_id: str, token_address: str) -> Dict:
        """Check orders paid for a token"""
        endpoint = DEXSCREENER_ENDPOINTS['token_orders'].format(
            chain_id=chain_id, 
            token_address=token_address
        )
        url = f"{self.base_url}{endpoint}"
        cache_key = f"orders_{chain_id}_{token_address}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, rate_limiter_key="tokens")
    
    def get_pair_data(self, chain_id: str, pair_id: str) -> Dict:
        """Get data for a specific trading pair"""
        endpoint = DEXSCREENER_ENDPOINTS['pairs'].format(
            chain_id=chain_id, 
            pair_id=pair_id
        )
        url = f"{self.base_url}{endpoint}"
        cache_key = f"pair_{chain_id}_{pair_id}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, rate_limiter_key="pairs")
    
    def search_pairs(self, query: str) -> Dict:
        """Search for trading pairs"""
        url = f"{self.base_url}{DEXSCREENER_ENDPOINTS['search_pairs']}"
        params = {"q": query}
        cache_key = f"search_{query}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, params, rate_limiter_key="search")
    
    def get_token_data(self, chain_id: str, token_address: str) -> Dict:
        """Get data for a specific token"""
        endpoint = DEXSCREENER_ENDPOINTS['tokens'].format(
            chain_id=chain_id, 
            token_address=token_address
        )
        url = f"{self.base_url}{endpoint}"
        cache_key = f"token_{chain_id}_{token_address}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, rate_limiter_key="tokens")

class BirdeyeAPI(APIClient):
    """Client for interacting with Birdeye API"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.base_url = BIRDEYE_BASE_URL
        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key} if api_key else {}
    
    def get_supported_networks(self) -> Dict:
        """Get a list of all supported networks"""
        url = f"{self.base_url}{BIRDEYE_ENDPOINTS['networks']}"
        return self._make_request(url, headers=self.headers)
    
    def get_token_price(self, token_address: str) -> Dict:
        """Get current price for a token"""
        url = f"{self.base_url}{BIRDEYE_ENDPOINTS['price']}"
        params = {"address": token_address, "chain": BIRDEYE_CHAIN_ID}
        cache_key = f"price_{token_address}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, params, self.headers)
    
    def get_multiple_token_prices(self, token_addresses: List[str]) -> Dict:
        """Get current prices for multiple tokens"""
        url = f"{self.base_url}{BIRDEYE_ENDPOINTS['price_multiple']}"
        params = {"list": ",".join(token_addresses), "chain": BIRDEYE_CHAIN_ID}
        cache_key = f"multi_price_{','.join(token_addresses)}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, params, self.headers)
    
    def get_historical_prices(self, token_address: str, timeframe: str = "1H", limit: int = 100) -> Dict:
        """Get historical price data for a token"""
        url = f"{self.base_url}{BIRDEYE_ENDPOINTS['price_historical']}"
        params = {
            "address": token_address,
            "chain": BIRDEYE_CHAIN_ID,
            "type": timeframe,
            "limit": limit
        }
        cache_key = f"history_{token_address}_{timeframe}_{limit}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, params, self.headers)
    
    def get_token_trades(self, token_address: str, limit: int = 100) -> Dict:
        """Get recent trades for a token"""
        url = f"{self.base_url}{BIRDEYE_ENDPOINTS['trades_token']}"
        params = {
            "address": token_address,
            "chain": BIRDEYE_CHAIN_ID,
            "limit": limit
        }
        # Don't cache trades as they're real-time data
        return self._make_request(url, params, self.headers)
    
    def get_ohlcv_data(self, token_address: str, timeframe: str = "1H", limit: int = 100) -> Dict:
        """Get OHLCV data for a token"""
        url = f"{self.base_url}{BIRDEYE_ENDPOINTS['ohlcv']}"
        params = {
            "address": token_address,
            "chain": BIRDEYE_CHAIN_ID,
            "type": timeframe,
            "limit": limit
        }
        cache_key = f"ohlcv_{token_address}_{timeframe}_{limit}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, params, self.headers)

class GeckoTerminalAPI(APIClient):
    """Client for interacting with GeckoTerminal API"""
    
    def __init__(self):
        super().__init__()
        self.base_url = GECKOTERMINAL_BASE_URL
    
    def get_networks(self) -> Dict:
        """Get all supported networks"""
        url = f"{self.base_url}{GECKOTERMINAL_ENDPOINTS['networks']}"
        return self._make_request(url)
    
    def get_dexes(self) -> Dict:
        """Get all DEXes for Solana"""
        endpoint = GECKOTERMINAL_ENDPOINTS['dexes'].format(
            network=GECKOTERMINAL_NETWORK
        )
        url = f"{self.base_url}{endpoint}"
        cache_key = "solana_dexes"
        return self._get_cached_or_fetch(cache_key, self._make_request, url)
    
    def get_pools(self, dex: str, page: int = 1) -> Dict:
        """Get pools for a specific DEX"""
        endpoint = GECKOTERMINAL_ENDPOINTS['pools'].format(
            network=GECKOTERMINAL_NETWORK,
            dex=dex
        )
        url = f"{self.base_url}{endpoint}"
        params = {"page": page}
        cache_key = f"pools_{dex}_page{page}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url, params)
    
    def get_pool_info(self, pool_address: str) -> Dict:
        """Get detailed information for a specific pool"""
        endpoint = GECKOTERMINAL_ENDPOINTS['pool_info'].format(
            network=GECKOTERMINAL_NETWORK,
            pool_address=pool_address
        )
        url = f"{self.base_url}{endpoint}"
        cache_key = f"pool_info_{pool_address}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url)
    
    def get_pool_ohlcv(self, pool_address: str, timeframe: str = "day") -> Dict:
        """Get OHLCV data for a specific pool"""
        endpoint = GECKOTERMINAL_ENDPOINTS['pool_ohlcv'].format(
            network=GECKOTERMINAL_NETWORK,
            pool_address=pool_address,
            timeframe=timeframe
        )
        url = f"{self.base_url}{endpoint}"
        cache_key = f"pool_ohlcv_{pool_address}_{timeframe}"
        return self._get_cached_or_fetch(cache_key, self._make_request, url)
    
    def get_pool_trades(self, pool_address: str) -> Dict:
        """Get recent trades for a specific pool"""
        endpoint = GECKOTERMINAL_ENDPOINTS['pool_trades'].format(
            network=GECKOTERMINAL_NETWORK,
            pool_address=pool_address
        )
        url = f"{self.base_url}{endpoint}"
        # Don't cache trades as they're real-time data
        return self._make_request(url)
    
    def get_trending_pools(self) -> Dict:
        """Get trending pools on Solana"""
        endpoint = GECKOTERMINAL_ENDPOINTS['trending_pools'].format(
            network=GECKOTERMINAL_NETWORK
        )
        url = f"{self.base_url}{endpoint}"
        # Refresh trending data frequently
        return self._make_request(url)

class DataCollector:
    """Main class for collecting and processing data from multiple sources"""
    
    def __init__(self, birdeye_api_key: Optional[str] = None):
        self.dexscreener = DexScreenerAPI()
        self.birdeye = BirdeyeAPI(api_key=birdeye_api_key)
        self.geckoterminal = GeckoTerminalAPI()
        logger.info("Data collector initialized")
    
    def determine_coin_age_category(self, creation_timestamp: int) -> str:
        """Determine the age category of a coin based on its creation timestamp"""
        current_time = int(time.time())
        age_in_seconds = current_time - creation_timestamp
        
        if age_in_seconds < 86400:  # Less than 24 hours
            return "new"
        elif age_in_seconds < 604800:  # Less than 7 days
            return "recent"
        elif age_in_seconds < 2592000:  # Less than 30 days
            return "established"
        else:  # More than 30 days
            return "mature"
    
    def determine_market_cap_tier(self, market_cap: float) -> str:
        """Determine the market cap tier of a coin"""
        for tier, (min_cap, max_cap) in MARKET_CAP_TIERS.items():
            if min_cap <= market_cap < max_cap:
                return tier
        return "micro"  # Default to micro if something goes wrong
    
    def get_appropriate_timeframe(self, coin_age_category: str) -> str:
        """Get the appropriate timeframe based on coin age"""
        return DATA_CONFIG["age_timeframe_mapping"][coin_age_category]["primary_timeframe"]
    
    def get_appropriate_history_length(self, market_cap_tier: str) -> int:
        """Get the appropriate history length based on market cap tier"""
        return DATA_CONFIG["market_cap_history_mapping"][market_cap_tier]
    
    def get_token_data(self, token_address: str, chain_id: str = "solana") -> Dict:
        """Get comprehensive token data from multiple sources"""
        logger.info(f"Fetching data for token {token_address} on {chain_id}")
        
        # Get basic token data from DexScreener
        dexscreener_data = self.dexscreener.get_token_data(chain_id, token_address)
        
        # Get price data from Birdeye
        try:
            birdeye_price = self.birdeye.get_token_price(token_address)
        except Exception as e:
            logger.warning(f"Failed to get Birdeye price data: {e}")
            birdeye_price = {}
        
        # Determine coin age category if creation timestamp is available
        creation_timestamp = dexscreener_data.get("pairs", [{}])[0].get("pairCreatedAt", 0)
        coin_age_category = self.determine_coin_age_category(creation_timestamp)
        
        # Determine market cap tier
        market_cap = dexscreener_data.get("pairs", [{}])[0].get("marketCap", 0)
        market_cap_tier = self.determine_market_cap_tier(market_cap)
        
        # Determine appropriate timeframe and history length
        timeframe = self.get_appropriate_timeframe(coin_age_category)
        history_length = self.get_appropriate_history_length(market_cap_tier)
        
        # Get historical price data with adaptive parameters
        try:
        
(Content truncated due to size limit. Use line ranges to read in chunks)