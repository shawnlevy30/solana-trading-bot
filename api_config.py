"""
API Configuration for Solana Memecoin Trading Bot

This module contains configuration settings for the APIs used to collect data
for the Solana memecoin trading bot.
"""

# Dexscreener API endpoints
DEXSCREENER_BASE_URL = "https://api.dexscreener.com"
DEXSCREENER_ENDPOINTS = {
    "token_profiles": "/token-profiles/latest/v1",
    "token_boosts_latest": "/token-boosts/latest/v1",
    "token_boosts_top": "/token-boosts/top/v1",
    "token_orders": "/orders/v1/{chain_id}/{token_address}",
    "pairs": "/latest/dex/pairs/{chain_id}/{pair_id}",
    "search_pairs": "/latest/dex/search",
    "tokens": "/latest/dex/tokens/{chain_id}/{token_address}"
}
DEXSCREENER_RATE_LIMITS = {
    "token_profiles": 60,  # requests per minute
    "token_boosts": 60,    # requests per minute
    "pairs": 300,          # requests per minute
    "search": 300,         # requests per minute
    "tokens": 300          # requests per minute
}

# Birdeye API endpoints
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
BIRDEYE_ENDPOINTS = {
    "networks": "/defi/networks",
    "price": "/defi/price",
    "price_multiple": "/defi/multi_price",
    "price_historical": "/defi/price_history",
    "trades_token": "/defi/trades_token",
    "trades_pair": "/defi/trades_pair",
    "ohlcv": "/defi/ohlcv",
    "ohlcv_pair": "/defi/ohlcv_pair",
    "price_volume": "/defi/price_volume"
}
BIRDEYE_CHAIN_ID = "solana"  # Solana chain ID for Birdeye API

# GeckoTerminal API endpoints
GECKOTERMINAL_BASE_URL = "https://api.geckoterminal.com/api/v2"
GECKOTERMINAL_ENDPOINTS = {
    "networks": "/networks",
    "dexes": "/networks/{network}/dexes",
    "pools": "/networks/{network}/dexes/{dex}/pools",
    "pool_info": "/networks/{network}/pools/{pool_address}",
    "pool_ohlcv": "/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}",
    "pool_trades": "/networks/{network}/pools/{pool_address}/trades",
    "trending_pools": "/networks/{network}/trending_pools"
}
GECKOTERMINAL_NETWORK = "solana"  # Solana network ID for GeckoTerminal API

# Timeframe configurations
TIMEFRAMES = {
    "1m": 60,             # 1 minute in seconds
    "5m": 300,            # 5 minutes in seconds
    "15m": 900,           # 15 minutes in seconds
    "30m": 1800,          # 30 minutes in seconds
    "1h": 3600,           # 1 hour in seconds
    "4h": 14400,          # 4 hours in seconds
    "1d": 86400,          # 1 day in seconds
    "1w": 604800          # 1 week in seconds
}

# Market cap tiers for strategy differentiation
MARKET_CAP_TIERS = {
    "micro": (0, 100000),           # $0 - $100K
    "small": (100000, 1000000),     # $100K - $1M
    "medium": (1000000, 10000000),  # $1M - $10M
    "large": (10000000, 100000000), # $10M - $100M
    "mega": (100000000, float('inf')) # $100M+
}

# Data collection configuration
DATA_CONFIG = {
    "default_timeframe": "5m",
    "default_history_length": 200,  # Number of candles to fetch by default
    "cache_expiry": 300,            # Cache expiry time in seconds (5 minutes)
    "retry_attempts": 3,            # Number of retry attempts for API calls
    "retry_delay": 2,               # Delay between retries in seconds
    "timeout": 30,                  # Timeout for API requests in seconds
    
    # Timeframe adaptability based on coin age
    "age_timeframe_mapping": {
        "new": {  # Less than 24 hours old
            "primary_timeframe": "1m",
            "secondary_timeframes": ["5m", "15m"],
            "history_length": 60
        },
        "recent": {  # 1-7 days old
            "primary_timeframe": "5m",
            "secondary_timeframes": ["15m", "1h"],
            "history_length": 120
        },
        "established": {  # 7-30 days old
            "primary_timeframe": "15m",
            "secondary_timeframes": ["1h", "4h"],
            "history_length": 200
        },
        "mature": {  # More than 30 days old
            "primary_timeframe": "1h",
            "secondary_timeframes": ["4h", "1d"],
            "history_length": 300
        }
    },
    
    # Market cap adaptability for historical data depth
    "market_cap_history_mapping": {
        "micro": 100,    # Fetch 100 candles for micro cap coins
        "small": 150,    # Fetch 150 candles for small cap coins
        "medium": 200,   # Fetch 200 candles for medium cap coins
        "large": 300,    # Fetch 300 candles for large cap coins
        "mega": 500      # Fetch 500 candles for mega cap coins
    }
}
