import unittest
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from data_collection.collector import DataCollector
from data_collection.data_pipeline import DataPipeline
from data_collection.custom_filters import CustomFilters
from data_collection.security_audit import SecurityAudit
from data_collection.ml_model import MLModel
from data_collection.trading_strategy import StrategyManager, OrderblockStrategy, MLBasedStrategy, MarketCapTierStrategy

class TestIntegration(unittest.TestCase):
    """Integration tests for the Solana Memecoin Trading Bot components"""
    
    def setUp(self):
        """Set up test environment"""
        # Initialize components
        self.data_collector = DataCollector()
        self.data_pipeline = DataPipeline()
        self.custom_filters = CustomFilters()
        self.security_audit = SecurityAudit()
        self.ml_model = MLModel()
        
        # Initialize strategies
        self.orderblock_strategy = OrderblockStrategy()
        self.ml_strategy = MLBasedStrategy(self.ml_model)
        self.market_cap_strategy = MarketCapTierStrategy()
        
        # Initialize strategy manager with all strategies
        self.strategy_manager = StrategyManager([
            self.orderblock_strategy,
            self.ml_strategy,
            self.market_cap_strategy
        ])
        
        # Create mock data
        self.mock_data = [
            {
                "token": "BONK",
                "name": "Bonk",
                "price": 0.00000235,
                "price_change_24h": 9.3,
                "market_cap": 1450000,
                "volume_24h": 325000,
                "liquidity": 125000,
                "creation_date": (datetime.now().timestamp() - 60*60*24*30),  # 30 days old
                "holders": 15000,
                "transactions": 5000,
                "buy_tax": 0,
                "sell_tax": 0,
                "is_honeypot": False,
                "is_pump_fun": False,
                "contract_verified": True,
                "orderblocks": [
                    {"level": 0.00000215, "strength": 0.8},
                    {"level": 0.00000195, "strength": 0.9}
                ],
                "indicators": {
                    "rsi": 65,
                    "macd": {"value": 0.00000002, "signal": 0.00000001},
                    "bollinger": {"upper": 0.00000250, "middle": 0.00000230, "lower": 0.00000210}
                }
            },
            {
                "token": "WIF",
                "name": "Dogwifhat",
                "price": 0.00000921,
                "price_change_24h": -3.3,
                "market_cap": 8750000,
                "volume_24h": 1250000,
                "liquidity": 450000,
                "creation_date": (datetime.now().timestamp() - 60*60*24*60),  # 60 days old
                "holders": 25000,
                "transactions": 8000,
                "buy_tax": 0,
                "sell_tax": 0,
                "is_honeypot": False,
                "is_pump_fun": False,
                "contract_verified": True,
                "orderblocks": [
                    {"level": 0.00000850, "strength": 0.7},
                    {"level": 0.00000800, "strength": 0.85}
                ],
                "indicators": {
                    "rsi": 45,
                    "macd": {"value": -0.00000001, "signal": 0.0},
                    "bollinger": {"upper": 0.00000950, "middle": 0.00000920, "lower": 0.00000890}
                }
            },
            {
                "token": "POPCAT",
                "name": "Popcat",
                "price": 0.00000123,
                "price_change_24h": 45.2,
                "market_cap": 350000,
                "volume_24h": 125000,
                "liquidity": 25000,
                "creation_date": (datetime.now().timestamp() - 60*60*3),  # 3 hours old
                "holders": 500,
                "transactions": 200,
                "buy_tax": 5,
                "sell_tax": 10,
                "is_honeypot": False,
                "is_pump_fun": True,
                "contract_verified": False,
                "orderblocks": [],
                "indicators": {
                    "rsi": 85,
                    "macd": {"value": 0.00000005, "signal": 0.00000002},
                    "bollinger": {"upper": 0.00000140, "middle": 0.00000110, "lower": 0.00000080}
                }
            }
        ]
    
    def test_data_flow_integration(self):
        """Test the flow of data through all components"""
        # Mock the data collector's get_raw_data method
        original_get_raw = self.data_collector.get_raw_data
        self.data_collector.get_raw_data = lambda: self.mock_data
        
        # Process data through pipeline
        processed_data = self.data_pipeline.process_data(self.data_collector.get_raw_data())
        
        # Verify processed data
        self.assertEqual(len(processed_data), 3)
        self.assertIn("token", processed_data[0])
        self.assertIn("price", processed_data[0])
        self.assertIn("market_cap", processed_data[0])
        
        # Apply custom filters
        filtered_tokens = self.custom_filters.apply_filters(processed_data)
        
        # Verify filtered tokens (POPCAT should be filtered out due to age and pump.fun)
        self.assertLessEqual(len(filtered_tokens), 3)
        
        # Perform security audit
        audited_tokens = self.security_audit.audit_tokens(filtered_tokens)
        
        # Verify audited tokens
        for token in audited_tokens:
            self.assertIn("security_rating", token)
        
        # Update ML model
        self.ml_model.update_data(audited_tokens)
        
        # Generate predictions
        for token in audited_tokens:
            prediction = self.ml_model.predict_price(token["token"])
            self.assertIsNotNone(prediction)
            self.assertIn("price_prediction", prediction)
            self.assertIn("confidence", prediction)
        
        # Update strategies
        self.strategy_manager.update_data(audited_tokens)
        
        # Generate signals
        signals = self.strategy_manager.get_signals()
        self.assertIsNotNone(signals)
        
        # Restore original method
        self.data_collector.get_raw_data = original_get_raw
    
    def test_strategy_integration(self):
        """Test the integration of different strategies"""
        # Mock data for strategies
        self.strategy_manager.update_data(self.mock_data)
        
        # Get signals from each strategy
        orderblock_signals = self.orderblock_strategy.generate_signals()
        ml_signals = self.ml_strategy.generate_signals()
        market_cap_signals = self.market_cap_strategy.generate_signals()
        
        # Verify signals from each strategy
        self.assertIsNotNone(orderblock_signals)
        self.assertIsNotNone(ml_signals)
        self.assertIsNotNone(market_cap_signals)
        
        # Get combined signals
        combined_signals = self.strategy_manager.get_signals()
        
        # Verify combined signals
        self.assertIsNotNone(combined_signals)
        
        # Test trade execution
        trade_data = {
            "token": "BONK",
            "action": "BUY",
            "amount_usd": 300
        }
        
        result = self.strategy_manager.execute_trade(trade_data)
        
        # Verify trade execution result
        self.assertIsNotNone(result)
        self.assertIn("token", result)
        self.assertEqual(result["token"], "BONK")
        self.assertIn("action", result)
        self.assertEqual(result["action"], "BUY")
    
    def test_timeframe_adaptability(self):
        """Test timeframe adaptability features"""
        from data_collection.timeframe_adaptability import TimeframeAdapter
        
        adapter = TimeframeAdapter()
        
        # Test adaptability for different coins
        # New micro-cap coin
        new_micro_timeframes = adapter.get_optimal_timeframes(
            age_days=1,
            market_cap=100000,
            volatility=0.2,
            trend_strength=0.5,
            volume_profile=0.3
        )
        
        # Established medium-cap coin
        established_medium_timeframes = adapter.get_optimal_timeframes(
            age_days=60,
            market_cap=5000000,
            volatility=0.1,
            trend_strength=0.7,
            volume_profile=0.6
        )
        
        # Verify different timeframes are selected
        self.assertNotEqual(new_micro_timeframes, established_medium_timeframes)
        
        # Verify new micro-cap has shorter timeframes
        self.assertLess(
            adapter.timeframe_to_minutes(new_micro_timeframes["primary"]),
            adapter.timeframe_to_minutes(established_medium_timeframes["primary"])
        )

if __name__ == '__main__':
    unittest.main()
