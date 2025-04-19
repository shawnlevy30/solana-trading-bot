import unittest
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from trading_bot import TradingBot
from data_collection.collector import DataCollector
from data_collection.data_pipeline import DataPipeline
from data_collection.custom_filters import CustomFilters
from data_collection.security_audit import SecurityAudit
from data_collection.ml_model import MLModel
from data_collection.trading_strategy import StrategyManager

class TestTradingBot(unittest.TestCase):
    """Test cases for the Solana Memecoin Trading Bot"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test config
        self.test_config = {
            "trading": {
                "risk_level": "moderate",
                "portfolio_allocation_per_trade": 5,
                "default_stop_loss": 10,
                "default_take_profit": 20,
                "preferred_strategy": "combined",
                "min_signal_confidence": 70
            },
            "data_collection": {
                "primary_source": "dexscreener",
                "refresh_interval": 5,
                "historical_data_depth": 30,
                "enable_timeframe_adaptability": True
            },
            "security": {
                "enable_security_audit": True,
                "max_risk_level": "moderate",
                "block_pump_fun": True,
                "block_new_tokens": True
            },
            "notifications": {
                "enable_email": True,
                "email_address": "test@example.com",
                "notify_new_signals": True,
                "notify_trade_execution": True,
                "notify_stop_loss": True,
                "notify_take_profit": True
            }
        }
        
        # Create test config file
        test_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        os.makedirs(test_config_dir, exist_ok=True)
        
        self.test_config_path = os.path.join(test_config_dir, 'test_config.json')
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f, indent=4)
        
        # Initialize trading bot with test config
        self.trading_bot = TradingBot(config_path=self.test_config_path)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test config file
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
    
    def test_trading_bot_initialization(self):
        """Test that the trading bot initializes correctly"""
        self.assertIsNotNone(self.trading_bot)
        self.assertIsInstance(self.trading_bot.data_collector, DataCollector)
        self.assertIsInstance(self.trading_bot.data_pipeline, DataPipeline)
        self.assertIsInstance(self.trading_bot.custom_filters, CustomFilters)
        self.assertIsInstance(self.trading_bot.security_audit, SecurityAudit)
        self.assertIsInstance(self.trading_bot.ml_model, MLModel)
        self.assertIsInstance(self.trading_bot.strategy_manager, StrategyManager)
    
    def test_config_loading(self):
        """Test that configuration is loaded correctly"""
        self.assertEqual(self.trading_bot.config['trading']['risk_level'], 'moderate')
        self.assertEqual(self.trading_bot.config['data_collection']['primary_source'], 'dexscreener')
        self.assertEqual(self.trading_bot.config['security']['enable_security_audit'], True)
        self.assertEqual(self.trading_bot.config['notifications']['email_address'], 'test@example.com')
    
    def test_config_update(self):
        """Test that configuration can be updated"""
        new_config = {
            "trading": {
                "risk_level": "low",
                "portfolio_allocation_per_trade": 3
            }
        }
        
        updated_config = self.trading_bot.update_config(new_config)
        
        self.assertEqual(updated_config['trading']['risk_level'], 'low')
        self.assertEqual(updated_config['trading']['portfolio_allocation_per_trade'], 3)
        self.assertEqual(updated_config['trading']['default_stop_loss'], 10)  # Unchanged
    
    def test_start_stop(self):
        """Test that the trading bot can start and stop"""
        start_result = self.trading_bot.start()
        self.assertTrue(start_result)
        
        stop_result = self.trading_bot.stop()
        self.assertTrue(stop_result)
    
    def test_mock_data_refresh(self):
        """Test data refresh with mock data"""
        # Mock the data collector's refresh_data method
        original_refresh = self.trading_bot.data_collector.refresh_data
        self.trading_bot.data_collector.refresh_data = lambda: None
        
        # Mock the data collector's get_raw_data method
        original_get_raw = self.trading_bot.data_collector.get_raw_data
        self.trading_bot.data_collector.get_raw_data = lambda: [
            {"token": "BONK", "price": 0.00000235, "market_cap": 1450000},
            {"token": "WIF", "price": 0.00000921, "market_cap": 8750000}
        ]
        
        # Mock the data pipeline's process_data method
        original_process = self.trading_bot.data_pipeline.process_data
        self.trading_bot.data_pipeline.process_data = lambda data: data
        
        # Mock the custom filters' apply_filters method
        original_filter = self.trading_bot.custom_filters.apply_filters
        self.trading_bot.custom_filters.apply_filters = lambda data: data
        
        # Mock the security audit's audit_tokens method
        original_audit = self.trading_bot.security_audit.audit_tokens
        self.trading_bot.security_audit.audit_tokens = lambda data: data
        
        # Test refresh_data
        refresh_result = self.trading_bot.refresh_data()
        
        self.assertIsNotNone(refresh_result)
        self.assertIn('timestamp', refresh_result)
        self.assertEqual(refresh_result['tokens_processed'], 2)
        
        # Restore original methods
        self.trading_bot.data_collector.refresh_data = original_refresh
        self.trading_bot.data_collector.get_raw_data = original_get_raw
        self.trading_bot.data_pipeline.process_data = original_process
        self.trading_bot.custom_filters.apply_filters = original_filter
        self.trading_bot.security_audit.audit_tokens = original_audit
    
    def test_mock_dashboard_summary(self):
        """Test dashboard summary with mock data"""
        # Mock the strategy manager's methods
        original_portfolio = self.trading_bot.strategy_manager.get_portfolio_summary
        self.trading_bot.strategy_manager.get_portfolio_summary = lambda: {
            "value": 12458.32,
            "change": 5.2,
            "active_positions": 2
        }
        
        original_signals = self.trading_bot.strategy_manager.get_latest_signals
        self.trading_bot.strategy_manager.get_latest_signals = lambda limit: [
            {"token": "BONK", "action": "BUY", "confidence": 0.85},
            {"token": "SAMO", "action": "HOLD", "confidence": 0.72}
        ]
        
        original_trades = self.trading_bot.strategy_manager.get_active_trades
        self.trading_bot.strategy_manager.get_active_trades = lambda: [
            {"token": "BONK", "entry_price": 0.00000215, "current_price": 0.00000235},
            {"token": "WIF", "entry_price": 0.00000952, "current_price": 0.00000921}
        ]
        
        # Test get_dashboard_summary
        summary = self.trading_bot.get_dashboard_summary()
        
        self.assertIsNotNone(summary)
        self.assertIn('portfolio', summary)
        self.assertIn('signals', summary)
        self.assertIn('active_trades', summary)
        self.assertEqual(len(summary['signals']), 2)
        self.assertEqual(len(summary['active_trades']), 2)
        
        # Restore original methods
        self.trading_bot.strategy_manager.get_portfolio_summary = original_portfolio
        self.trading_bot.strategy_manager.get_latest_signals = original_signals
        self.trading_bot.strategy_manager.get_active_trades = original_trades
    
    def test_mock_execute_trade(self):
        """Test trade execution with mock data"""
        # Mock the strategy manager's execute_trade method
        original_execute = self.trading_bot.strategy_manager.execute_trade
        self.trading_bot.strategy_manager.execute_trade = lambda trade_data: {
            "success": True,
            "token": trade_data['token'],
            "action": trade_data['action'],
            "price": 0.00000235,
            "amount": 1250000000,
            "value": 293.75,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test execute_trade
        trade_data = {
            "token": "BONK",
            "action": "BUY",
            "amount_usd": 300
        }
        
        result = self.trading_bot.execute_trade(trade_data)
        
        self.assertIsNotNone(result)
        self.assertTrue(result['success'])
        self.assertEqual(result['token'], 'BONK')
        self.assertEqual(result['action'], 'BUY')
        
        # Test invalid trade data
        with self.assertRaises(ValueError):
            self.trading_bot.execute_trade({"invalid": "data"})
        
        # Restore original method
        self.trading_bot.strategy_manager.execute_trade = original_execute
    
    def test_mock_generate_report(self):
        """Test report generation with mock data"""
        # Mock the strategy manager's generate_report method
        original_report = self.trading_bot.strategy_manager.generate_report
        self.trading_bot.strategy_manager.generate_report = lambda: {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "portfolio_value": 12458.32,
            "daily_change": 5.2,
            "trades": [
                {"token": "BONK", "action": "BUY", "price": 0.00000235, "value": 293.75},
                {"token": "CATO", "action": "SELL", "price": 0.00000789, "value": 394.50}
            ],
            "signals": [
                {"token": "BONK", "action": "BUY", "confidence": 0.85},
                {"token": "SAMO", "action": "HOLD", "confidence": 0.72},
                {"token": "CATO", "action": "SELL", "confidence": 0.78}
            ]
        }
        
        # Test generate_daily_report
        report = self.trading_bot.generate_daily_report()
        
        self.assertIsNotNone(report)
        self.assertIn('date', report)
        self.assertIn('portfolio_value', report)
        self.assertIn('trades', report)
        self.assertIn('signals', report)
        self.assertEqual(len(report['trades']), 2)
        self.assertEqual(len(report['signals']), 3)
        
        # Restore original method
        self.trading_bot.strategy_manager.generate_report = original_report

if __name__ == '__main__':
    unittest.main()
