import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from data_collection.collector import DataCollector
from data_collection.data_pipeline import DataPipeline
from data_collection.custom_filters import CustomFilters
from data_collection.security_audit import SecurityAudit
from data_collection.ml_model import MLModel
from data_collection.trading_strategy import StrategyManager, OrderblockStrategy, MLBasedStrategy, MarketCapTierStrategy

class TradingBot:
    """
    Main class for the Solana Memecoin Trading Bot that integrates all components
    """
    
    def __init__(self, config_path=None):
        """Initialize the trading bot with all components"""
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.logger.info("Initializing trading bot components...")
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
        
        self.logger.info("Trading bot initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('trading_bot')
    
    def load_config(self, config_path=None):
        """Load configuration from file or use default"""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'config', 
                'app_config.json'
            )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Configuration loaded from {config_path}")
        else:
            # Default configuration
            config = {
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
                    "email_address": "user@example.com",
                    "notify_new_signals": True,
                    "notify_trade_execution": True,
                    "notify_stop_loss": True,
                    "notify_take_profit": True
                }
            }
            
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                self.logger.info(f"Default configuration created at {config_path}")
        
        return config
    
    def update_config(self, new_config):
        """Update configuration with new values"""
        for category in new_config:
            if category in self.config:
                for setting in new_config[category]:
                    if setting in self.config[category]:
                        self.config[category][setting] = new_config[category][setting]
        
        # Save updated config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'config', 
            'app_config.json'
        )
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            self.logger.info("Configuration updated successfully")
        
        # Apply configuration changes to components
        self.apply_config()
        
        return self.config
    
    def apply_config(self):
        """Apply configuration to all components"""
        # Apply to data collector
        self.data_collector.set_primary_source(self.config['data_collection']['primary_source'])
        self.data_collector.set_refresh_interval(self.config['data_collection']['refresh_interval'])
        self.data_collector.set_historical_data_depth(self.config['data_collection']['historical_data_depth'])
        self.data_collector.set_timeframe_adaptability(self.config['data_collection']['enable_timeframe_adaptability'])
        
        # Apply to security audit
        self.security_audit.set_enabled(self.config['security']['enable_security_audit'])
        self.security_audit.set_max_risk_level(self.config['security']['max_risk_level'])
        self.security_audit.set_block_pump_fun(self.config['security']['block_pump_fun'])
        self.security_audit.set_block_new_tokens(self.config['security']['block_new_tokens'])
        
        # Apply to strategy manager
        self.strategy_manager.set_risk_level(self.config['trading']['risk_level'])
        self.strategy_manager.set_portfolio_allocation(self.config['trading']['portfolio_allocation_per_trade'])
        self.strategy_manager.set_default_stop_loss(self.config['trading']['default_stop_loss'])
        self.strategy_manager.set_default_take_profit(self.config['trading']['default_take_profit'])
        self.strategy_manager.set_preferred_strategy(self.config['trading']['preferred_strategy'])
        self.strategy_manager.set_min_signal_confidence(self.config['trading']['min_signal_confidence'])
        
        self.logger.info("Configuration applied to all components")
    
    def start(self):
        """Start the trading bot"""
        self.logger.info("Starting trading bot...")
        
        # Initial data collection
        self.refresh_data()
        
        # Apply configuration
        self.apply_config()
        
        self.logger.info("Trading bot started successfully")
        return True
    
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot...")
        # Perform cleanup if needed
        self.logger.info("Trading bot stopped successfully")
        return True
    
    def refresh_data(self):
        """Refresh all data"""
        self.logger.info("Refreshing data...")
        
        # Collect data
        self.data_collector.refresh_data()
        
        # Process data through pipeline
        processed_data = self.data_pipeline.process_data(self.data_collector.get_raw_data())
        
        # Apply custom filters
        filtered_tokens = self.custom_filters.apply_filters(processed_data)
        
        # Perform security audit
        audited_tokens = self.security_audit.audit_tokens(filtered_tokens)
        
        # Update ML model
        self.ml_model.update_data(audited_tokens)
        
        # Update strategies
        self.strategy_manager.update_data(audited_tokens)
        
        self.logger.info("Data refresh completed")
        return {
            "timestamp": datetime.now().isoformat(),
            "tokens_processed": len(processed_data),
            "tokens_filtered": len(filtered_tokens),
            "tokens_audited": len(audited_tokens)
        }
    
    def get_dashboard_summary(self):
        """Get summary data for dashboard"""
        self.logger.info("Generating dashboard summary")
        
        # Get portfolio data
        portfolio_data = self.strategy_manager.get_portfolio_summary()
        
        # Get latest signals
        signals = self.strategy_manager.get_latest_signals(limit=3)
        
        # Get active trades
        active_trades = self.strategy_manager.get_active_trades()
        
        return {
            "portfolio": portfolio_data,
            "signals": signals,
            "active_trades": active_trades
        }
    
    def get_signals(self, action='all', min_confidence=0, market_cap='all', risk_level='all'):
        """Get trading signals with filters"""
        self.logger.info(f"Getting signals with filters: action={action}, min_confidence={min_confidence}, market_cap={market_cap}, risk_level={risk_level}")
        
        return self.strategy_manager.get_signals(
            action=action,
            min_confidence=min_confidence,
            market_cap=market_cap,
            risk_level=risk_level
        )
    
    def get_tokens(self, market_cap='all', change='all', signal='all', risk_level='all', search=''):
        """Get token list with filters"""
        self.logger.info(f"Getting tokens with filters: market_cap={market_cap}, change={change}, signal={signal}, risk_level={risk_level}, search={search}")
        
        return self.data_collector.get_tokens(
            market_cap=market_cap,
            change=change,
            signal=signal,
            risk_level=risk_level,
            search=search
        )
    
    def get_token_details(self, token_address):
        """Get detailed information about a specific token"""
        self.logger.info(f"Getting details for token: {token_address}")
        
        # Get token details from data collector
        token_details = self.data_collector.get_token_details(token_address)
        
        # Get security audit
        security_rating = self.security_audit.audit_token(token_address)
        
        # Get trading signal
        signal = self.strategy_manager.get_token_signal(token_address)
        
        return {
            "details": token_details,
            "security": security_rating,
            "signal": signal
        }
    
    def execute_trade(self, trade_data):
        """Execute a trade"""
        self.logger.info(f"Executing trade: {trade_data}")
        
        # Validate trade data
        if 'token' not in trade_data or 'action' not in trade_data:
            self.logger.error("Invalid trade data: missing token or action")
            raise ValueError("Invalid trade data: missing token or action")
        
        # Execute trade
        result = self.strategy_manager.execute_trade(trade_data)
        
        # Log result
        self.logger.info(f"Trade executed: {result}")
        
        return result
    
    def get_portfolio(self):
        """Get portfolio data"""
        self.logger.info("Getting portfolio data")
        
        return self.strategy_manager.get_portfolio()
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        self.logger.info("Generating daily report")
        
        report = self.strategy_manager.generate_report()
        
        # Save report to file
        report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f'daily_report_{datetime.now().strftime("%Y%m%d")}.json')
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
            self.logger.info(f"Daily report saved to {report_file}")
        
        return report
