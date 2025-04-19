from flask import Flask, render_template, jsonify, request
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from data_collection.collector import DataCollector
from data_collection.data_pipeline import DataPipeline
from data_collection.custom_filters import CustomFilters
from data_collection.security_audit import SecurityAudit
from data_collection.ml_model import MLModel
from data_collection.trading_strategy import StrategyManager

app = Flask(__name__)

# Initialize components
data_collector = DataCollector()
data_pipeline = DataPipeline()
custom_filters = CustomFilters()
security_audit = SecurityAudit()
ml_model = MLModel()
strategy_manager = StrategyManager()

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'app_config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
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

# Routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/signals')
def signals():
    return render_template('signals.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/tokens')
def tokens():
    return render_template('tokens.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# API Endpoints
@app.route('/api/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    """Get summary data for dashboard"""
    try:
        # Get portfolio data
        portfolio_data = {
            "value": 12458.32,
            "change": 5.2,
            "active_positions": 2,
            "unrealized_pnl": 325.75,
            "realized_pnl": 1245.50,
            "win_rate": 68
        }
        
        # Get latest signals
        signals = strategy_manager.get_latest_signals(limit=3)
        
        # Get active trades
        active_trades = strategy_manager.get_active_trades()
        
        return jsonify({
            "success": True,
            "data": {
                "portfolio": portfolio_data,
                "signals": signals,
                "active_trades": active_trades
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get trading signals with optional filters"""
    try:
        # Get query parameters
        action = request.args.get('action', 'all')
        min_confidence = float(request.args.get('min_confidence', 0))
        market_cap = request.args.get('market_cap', 'all')
        risk_level = request.args.get('risk_level', 'all')
        
        # Get signals from strategy manager
        signals = strategy_manager.get_signals(
            action=action,
            min_confidence=min_confidence,
            market_cap=market_cap,
            risk_level=risk_level
        )
        
        return jsonify({
            "success": True,
            "data": signals
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/tokens', methods=['GET'])
def get_tokens():
    """Get token list with optional filters"""
    try:
        # Get query parameters
        market_cap = request.args.get('market_cap', 'all')
        change = request.args.get('change', 'all')
        signal = request.args.get('signal', 'all')
        risk_level = request.args.get('risk_level', 'all')
        search = request.args.get('search', '')
        
        # Get tokens from data collector
        tokens = data_collector.get_tokens(
            market_cap=market_cap,
            change=change,
            signal=signal,
            risk_level=risk_level,
            search=search
        )
        
        return jsonify({
            "success": True,
            "data": tokens
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get portfolio data"""
    try:
        # Get portfolio data from strategy manager
        portfolio_data = strategy_manager.get_portfolio()
        
        return jsonify({
            "success": True,
            "data": portfolio_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    try:
        return jsonify({
            "success": True,
            "data": config
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    try:
        # Get settings from request
        new_settings = request.json
        
        # Update config
        for category in new_settings:
            if category in config:
                for setting in new_settings[category]:
                    if setting in config[category]:
                        config[category][setting] = new_settings[category][setting]
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return jsonify({
            "success": True,
            "message": "Settings updated successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    try:
        # Get trade data from request
        trade_data = request.json
        
        # Execute trade
        result = strategy_manager.execute_trade(trade_data)
        
        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/token/<token_address>', methods=['GET'])
def get_token_details(token_address):
    """Get detailed information about a specific token"""
    try:
        # Get token details from data collector
        token_details = data_collector.get_token_details(token_address)
        
        # Get security audit
        security_rating = security_audit.audit_token(token_address)
        
        # Get trading signal
        signal = strategy_manager.get_token_signal(token_address)
        
        return jsonify({
            "success": True,
            "data": {
                "details": token_details,
                "security": security_rating,
                "signal": signal
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """Refresh all data"""
    try:
        # Refresh data
        data_collector.refresh_data()
        
        return jsonify({
            "success": True,
            "message": "Data refreshed successfully",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
