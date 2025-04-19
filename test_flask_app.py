import unittest
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from ui.app import app

class TestFlaskApp(unittest.TestCase):
    """Test cases for the Flask application"""
    
    def setUp(self):
        """Set up test environment"""
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    def test_routes(self):
        """Test that all routes return 200 status code"""
        routes = ['/', '/dashboard', '/signals', '/portfolio', '/tokens', '/settings']
        
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200)
    
    def test_api_dashboard_summary(self):
        """Test the dashboard summary API endpoint"""
        response = self.client.get('/api/dashboard/summary')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        self.assertIn('portfolio', data['data'])
        self.assertIn('signals', data['data'])
        self.assertIn('active_trades', data['data'])
    
    def test_api_signals(self):
        """Test the signals API endpoint"""
        response = self.client.get('/api/signals')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_api_signals_with_filters(self):
        """Test the signals API endpoint with filters"""
        response = self.client.get('/api/signals?action=buy&min_confidence=70&market_cap=small&risk_level=moderate')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_api_tokens(self):
        """Test the tokens API endpoint"""
        response = self.client.get('/api/tokens')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_api_tokens_with_filters(self):
        """Test the tokens API endpoint with filters"""
        response = self.client.get('/api/tokens?market_cap=micro&change=positive&signal=buy&risk_level=low&search=bonk')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_api_portfolio(self):
        """Test the portfolio API endpoint"""
        response = self.client.get('/api/portfolio')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_api_settings_get(self):
        """Test getting settings via API"""
        response = self.client.get('/api/settings')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        self.assertIn('trading', data['data'])
        self.assertIn('data_collection', data['data'])
        self.assertIn('security', data['data'])
        self.assertIn('notifications', data['data'])
    
    def test_api_settings_update(self):
        """Test updating settings via API"""
        new_settings = {
            "trading": {
                "risk_level": "low",
                "portfolio_allocation_per_trade": 3
            }
        }
        
        response = self.client.post('/api/settings', 
                                   data=json.dumps(new_settings),
                                   content_type='application/json')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('message', data)
    
    def test_api_trade_execution(self):
        """Test trade execution API endpoint"""
        trade_data = {
            "token": "BONK",
            "action": "BUY",
            "amount_usd": 300
        }
        
        response = self.client.post('/api/trade', 
                                   data=json.dumps(trade_data),
                                   content_type='application/json')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_api_token_details(self):
        """Test token details API endpoint"""
        response = self.client.get('/api/token/BONK')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        self.assertIn('details', data['data'])
        self.assertIn('security', data['data'])
        self.assertIn('signal', data['data'])
    
    def test_api_refresh(self):
        """Test data refresh API endpoint"""
        response = self.client.post('/api/refresh')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('timestamp', data)

if __name__ == '__main__':
    unittest.main()
