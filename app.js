// Solana Memecoin Trading Bot UI JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initCharts();
    
    // Initialize event listeners
    initEventListeners();
    
    // Fetch initial data
    fetchDashboardData();
});

// Initialize Chart.js charts
function initCharts() {
    // Portfolio Value Chart
    const portfolioCtx = document.getElementById('portfolioChart');
    if (portfolioCtx) {
        new Chart(portfolioCtx, {
            type: 'line',
            data: {
                labels: generateDateLabels(30),
                datasets: [{
                    label: 'Portfolio Value (USD)',
                    data: generateRandomData(30, 10000, 15000),
                    borderColor: '#9945FF',
                    backgroundColor: 'rgba(153, 69, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#1E1E1E',
                        titleColor: '#FFFFFF',
                        bodyColor: '#AAAAAA',
                        borderColor: '#333333',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: '#AAAAAA',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 7
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#AAAAAA',
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }

    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart');
    if (performanceCtx) {
        new Chart(performanceCtx, {
            type: 'bar',
            data: {
                labels: ['Win Rate', 'Avg. Profit', 'Avg. Loss', 'Profit Factor'],
                datasets: [{
                    label: 'Performance Metrics',
                    data: [68, 25, 12, 2.1],
                    backgroundColor: [
                        '#14F195',
                        '#9945FF',
                        '#FF5252',
                        '#00C2FF'
                    ],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#1E1E1E',
                        titleColor: '#FFFFFF',
                        bodyColor: '#AAAAAA',
                        borderColor: '#333333',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                let value = context.raw;
                                
                                if (context.dataIndex === 0) {
                                    return `${label}: ${value}%`;
                                } else if (context.dataIndex === 1 || context.dataIndex === 2) {
                                    return `${label}: ${value}%`;
                                } else {
                                    return `${label}: ${value}`;
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: '#AAAAAA'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#AAAAAA',
                            callback: function(value) {
                                if (value % 1 === 0) {
                                    return value;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    // Token Price Chart
    const tokenPriceCtx = document.getElementById('tokenPriceChart');
    if (tokenPriceCtx) {
        new Chart(tokenPriceCtx, {
            type: 'line',
            data: {
                labels: generateTimeLabels(24),
                datasets: [{
                    label: 'Price (USD)',
                    data: generateRandomData(24, 0.00001, 0.0001),
                    borderColor: '#14F195',
                    backgroundColor: 'rgba(20, 241, 149, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#1E1E1E',
                        titleColor: '#FFFFFF',
                        bodyColor: '#AAAAAA',
                        borderColor: '#333333',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                let value = context.raw;
                                return `${label}: $${value.toFixed(8)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: '#AAAAAA',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 6
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#AAAAAA',
                            callback: function(value) {
                                return '$' + value.toFixed(8);
                            }
                        }
                    }
                }
            }
        });
    }
}

// Initialize event listeners
function initEventListeners() {
    // Tab switching
    const tabs = document.querySelectorAll('.tab-link');
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Remove active class from all tabs
                tabs.forEach(t => t.classList.remove('active'));
                
                // Add active class to clicked tab
                this.classList.add('active');
                
                // Show corresponding content
                const targetId = this.getAttribute('data-target');
                const tabContents = document.querySelectorAll('.tab-content');
                
                tabContents.forEach(content => {
                    content.style.display = 'none';
                });
                
                document.getElementById(targetId).style.display = 'block';
            });
        });
    }

    // Token search
    const searchInput = document.getElementById('tokenSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const tokenRows = document.querySelectorAll('.token-row');
            
            tokenRows.forEach(row => {
                const tokenName = row.querySelector('.token-name').textContent.toLowerCase();
                const tokenSymbol = row.querySelector('.token-symbol').textContent.toLowerCase();
                
                if (tokenName.includes(searchTerm) || tokenSymbol.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }

    // Settings form
    const settingsForm = document.getElementById('settingsForm');
    if (settingsForm) {
        settingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show saving indicator
            const saveBtn = this.querySelector('button[type="submit"]');
            const originalText = saveBtn.textContent;
            saveBtn.textContent = 'Saving...';
            saveBtn.disabled = true;
            
            // Simulate saving settings
            setTimeout(() => {
                // Show success message
                const alertContainer = document.getElementById('alertContainer');
                alertContainer.innerHTML = `
                    <div class="alert alert-success">
                        <span class="alert-icon">✓</span>
                        Settings saved successfully!
                    </div>
                `;
                
                // Reset button
                saveBtn.textContent = originalText;
                saveBtn.disabled = false;
                
                // Hide alert after 3 seconds
                setTimeout(() => {
                    alertContainer.innerHTML = '';
                }, 3000);
            }, 1000);
        });
    }

    // Refresh data button
    const refreshBtn = document.getElementById('refreshData');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            this.classList.add('rotating');
            fetchDashboardData();
            
            // Remove rotating class after animation completes
            setTimeout(() => {
                this.classList.remove('rotating');
            }, 1000);
        });
    }

    // Trade action buttons
    const tradeButtons = document.querySelectorAll('.trade-btn');
    if (tradeButtons.length > 0) {
        tradeButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                const action = this.getAttribute('data-action');
                const tokenAddress = this.getAttribute('data-token');
                const tokenName = this.getAttribute('data-name');
                
                // Show trade modal
                showTradeModal(action, tokenAddress, tokenName);
            });
        });
    }
}

// Fetch dashboard data
function fetchDashboardData() {
    // Show loading indicators
    const loadingElements = document.querySelectorAll('.loading-indicator');
    loadingElements.forEach(el => {
        el.style.display = 'block';
    });
    
    // Simulate API call
    setTimeout(() => {
        // Update dashboard data
        updateDashboardData();
        
        // Hide loading indicators
        loadingElements.forEach(el => {
            el.style.display = 'none';
        });
    }, 1500);
}

// Update dashboard data
function updateDashboardData() {
    // Update portfolio value
    const portfolioValue = document.getElementById('portfolioValue');
    if (portfolioValue) {
        portfolioValue.textContent = '$12,458.32';
    }
    
    // Update portfolio change
    const portfolioChange = document.getElementById('portfolioChange');
    if (portfolioChange) {
        portfolioChange.textContent = '+5.2%';
        portfolioChange.classList.add('positive');
    }
    
    // Update trading signals
    updateTradingSignals();
    
    // Update active trades
    updateActiveTrades();
    
    // Update token list
    updateTokenList();
}

// Update trading signals
function updateTradingSignals() {
    const signalsContainer = document.getElementById('tradingSignals');
    if (!signalsContainer) return;
    
    // Sample signals data
    const signals = [
        {
            token: 'BONK',
            name: 'Bonk',
            action: 'BUY',
            confidence: 0.85,
            reasoning: [
                'Strong upward momentum detected',
                'RSI indicates oversold condition',
                'Price is in identified orderblock'
            ],
            price: 0.00000235
        },
        {
            token: 'SAMO',
            name: 'Samoyedcoin',
            action: 'HOLD',
            confidence: 0.72,
            reasoning: [
                'Price consolidating in range',
                'Volume decreasing',
                'Waiting for clearer signal'
            ],
            price: 0.00521
        },
        {
            token: 'CATO',
            name: 'Cato',
            action: 'SELL',
            confidence: 0.78,
            reasoning: [
                'Bearish divergence on RSI',
                'Price approaching resistance',
                'Volume spike with price decrease'
            ],
            price: 0.00000789
        }
    ];
    
    // Clear container
    signalsContainer.innerHTML = '';
    
    // Add signals
    signals.forEach(signal => {
        const signalCard = document.createElement('div');
        signalCard.className = `card signal-card signal-${signal.action.toLowerCase()}`;
        
        signalCard.innerHTML = `
            <div class="signal-header">
                <div class="token-card">
                    <div class="token-icon">${signal.token.charAt(0)}</div>
                    <div class="token-info">
                        <div class="token-name">${signal.name}</div>
                        <div class="token-symbol">${signal.token}</div>
                    </div>
                </div>
                <div>
                    <div class="signal-action ${signal.action.toLowerCase()}">${signal.action}</div>
                    <div class="signal-confidence">${(signal.confidence * 100).toFixed(0)}% confidence</div>
                </div>
            </div>
            <div class="signal-reasoning">
                <ul>
                    ${signal.reasoning.map(reason => `<li>${reason}</li>`).join('')}
                </ul>
            </div>
            <div class="card-footer">
                <div>Current price: $${signal.price.toFixed(8)}</div>
                <button class="btn btn-sm ${signal.action === 'BUY' ? 'btn-primary' : signal.action === 'SELL' ? 'btn-accent' : 'btn-outline'} tra
(Content truncated due to size limit. Use line ranges to read in chunks)