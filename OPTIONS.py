import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import plotly.graph_objects as go
import ssl
import urllib.request

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Helper functions
def get_stock_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Adj Close']

def get_ytd_volatility(tickers):
    volatilities = {}
    start_date = datetime(datetime.today().year, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    for ticker in tickers:
        data = yf.Ticker(ticker)
        hist = data.history(start=start_date, end=end_date)
        daily_returns = hist['Close'].pct_change().dropna()
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        volatilities[ticker] = annualized_volatility
    return volatilities

def get_options_data(ticker):
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    if not options_dates:
        return None
    nearest_expiry = options_dates[0]
    options_chain = stock.option_chain(nearest_expiry)
    return options_chain

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
    return price

def calculate_var(returns, confidence_level=0.95):
    var = np.percentile(returns, (1-confidence_level)*100)
    return var

def optimize_portfolio(returns, risk_free_rate):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    results = np.zeros((4, num_assets))
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []
    
    num_portfolios = 10000
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (returns - risk_free_rate) / volatility
        port_returns.append(returns)
        port_volatility.append(volatility)
        sharpe_ratio.append(sharpe)
        stock_weights.append(weights)
        
    max_index = np.argmax(sharpe_ratio)
    return stock_weights[max_index], port_returns[max_index], port_volatility[max_index], sharpe_ratio[max_index]

def simulate_stock_prices(data, num_simulations, num_days):
    log_returns = np.log(data / data.shift(1)).dropna()
    simulations = np.zeros((num_days, num_simulations, data.shape[1]))
    initial_prices = data.iloc[-1].values

    for i in range(data.shape[1]):
        mu = log_returns.mean()[i]
        sigma = log_returns.std()[i]
        simulations[:, :, i] = np.exp(
            (mu - 0.5 * sigma**2) * (1/252) + sigma * np.random.normal(size=(num_days, num_simulations)) * np.sqrt(1/252)
        ) * initial_prices[i]
    return simulations

def model_option_prices(portfolio, risk_free_rate, num_simulations=10000, num_days=21):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Step 1: Get stock data
    stock_data = get_stock_data(portfolio, start_date, end_date)
    if stock_data.empty:
        raise ValueError("No stock data available for the given tickers.")
    
    # Step 2: Get YTD volatility data
    volatilities = get_ytd_volatility(portfolio)
    
    # Step 3: Simulate stock prices
    stock_prices = simulate_stock_prices(stock_data, num_simulations, num_days)
    
    # Step 4: Calculate daily returns from simulated prices
    daily_returns = (stock_prices[1:] - stock_prices[:-1]) / stock_prices[:-1]
    
    # Step 5: Calculate Value at Risk (VaR)
    var = calculate_var(daily_returns.flatten())
    
    # Step 6: Optimize the portfolio
    returns_df = pd.DataFrame(daily_returns.mean(axis=1), columns=portfolio)
    weights, port_return, port_volatility, sharpe_ratio = optimize_portfolio(returns_df, risk_free_rate)
    
    # Step 7: Fetch options data and calculate Option Prices using Black-Scholes
    best_option = None
    best_profit_potential = -np.inf
    for stock in portfolio:
        options_data = get_options_data(stock)
        if options_data is None:
            continue
        S = stock_prices[-1, :, portfolio.index(stock)].mean()
        sigma = volatilities[stock]
        T = 1/12  # 1 month expiration
        
        # Filter strike prices within 5-10% of the current stock price
        min_strike = S * 0.95
        max_strike = S * 1.1
        
        filtered_calls = options_data.calls[(options_data.calls['strike'] >= min_strike) & (options_data.calls['strike'] <= max_strike)]
        filtered_puts = options_data.puts[(options_data.puts['strike'] >= min_strike) & (options_data.puts['strike'] <= max_strike)]
        
        for index, row in filtered_calls.iterrows():
            K = row['strike']
            call_price = black_scholes(S, K, T, risk_free_rate, sigma, option_type='call')
            put_price = black_scholes(S, K, T, risk_free_rate, sigma, option_type='put')
            for option_type, option_price in [('call', call_price), ('put', put_price)]:
                risk_amount = (K - S) if option_type == 'call' else (S - K)
                profit_amount = 2 * abs(risk_amount)  # 1:2 risk to reward ratio
                stop_loss = S - abs(risk_amount) if option_type == 'call' else S + abs(risk_amount)
                take_profit = S + abs(profit_amount) if option_type == 'call' else S - abs(profit_amount)
                potential_profit = (take_profit - S) / S if option_type == 'call' else (S - take_profit) / S
                if potential_profit > best_profit_potential and ((option_type == 'call' and stop_loss < S) or (option_type == 'put' and stop_loss > S)):
                    best_profit_potential = potential_profit
                    best_option = {
                        'stock': stock,
                        'strike': K,
                        'option_type': option_type,
                        'option_price': option_price,
                        'volatility': sigma,
                        'weight': weights[portfolio.index(stock)],
                        'potential_profit': best_profit_potential
                    }
    
    return best_option, var, weights, port_return, port_volatility, sharpe_ratio

def get_top_50_sp500():
    # Download S&P 500 companies data from Wikipedia
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url, header=0)[0]
    
    # Extract the tickers
    tickers = sp500_table['Symbol'].tolist()
    
    # Get market cap for each ticker and filter those with stock price under $450
    market_caps = []
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            market_cap = stock_info['marketCap']
            current_price = stock_info.get('regularMarketPrice') or stock_info.get('previousClose')
            if current_price is None:
                print(f"Skipping ticker {ticker} due to missing price information.")
                continue
            print(f"Ticker: {ticker}, Market Cap: {market_cap}, Current Price: {current_price}")
            if current_price < 450:
                market_caps.append((ticker, market_cap))
        except KeyError as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
            continue

    # Sort by market cap and get top 50
    sorted_tickers = sorted(market_caps, key=lambda x: x[1], reverse=True)[:50]
    top_50_tickers = [ticker for ticker, _ in sorted_tickers]
    
    return top_50_tickers

# Get the top 50 companies in the S&P 500
portfolio = get_top_50_sp500()
if not portfolio:
    raise ValueError("No valid tickers found for the top 50 companies in the S&P 500.")
print(f"Selected Portfolio: {portfolio}")

risk_free_rate = 0.0533  # Fixed federal funds rate of 5.33%
best_option, var, weights, port_return, port_volatility, sharpe_ratio = model_option_prices(portfolio, risk_free_rate)

# Display results using plotly
fig = go.Figure()

# Best Option
fig.add_trace(go.Table(
    header=dict(values=["Metric", "Value"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[
        ["Stock", "Strike", "Option Type", "Option Price", "Volatility", "Weight in Portfolio", "Potential Profit"],
        [best_option['stock'], f"{best_option['strike']:.2f}", best_option['option_type'], f"{best_option['option_price']:.2f}", f"{best_option['volatility']:.2f}", f"{best_option['weight']:.2%}", f"{best_option['potential_profit']:.2%}"]
    ],
    fill_color='lavender',
    align='left'))
)

# Portfolio Metrics
metrics = {
    "Value at Risk (VaR)": f"{var:.2%}",
    "Portfolio Return": f"{port_return:.2%}",
    "Portfolio Volatility": f"{port_volatility:.2%}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}"
}

for i, (metric, value) in enumerate(metrics.items()):
    fig.add_trace(go.Indicator(
        mode="number",
        value=float(value.strip('%')),
        title={"text": metric},
        domain={'x': [0, 1], 'y': [1 - (i + 1) * 0.2, 1 - i * 0.2]}
    ))

# Update layout to avoid overlap
fig.update_layout(
    height=1000,
    margin=dict(t=50, b=50),
    showlegend=False
)

# Show plot
fig.show()
