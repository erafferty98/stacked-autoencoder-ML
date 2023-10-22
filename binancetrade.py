import time
import json
import pandas as pd
from binance.client import Client
import talib
import sqlite3
from datetime import datetime
import threading
import math


# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

API_KEY = config["API_1"]
API_SECRET = config["API_2"]

client = Client(API_KEY, API_SECRET)

MINIMUM_USDT_THRESHOLD = 5
CURRENCIES = ["BTCUSDT"] #,"ROSEUSDT"]
rsi_buy = 45
rsi_sell = 70
macd_buy = -28.5
percentage_sell = 1.00409

# Assuming you have Binance client initialized as `client`

def api_error_handling(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")

    return wrapper

@api_error_handling
def get_df_klines(symbol, interval):
    klines = client.get_klines(symbol=symbol, interval=interval)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['close'] = df['close'].astype(float)
    return df

@api_error_handling
def get_balance(currency):
    return float(client.get_asset_balance(asset=currency)['free'])
 
def get_rsi(df, lookback_period):
    rsi = talib.RSI(df['close'], timeperiod=lookback_period)
    return rsi.iloc[-1]

def get_macd(df):
    macd, signal, hist = talib.MACD(df['close'])
    return hist.iloc[-1]

def get_bollinger(df):
    upper, middle, lower = talib.BBANDS(df['close'])
    return upper.iloc[-1]

def get_fibonacci_levels(df):
    max_price = df['close'].max()
    min_price = df['close'].min()
    diff = max_price - min_price

    return {
        '23.6%': max_price - diff * 0.236,
        '38.2%': max_price - diff * 0.382,
        '50%': max_price - diff * 0.5,
        '61.8%': max_price - diff * 0.618,
        '78.6%': max_price - diff * 0.786,
    }

@api_error_handling
def get_avg_weekly_volume(symbol):
    df_weekly = get_df_klines(symbol, Client.KLINE_INTERVAL_1WEEK)
    if df_weekly is not None:
        # Removing the current week's volume and calculating the average of the previous weeks
        avg_weekly_volume = df_weekly['volume'].iloc[:-1].astype(float).mean()
        return avg_weekly_volume
    return None

@api_error_handling
def get_balance(currency):
    try:
        balance_info = client.get_asset_balance(asset=currency)
        return float(balance_info['free'])
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

@api_error_handling    
def place_buy_order(symbol, volume, price):
    try:
        order = client.order_limit_buy(
            symbol=symbol, 
            quantity=volume,
            price=str(price)  # Binance client often requires price to be a string
        )
        return order
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

@api_error_handling
def place_sell_order(symbol, volume, price):
    try:
        order = client.order_limit_sell(
            symbol=symbol, 
            quantity=volume,
            price=str(price)  # Binance client often requires price to be a string
        )
        return order
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

@api_error_handling
def place_take_profit_order(symbol, volume, price):
    try:
        # Binance requires the stopPrice for a TAKE_PROFIT_LIMIT order
        # We will set it to the same value as the desired sell price for simplicity
        stop_price = price
        
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_TAKE_PROFIT_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,  # Good Till Cancel
            quantity=volume,
            price=price,   # This is the price you want to sell at for profit
            stopPrice=stop_price
        )
        return order
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

def fetch_data_and_calculate_indicators(symbol, shorter_interval, longer_interval):
    df_long = get_df_klines(symbol, longer_interval)
    df_short = get_df_klines(symbol, shorter_interval)
    
    current_volume = df_short['volume'].astype(float).mean()
    avg_weekly_volume = get_avg_weekly_volume(symbol)
    is_high_volume = current_volume > 1.5 * avg_weekly_volume if avg_weekly_volume else False
    fib_levels = get_fibonacci_levels(df_long)
    rsi = get_rsi(df_short, 14)
    macd_hist = get_macd(df_short)
    
    return df_long, df_short, is_high_volume, fib_levels, rsi, macd_hist

def determine_bullish_trend(df_long):
    short_ma = talib.SMA(df_long['close'], timeperiod=50).iloc[-1]
    long_ma = talib.SMA(df_long['close'], timeperiod=200).iloc[-1]
    return short_ma > long_ma

def check_valid_fib_signal(current_price, fib_levels):
    some_threshold = 0.01 * current_price
    fib_levels_to_check = ['23.6%', '38.2%', '50%', '61.8%']
    return any([abs(current_price - fib_levels[level]) < some_threshold for level in fib_levels_to_check])

def fibonacci_bullish_signal(current_price, previous_high, previous_low):
    # Calculate retracement and extension levels
    retracement_100 = previous_high
    extension_1618 = previous_low + 1.618 * (previous_high - previous_low)
    
    # Check for bullish signal
    bullish_signal = current_price > retracement_100
    
    # Dynamic sell price
    sell_price = extension_1618
    
    return bullish_signal, sell_price

def get_previous_high(df):
    return df['high'].iloc[-2]

def get_previous_low(df):
    return df['low'].iloc[-2]


class TradingBot:
    def __init__(self, symbol, shorter_interval, longer_interval, percentage_sell):
        self.symbol = symbol
        self.percentage_sell = percentage_sell
        self.shorter_interval = shorter_interval
        self.longer_interval = longer_interval
        self.buy_orders = []
        self.has_bought = False
        self.trailing_stop = None

    def log_trade_to_db(self, trade_details):
        with sqlite3.connect('mytrades.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO trades (timestamp, crypto_asset, balance_before, balance_after, profit, action)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (trade_details['timestamp'], trade_details['crypto_asset'], trade_details['balance_before'],
                  trade_details['balance_after'], trade_details['profit'], trade_details['action']))
            conn.commit()
    
    def get_precisions(self):
        try:
            data = client.get_exchange_info()

            quantityPrecision = 0
            pricePrecision = 0
            minNotional = 0.0
            max_orders = 0

            for s in data['symbols']:
                if s['symbol'] == self.symbol:  # Use the instance attribute
                    for filter in s['filters']:
                        if filter['filterType'] == 'LOT_SIZE':
                            quantityPrecision = round(-math.log10(float(filter['stepSize'])))
                        if filter['filterType'] == 'PRICE_FILTER':
                            pricePrecision = round(-math.log10(float(filter['tickSize'])))
                        if filter['filterType'] == 'NOTIONAL':
                            minNotional = float(filter['minNotional'])
                        if filter['filterType'] == 'MAX_NUM_ALGO_ORDERS':
                            max_orders = filter['maxNumAlgoOrders']

            return {
                'quantityPrecision': quantityPrecision,
                'pricePrecision': pricePrecision,
                'minNotional': minNotional,
                'maxOrders': max_orders
            }

        except KeyError:
            print(f"Failed to retrieve precision and notional details for {self.symbol}.")
            return None
        except Exception as e:
            print(f"Error fetching precision and notional details: {e}")
            return None


    def set_trailing_stop(self, buy_price):
        self.trailing_stop = buy_price * (percentage_sell)

    def update_trailing_stop(self, current_price):
        self.trailing_stop = current_price * 0.999

    
    def execute_buy_logic(self, is_high_volume, rsi, macd_hist, df_short):
        balance_before = get_balance('USDT')
        if balance_before < MINIMUM_USDT_THRESHOLD:
            return
        
        precisions = self.get_precisions()
        if len(self.buy_orders) >= precisions['maxOrders']:
            return

        if (rsi < rsi_buy and macd_hist > macd_buy):

            USDT_balance = get_balance('USDT')
            crypto_symbol = self.symbol[:-4]
            crypto_balance = get_balance(crypto_symbol)
            ccy_price = float(client.get_symbol_ticker(symbol=self.symbol)["price"])

            buy_volume = (USDT_balance * 0.5)/ccy_price
            # Get precision details for your trading pair


            # Now, when placing an order or doing calculations, round to the correct precision
            quantity_to_buy = round(buy_volume, precisions['quantityPrecision'])
            price_to_buy = round(ccy_price, precisions['pricePrecision'])
            notional_value = quantity_to_buy * price_to_buy

            if notional_value < precisions['minNotional']:
                return
            
            
            print(f"Placing buy order for {quantity_to_buy} {crypto_symbol} for {price_to_buy}.")
            buy_order = place_buy_order(self.symbol, quantity_to_buy,price_to_buy)
            if buy_order and 'fills' in buy_order and len(buy_order['fills']) > 0:
                self.has_bought = True
                buy_order_details = {
                    'order_id': buy_order['orderId'],
                    'price': float(buy_order['fills'][0]['price']),
                    'volume': buy_volume,
                    'sold': False,
                    'sell_order_id': None  # will store the sell order ID once it's placed
                }

                self.buy_orders.append(buy_order_details)

                buy_price = float(buy_order['fills'][0]['price'])

                if buy_order is not None:
                    self.set_trailing_stop(buy_price)
                
                current_price = float(client.get_symbol_ticker(symbol=self.symbol)["price"])

                profit_target = buy_price * percentage_sell
                upper_bollinger = get_bollinger(df_short)
                profit_target = max(upper_bollinger, profit_target)
                place_take_profit_order(self.symbol, buy_volume, profit_target)
                print(f"Bought at {buy_price}. Setting sell order at {profit_target}.")

                profit = (buy_volume * current_price) - balance_before  # This might be negative if it's a buy
                self.log_trade_to_db({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'crypto_asset': self.symbol,
                    'balance_before': balance_before,
                    'balance_after': USDT_balance,
                    'profit': profit,
                    'action': 'buy'
                })
                


    def execute_sell_logic(self, current_price, fib_levels, rsi, macd_hist, previous_high, previous_low, is_bullish):
        
        if not self.has_bought:
            return
        
        balance_before = get_balance('USDT')

        crypto_symbol = self.symbol[:-4]
        crypto_balance = get_balance(crypto_symbol)

        if crypto_balance * current_price < MINIMUM_USDT_THRESHOLD:
            print("Insufficient crypto balance for selling.")
            time.sleep(60)
            return
        
        for buy_order in self.buy_orders:
            precisions = self.get_precisions()


            if self.trailing_stop and current_price <= self.trailing_stop:
                # Execute the sell logic based on trailing stop being hit
                
                # Assuming each buy_order contains data about a singular buy transaction
                for buy_order in self.buy_orders:
                    if not buy_order['sold']:
                        # Determine the quantity to sell using your defined precisions
                        precisions = self.get_precisions()
                        quantity_to_sell = round(buy_order['volume'], precisions['quantityPrecision'])

                        # Place a market sell order since it's based on trailing stop being hit
                        # Replace this with your actual sell function/method call
                        sell_order = place_sell_order(self.symbol, quantity_to_sell)

                        if sell_order:
                            # Update the buy_order to mark it as sold
                            buy_order['sold'] = True
                            print(f"Selling {self.symbol} based on trailing stop at {current_price}.")
                            # Calculate profit and log the trade
                            profit = (quantity_to_sell * current_price) - (quantity_to_sell * buy_order['price'])
                            self.log_trade_to_db({
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'crypto_asset': self.symbol,
                                'balance_before': get_balance('USDT'),  # Use your actual function call here
                                'balance_after': get_balance('USDT') + (quantity_to_sell * current_price),  # Use your actual function call here
                                'profit': profit,
                                'action': 'sell'
                            })
                            
                            # You can break out of the loop if you only want to sell one order at a time
                            # If you wish to sell all buy_orders that haven't been sold yet, you can remove this break
                            break

                    elif (not buy_order['sold']) and (current_price >= min_sell_price):          
                        fib_bullish_signal, fib_sell_price = fibonacci_bullish_signal(current_price, previous_high, previous_low)
                        max_sell_price = max(fib_sell_price, (buy_order['price'] * percentage_sell),current_price),
                        min_sell_price = buy_order['price'] * percentage_sell
                        min_sell_price = round(min_sell_price, precisions['pricePrecision'])
                        max_sell_price = round(max_sell_price, precisions['pricePrecision'])
                        quantity_to_sell = round(buy_order['volume'], precisions['quantityPrecision'])
                        current_price = round(current_price,precisions['pricePrecision'])
                        notional_value = quantity_to_sell * min_sell_price
                
                        if current_price >= min_sell_price:
                            if (fib_bullish_signal and fib_sell_price > buy_order['price']) or current_price>=max_sell_price:
                                sell_order = place_sell_order(symbol, quantity_to_sell,max_sell_price)
                                profit = (crypto_balance * max_sell_price) - balance_before
                            else:
                                min_sell_price = max(current_price,min_sell_price)
                                sell_order = place_sell_order(symbol, quantity_to_sell,min_sell_price)
                                profit = (crypto_balance * min_sell_price) - balance_before

                            print(f"Selling {symbol} at {current_price}.")
                            if sell_order:
                                buy_order['sold'] = True
                                sell_order = None  # Replace with the sell logic based on conditions


                                # Calculate profit and log the trade
                                profit = (quantity_to_sell * current_price) - (quantity_to_sell * buy_order['price'])
                                self.log_trade_to_db({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'crypto_asset': self.symbol,
                                    'balance_before': get_balance('USDT'),  # Use your actual function call here
                                    'balance_after': get_balance('USDT') + (quantity_to_sell * current_price),  # Use your actual function call here
                                    'profit': profit,
                                    'action': 'sell'
                                })
            return

    def run(self, iterations=3):  # default to 3 iterations if not provided
        for i in range(iterations):
            try:
                df_long, df_short, is_high_volume, fib_levels, rsi, macd_hist = fetch_data_and_calculate_indicators(self.symbol, self.shorter_interval, self.longer_interval)

                if df_long is None or df_short is None:
                    print("Error fetching klines.")
                    time.sleep(120)
                    continue

                is_bullish = determine_bullish_trend(df_long)
                current_price = float(client.get_symbol_ticker(symbol=self.symbol)["price"])
                valid_fib_signal = check_valid_fib_signal(current_price, fib_levels)
                previous_high = get_previous_high(df_long)
                previous_low = get_previous_low(df_long)


                if valid_fib_signal:
                    current_price = float(client.get_symbol_ticker(symbol=self.symbol)["price"])
                    self.execute_buy_logic(is_high_volume, rsi, macd_hist, df_short)

                    if hasattr(self, 'trailing_stop') and self.trailing_stop and current_price >= self.trailing_stop:
                        self.update_trailing_stop(current_price)

                    self.execute_sell_logic(current_price, fib_levels, rsi, macd_hist, previous_high, previous_low,is_bullish)

            except Exception as e:
                print(f"Unexpected error in main loop: {e}")
                time.sleep(120)


if __name__ == "__main__":
    while True:
        for symbol in CURRENCIES:
            bot = TradingBot(symbol, Client.KLINE_INTERVAL_15MINUTE, Client.KLINE_INTERVAL_1HOUR,percentage_sell)
            bot.run(3)  # Running for 3 iterations per currency
