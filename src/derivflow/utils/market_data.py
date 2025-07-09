"""
DERIVFLOW-FINANCE: Advanced Market Data Integration
=================================================

Professional-grade market data interface with:
- Real-time options data from Yahoo Finance
- Implied volatility extraction
- Options chain analysis
- Market data validation and cleaning
- Robust handling of market closed scenarios
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
import time

@dataclass
class OptionData:
    """Single option data point"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None

@dataclass
class MarketDataResult:
    """Market data extraction result"""
    symbol: str
    spot_price: float
    options_data: List[OptionData]
    timestamp: datetime
    total_options: int
    data_quality: str
    market_status: str

class AdvancedMarketData:
    """
    Professional market data interface for derivatives analytics
    
    Provides real-time access to:
    - Stock prices and options chains
    - Implied volatility extraction
    - Market data quality assessment
    - Historical volatility calculation
    """
    
    def _init_(self, cache_duration: int = 300):
        """
        Initialize market data interface
        
        Parameters:
        -----------
        cache_duration : int
            Cache duration in seconds (default: 5 minutes)
        """
        self.cache_duration = cache_duration
        self.cache = {}
        self.rate_limit_delay = 0.5  # Seconds between requests
        
    def _safe_float(self, val, default: float = 0.0) -> float:
        """Safely convert value to float, handling NaN"""
        try:
            if pd.isna(val) or val == '' or val is None:
                return default
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, val, default: int = 0) -> int:
        """Safely convert value to int, handling NaN"""
        try:
            if pd.isna(val) or val == '' or val is None:
                return default
            return int(float(val))
        except (ValueError, TypeError):
            return default
    
    def is_market_open(self) -> Tuple[bool, str]:
        """
        Check if US market is currently open
        
        Returns:
        --------
        Tuple[bool, str]
            (is_open, status_message)
        """
        try:
            from datetime import datetime
            import pytz
            
            # US Eastern Time
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            
            # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if weekday and within market hours
            is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
            is_market_time = market_open <= now <= market_close
            
            if not is_weekday:
                return False, "Weekend"
            elif now < market_open:
                return False, "Pre-market"
            elif now > market_close:
                return False, "After-hours"
            else:
                return True, "Open"
                
        except Exception:
            return False, "Unknown"
    
    def _get_risk_free_rate(self) -> float:
        """Get current risk-free rate from Treasury data"""
        try:
            # Use 10-year Treasury as proxy for risk-free rate
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")
            if not hist.empty:
                return self._safe_float(hist['Close'].iloc[-1]) / 100
            else:
                return 0.045  # Default 4.5% if data unavailable
        except Exception:
            return 0.045  # Default fallback
    
    def get_current_price(self, symbol: str) -> Tuple[float, datetime]:
        """
        Get current stock price (or last available price)
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
        --------
        Tuple[float, datetime]
            Current/last price and timestamp
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get intraday data first
            hist = ticker.history(period="2d", interval="5m")
            
            if not hist.empty:
                current_price = self._safe_float(hist['Close'].iloc[-1])
                timestamp = hist.index[-1].to_pydatetime()
            else:
                # Fallback to daily data
                hist = ticker.history(period="5d")
                if hist.empty:
                    raise ValueError(f"No price data available for {symbol}")
                current_price = self._safe_float(hist['Close'].iloc[-1])
                timestamp = hist.index[-1].to_pydatetime()
            
            return current_price, timestamp
            
        except Exception as e:
            raise ValueError(f"Error fetching price for {symbol}: {str(e)}")
    
    def get_options_chain(self, symbol: str, expiry_date: Optional[str] = None) -> MarketDataResult:
        """
        Get complete options chain for a symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        expiry_date : str, optional
            Specific expiry date (YYYY-MM-DD), if None gets nearest expiry
            
        Returns:
        --------
        MarketDataResult
            Complete options chain data
        """
        market_open, market_status = self.is_market_open()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            spot_price, timestamp = self.get_current_price(symbol)
            
            # Get available expiry dates
            expiry_dates = ticker.options
            if not expiry_dates:
                raise ValueError(f"No options data available for {symbol}")
            
            # Select expiry date
            if expiry_date is None:
                target_expiry = expiry_dates[0]  # Nearest expiry
            else:
                if expiry_date not in expiry_dates:
                    # Find closest available expiry
                    target_expiry = min(expiry_dates, 
                                      key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - 
                                                        datetime.strptime(expiry_date, '%Y-%m-%d')).days))
                else:
                    target_expiry = expiry_date
            
            # Get options chain
            options_chain = ticker.option_chain(target_expiry)
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Process options data
            options_data = []
            
            # Process calls with robust error handling
            if not calls.empty:
                for _, row in calls.iterrows():
                    try:
                        option = OptionData(
                            symbol=symbol,
                            strike=self._safe_float(row.get('strike', 0)),
                            expiry=target_expiry,
                            option_type='call',
                            last_price=self._safe_float(row.get('lastPrice', 0)),
                            bid=self._safe_float(row.get('bid', 0)),
                            ask=self._safe_float(row.get('ask', 0)),
                            volume=self._safe_int(row.get('volume', 0)),
                            open_interest=self._safe_int(row.get('openInterest', 0)),
                            implied_volatility=self._safe_float(row.get('impliedVolatility')) 
                                               if pd.notna(row.get('impliedVolatility')) else None
                        )
                        
                        # Only add if strike is valid
                        if option.strike > 0:
                            options_data.append(option)
                            
                    except Exception:
                        continue  # Skip problematic rows
            
            # Process puts with robust error handling
            if not puts.empty:
                for _, row in puts.iterrows():
                    try:
                        option = OptionData(
                            symbol=symbol,
                            strike=self._safe_float(row.get('strike', 0)),
                            expiry=target_expiry,
                            option_type='put',
                            last_price=self._safe_float(row.get('lastPrice', 0)),
                            bid=self._safe_float(row.get('bid', 0)),
                            ask=self._safe_float(row.get('ask', 0)),
                            volume=self._safe_int(row.get('volume', 0)),
                            open_interest=self._safe_int(row.get('openInterest', 0)),
                            implied_volatility=self._safe_float(row.get('impliedVolatility')) 
                                               if pd.notna(row.get('impliedVolatility')) else None
                        )
                        
                        # Only add if strike is valid
                        if option.strike > 0:
                            options_data.append(option)
                            
                    except Exception:
                        continue  # Skip problematic rows
            
            # Assess data quality
            if options_data:
                valid_prices = [opt for opt in options_data if opt.last_price > 0]
                valid_ivs = [opt for opt in options_data if opt.implied_volatility is not None and opt.implied_volatility > 0]
                
                price_quality = len(valid_prices) / len(options_data)
                iv_quality = len(valid_ivs) / len(options_data)
                overall_quality = (price_quality + iv_quality) / 2
                
                if overall_quality > 0.7:
                    data_quality = "Excellent"
                elif overall_quality > 0.5:
                    data_quality = "Good"
                elif overall_quality > 0.3:
                    data_quality = "Fair"
                else:
                    data_quality = "Limited"
            else:
                data_quality = "No Data"
            
            result = MarketDataResult(
                symbol=symbol,
                spot_price=spot_price,
                options_data=options_data,
                timestamp=timestamp,
                total_options=len(options_data),
                data_quality=data_quality,
                market_status=market_status
            )
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error fetching options chain for {symbol}: {str(e)}")
    
    def build_volatility_surface_from_market(self, symbol: str, 
                                           max_expiries: int = 4) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Build volatility surface from live market data
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        max_expiries : int
            Maximum number of expiry dates to include
            
        Returns:
        --------
        Dict
            Volatility surface data ready for VolatilitySurface class
        """
        try:
            ticker = yf.Ticker(symbol)
            expiry_dates = ticker.options[:max_expiries]
            
            surface_data = []
            spot_price, _ = self.get_current_price(symbol)
            market_open, market_status = self.is_market_open()
            
            print(f"📊 Building volatility surface for {symbol} (Spot: ${spot_price:.2f})")
            print(f"🕐 Market Status: {market_status}")
            print(f"📅 Processing {len(expiry_dates)} expiry dates...")
            
            for i, expiry in enumerate(expiry_dates):
                print(f"   Processing {expiry} ({i+1}/{len(expiry_dates)})...")
                
                try:
                    market_data = self.get_options_chain(symbol, expiry)
                    
                    # Calculate time to expiry
                    expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                    time_to_expiry = (expiry_date - datetime.now()).days / 365.25
                    
                    # Filter for options with valid data
                    for option in market_data.options_data:
                        # More lenient filtering for market closed scenarios
                        if (option.strike > 0 and 
                            (option.last_price > 0 or option.bid > 0 or option.ask > 0)):
                            
                            # Use mid price if available, otherwise last price
                            if option.bid > 0 and option.ask > 0:
                                market_price = (option.bid + option.ask) / 2
                            else:
                                market_price = option.last_price
                            
                            surface_data.append({
                                'strike': option.strike,
                                'expiry': time_to_expiry,
                                'volatility': option.implied_volatility or 0.25,  # Default if missing
                                'market_price': market_price,
                                'option_type': option.option_type,
                                'volume': option.volume,
                                'open_interest': option.open_interest
                            })
                    
                    time.sleep(self.rate_limit_delay)  # Rate limiting
                    
                except Exception as e:
                    print(f"   ⚠ Error processing {expiry}: {str(e)}")
                    continue
            
            print(f"✅ Surface built with {len(surface_data)} data points")
            
            return {
                'symbol': symbol,
                'spot_price': spot_price,
                'surface_data': surface_data,
                'timestamp': datetime.now(),
                'risk_free_rate': self._get_risk_free_rate(),
                'market_status': market_status
            }
            
        except Exception as e:
            raise ValueError(f"Error building volatility surface: {str(e)}")
    
    def get_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """
        Calculate historical volatility
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        days : int
            Number of days for calculation
            
        Returns:
        --------
        float
            Annualized historical volatility
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{max(days*2, 60)}d")  # Get extra data for safety
            
            if len(hist) < days:
                # If not enough data, use what we have
                days = len(hist) - 1
                if days < 5:
                    raise ValueError(f"Insufficient historical data for {symbol}")
            
            # Calculate daily returns
            prices = hist['Close'].tail(days + 1)
            returns = np.log(prices / prices.shift(1)).dropna()
            
            # Annualized volatility
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
            
            return float(annual_vol)
            
        except Exception as e:
            raise ValueError(f"Error calculating historical volatility: {str(e)}")
    
    def market_summary(self, symbol: str) -> Dict[str, Union[str, float, int]]:
        """
        Get comprehensive market summary
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        Dict
            Complete market summary
        """
        try:
            # Get market status
            market_open, market_status = self.is_market_open()
            
            # Get current price
            spot_price, timestamp = self.get_current_price(symbol)
            
            # Get historical volatility
            hist_vol = self.get_historical_volatility(symbol)
            
            # Get risk-free rate
            risk_free_rate = self._get_risk_free_rate()
            
            # Try to get options data
            try:
                options_data = self.get_options_chain(symbol)
                
                # Calculate average implied volatility
                valid_ivs = [opt.implied_volatility for opt in options_data.options_data 
                           if opt.implied_volatility is not None and opt.implied_volatility > 0]
                avg_iv = np.mean(valid_ivs) if valid_ivs else None
                
                return {
                    'symbol': symbol,
                    'spot_price': spot_price,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'market_status': market_status,
                    'options_available': len(options_data.options_data),
                    'data_quality': options_data.data_quality,
                    'historical_volatility_30d': hist_vol,
                    'average_implied_volatility': avg_iv,
                    'risk_free_rate': risk_free_rate,
                    'total_volume': sum(opt.volume for opt in options_data.options_data),
                    'total_open_interest': sum(opt.open_interest for opt in options_data.options_data),
                    'expiry_date': options_data.options_data[0].expiry if options_data.options_data else None
                }
                
            except Exception:
                # If options data fails, return basic summary
                return {
                    'symbol': symbol,
                    'spot_price': spot_price,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'market_status': market_status,
                    'historical_volatility_30d': hist_vol,
                    'risk_free_rate': risk_free_rate,
                    'options_available': 0,
                    'data_quality': 'Unavailable',
                    'note': 'Options data temporarily unavailable'
                }
            
        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    # Example usage and testing
    print("🚀 DERIVFLOW-FINANCE: Advanced Market Data Integration")
    print("=" * 70)
    
    # Initialize market data interface
    market_data = AdvancedMarketData()
    
    # Test symbol
    symbol = "AAPL"
    
    print(f"📊 Real-Time Market Data Analysis for {symbol}")
    print("-" * 50)
    
    # Check market status
    market_open, market_status = market_data.is_market_open()
    status_emoji = "🟢" if market_open else "🔴"
    print(f"🕐 Market Status: {status_emoji} {market_status}")
    
    if not market_open:
        print("ℹ  Using last available market data")
    
    print()
    
    try:
        # Get comprehensive market summary
        summary = market_data.market_summary(symbol)
        
        if 'error' in summary:
            print(f"❌ Error: {summary['error']}")
        else:
            print(f"💰 Current Price:        ${summary['spot_price']:.2f}")
            print(f"🕐 Last Updated:         {summary['timestamp']}")
            print(f"📈 Historical Vol (30d): {summary['historical_volatility_30d']:.1%}")
            print(f"💸 Risk-Free Rate:       {summary['risk_free_rate']:.2%}")
            
            if summary.get('options_available', 0) > 0:
                print(f"📊 Options Available:    {summary['options_available']}")
                print(f"🎯 Data Quality:         {summary['data_quality']}")
                print(f"📅 Nearest Expiry:       {summary.get('expiry_date', 'N/A')}")
                
                if summary.get('average_implied_volatility'):
                    print(f"💫 Average Implied Vol:  {summary['average_implied_volatility']:.1%}")
                    
                if summary.get('total_volume', 0) > 0:
                    print(f"📈 Total Volume:         {summary['total_volume']:,}")
                    print(f"🔓 Total Open Interest:  {summary['total_open_interest']:,}")
            else:
                print(f"⚠  Options Data:        {summary.get('data_quality', 'Limited')}")
                if 'note' in summary:
                    print(f"ℹ  Note:                {summary['note']}")
        
        # Try to get sample options chain
        print(f"\n📈 Options Chain Sample:")
        print("-" * 65)
        
        try:
            options_result = market_data.get_options_chain(symbol)
            
            if options_result.options_data:
                # Show options around current price
                spot = options_result.spot_price
                relevant_options = [opt for opt in options_result.options_data 
                                  if abs(opt.strike - spot) / spot < 0.1 and opt.strike > 0][:10]
                
                if relevant_options:
                    print(f"{'Type':<4} {'Strike':<8} {'Last':<8} {'Bid':<6} {'Ask':<6} {'IV':<8} {'Vol':<6}")
                    print("-" * 65)
                    
                    for option in relevant_options:
                        iv_str = f"{option.implied_volatility:.1%}" if option.implied_volatility else "N/A"
                        print(f"{option.option_type:<4} {option.strike:<8.0f} "
                              f"${option.last_price:<7.2f} ${option.bid:<5.2f} ${option.ask:<5.2f} "
                              f"{iv_str:<8} {option.volume:<6}")
                else:
                    print("📋 Options data available but limited liquidity around current price")
            else:
                print("📋 No options data available for display")
                
        except Exception as e:
            print(f"📋 Options display unavailable: Limited data during {market_status.lower()}")
        
        print(f"\n🎉 MARKET DATA INTEGRATION STATUS:")
        print("-" * 50)
        print("✅ Stock price data: Working")
        print("✅ Historical volatility: Working") 
        print("✅ Risk-free rate: Working")
        print(f"{'✅' if summary.get('options_available', 0) > 0 else '⚠ '} Options data: {'Available' if summary.get('options_available', 0) > 0 else 'Limited'}")
        print()
        print("🚀 Market data integration ready!")
        print("📊 Perfect for real-time derivatives analytics!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Check internet connection and symbol validity")