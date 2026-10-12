import pandas as pd
import numpy as np

def generate_entsoe_mock_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic ENTSO-E Day-Ahead market data.
    Simulates:
    - Diurnal solar cycles (peaks at noon)
    - Stochastic wind forecasting
    - Target spot prices inversely correlated with renewable generation
    """
    # 1. Create the Quarter-Hourly Timeline (ENTSO-E standard)
    dates = pd.date_range(start=start_date, end=end_date, freq="15min", tz="Europe/Copenhagen")
    n = len(dates)
    
    # 2. Extract Time Features for Seasonality
    hour_of_day = dates.hour + dates.minute / 60.0
    
    # 3. Simulate Exogenous Variables (Available at 12:00 Day D for D+1)
    # Solar follows a sine wave peaking at 12:00, zero at night
    solar_norm = np.clip(np.sin((hour_of_day - 6) * np.pi / 12), 0, 1)
    forecast_solar_dk1 = solar_norm * 800 + np.random.normal(0, 50, n)
    
    # Wind is mean-reverting stochastic noise
    forecast_wind_dk1 = 1500 + 500 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 200, n)
    
    # Load follows a double-peak profile (Morning at 8:00, Evening at 18:00)
    load_profile = 2500 + 500 * np.sin((hour_of_day - 5) * np.pi / 12) + 300 * np.sin((hour_of_day - 15) * np.pi / 6)
    forecast_load_dk1 = load_profile + np.random.normal(0, 100, n)
    
    # 4. Calculate Residual Load (The primary driver of power prices)
    residual_load = forecast_load_dk1 - (forecast_solar_dk1 + forecast_wind_dk1)
    
    # 5. Simulate Target Spot Prices (With occasional negative spikes)
    # Base price driven by gas generation, dropping sharply when renewables overproduce
    base_price = 40.0 + (residual_load / 100) * 5
    
    # Introduce heavy tails (simulating negative prices during high wind/solar)
    tail_noise = np.random.standard_t(df=3, size=n) * 15
    spot_price_dk1 = base_price + tail_noise
    
    # Clean up negative outliers slightly to keep it realistic
    spot_price_dk1 = np.where(residual_load < -500, spot_price_dk1 - 50, spot_price_dk1)

    # 6. Build the DataFrame
    df = pd.DataFrame({
        "timestamp": dates,
        "forecast_solar_dk1": forecast_solar_dk1.clip(min=0),
        "forecast_wind_dk1": forecast_wind_dk1.clip(min=0),
        "forecast_load_dk1": forecast_load_dk1,
        "spot_price_dk1": spot_price_dk1
    }).set_index("timestamp")
    
    return df