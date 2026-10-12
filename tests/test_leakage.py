import pandas as pd
import pytest

def test_production_gate_closure_leakage():
    """
    Simulate a realistic SDAC 12:00 CET gate closure.
    Ensures realized prices stop at 12:00 Day D, while day-ahead 
    TSO forecasts are available for Day D+1.
    """
    # Create timeline: Day D and Day D+1
    dates = pd.date_range(start="2026-10-12 00:00", end="2026-10-13 23:00", freq="h", tz="Europe/Copenhagen")
    
    # Simulate our master feature table
    df = pd.DataFrame({
        "timestamp": dates,
        "spot_price_dk1": [50.0] * len(dates),         # Realized price (Must be masked for D+1)
        "forecast_wind_dk1": [1200.0] * len(dates)     # Exogenous forecast (Allowed for D+1)
    }).set_index("timestamp")

    target_delivery_day = pd.Timestamp("2026-10-13").tz_localize("Europe/Copenhagen")
    gate_closure_time = pd.Timestamp("2026-10-12 12:00").tz_localize("Europe/Copenhagen")

    # --- SIMULATE THE SPLITTER ---
    # 1. Mask future realized prices (set to NaN after 12:00 Day D)
    realized_cols = ["spot_price_dk1"]
    df.loc[df.index > gate_closure_time, realized_cols] = float("nan")
    
    # 2. Extract the feature set we pass to the model for Day D+1
    model_inputs_d_plus_1 = df[df.index >= target_delivery_day]

    # --- THE GUARDRAILS ---
    
    # Guardrail 1: Tomorrow's ACTUAL prices must be completely missing (NaN)
    assert model_inputs_d_plus_1["spot_price_dk1"].isna().all(), \
        "FATAL LEAKAGE: Tomorrow's realized spot prices leaked into the model inputs!"

    # Guardrail 2: Tomorrow's FORECASTS (Wind/Solar/Load) must be fully populated
    assert model_inputs_d_plus_1["forecast_wind_dk1"].notna().all(), \
        "MISSING DATA: The model is blind! TSO wind forecasts for tomorrow were accidentally dropped."
        
    print("Production gate closure integrity check passed.")