import pandas as pd
import numpy as np
import pickle
import os
from new_iv import blackScholes, implied_volatility


index_mapper={"NIFTY":"NIFTY","BANKNIFTY":"BANKNIFTY","FINNIFTY":"FINNIFTY","MIDCPNIFTY":"MIDCPNIFTY"}

def get_strike_from_symbol(symbol: str, index:str) -> int:
   # print(symbol,index)
    """Extract strike price from option symbol"""
    return int(symbol[len(index_mapper.get(index))+7:-6])

def process_option_chain(df: pd.DataFrame, date: str,index_name :str,out_put_dir:str,spot_dir:str) -> bool:
    """Convert single day data into option chains for each expiry and save to Parquet
    
    Returns:
        bool: True if spot data was found and valid, False otherwise
    """
    
    # Check for spot data availability first before any processing
    #spot_file = f"e:/midcap_files/spot_data/{date}.pkl"
    spot_file=os.path.join(spot_dir,f"{date}.pkl")
    #base_dir = "e:/mid_chain"

    #spot_file = spot_dir
    base_dir = out_put_dir
    print(f"Loading spot data from: {spot_file}")
    
    # Spot file existence should already be checked by process_all_data.py
    try:
        with open(spot_file, 'rb') as f:
            spot_df = pickle.load(f)
        
        if spot_df.empty:
            print(f"Spot data file is empty for date {date}. Skipping all processing.")
            return False
    except Exception as e:
        print(f"Error loading spot data for date {date}: {str(e)}. Skipping all processing.")
        return False
    
    print("\nInput data columns:", df.columns.tolist()) 
    
    if index_name!="SENSEX":
        df = df[~df['expiry'].str.contains('FUT', na=False)].copy()
        

    if df.empty:

        print("No options data found after filtering futures. Skipping processing.")

        return False
    
    # Print all unique expiries
    unique_expiries = pd.to_datetime(df['expiry']).dt.date.unique()
    print(f"\nFound {len(unique_expiries)} expiry dates:")
    for exp in sorted(unique_expiries):
        print(f"- {exp}")
    print("\nProcessing each expiry...")
    
    # Add minute-level timestamps
    df['minute'] = df['dt'].dt.strftime('%H:%M:00')
    
    # Convert spot timestamps to minutes and get spot prices
    spot_df['minute'] = spot_df['dt'].dt.strftime('%H:%M:00')
    spot_prices = spot_df.groupby('minute')['close'].first().reset_index()
    
    # Extract strike prices
    #df['strike'] = df['ticker'].apply(get_strike_from_symbol)
    if index_name=="MIDCP":
        index_name="MIDCPNIFTY"

   # df['strike'] = df['ticker'].apply(lambda x: get_strike_from_symbol(x, index_name))
    df["strike"] = df["ticker"].apply(lambda x: float(x[len(index_name)+7:-6]))

  
    # Create output directory structure
    date_dir = os.path.join(base_dir, date)
    print(f"\nCreating output directory: {date_dir}")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(date_dir, exist_ok=True)
    
    # Set fixed day start (09:15) and end times (15:30)
    date_obj = pd.to_datetime(date)
    day_start = date_obj.replace(hour=9, minute=15, second=0, microsecond=0)
    day_end = date_obj.replace(hour=15, minute=30, second=0, microsecond=0)
    total_day_seconds = (day_end - day_start).total_seconds()
    
    # Process each expiry
    unique_expiries = [e for e in unique_expiries if pd.notna(e)]
    for expiry in unique_expiries:
        print(f"\nProcessing expiry: {expiry}")
        
        # Set fixed expiry time (15:30)
        expiry_dt = pd.to_datetime(expiry).replace(hour=15, minute=30)
        
        # Filter data for this expiry
        expiry_df = df[pd.to_datetime(df['expiry']).dt.date == expiry].copy()
        print(f"Found {len(expiry_df)} records for expiry {expiry}")
        
        if len(expiry_df) == 0:
            print(f"No data found for expiry {expiry}, skipping...")
            continue
        
        # Split into calls and puts
        calls = expiry_df[expiry_df['ticker'].str.contains('CE')].copy()
        puts = expiry_df[expiry_df['ticker'].str.contains('PE')].copy()

        print(calls)
        print(puts)


        #print(calls)
        
        # Prepare market data columns
        market_cols = ['open', 'high', 'low', 'close', 'volume', 'oi']
        
        # Rename columns to distinguish between calls and puts
        for col in market_cols:
            calls = calls.rename(columns={col: f'call_{col}'})
            puts = puts.rename(columns={col: f'put_{col}'})
        
        # Select relevant columns for merging
        call_cols = ['minute', 'strike'] + [f'call_{col}' for col in market_cols]
        put_cols = ['minute', 'strike'] + [f'put_{col}' for col in market_cols]
        
        # Create the chain by merging calls and puts
        chain_df = pd.merge(
            calls[call_cols],
            puts[put_cols],
            on=['minute', 'strike'],
            how='outer'
        )
        
        # Add spot prices
        chain_df = pd.merge(chain_df, spot_prices, on='minute', how='left')
        chain_df.rename(columns={'close': 'spot_price'}, inplace=True)
        
        # Calculate positions for each minute group
        def calculate_positions(group):
            # Find ATM strike (closest to spot)
            spot = group['spot_price'].iloc[0]
            strikes = group['strike'].values
            
            # Initialize position columns
            group['call_position'] = np.nan
            group['put_position'] = np.nan
            
            # Calculate call positions - vectorized
            call_mask = ~group['call_close'].isna()
            if call_mask.any():
                valid_call_strikes = sorted(strikes[call_mask])  # Sort ascending for calls
                atm_idx = np.abs(np.array(valid_call_strikes) - spot).argmin()
                call_positions = np.arange(-atm_idx, len(valid_call_strikes)-atm_idx)  # Centered around ATM
                call_position_map = {strike: pos for strike, pos in zip(valid_call_strikes, call_positions)}
                group.loc[call_mask, 'call_position'] = group.loc[call_mask, 'strike'].map(call_position_map)
            
            # Calculate put positions - vectorized
            put_mask = ~group['put_close'].isna()
            if put_mask.any():
                valid_put_strikes = sorted(strikes[put_mask], reverse=True)  # Sort descending for puts
                atm_idx = np.abs(np.array(valid_put_strikes) - spot).argmin()
                put_positions = np.arange(-atm_idx, len(valid_put_strikes)-atm_idx)  # Centered around ATM
                put_position_map = {strike: pos for strike, pos in zip(valid_put_strikes, put_positions)}
                group.loc[put_mask, 'put_position'] = group.loc[put_mask, 'strike'].map(put_position_map)

            # Calculate synthetic price for this group
            atm_call = group[group['call_position'] == 0]['call_close'].iloc[0] if len(group[group['call_position'] == 0]) > 0 else np.nan
            atm_put = group[group['put_position'] == 0]['put_close'].iloc[0] if len(group[group['put_position'] == 0]) > 0 else np.nan
            synthetic = spot
            if pd.notna(atm_call) and pd.notna(atm_put):
                synthetic = spot + atm_call - atm_put
            group['synthetic'] = synthetic
            group['atm_call'] = atm_call
            group['atm_put'] = atm_put
            
            # Calculate TTE for this minute using actual time but fixed start/end
            current_time = pd.to_datetime(df[df['minute'] == group['minute'].iloc[0]]['dt'].iloc[0])
            current_date = current_time.date()
            
            # Calculate days to expiry (whole days)
            days_to_expiry = (expiry_dt.date() - current_date).days
            
            # Calculate intraday decay
            seconds_from_start = (current_time - day_start).total_seconds()
            dd = 1 - (seconds_from_start / total_day_seconds)
            
            tte = days_to_expiry + dd
            group['tte'] = round(tte, 2)
            
            # Initialize Greek columns
            greek_cols = ['delta', 'gamma', 'theta', 'vega']
            for prefix in ['call_', 'put_']:
                for greek in greek_cols:
                    group[f'{prefix}{greek}'] = np.nan
            
            # Initialize IV columns
            group['call_iv'] = np.nan
            group['put_iv'] = np.nan
            
            # Vectorized Greek calculations for calls
            if call_mask.any():
                call_data = group[call_mask]
                ivs = np.zeros(len(call_data))
                for i, (_, row) in enumerate(call_data.iterrows()):
                    try:
                        ivs[i] = implied_volatility('c', int(row['strike']), float(synthetic), float(tte), float(row['call_close']))
                    except:
                        ivs[i] = np.nan
                
                valid_iv_mask = ~np.isnan(ivs)
                if valid_iv_mask.any():
                    valid_calls = call_data[valid_iv_mask]
                    valid_ivs = ivs[valid_iv_mask]
                    
                    # Store IVs
                    group.loc[valid_calls.index, 'call_iv'] = valid_ivs
                    
                    for greek, code in zip(greek_cols, ['d', 'g', 't', 'v']):
                        results = np.array([
                            blackScholes(code, 'c', int(strike), float(synthetic), float(tte), iv, 0)
                            for strike, iv in zip(valid_calls['strike'], valid_ivs)
                        ])
                        group.loc[valid_calls.index, f'call_{greek}'] = results
            
            # Vectorized Greek calculations for puts
            if put_mask.any():
                put_data = group[put_mask]
                ivs = np.zeros(len(put_data))
                for i, (_, row) in enumerate(put_data.iterrows()):
                    try:
                        ivs[i] = implied_volatility('p', int(row['strike']), float(synthetic), float(tte), float(row['put_close']))
                    except:
                        ivs[i] = np.nan
                
                valid_iv_mask = ~np.isnan(ivs)
                if valid_iv_mask.any():
                    valid_puts = put_data[valid_iv_mask]
                    valid_ivs = ivs[valid_iv_mask]
                    
                    # Store IVs
                    group.loc[valid_puts.index, 'put_iv'] = valid_ivs
                    
                    for greek, code in zip(greek_cols, ['d', 'g', 't', 'v']):
                        results = np.array([
                            blackScholes(code, 'p', int(strike), float(synthetic), float(tte), iv, 0)
                            for strike, iv in zip(valid_puts['strike'], valid_ivs)
                        ])
                        group.loc[valid_puts.index, f'put_{greek}'] = results
            
            return group
        
        # Apply position calculation and synthetic data
        chain_df = chain_df.groupby('minute', group_keys=False).apply(calculate_positions)
        
        # Sort by timestamp and strike
        chain_df = chain_df.sort_values(['minute', 'strike'])
        
        # Filter out 15:30 data points
        chain_df = chain_df[chain_df['minute'] != '15:30:00']
        
        # Save to parquet
        expiry_str = expiry.strftime('%Y-%m-%d')
        parquet_file = os.path.join(date_dir, f"{expiry_str}.parquet")
        print(f"\nSaving to {parquet_file}...")
        try:
            chain_df.to_parquet(parquet_file, index=False)
            print(f"Successfully saved {len(chain_df)} rows to {parquet_file}")
            #chain_df.to_csv(os.path.join("E:\\nifty_chain","sas.csv"))
           
        except Exception as e:
            print(f"Error saving to {parquet_file}: {str(e)}")
            raise
        
        print(f"Total timestamps: {len(chain_df['minute'].unique())}")
        print(f"Total strikes per timestamp: {len(chain_df['strike'].unique())}")
        
        print("\nSample of option chain data:")
        sample_time = chain_df['minute'].iloc[0]
       

    return True

def example_usage():
    """Example of how to use the conversion"""
    import time
    start_time = time.time()
    
    # Example date in YYYY-MM-DD format
    date = "2025-02-03"
    
    # Load your pickle files
    with open(f"e:/midcap_files/data_files/{date}.pkl", 'rb') as f:
        df = pickle.load(f)
    
    # Process and save option chain using the actual date
    if process_option_chain(df, date):
        print("Processing completed successfully.")
    else:
        print("Processing failed or skipped due to missing spot data.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    example_usage()
