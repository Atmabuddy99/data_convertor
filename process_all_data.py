import os
import pandas as pd
from convert_single_day import process_option_chain
from datetime import datetime
import traceback
from time import sleep
import duckdb


def process_data_from_db(spot_data_dir, out_dir, index):
    # Database connection setup
    db_config = {
        "BANKNIFTY": {
            "db_file": "bankniftydb.db",
            "table": "bn_opt"
        },
        "NIFTY": {
            "db_file": "niftydb.db",
            "table": "options_data"
        },
        "FINNIFTY": {
            "db_file": "finnifty.db",
            "table": "fin_opt"
        },
        "MIDCP": {
            "db_file": "midcap.db",
            "table": "midcap_opt"
        },
        "SENSEX": {
            "db_file": "sensex.db",
            "table": "options_data"
        }
    }
    
    # Validate index
    if index not in db_config:
        raise ValueError(f"Invalid index: {index}")

    
    
    # Setup database connection
    db_path = "D:\\new_db"
    db_info = db_config[index]
    db_file = os.path.join(db_path, db_info["db_file"])
    
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file not found: {db_file}")
    
    conn = duckdb.connect(db_file)
    table_name = db_info["table"]
    
    try:
        # Get unique dates from database
        query = f"SELECT DISTINCT date FROM {table_name}"
        dates = conn.execute(query).fetchall()
        dates = [pd.to_datetime(date[0]) for date in dates if date[0] != "FUT"]
        dates = sorted(dates)
        
        print(f"Found {len(dates)} unique dates to process for {index}")
        
        success_count = 0
        error_count = 0
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            print(f"\nProcessing {index} data for date: {date_str}")
            print("-" * 50)
            
            # Check if spot data exists
            spot_file = os.path.join(spot_data_dir, f"{date_str}.pkl")
            
            # Check if output directory already exists
            check_out = os.path.isdir(os.path.join(out_dir, date_str))
            if check_out:
                print(f"Option chain directory already exists for {date_str}, skipping...")
                continue
            
            if not os.path.exists(spot_file):
                print(f"No spot data file found for {date_str}, skipping...")
                continue
            
            try:
                # Get data from database for this date
                query = f"""
                    SELECT *
                    FROM {table_name}
                    WHERE date = '{date_str}'
                """
                if index=="SENSEX":
                    query = f"""
                    SELECT *
                    FROM {table_name}
                    WHERE ticker LIKE'{index}%'
                    AND date = '{date_str}'
                """
                df = conn.execute(query).df()
                
                print(f"Loaded {len(df)} rows for {date_str}")
                
                # Process the option chain
                if process_option_chain(df, date_str, index, out_dir, spot_data_dir):
                    success_count += 1
                    print(f"Successfully processed {date_str}")
                else:
                    print(f"Failed to process {date_str}")
                    error_count += 1
            
            except Exception as e:
                error_count += 1
                print(f"Error processing {date_str}:")
                print(traceback.format_exc())
                continue
        
        print(f"\nProcessing complete for {index}!")
        print(f"Successfully processed: {success_count} dates")
        print(f"Failed to process: {error_count} dates")
    
    finally:
        conn.close()

if __name__ == "__main__":
    # Start processing
    print("Starting to process all data files...")
    print("=" * 50)
    
    start_time = datetime.now()
    data_dir = "d:/pickle_nifty/data_files"
    spot_data_dir = "d:/pickle_nifty/spot_data"
    # Base directory for output
    out_dir = r"d:/OPTION_CHAINS/nifty_chain"
    index="NIFTY"

    indexes = {
        "NIFTY": {
            "spot_dir": r"D:\SPOT_DATA_PKL\NIFTY",
            "out_dir": "d:/OPTION_CHAINS/nifty_chain"
        },
        "BANKNIFTY": {
            "spot_dir": r"D:\SPOT_DATA_PKL\BANKNIFTY",
            "out_dir": "d:/OPTION_CHAINS/bank_chain"
        },
        "FINNIFTY": {
            "spot_dir": r"D:\SPOT_DATA_PKL\FINNIFTY",
            "out_dir": "d:/OPTION_CHAINS/fin_chain"
        },
        "MIDCP": {
            "spot_dir": r"D:\SPOT_DATA_PKL\MIDCP",
            "out_dir": "d:/OPTION_CHAINS/mid_chain"
        },
        "SENSEX":{
            "spot_dir": r"D:\SPOT_DATA_PKL\SENSEX",
            "out_dir": "d:/OPTION_CHAINS/option_chain_sensex"
            }
    }
    
    # Process each index
    for index, paths in indexes.items():
        print(f"\nProcessing {index}...")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            process_data_from_db(
                spot_data_dir=paths["spot_dir"],
                out_dir=paths["out_dir"],
                index=index
            )
        except Exception as e:
            print(f"Error processing {index}:")
            print(traceback.format_exc())
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Total processing time for {index}: {duration}")
