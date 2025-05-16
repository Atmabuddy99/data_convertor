import duckdb
import os
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    # Step 1: Connect to database
    print("Step 1: Connecting to database...")
    db_path = r'D:\MAIN_DB\data_backtesting.db'
    print(f"Database path: {db_path}")
    conn = duckdb.connect(db_path)
    
    # Step 2: Get unique dates
    print("\nStep 2: Getting unique dates...")
    query = """
    SELECT DISTINCT date 
    FROM options_data 
    WHERE date != 'FUT' 
    ORDER BY date
    """
    dates = []
    try:
        result = conn.execute(query).fetchall()
        for row in result:
            try:
                date = pd.to_datetime(row[0])
                dates.append(date)
            except:
                continue
        dates = sorted(dates)
        print(f"Found {len(dates)} unique dates")
    except Exception as e:
        print(f"Error getting dates: {str(e)}")
        return
    
    # Step 3 & 4: Create directory and process each date
    print("\nStep 3 & 4: Processing each date...")
    output_dir = "date_wise_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 5: Process all dates
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        output_file = os.path.join(output_dir, f'{date_str}.csv')
        print(f"\nProcessing {date_str}...")
        
        # Fetch data for this date
        query = f"""
        SELECT *
        FROM options_data
        WHERE date = '{date_str}'
        """
        
        try:
            df = conn.execute(query).df()
            if not df.empty:
                # Save to CSV
                df.to_csv(output_file, index=False)
                print(f"Created {output_file} with {len(df)} rows")
            else:
                print(f"No data found for {date_str}")
        except Exception as e:
            print(f"Error processing {date_str}: {str(e)}")
    
    conn.close()
    print("\nCompleted! All CSV files are in the 'date_wise_data' directory")

if __name__ == "__main__":
    main()
