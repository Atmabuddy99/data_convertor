import duckdb
import pandas as pd
import os
from collections import defaultdict

def check_data_by_year():
    # Connect to database
    print("Connecting to database...")
    main_path = "D:\\"
    db_path = "MAIN_DB"
    path_db = os.path.join(main_path, db_path)
    
    try:
        os.chdir(path_db)
    except Exception as e:
        print(f"Error changing directory: {str(e)}")
        return
        
    conn = None
    try:
        conn = duckdb.connect("data_backtesting.db")
        print("\nAvailable tables:")
        tables = conn.execute("SHOW TABLES").fetchdf()
        print(tables["name"].tolist())
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return
    
    # Query all distinct years and months
    query = """
    SELECT DISTINCT date
    FROM options_data
    WHERE date != 'FUT'
    ORDER BY date
    """
    
    try:
        result = conn.execute(query).fetchall()
        dates_by_year = defaultdict(lambda: set())
        
        for row in result:
            try:
                date = pd.to_datetime(row[0])
                dates_by_year[date.year].add(date.month)
            except:
                continue
        
        print("\nData availability by year:")
        for year in sorted(dates_by_year.keys()):
            months = sorted(dates_by_year[year])
            missing_months = set(range(1, 13)) - set(months)
            
            print(f"\nYear {year}:")
            print(f"Available months: {', '.join(str(m) for m in months)}")
            if missing_months:
                print(f"Missing months: {', '.join(str(m) for m in sorted(missing_months))}")
            
            # Get record count for first and last date of the year
            if months:
                first_month = min(months)
                last_month = max(months)
                
                first_query = f"""
                SELECT MIN(date), COUNT(*) 
                FROM options_data 
                WHERE date LIKE '{year}-{first_month:02d}%'
                """
                last_query = f"""
                SELECT MAX(date), COUNT(*) 
                FROM options_data 
                WHERE date LIKE '{year}-{last_month:02d}%'
                """
                
                first_date, first_count = conn.execute(first_query).fetchone()
                last_date, last_count = conn.execute(last_query).fetchone()
                
                print(f"First date: {first_date} ({first_count} records)")
                print(f"Last date: {last_date} ({last_count} records)")
                
    except Exception as e:
        print(f"Error: {str(e)}")
    
    conn.close()

if __name__ == "__main__":
    check_data_by_year()
