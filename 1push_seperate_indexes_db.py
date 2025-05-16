



import cProfile
import duckdb
import os
import shutil
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')
INDEXES=["BANKNIFTY","NIFTY","FINNIFTY","MIDCP"]
dbmapper={"BANKNIFTY":"bankniftydb.db","NIFTY":"niftydb.db","FINNIFTY":"finnifty.db","MIDCP":"midcap.db"}
table_names={"BANKNIFTY":"bn_opt","NIFTY":"options_data","FINNIFTY":"fin_opt","MIDCP":"midcap_opt"}




def apply_function(x):
    try:
        idx=x.find(".NFO")
        ticker=x[:idx]
        if ticker[-1]!="I":
            for j,k in enumerate(ticker):
                if k.isdigit():
                    break
            expiry=ticker[len(ticker[:j]):len(ticker[:j])+7]
            
            expiry=datetime.datetime.strptime(expiry,"%d%b%y")
            return expiry
        else:
            return "FUT"
    except:
        print(ticker,expiry)

        
def get_dates(dbname):
    query = f"""
            SELECT DISTINCT date
            FROM {dbname}
            """
    resultqqq = conn.execute(query).fetchall()
    s=[]
    for j in resultqqq:
        if j[0]!="FUT":
            w=pd.to_datetime(j[0])
            s.append(w)
    return s


#tbname=table_names.get(INDEX)
#dates=get_dates(tbname)
#dates=sorted(dates)
        



for i in INDEXES:
    print(i)
    INDEX=i
    main_path="D:\\"
    db_path="new_db"
    csv_gdfl_path=r"fromfebrest"
    path_data=os.path.join(main_path,csv_gdfl_path)
    path_db=os.path.join(main_path,db_path)
    x=os.listdir(path_data)
    path_data=os.path.join(main_path,csv_gdfl_path)
    path_db=os.path.join(main_path,db_path)
    os.chdir(path_db)
    db_to_connect=dbmapper.get(INDEX)
    #if db_to_connect is None:
    #   db_to_connect="midcap.db"
    conn=duckdb.connect(db_to_connect)
    existing_tables = conn.execute("SHOW TABLES").fetchdf()
    print(existing_tables["name"].tolist())
    
    os.chdir(path_db)

    fut1map="-I"
    fut2map="-II"
    fut3map="-III"
    target_variable=-1
    opt=table_names.get(INDEX)
    w=0
    print(datetime.datetime.now(),"initial")
    result=None
    entry=0
    for i in range(len(x)):
            
        try:

            print(x[i])

            a=datetime.datetime.now()
            df=pd.read_csv(os.path.join(path_data,x[i]))
            df=df[df.Ticker.str.startswith(INDEX)]
            if INDEX=="NIFTY":
                df=df[~df['Ticker'].str.contains("NIFTYNXT")]

            df["dt"]=df.Date.astype(str)+" "+df.Time.astype(str)
            df["dt"] = df["dt"].apply(lambda x:x.strip())

            df["dt"]=pd.to_datetime(df["dt"],format='%d/%m/%Y %H:%M:%S')



            df.drop("Time",axis=1,inplace=True)
            df.rename(columns={"Ticker":"ticker",
                            "Date":"date",
                            "Open":"open",
                            "High":"high",
                            "Low":"low",
                            "Close":"close",
                            "Volume":"volume",
                            "Open Interest":"oi"},inplace=True)

            df["date"] = df["date"].apply(lambda x:x.strip())
            df["date"]=pd.to_datetime(df["date"],format='%d/%m/%Y')
            date_to_check=df.date.iloc[0]

            try:
                query = f"""
                        SELECT COUNT(*)
                        FROM {opt}
                        WHERE CAST(date AS DATE) = DATE '{date_to_check}'
                        """
                result = conn.execute(query).fetchall() 
            except Exception as e:
                print(f"{x[i]} for this file in querying")


            if result:
                if result[0][0] > 0:
                    print(f"The date {date_to_check} exists in the table.")
                else:
                    df["expiry"]=df["ticker"].apply(apply_function)
                    print(f"The date {date_to_check} does not exist in the table.")
                    if i==0:
                        conn.sql(f"CREATE TABLE IF NOT EXISTS {opt} AS SELECT * FROM df")
                        conn.sql(f"INSERT INTO {opt}  SELECT * FROM df")
                    else:
                        conn.sql(f"INSERT INTO {opt}  SELECT * FROM df")
            else:
                df["expiry"]=df["ticker"].apply(apply_function)
                if entry==0:
                    print(f"The date {date_to_check} deesnt exists in the table.")
                    conn.sql(f"CREATE TABLE IF NOT EXISTS {opt} AS SELECT * FROM df")
                    entry=1

                else:
                    print(f"The date {date_to_check} deesnt exists in the table.")
                    conn.sql(f"INSERT INTO {opt}  SELECT * FROM df")

            b=datetime.datetime.now()

            print(i,b-a)
        except:
            print("errr",x[i],i)



    print(datetime.datetime.now(),"final",i)
    conn.close()




















   








