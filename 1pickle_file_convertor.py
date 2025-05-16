

import cProfile
import duckdb
import os
import shutil
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')



def db_call_between_dates_startswith(symbol,starttime=None,endtime=None,conn=None,dbname="bn_opt"):
    """
    symbol:symbol of the instrument
    starttime:not mandatory(datetime)
    endtime:not mandatory(datetime)
    this functions pulls out the necessary data needed
    
    """
    if not isinstance(symbol,str):
        symbol=str(symbol)
    if (starttime==None) & (endtime==None):
        query=f"SELECT * from {dbname} WHERE ticker LIKE'{symbol}%'"
    else:
        print("HERE")
        starttime= starttime.replace(hour=9,minute=15)
        endtime=endtime.replace(hour=15,minute=30)
        query=f"SELECT * from {dbname} WHERE ticker LIKE'{symbol}%' AND dt BETWEEN '{starttime}' AND '{endtime}'"
    data=conn.sql(query).fetchdf()
    return data

def generate_options_dict(date,df):
    """
    generate spot prices dictionary for the given date
    """
    df3 = df.query(f"date=='{date}'").copy().sort_values(by="dt")
    if len(df3) > 0:
        df3["dt"] = df3["dt"].dt.floor('T')
        return df3
    else:
        return None




main_path_finnifty=r"E:\\finnifty_pickle\\data_files"
main_path_nifty=r"E:\\pickle_nifty\\data_files"
main_path_banknifty=r"E:\pickle_banknifty\2023-2024"
main_path_midcap="E:\midcap_files\\data_files"
dbmapper={"BANKNIFTY":"bankniftydb.db","NIFTY":"niftydb.db","FINNIFTY":"finnifty.db","MIDCP":"midcap.db"}
table_names={"BANKNIFTY":"bn_opt","NIFTY":"options_data","FINNIFTY":"fin_opt","MIDCP":"midcap_opt"}
filenames_to_fetch={"BANKNIFTY":main_path_banknifty,"NIFTY":main_path_nifty,"FINNIFTY":main_path_finnifty,"MIDCP":main_path_midcap}




totaltb2=[]
from collections import Counter
from collections import defaultdict
daterange=pd.date_range("12-1-2024","10-1-2025")
port_delta={}
orderbook_list=[]
holidays=["Saturday","Sunday"]
adf=[]
trade_df__=pd.DataFrame(columns=["date","mtm"])
daterange[0].strftime("%A")
port_delta={}
orderbook_list=[]
import pickle



main_path="E:\\"
db_path="new_db"
csv_gdfl_path=r"gfdl_data1"
path_data=os.path.join(main_path,csv_gdfl_path)
path_db=os.path.join(main_path,db_path)
#x=os.listdir(path_data)
path_data=os.path.join(main_path,csv_gdfl_path)
path_db=os.path.join(main_path,db_path)
os.chdir(path_db)


INDEXES=["BANKNIFTY","NIFTY","FINNIFTY","MIDCP"]
for i in INDEXES:
    INDEX=i
    db_to_connect=dbmapper.get(INDEX)
    conn=duckdb.connect(db_to_connect)
    existing_tables = conn.execute("SHOW TABLES").fetchdf()
    print(existing_tables["name"].tolist(),INDEX)
    table_names.get(INDEX)
    
    options_data=db_call_between_dates_startswith(INDEX,daterange[0],daterange[-1],conn=conn,dbname=table_names.get(INDEX))
    dates = set(options_data["date"])
    options_dict={dt:generate_options_dict(dt,options_data) for dt in dates}
    
    folder_to_fetch=filenames_to_fetch.get(INDEX)
    folder_to_fetch_speical=os.path.join(folder_to_fetch,"special_trading_day")
    print(folder_to_fetch)
    filenames=os.listdir(folder_to_fetch)
    for i ,j  in options_dict.items():
        fil_name=f'{str(i)[:10]}.pkl'
        if os.path.isfile(os.path.join(folder_to_fetch,fil_name)):
            print("yes",fil_name,"do nothing as file already exists")


        elif os.path.isfile(os.path.join(folder_to_fetch_speical,fil_name)):
            print("yes",fil_name,"do nothing as file already exists in special trading day file")

        else:

            pickle_file_path = os.path.join(folder_to_fetch,fil_name)
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(j, file)
            print("saving the pickle file in directory",fil_name)
    conn.close()
print("done baby")





