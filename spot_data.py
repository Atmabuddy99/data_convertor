import os
import pickle
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')





def apply_function(x):
    idx=x.find(".BFO")
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
  
        

main_path_finnifty=r"D:\SPOT_DATA_PKL\pickle_finnifty"
main_path_nifty=r"D:\SPOT_DATA_PKL\pickle_nifty"
main_path_banknifty=r"D:\SPOT_DATA_PKL\pickle_banknifty"
main_path_midcap=r"D:\SPOT_DATA_PKL\pickle_midcp"
vix_data_path=r"D:\SPOT_DATA_PKL\vix"


fn=r'D:\NSE_INDEX_MINUTE'





os.chdir(fn)
os.getcwd()
x=os.listdir()



path_index={"NIFTY":main_path_nifty,"BANKNIFTY":main_path_banknifty,"FINNIFTY":main_path_finnifty,"MIDCP":main_path_midcap,"VIX":vix_data_path}


mapper_index={"NIFTY":"NIFTY 50.NSE_IDX","BANKNIFTY":"NIFTY BANK.NSE_IDX",
              "FINNIFTY":"NIFTY FIN SERVICE.NSE_IDX","MIDCP":"NIFTY MID SELECT.NSE_IDX","VIX":"INDIA VIX.NSE_IDX"}



opt="options_data"
entry=0
no_count=0
yes_count=0
total_count=0
nocount_last=0


INDEXES=["NIFTY","BANKNIFTY","FINNIFTY","MIDCP","VIX"]
for i in x:
    try:
        df=pd.read_csv(os.path.join(fn,i))
        for index in INDEXES:
            print(index)
            copydf=df.copy()
            copydf=copydf[copydf["Ticker"].str.startswith(mapper_index.get(index))]
            copydf['Time'] = pd.to_datetime(copydf['Time'], format='%H:%M:%S').dt.time
            start_time = pd.to_datetime("09:15:00").time()
            end_time = pd.to_datetime("15:30:00").time()
            copydf = copydf[(copydf['Time'] >= start_time) & (copydf['Time'] <= end_time)]
            
            copydf["dt"]=copydf.Date.astype(str)+" "+copydf.Time.astype(str)
            copydf["dt"] = copydf["dt"].apply(lambda x:x.strip())
            copydf["dt"]=pd.to_datetime(copydf["dt"],format='%d/%m/%Y %H:%M:%S',errors='coerce')#.dt.strftime('%Y-%m-%d %H:%M:%S')
            copydf.drop("Time",axis=1,inplace=True)
            copydf.columns = copydf.columns.str.replace(' ', '', regex=False)
            copydf.rename(columns={"Ticker":"ticker",
                            "Date":"date",
                            "Open":"open",
                            "High":"high",
                            "Low":"low",
                            "Close":"close",
                            "Volume":"volume",
                            "OpenInterest":"oi"},inplace=True)
            copydf["date"] = copydf["date"].apply(lambda x:x.strip())
            copydf["date"]=pd.to_datetime(copydf["date"],format='%d/%m/%Y')
            date_to_check=copydf.date.iloc[0]
            fil_name=f'{str(date_to_check)[:10]}.pkl'
            
            fn2=path_index.get(index)
            if os.path.isfile(os.path.join(fn2,fil_name)):
                aaaaaaa=1
            elif os.path.isfile(os.path.join(fn2,fil_name)):
                print("yes",fil_name,"do nothing as file already exists in special trading day file")
            else:
                pickle_file_path = os.path.join(fn2,fil_name)
                with open(pickle_file_path, 'wb') as file:
                    pickle.dump(copydf ,file)
                print("saving the pickle file in directory",fil_name)

    except Exception as e:
        print(f"{i} for this file in querying",e)

















