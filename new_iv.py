#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cProfile
import duckdb
import os
import shutil
import pandas as pd

import datetime
import warnings
warnings.filterwarnings('ignore')
import calendar
import math


# In[2]:


import os 
import warnings

warnings.filterwarnings('ignore')
import glob
import pandas as pd
import duckdb
import datetime
from datetime import timedelta
import numpy as np
import json
import numpy as np
import math
import warnings
import random
import numpy as np
import math
import warnings
import random
warnings.simplefilter('ignore')

def normal_pdf(x, mu=0, sigma=1):
    return (1 / math.sqrt(2 * math.pi * sigma**2)) * math.exp(-1 * ((x - mu)**2) / (2 * sigma**2))


def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2 * sigma**2))) / 2

data = [random.normalvariate(0, 1) for _ in range(100)]

sample_mean = sum(data) / len(data)
sample_stddev = math.sqrt(sum([(x - sample_mean)**2 for x in data]) / (len(data) - 1))


def isnan(x):
    if int(x) == -9223372036854775808:
        return True
    else:
        return False


def blackScholes(calculation_type, Option_type, K, S, T, sigma, r):
    
    
    K = float(K)
    S = float(S)
    T = float(T)
    T = T/365
    calculation_type = calculation_type.lower()
    Option_type = Option_type.lower()
    #PRICE
    
    # Exceptions 
    
    if isnan(sigma) or sigma == 0: 
        return 0
    
    if isnan(K) or K == 0: 
        return 0
    
    if isnan(S) or S == 0: 
        return 0
   
    if isnan(T) or T == 0: 
        
        return 0
    
    if calculation_type=="p":
        
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        try:
            if Option_type == "c":
                price = S*normal_cdf(d1, 0, 1) - K*np.exp(-r*T)*normal_cdf(d2, 0, 1)
            elif Option_type == "p":
                price = K*np.exp(-r*T)*normal_cdf(-d2, 0, 1) - S*normal_cdf(-d1, 0, 1)
            return price
        except:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")
            
    
    #DELTA
    elif calculation_type=="d":
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        try:
            if Option_type == "c":
                delta_calc = normal_cdf(d1, 0, 1)
            elif Option_type == "p":
                delta_calc = -normal_cdf(-d1, 0, 1)
            return delta_calc
        except:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")
            
    #GAMMA
    elif calculation_type=="g":
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        try:
            gamma_calc = normal_pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))
            return gamma_calc
        except:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")
    
    #VEGA
    elif calculation_type=="v":
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        try:
            vega_calc = S*normal_pdf(d1, 0, 1)*np.sqrt(T)
            return vega_calc*0.01
        except:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")
    
    
    #THETA
    elif calculation_type=="t":
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        try:
            if Option_type == "c":
                theta_calc = -S*normal_pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*normal_cdf(d2, 0, 1)
            elif Option_type == "p":
                theta_calc = -S*normal_pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*normal_cdf(-d2, 0, 1)
            return theta_calc/365
        except:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")


    #RHO
    elif calculation_type=="r":
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        try:
            if Option_type == "c":
                rho_calc = K*T*np.exp(-r*T)*normal_cdf(d2, 0, 1)
            elif Option_type == "p":
                rho_calc = -K*T*np.exp(-r*T)*normal_cdf(-d2, 0, 1)
            return rho_calc*0.01
        except:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")


# In[3]:


def implied_volatility(Option_type, K, S, T, Option_price, r=0, tol=0.0001, max_iterations=100):
    # Option_type = 'c'
    # K = 750
    # S = 1000
    # T = 1
    # Option_price = 1000
    
    
    # print("Option_type: "+str(Option_type))
    # print("STrike::"+str(K))
    # print("Spot:"+str(S))
    # print("DTE"+str(T))
    # print("OPtion Price:"+str(Option_price))
    # print("r:"+str(r))
    
    K = float(K)
    S = float(S)
    T = float(T)
    # num!= num
    # if (Option_price!= Option_price) or Option_price == 0: 
    #     print(f'Option_price: {Option_price}' )
    #     return 0
    
    # if isnan(float(K)) or K == 0:
    #     print(f'Strike Price: {K}')
    #     return 0
    
    # if isnan(float(S)) or S == 0:
    #     print(f'UnderlyingSpot: {S}')
    #     return 0
    
    # if isnan(float(T)) or T == 0:
    #     print(f'DTE: {T}')
    #     return 0
    
   
    
    Option_type = Option_type.lower()
    
    Option_type=Option_type[0]
    #print(Option_type)
    if Option_type == 'c':
        intrinsic = max(S - K, 0)
    else:
        intrinsic = max(K - S, 0)
    
    if Option_price <= intrinsic:
        return 0
    # T = T
    # N_prime = norm.pdf
    # N = norm.cdf
    # # print(N)
    
    
    if ((math.sqrt(2*math.pi)/math.sqrt(T/365))*(Option_price/S) <= 0.03) :
        sigma = 0.2
    elif (math.sqrt(2*math.pi)/math.sqrt(T/365))*(Option_price/S) >= 4 :
        sigma = 2.5
    else:
        sigma = (math.sqrt(2*math.pi)/math.sqrt(T/365))*(Option_price/S)
    
    # print('Sigma',sigma)
    for i in range(max_iterations):
        # print(i)
        if(Option_type == "c"):
            # print(S,K,T,sigma)
            if blackScholes("p", "c", K, S, T, sigma, r)<0.05:
                while(blackScholes("p", "c", K, S, T, sigma, r)<0.05):
                    sigma+=0.1
                    # print(sigma)
            
            diff = blackScholes("p", "c", K, S, T, sigma, r) - Option_price
            # print('Diff',diff)
            # print(diff)
            if abs(diff) < tol:
                # print(f'found on {i}th iteration')
                # print(f'difference is equal to {diff}')
                break
            
            sigma = sigma - (diff / blackScholes("v", "c", K, S, T, sigma, r))/100
            if sigma > 4:
                sigma = 4
            # print('Sigma2',sigma)
            # print(sigma)
        
        else:
            if blackScholes("p", "p", K, S, T, sigma, r)<0.05:
                while(blackScholes("p", "p", K, S, T, sigma, r)<0.05):
                    sigma+=0.01
            diff = blackScholes("p", "p", K, S, T, sigma, r) - Option_price
            # print('diff',diff)
            # print(S,K,T,sigma)
            if abs(diff) < tol:
                # print(f'found on {i}th iteration')
                # print(f'difference is equal to {diff}')
                break

            sigma = sigma - (diff / blackScholes("v", "p", K, S, T, sigma, r))/100
            if sigma > 4:
                sigma = 4
            # print('Sigma3',sigma)
            # print(sigma)
        # if isnan(sigma):
        #     return (math.sqrt(2*math.pi)/math.sqrt(T/365))*(Option_price/S)
        # print (sigma)
    # print('Sigma',sigma)
    return sigma


def get_unique_expirires(conn):
    query = """
            SELECT DISTINCT expiry
            FROM bn_opt
            """
    resultqqq = conn.execute(query).fetchall()
    s=[]
    for j in resultqqq:
        if j[0]!="FUT":
            w=pd.to_datetime(j[0])
            s.append(w)
    return s

#exp_lst=get_unique_expirires()



    

def sel_exp_on_index(index,conn,dbname="options_data"):
    query = f"""
                SELECT DISTINCT expiry
                FROM {dbname} WHERE ticker like '{index}%'
                """
    resultqqq = conn.execute(query).fetchall()
    s=[]
    for j in resultqqq:
        if j[0]!="FUT":
            w=pd.to_datetime(j[0])
            s.append(w)
    return s


weekday_mapping = {
        "Monday": calendar.MONDAY,
        "Tuesday": calendar.TUESDAY,
        "Wednesday": calendar.WEDNESDAY,
        "Thursday": calendar.THURSDAY,
        "Friday": calendar.FRIDAY,
        "Saturday": calendar.SATURDAY,
        "Sunday": calendar.SUNDAY
    }

def get_thursdays(year, month,day):
    week_day_code = weekday_mapping.get(day)
    month_matrix = calendar.monthcalendar(year, month)
    print(week_day_code)
    
    thursdays = [week[week_day_code] for week in month_matrix if week[week_day_code] != 0]
    
    thursday_dates = [datetime.date(year, month, day) for day in thursdays]
    
    return thursday_dates

from datetime import timedelta
def get_expiry_if_holiday(expiry,exp_list):
    #time_=None
    expiry=expiry
    while (expiry  not in exp_list):
        expiry=expiry-timedelta(days=1)
    return expiry


def get_all_symbols_nearby(atm,underlying,expiry,step):
        #atm=int(atm)
        atm=int(step*round(atm/step))
       
        if atm>0:
            ce=[]
            pe=[]
            
            today=datetime.datetime.now()
            if today.strftime("%A")=="Thursday":
                ra=3
            else:
                ra=3
            for u in range(ra):
                e=create_option_symbol(underlying,expiry,atm+(u*step),"ce")
                g=create_option_symbol(underlying,expiry,atm-(u*step),"ce")
                ce.append(e)
                ce.append(g)
            for q in range(ra):
                p=create_option_symbol(underlying,expiry,atm-(q*step),"pe")
                r=create_option_symbol(underlying,expiry,atm+(q*step),"pe")
                pe.append(r)
                pe.append(p)
            combined=ce+pe
            return combined
        else:
            return []



def create_option_symbol(underlying,exp_date,strike,opt_type):
    """
    creating option ticker symbol according to the gdfl data/database data
    
    """
    m=exp_date.strftime("%b").upper()
    y=exp_date.strftime("%Y")[-2:]
    d=exp_date.strftime("%d")
    if not isinstance(opt_type,str):
        opt_type=str(opt_type).upper()
    opt_type=opt_type.upper()
    if not isinstance(underlying,str):
        underlying=str(underlying).upper()
    underlying=underlying.upper()
    symbol=f'{underlying}{d}{m}{y}{strike}{opt_type}.BFO'
    return symbol


def create_future_symbol(underlying,type_):
    """
    here type can be
    near(current_month)
    next(next_month)
    far(far_month)
    defualt is near
    
    """
    if not isinstance(underlying,str):
        underlying=str(underlying).upper()
    underlying=underlying.upper()
    if type_=="near":
        type_='-I'
    if type_=="next":
        type_='-II'
    if type_=="far":
        type_='-III'
    return_fut=f'{underlying}{type_}.BFO'
    return return_fut
    
def db_call_between_dates(symbol,starttime=None,endtime=None,conn=None,dbname="bn_opt"):
    """
    symbol:symbol of the instrument
    starttime:not mandatory(datetime)
    endtime:not mandatory(datetime)
    this functions pulls out the necessary data needed
    
    """
    if not isinstance(symbol,str):
        symbol=str(symbol)
    if (starttime==None) & (endtime==None):
        query=f"SELECT * from {dbname} WHERE ticker='{symbol}'"
    else:
        query=f"SELECT * from {dbname} WHERE ticker='{symbol}' ANd dt BETWEEN '{starttime}' AND '{endtime}'"
    data=conn.sql(query).fetchdf()
    data=data.sort_values(by="date")
    return data


def get_fut(symbol,date,conn):
    """
    to get future dataframe from db
    
    """
    if not isinstance(symbol,str):
        symbol=str(symbol)
    symbol=symbol.upper()
    
    
    st=date.replace(hour=9,minute=15)
    #print(st)
    en=date.replace(hour=15,minute=30)
    
    #print("start",st,"end",en)
    x=db_call_between_dates(symbol,st,en,conn=conn)
    #print(x)
    x=x.drop_duplicates()
    return x

    
def select_particular_exp_ka_data(exp,underlying,conn,db="bn_opt"):
    year="%#y"
    month="%b"
    date="%d"
    ud=underlying.upper()
    y=exp.strftime(year)
    m=exp.strftime(month).upper()
    d=exp.strftime(date)
    tic=f'{ud}{d}{m}{y}'
    query = f"SELECT * FROM {db} WHERE ticker LIKE '{tic}%'"
    data=conn.sql(query).fetchdf()
    data=data.sort_values(by="date")
    return data   

def startswith_fetching_from_db(startswith,conn,db="bn_opt"):
    """
    any data which is starting with any parituclar can be fetched
    with this function
    
    """
    startswith=startswith.upper()
    print(startswith)
    query = f"SELECT * FROM {db} WHERE ticker LIKE '{startswith}%'"
    data=conn.sql(query).fetchdf()
    return data

def options_data_for_a_particular_day(options_df,date,underlying):
    st=date.replace(hour=9,minute=15)
    en=date.replace(hour=15,minute=30)
    options_df=options_df[(options_df["dt"]>=st)&(options_df["dt"]<=en)]
    options_df["otype"]=options_df["ticker"].apply(lambda x :x[-6:-4])
    options_df["strike"]=options_df["ticker"].apply(lambda x:x[len(underlying)+7:17] if len(x)==23 else x[len(underlying)+7:16])
    return options_df

def sum_oi_cepe(options_df,time):
    options_df=options_df[options_df["dt"]==time]
    ce=options_df[options_df["otype"]=="CE"]
    pe=options_df[options_df["otype"]=="PE"]
    ce_oi=int(sum(ce["oi"]))
    pe_oi=int(sum(pe["oi"]))
    return ce_oi,pe_oi




def option_chain_replica(df,atm,datetime,underlying):
    underlying=underlying.upper()
    df["strike"]=df["ticker"].apply(lambda x:int(x[len(underlying)+7:17]) if len(x)==23 else int(x[len(underlying)+7:16]))
    
    




    







def get_symbols(orderbook,options_data,atm,ulying,expiry,step):
    s=set()
    if len(orderbook)>0:
        for order in range(len(orderbook)):
            current_row=orderbook.iloc[order]
            symbol=current_row["symbol"]
            s.add(symbol)
    all_symbols=get_all_symbols_nearby(atm,ulying,expiry,step)
    for symb in all_symbols:
        s.add(symb)
    return s




# to be used to fetch data


    #print(error_symbols)
       
    
    return ltp_mapper,error_symbols


    
    

def get_second_leg_qty_in_lot(int_ratio, lot_size, order_lot_size):
    return int(int_ratio * lot_size * order_lot_size + lot_size / 2) // lot_size

    




def create_orderbook():
    orderbook=pd.DataFrame(columns=["status","symbol","price","time","lots","strike","close","synthetic"])
    return orderbook
    
def analysis(df):
    df.set_index("date",inplace=True)
    df.mtm.cumsum().plot()
    return "plotted"

#import quantstats as qs


def metrics(df,capital):
    d=df.copy()
    capital=capital
    capital
    d["precentage_returns"]=d["mtm"]/capital
    #df_removed.mtm.describe()
    
    qs.reports.metrics(d["precentage_returns"])
    return "Success"


def get_all_symbols_nearby(atm,underlying,expiry,step):
        #atm=int(atm)
        atm=int(step*round(atm/step))
       
        if atm>0:
            ce=[]
            pe=[]
            
            today=datetime.datetime.now()
            if today.strftime("%A")=="Thursday":
                ra=3
            else:
                ra=3
            for u in range(ra):
                e=create_option_symbol(underlying,expiry,atm+(u*step),"ce")
                g=create_option_symbol(underlying,expiry,atm-(u*step),"ce")
                ce.append(e)
                ce.append(g)
            for q in range(ra):
                p=create_option_symbol(underlying,expiry,atm-(q*step),"pe")
                r=create_option_symbol(underlying,expiry,atm+(q*step),"pe")
                pe.append(r)
                pe.append(p)
            combined=ce+pe
            return combined
        else:
            return []





def get_symbols(orderbook,options_data,atm,ulying,expiry,step):
    s=set()
    if len(orderbook)>0:
        for order in range(len(orderbook)):
            current_row=orderbook.iloc[order]
            symbol=current_row["symbol"]
            s.add(symbol)
    all_symbols=get_all_symbols_nearby(atm,ulying,expiry,step)
    for symb in all_symbols:
        s.add(symb)
    return s




# to be used to fetch data



def create_ltp_new(options_data,ulying,expiry,step,time,orderbook):
    x=options_data[(options_data["dt"]==time)]
    ltp_mapper = dict(zip(x["ticker"], x["close"]))
    #vwap_mapper=dict(zip(x["ticker"],x["vwap"]))
    
    return ltp_mapper



def get_second_leg_qty_in_lot(int_ratio, lot_size, order_lot_size):
    return int(int_ratio * lot_size * order_lot_size + lot_size / 2) // lot_size

    


def get_nearest_premium(df,premium,time):
    ce_prem=df[(df["close"]>=premium) & (df["opt_type"]=="CE") & (df["dt"]==time)].sort_values(by="close").iloc[0]
    pe_prem=df[(df["close"]>=premium)& (df["opt_type"]=="PE") & (df["dt"]==time)].sort_values(by="close").iloc[0]
    return ce_prem,pe_prem


def get_ntrading_days(today,previous_day,trading_holidays,nodays):
    #time_=None
    day_list=[]
    previous_day=previous_day
    holidays=["Saturday","Sunday"]
    while True:
        if len(day_list)>=nodays:
            break
        
        while (previous_day in trading_holidays) or (previous_day.strftime("%A") in holidays):
            previous_day=previous_day-timedelta(days=1)
        day_list.append(previous_day)
        previous_day=previous_day-timedelta(days=1)
    return day_list



def add_synthetic___(optiondata,fut_data):
    o=optiondata.copy()
    for index ,row in o.iterrows():
        try:
            ct=row["dt"]
            #print(ct)
            fc=fut_data[fut_data["dtt"]==ct].syn.iloc[-1]
            o.loc[index,"fur"]=fc
        except:
            pass
    return o


def add_synthetic(df,options_data,ulying,expiry,step=100):
    o=df.copy()
    for index,row in o.iterrows():
        ct=row["dtt"]
        c=row["close"]
        atm=int(step*round(c/step))
        fixed_ce=create_option_symbol(ulying,expiry,atm,"ce")
        fixed_pe=create_option_symbol(ulying,expiry,atm,"pe")
        ce_=options_data[(options_data["ticker"]==fixed_ce) & (options_data["dt"]==ct)].close.iloc[-1]
        pe_=options_data[(options_data["ticker"]==fixed_pe) & (options_data["dt"]==ct)].close.iloc[-1]
        sync=atm+ce_-pe_
        o.loc[index,"syn"]=sync
    return o

def addtime_value(options_data):
    options_data["strike"]=options_data["strike"].apply(lambda x :int(x))
    options_data["so"]=np.where(options_data["fur"]>(options_data["strike"]),1,0)
    options_data['result1'] = np.where(options_data['otype'] == 'call', 
                                  np.where(options_data['fur'] > options_data['strike'], options_data["close"]-(options_data["fur"] - options_data["strike"]), options_data["close"]),0)
                                          
    options_data['result2'] = np.where(options_data['otype'] == "put", np.where(options_data['fur'] < options_data['strike'], options_data["close"]-(options_data["strike"] - options_data["fur"]), options_data["close"]), 0)                                      
                                           
    options_data["tv"]=options_data["result1"]+options_data["result2"]
    options_data = options_data.drop(columns=['result1', 'result2'])

    return options_data



def get_atm_strikes(atm,options_df,expiry,u=0,ulying="BANKNIFTY",step=100):
    atm=int(atm)
    current_ce_=create_option_symbol(ulying,expiry,atm+u*(step),"ce")
    current_pe_=create_option_symbol(ulying,expiry,atm+u*(step),"pe")
    return current_ce_,current_pe_


def iron_condor(atm,ulying,expiry,sell_distance,buy_distance):
    sell_strikece=create_option_symbol(ulying,expiry,atm+sell_distance,"ce")
    sell_strikepe=create_option_symbol(ulying,expiry,atm-sell_distance,"pe")
    buy_strikece=create_option_symbol(ulying,expiry,atm+buy_distance,"ce")
    buy_strikepe=create_option_symbol(ulying,expiry,atm-buy_distance,"pe")
    return sell_strikece,sell_strikepe,buy_strikece,buy_strikepe

def get_fut(symbol,date,conn,en=None):
    """
    to get future dataframe from db
    
    """
    if not isinstance(symbol,str):
        symbol=str(symbol)
    symbol=symbol.upper()
    
    
    st=date.replace(hour=9,minute=15)
    if en==None:
        en=date.replace(hour=15,minute=30)
    else:
        en=en.replace(hour=15,minute=30)
    x=db_call_between_dates(symbol,st,en,conn=conn)
    x=x.drop_duplicates()
    return x


def fut_for_n_days(n,start_date,end_date,conn,symbol):
    ff=get_fut(symbol,start_date,conn,end_date)
    return ff



def do_all_the_neccessary_data_preprocessing(date,ulying,conn,exp_lst):
    fut_symbol=create_future_symbol(ulying,"near")
    df=get_fut(fut_symbol,date,conn)
    df=df.sort_values(by="dt")
    df["dt"]=df["dt"].apply(lambda x:x.replace(second=0))
    expiry,next_,monthly=get_nearest_expiry(date,exp_lst,"Thursday")
    options_data=select_particular_exp_ka_data(expiry,ulying,conn)
    options_data["dt"]=options_data["dt"].apply(lambda x:x.replace(second=0))
    options_data["strike"]=options_data["ticker"].apply(lambda x:x[len(ulying)+7:17] if len(x)==23 else x[len(ulying)+7:16])
    options_data["opt_type"]=options_data["ticker"].apply(lambda x:x[-6:-4])
    options_data=options_data_for_a_particular_day(options_data,date,ulying)
    return expiry,next_,monthly,df,options_data




def do_all_the_neccessary_data_preprocessing_positional(date,end_date,ulying,conn,exp_lst,expiry):
    fut_symbol=create_future_symbol(ulying,"near")
    df=fut_for_n_days(1,date,end_date,conn,fut_symbol)
    df=df.sort_values(by="dt")
    df["dt"]=df["dt"].apply(lambda x:x.replace(second=0))
    options_data=select_particular_exp_ka_data(expiry,ulying,conn)
    options_data["dt"]=options_data["dt"].apply(lambda x:x.replace(second=0))
    options_data["strike"]=options_data["ticker"].apply(lambda x:x[len(ulying)+7:17] if len(x)==23 else x[len(ulying)+7:16])
    options_data["opt_type"]=options_data["ticker"].apply(lambda x:x[-6:-4])
    return df,options_data

def get_pnl(orderbook,ltps):
    oo=orderbook.copy()
    oo["ltp"]=oo["symbol"].apply(lambda x :ltps.get(x))
    oo["pnl"]=np.where(oo["status"]=="sell",oo["price"]-oo['ltp'],oo["ltp"]-oo["price"])*oo["lots"]
    return oo.pnl.sum()*25


from datetime import timedelta

def get_all_thursdays(year, month):
    # Get the first day of the month
    first_day = datetime.datetime(year, month, 1)

    # Calculate the difference in days to the next Thursday
    days_until_thursday = (3 - first_day.weekday() + 7) % 7

    # Calculate the first Thursday of the month
    first_thursday = first_day + timedelta(days=days_until_thursday)

    # Generate all Thursdays of the month
    all_thursdays = [first_thursday + timedelta(weeks=i) for i in range(5)]

    # Filter out Thursdays that belong to the next month
    all_thursdays = [thursday for thursday in all_thursdays if thursday.month == month]

    return all_thursdays

def get_nearest_expiry_post_sept(date,list_of_expiries,day,ulying="BANKNIFTY"):
    """
    date as to be given as a datetime without any time 
    mentioned in the datetime
    
    """
    
    if ulying=="BANKNIFTY":
        day2="Thursday"
    kk=[]
    for index in list_of_expiries:
        if index>=date:
            #if (index.strftime("%A")==day):
            kk.append(index)
    mmm=None
    nn=None
    date_ka_month=date.month
    date_ka_year=date.year
    #print(date_ka_month,date_ka_year)

    currrent_month=[date_ for date_ in kk if (date_.month == date_ka_month) & (date_.year==date_ka_year)]
    for iaa in kk:
        #if iaa.strftime("%A")==day:
            diff=abs(iaa-date)
            if mmm is None or (diff< mmm):
                mmm=diff
                nn=iaa
    #print("before",kk)

    kk=sorted(kk)
    #print(kk)
    #print("after",kk)
    kk1=kk[1]
    #print(currrent_month)
    if len(currrent_month)<=1:
        if date_ka_month==12:
            date_ka_month=1
            date_ka_year=date_ka_year+1
            currrent_month=[date_ for date_ in kk if (date_.month == date_ka_month) & (date_.year==date_ka_year)]
        else:
            date_ka_month=date_ka_month+1
            currrent_month=[date_ for date_ in kk if (date_.month == date_ka_month) & (date_.year==date_ka_year)]
    #print("current_month is:",currrent_month) 
    monthly=max(currrent_month)
    return nn,kk1,monthly
    
    
    


def get_nearest_expiry(date,list_of_expiries,day,ulying="BANKNIFTY"):
    """
    date as to be given as a datetime without any time 
    mentioned in the datetime
    
    """
    
    if ulying=="BANKNIFTY":
        day2="Thursday"
    kk=[]
    for index in list_of_expiries:
        if index>=date:
            if (index.strftime("%A")==day):
                kk.append(index)
    mmm=None
    nn=None
    date_ka_month=date.month
    date_ka_year=date.year
    #print(date_ka_month,date_ka_year)

    currrent_month=[date_ for date_ in kk if (date_.month == date_ka_month) & (date_.year==date_ka_year)]
    for iaa in kk:
        if iaa.strftime("%A")==day:
            diff=abs(iaa-date)
            if mmm is None or (diff< mmm):
                mmm=diff
                nn=iaa
    #print("before",kk)

    kk=sorted(kk)
    #print(kk)
    #print("after",kk)
    kk1=kk[1]
    #print(currrent_month)
    if len(currrent_month)<=1:
        if date_ka_month==12:
            date_ka_month=1
            date_ka_year=date_ka_year+1
            currrent_month=[date_ for date_ in kk if (date_.month == date_ka_month) & (date_.year==date_ka_year)]
        else:
            date_ka_month=date_ka_month+1
            currrent_month=[date_ for date_ in kk if (date_.month == date_ka_month) & (date_.year==date_ka_year)]
    #print("current_month is:",currrent_month) 
    monthly=max(currrent_month)
    return nn,kk1,monthly



def get_iv(df,time,date,days_back):
    #da=date-timedelta(days=days_back)
    
    df=df[(df["day"]>=days_back) & (df["day"]<date)]
    
    df=df[(df["time"]==time)]
    return df
    


def generate_spot_dict(date,df):
    """
    generate spot prices dictionary for the given date
    """
    #print(date,type(date))
    df3 = df.query(f"date=='{date}'").copy().sort_values(by="dt")
    #print(df3)
    if len(df3) > 0:
        #df3["ts"] = df3["dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        #cols = ["ts", "spot"]
        df3["dt"]=df3["dt"].apply(lambda x:x.replace(second=0))
        return dict(zip(df3["dt"], df3["close"]))
    else:
        return None

            
    
        
        
    
    
    
        
        
    
        
        
    






