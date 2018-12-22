import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

def load_messages(path, name, date):
    data = h5py.File(path, 'r')
    messages = data['/messages/' + name + '/' + date]
    mdata = messages[:, :]
    t, n = mdata.shape
    data.close()
    mcolumns = ['msec',
                'type',
                'buysell',
                'price',
                'shares',
                'refno']
    mout = pd.DataFrame(mdata, index=np.arange(0, t), columns=mcolumns)
#     mout["time"]= pd.to_datetime(mout["msec"],unit='ms',origin=pd.Timestamp(date)) 
#     mout = mout[['time','msec', 'type', 'buysell', 'price', 'shares', 'refno']]    
    return mout


def load_books(path, name, date):
    data = h5py.File(path, 'r')
    orderbooks = data['/orderbooks/' + name + '/' + date]
    mdata = orderbooks[:, :]
    t, n = mdata.shape
    data.close()
    # columns names
    time = ['msec']
    bid_price = ['bp'+str(i) for i in range(1,11)]
    ask_price = ['ap'+str(i) for i in range(1,11)]
    bid_volumn = ['bv'+str(i) for i in range(1,11)]
    ask_volumn = ['av'+str(i) for i in range(1,11)]
    mcolumns = time+bid_price+ask_price+bid_volumn+ask_volumn
    mout = pd.DataFrame(mdata, index=np.arange(0, t),columns=mcolumns)
    return mout

def vwap_series(df, tinterval):
    df['sec'] = df['msec']/1000
    vwap_list = []
    df_v = df.values
    time = 34200
    temp = []
    for i in range(len(df_v)):
        if df_v[i][6]>= time and df_v[i][6]< time+tinterval:
            temp.append([df_v[i][3],df_v[i][4]])
        if df_v[i][6]>= time+tinterval:
            time = time+tinterval
            vol_time_price = [x[0]*x[1] for x in temp]
            if sum([x[1] for x in temp]) != 0:
                vwap_list.append(sum(vol_time_price)/sum([x[1] for x in temp]))
                temp = []
            else:
                vwap_list.append(np.nan)
                temp = []
        if i == len(df_v)-1:
            # multiply volume by price for each row in the 10s interval
            vol_time_price = [x[0]*x[1] for x in temp]
            if sum([x[1] for x in temp]) != 0:
                # sum all the vol*p and divide by total volume to get vwap
                vwap_list.append(sum(vol_time_price)/sum([x[1] for x in temp]))
            else:
                vwap_list.append(np.nan)
    return vwap_list 





DATE_list = ['20181105','20181106','20181107','20181108','20181109',
             '20181112','20181113','20181114','20181115','20181116',
             '20181119','20181120','20181121','20181126',
             '20181127','20181128','20181129','20181130','20181203',
             '20181204' ]

df_mult_date = pd.DataFrame()

for DATE in DATE_list:

    # Goldman Sachs message data
    # df = load_messages('gs_tick_data.hdf5', 'GS', DATE)
    # Apple message data
    df = load_messages('/Volumes/easystore/FML_project/aapl_tick_data.hdf5', 'AAPL', DATE)
    df = df[(df['msec'] >= 34200000) & (df['msec'] <= 57600000)]
    ex = df[df['type'].isin([2,4,6,7])]
    ex = ex.reset_index(drop=True)
    ex['price'] = ex['price']/10000

    # computing 10 second vwap series
    vwap = vwap_series(ex,10)
    sec = list(range(34210,57601,10))
    vwap_df = pd.DataFrame()
    vwap_df['vwap'] = vwap
    vwap_df['sec'] = sec
    vwap_df['msec'] = vwap_df['sec']*1000
    vwap_df["time"]= pd.to_datetime(vwap_df["msec"],unit='ms',origin=pd.Timestamp(20181204)) 
    vwap_df = vwap_df.dropna()
    # l.plot(x='time', y='vwap')
    # plt.show()

    # Goldman Sachs Tick data
    # book = load_books('gs_tick_data.hdf5', 'GS', DATE)
    # Apple tick data
    book = load_books('/Volumes/easystore/FML_project/aapl_tick_data.hdf5','AAPL',DATE)
    book = book[(book['msec'] >= 34200000) & (book['msec'] <= 57600000)]
    book = book.reset_index(drop=True)
    book = book[['msec', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'bv1', 'bv2', 'bv3', 'bv4', 'bv5', 'av1', 'av2', 'av3', 'av4', 'av5']]
    book['origion'] = 1

    msec = [x*1000 for x in list(range(34200,57600,10))]
    mcolumns = ['msec', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'bv1', 'bv2', 'bv3', 'bv4', 'bv5', 'av1', 'av2', 'av3', 'av4', 'av5','origion']
    a = np.empty((len(msec),22,))
    a[:] = np.nan
    insert_book = pd.DataFrame(a,index=np.arange(0, len(msec)),columns=mcolumns)
    insert_book['msec'] = msec
    insert_book['origion'] = 0

    # merge
    frames = [book, insert_book]
    y = pd.concat(frames,ignore_index=True)
    y = y.sort_values(by=['msec'])
    y = y.fillna(method='ffill')
    # pull out
    new_book = y[y['origion']==0]
    new_book = new_book.dropna()
    new_book = new_book.drop(columns=['origion'])

    d = vwap_df.join(new_book.set_index('msec'), on='msec')
    d = d.dropna()
    d = d.drop(columns=['time','sec'])

    cols = ['bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'bv1', 'bv2', 'bv3', 'bv4', 'bv5', 'av1', 'av2', 'av3', 'av4', 'av5']
    for col in cols:
        d['delta_'+col] = d[col].diff(1)

    d['mean_volumn_diff'] = (d['bv1']+d['bv2']+d['bv3']+d['bv4']+d['bv5'])/5 - (d['av1']+d['av2']+d['av3']+d['av4']+d['av5'])/5
    d['spread'] = d['ap1'] - d['bp1']
    d['vol_unb1'] = (d['bv1'] - d['av1'])/d['bv1']
    d['vol_unb2'] = (d['bv2'] - d['av2'])/d['bv2']
    d['vol_unb3'] = (d['bv3'] - d['av3'])/d['bv3']
    d['vol_unb4'] = (d['bv4'] - d['av4'])/d['bv4']
    d['vol_unb5'] = (d['bv5'] - d['av5'])/d['bv5']

    d_v = d.values
    mom_b = [np.nan,np.nan,np.nan,np.nan,np.nan]
    volat_b = [np.nan,np.nan,np.nan,np.nan,np.nan]
    mom_a = [np.nan,np.nan,np.nan,np.nan,np.nan]
    volat_a = [np.nan,np.nan,np.nan,np.nan,np.nan]

    # why volataility for last 5 
    for i in range(5,len(d_v)):
        bp_past5 = np.asarray([d_v[i-1][2]/10000,d_v[i-2][2]/10000,d_v[-3][2]/10000,d_v[-4][2]/10000,d_v[i-5][2]/10000])
        ap_past5 = np.asarray([d_v[i-1][7]/10000,d_v[i-2][7]/10000,d_v[-3][7]/10000,d_v[-4][7]/10000,d_v[i-5][7]/10000])
        mom_b.append((d_v[i][2]-d_v[i-5][2])/d_v[i-5][2])
        volat_b.append(bp_past5.std())
        mom_a.append((d_v[i][7]-d_v[i-5][7])/d_v[i-5][7])
        volat_a.append(ap_past5.std())
    d['mom_bp1'] = mom_b
    d['mom_ap1'] = mom_a
    d['vola_bp1'] = volat_b
    d['vola_ap1'] = volat_a

    label1 = []
    label2 = []
    for i in range(len(d_v)-1):
        if d_v[i+1][0]>d_v[i][0]:
            label1.append(1)
        if d_v[i+1][0]<d_v[i][0]:
            label1.append(-1)
        label2.append(d_v[i+1][0])

    label1.append(np.nan)
    label2.append(np.nan)

    d['vwap_d'] = label1
    d['vwap_v'] = label2
    d = d.dropna()
    d = d.reset_index(drop = True)

    frames = [df_mult_date, d]
    df_mult_date = pd.concat(frames,ignore_index=True)

# AAPL limit order features data
df_mult_date.to_csv('labelled_data_10s_AAPL', index=False)
    
