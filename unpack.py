import numpy as np
import pandas as pd
import itch 
import h5py
import sys
import time as timer


DATE_list = ['20181105','20181106','20181107','20181108','20181109',
             '20181112','20181113','20181114','20181115','20181116',
             '20181119','20181120','20181121','20181123','20181126',
             '20181127','20181128','20181129','20181130','20181203',
             '20181204' ]
# stock = 'AAPL'
stock = 'GS'

fout = h5py.File('gs_tick_data.hdf5', 'a')                           
LEVELS = 10

for DATE in DATE_list:
    print(DATE)
    start = timer.time()

    df = pd.read_csv("~/Downloads/"+DATE+"_"+stock+".csv")                            
    df = df.drop('MPID', axis = 1)
    df = df.drop('X', axis = 1)                                                                              
    df_v = df.values

    orderpool = itch.Orderpool() 
    book = itch.Book(LEVELS)
    messagedata = []
    bookdata = []

    for i in range(len(df_v)):
        line  = df_v[i]
        message_type = line[3]
        message = itch.get_message(line,message_type)
        
        # complete message...
        if message_type in ('E', 'C', 'F', 'D'):
            orderpool.complete_message(message)
        
        # update orderpool...
        if message_type in ('B','S','E', 'C', 'F', 'D'):
            orderpool.update(message)

        # update booklist...
        if message_type in ('B','S','E', 'C', 'F', 'D'):
            book.update(message)

        # update messagedata...
        messagedata.append(message.values())

        # update bookdata...
        # check if bookdata is all zero
        book_v = book.values()
        if np.any(book_v[1:]):
            bookdata.append(book_v)

    # messagedata to HDF5...
    messagedata = np.asarray(messagedata)
    group = 'messages'
    itch.writedata(messagedata,fout,group,stock,DATE)

    # bookdata to HDF5...
    bookdata = np.asarray(bookdata)
    group = 'orderbooks'
    itch.writedata(bookdata,fout,group,stock,DATE)

    stop = timer.time()

    # OUTPUT #
    print('Elapsed time:', stop - start, 'sec')

fout.close()
