import numpy as np
import struct
import pandas as pd
import h5py

# CLASSES #

class Message:
    def __init__(self, msec=-1, type = '.',name = '.', buysell = '.',price = -1, shares = 0,refno=-1):
        self.msec = msec
        self.type = type
        self.name = name
        self.buysell = buysell
        self.price = price
        self.shares = shares
        self.refno = refno

    def values(self):
        values = [int(self.msec)]
        
        if self.type in ('B', 'S'):  # adds
            values.append(1)
        elif self.type == 'E':  # partial execute
            values.append(2)
        elif self.type == 'C':  # partial cancel
            values.append(3)
        elif self.type == 'F':  # execute outstanding in full
            values.append(4)
        elif self.type == 'D':  # delete outstanding in full
            values.append(5)
        elif self.type == 'X':  # bulk volume for the cross event
            values.append(6)
        elif self.type == 'T':  # Execute non-displayed order
            values.append(7)
        else:
            values.append(-1)  # other (ignored)

        values.append(int(self.buysell))
        values.append(int(self.price))
        values.append(int(self.shares))
        values.append(int(self.refno))
        # values.append(int(self.newrefno))

        return np.array(values)

class Order:
    def __init__(self, name='.', buysell='.', price='.', shares='.'):
        self.name = name
        self.buysell = buysell
        self.price = price
        self.shares = shares

class Orderpool:
    def __init__(self):
        self.orders = {}

    # Changes message by REFERENCE.
    def complete_message(self, message):
        if message.refno in self.orders.keys():
            ref_order = self.orders[message.refno]
            # partial execute
            if message.type == 'E':
                message.buysell = ref_order.buysell
                message.price = ref_order.price
            # partial cancel
            if message.type == 'C':
                message.buysell = ref_order.buysell
                message.price = ref_order.price
            # execute outstanding in full
            if message.type == 'F':
                message.buysell = ref_order.buysell
                message.price = ref_order.price
                message.shares = ref_order.shares
            # delete outstanding in full
            if message.type == 'D':
                message.buysell = ref_order.buysell
                message.price = ref_order.price
                message.shares = ref_order.shares

    def update(self, message):
        if message.type in ('B', 'S'):
            self.add_order(message)
        elif message.type in ('E', 'C', 'F', 'D'):
            self.update_order(message)

    def add_order(self, message):
        order = Order()
        order.name = message.name
        order.price = message.price
        order.shares = message.shares
        if message.type == 'B':
            order.buysell = 1
        elif message.type == 'S':
            order.buysell = -1
        self.orders[message.refno] = order

    def update_order(self, message):
        if message.refno in self.orders.keys():
            if message.type == 'E':  # partial execute
                self.orders[message.refno].shares -= message.shares
            elif message.type == 'C':  # partial cancel
                self.orders[message.refno].shares -= message.shares
            elif message.type == 'F':  # execute outstanding in full
                self.orders.pop(message.refno)
            elif message.type == 'D':  # delete outstanding in full
                self.orders.pop(message.refno)

class Book:
    def __init__(self, levels):
        self.bids = {}
        self.asks = {}
        self.levels = levels
        self.msec = -1

    def update(self, message):
        self.msec = message.msec

        if message.buysell == 1:
            if message.price in self.bids.keys():
                if message.type == 'B':
                    self.bids[message.price] += message.shares
                elif message.type in ('E', 'C', 'F','D'):
                    self.bids[message.price] -= message.shares

                    if self.bids[message.price] < 0:
                        print('Warning!!! depth of LOB become negative bid', message.refno)

                    if self.bids[message.price] == 0:
                        self.bids.pop(message.price)
            else:
                if message.type == 'B':
                    self.bids[message.price] = message.shares

        elif message.buysell == -1:
            if message.price in self.asks.keys():
                if message.type == 'S':
                    self.asks[message.price] += message.shares
                elif message.type in ('E', 'C', 'F','D'):
                    self.asks[message.price] -= message.shares

                    if self.asks[message.price] < 0:
                        print('Warning!!! depth of LOB become negative ask', message.refno)
                        
                    if self.asks[message.price] == 0:
                        self.asks.pop(message.price)
            else:
                if message.type == 'S':
                    self.asks[message.price] = message.shares

    def values(self):
        """Convert book to numpy array."""
        values = [int(self.msec)]
        sorted_bids = sorted(self.bids.keys(), reverse=True)
        sorted_asks = sorted(self.asks.keys())
        for i in range(0, self.levels):  # bid price
            if i < len(self.bids):
                values.append(sorted_bids[i])
            else:
                values.append(0)
        for i in range(0, self.levels):  # ask price
            if i < len(self.asks):
                values.append(sorted_asks[i])
            else:
                values.append(0)
        for i in range(0, self.levels):  # bid depth
            if i < len(self.bids):
                values.append(self.bids[sorted_bids[i]])
            else:
                values.append(0)
        for i in range(0, self.levels):  # ask depth
            if i < len(self.asks):
                values.append(self.asks[sorted_asks[i]])
            else:
                values.append(0)
        return np.array(values)


# METHODS #

def get_message(line, message_type):
    message = Message()
    message.type = message_type
    if message_type == 'B':
        message.msec = line[0]
        message.name = line[1]
        message.refno = line[2]
        message.shares = line[4]
        message.price = line[5]
        message.buysell = 1
    if message_type == 'S':
        message.msec = line[0]
        message.name = line[1]
        message.refno = line[2]
        message.shares = line[4]
        message.price = line[5]
        message.buysell = -1
    if message_type in ['E','C']:
        message.msec = line[0]
        message.name = line[1]
        message.refno = line[2]
        message.shares = line[4]
    if message_type in ['F','D']:
        message.msec = line[0]
        message.name = line[1]
        message.refno = line[2]
    if message_type in ['X','T']:
        message.msec = line[0]
        message.name = line[1]
        message.refno = line[2]
        message.shares = line[4]
        message.price = line[5]
        message.buysell = 0
    return message

def writedata(data, file, group, name, date):
    n, m = data.shape
    grp = file.require_group(group)
    name_grp = grp.require_group(name)
        
    date_name_grp = name_grp.require_dataset(date,
                                             shape=(n, m),
                                             maxshape=(None, None),
                                             dtype='i')
    date_name_grp[:, :] = data