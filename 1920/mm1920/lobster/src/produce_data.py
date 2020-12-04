#!/usr/bin/env python
# coding: utf-8
import os 
import sys
import pickle
path_lobster = os.path.abspath('./')
n=0
while (not os.path.basename(path_lobster)=='lobster') and (n<3):
    path_lobster=os.path.dirname(path_lobster)
    n+=1 
if not os.path.basename(path_lobster)=='lobster':
    raise ValueError("path_lobster not found. Instead: {}".format(path_lobster))
path_data = path_lobster+'/data'
sys.path.append(path_lobster+'/src')
import numpy as np
import pandas as pd
import prepare_from_lobster as from_lobster

# Parameters for data cleaning
n_levels = 2 #Number of levels to take into account in the volumebook
aggregate_time_stamp=True
eventTypes_to_aggregate=[1,2,3,4,5]
eventTypes_to_drop_with_nonunique_time=[]
eventTypes_to_drop_after_aggregation=[3]
only_4_events=False
separate_directions=True
separate_13_events=False
separate_31_events=False
separate_41_events=True
separate_34_events=True
separate_43_events=True
equiparate_45_events_with_same_time_stamp=True
drop_all_type3_events_with_nonunique_time=False
drop_5_events_with_same_time_stamp_as_4=True
drop_5_events_after_aggregation=True
tolerance_when_dropping=1.0e-8
clear_same_time_stamp=True
num_iter=4

def produce(
    symbol='MSFT',
    date='2012-06-21',
    initial_time=float(9.0*60*60),
    final_time=float(16*60*60),
    first_read_fromLOBSTER=True,
    dump_after_reading=False,
    add_level_to_messagefile=True,
    ):
    time_window=str('{}-{}'.format(int(initial_time),int(final_time)))
    print('symbol={}, date={}, time_window={}\n'.format(symbol,date,time_window))
    if (first_read_fromLOBSTER):
        LOB,messagefile=from_lobster.read_from_LOBSTER(symbol,date,
                                          dump_after_reading=dump_after_reading,
                                          add_level_to_messagefile=add_level_to_messagefile)
    else:
        LOB,messagefile=from_lobster.load_from_pickleFiles(symbol,date)
    LOB,messagefile=from_lobster.select_subset(LOB,messagefile,
                                  initial_time=initial_time,
                                  final_time=final_time)
    print('\nDATA CLEANING\n')
    man_mf=from_lobster.ManipulateMessageFile(
         LOB,messagefile,
         symbol=symbol,
         date=date,
         n_levels = n_levels,
         aggregate_time_stamp=aggregate_time_stamp,
         eventTypes_to_aggregate=eventTypes_to_aggregate,  
         only_4_events=only_4_events,
         separate_directions=separate_directions,
         separate_13_events=separate_13_events,
         separate_31_events=separate_31_events,
         separate_41_events=separate_41_events,
         separate_34_events=separate_34_events,
         separate_43_events=separate_43_events,
         equiparate_45_events_with_same_time_stamp=
         equiparate_45_events_with_same_time_stamp,
         eventTypes_to_drop_with_nonunique_time=eventTypes_to_drop_with_nonunique_time,
         eventTypes_to_drop_after_aggregation=eventTypes_to_drop_after_aggregation,
         drop_all_type3_events_with_nonunique_time=drop_all_type3_events_with_nonunique_time,  
         drop_5_events_with_same_time_stamp_as_4=drop_5_events_with_same_time_stamp_as_4,
         drop_5_events_after_aggregation=drop_5_events_after_aggregation,
         tolerance_when_dropping=tolerance_when_dropping,
         clear_same_time_stamp=clear_same_time_stamp,
         num_iter=num_iter
    )
    print("END OF DATA CLEANING")
    man_ob=from_lobster.ManipulateOrderBook(
        man_mf.aggregated_LOB,symbol=symbol,date=date,
        ticksize=man_mf.ticksize,n_levels=man_mf.n_levels,volume_imbalance_upto_level=2)
    data=from_lobster.DataToStore(man_ob,man_mf,time_origin=initial_time)
    return data
