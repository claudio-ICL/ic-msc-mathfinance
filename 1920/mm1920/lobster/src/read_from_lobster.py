#!/usr/bin/env python
# coding: utf-8

import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<6):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
path_sdhawkes=path_pipest+'/sdhawkes'
path_lobster=path_pipest+'/lobster'
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'


import numpy as np
import pandas as pd
import pickle
import datetime
import time

import sys
sys.path.append(path_lobster_pyscripts+'/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_sdhawkes+'/resources/')

import model as sd_hawkes_model
import lob_model
import prepare_from_lobster as from_lobster


symbol='INTC'
date='2019-01-22'
initial_time=float(9.0*60*60)
final_time=float(16*60*60)
time_window=str('{}-{}'.format(int(initial_time),int(final_time)))
first_read_fromLOBSTER=True
dump_after_reading=False
add_level_to_messagefile=True

print('I am reading from lobster')
print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))

saveout=sys.stdout
print('Output is being redirected to '+path_lobster_data+'/{}/{}_{}_{}_readout'.format(symbol,symbol,date,time_window))
fout=open(path_lobster_data+'/{}/{}_{}_{}_readout'.format(symbol,symbol,date,time_window),'w')
sys.stdout=fout


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
add_hawkes_marks=True
clear_same_time_stamp=True
num_iter=4


print('\n\nDATA CLEANING\n')

man_mf=from_lobster.ManipulateMessageFile(
     LOB,messagefile,
     symbol=symbol,
     date=date,
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
     add_hawkes_marks=add_hawkes_marks,
     num_iter=num_iter
)


man_ob=from_lobster.ManipulateOrderBook(
    man_mf.LOB_sdhawkes,symbol=symbol,date=date,
    ticksize=man_mf.ticksize,n_levels=man_mf.n_levels,volume_imbalance_upto_level=2)


data=from_lobster.DataToStore(man_ob,man_mf,time_origin=initial_time)

sym = data.symbol
d = data.date
t_0 = data.initial_time
t_1 = data.final_time

assert sym==symbol
assert date==d

print('\nData is being stored in {}'.format(path_lobster_data))
with open(path_lobster_data+'/{}/{}_{}_{}-{}'.format(sym,sym,d,t_0,t_1), 'wb') as outfile:
    pickle.dump(data,outfile)
    
print('read_from_lobster.py: END OF FILE')

sys.stdout=saveout
fout.close()

print('read_from_lobster.py: END OF FILE')

