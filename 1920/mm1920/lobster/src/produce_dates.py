import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<6):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
path_lobster=path_pipest+'/lobster_for_sdhawkes'
path_lobster_data=path_lobster+'/data'

import sys
import pickle


def main():
    symbol=sys.argv[1]
    path_symbol=path_lobster_data+'/'+symbol
    dates_txt=open(path_symbol+'/dates.txt','r')
    dates=dates_txt.readlines()
    dates_txt.close()
    dates=[dates[n][:-1] for n in range(len(dates))]
    with open(path_symbol+'/dates','wb') as outfile:
       pickle.dump(dates,outfile)
    print('I have pickled "dates" as a python list at '+path_symbol) 


if __name__=="__main__":
    main()

