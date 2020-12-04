#cython: boundscheck=False, wraparound=False, nonecheck=False 

import os
cdef str path_lobster = os.path.abspath('./')
n=0
while (not os.path.basename(path_lobster)=='lobster') and (n<6):
    path_lobster=os.path.dirname(path_lobster)
    n+=1 
if not os.path.basename(path_lobster)=='lobster':
    raise ValueError("path_lobster not found. Instead: {}".format(path_lobster))
cdef str path_src=path_lobster+'/src'
cdef str path_data=path_lobster+'/data'

import sys
sys.path.append(path_src)
import pandas as pd
import numpy as np
cimport numpy as np
import pickle
import bisect
from libc.math cimport ceil,floor

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t


"""
Dictionary for trades and orders' direction.
We follow LOBSTER convention, i.e. we adopt the perspective of the orderbook and we label events that happend on the bid side with the integer 1, and event that happened on the ask side with the integer -1. This means for example that a limit order to buy a certain number of shares at a certain price is referred to as having direction 1, and a limit order to sell a certain number of shares at a certain price is referred to as having direction -1. Consistently, executions of sell limit orders have direction -1, but notice that such executions are a buyer initiated trades. Analogously, executions of buy limit orders have direction 1, but they are seller initiated trades. As a consequence, market orders in the LOBSTER message file are identified as being the counterparts to  executions of limit orders, and a market order is a buy market order when it causes the execution of a sell limit order (hence direction -1), and a market order is a sell market order when it causes the execution of a buy limit order (hence direction 1). 
"""
        
        
class volume_encoding:
    def __init__(self, np.ndarray[DTYPEf_t, ndim=1] volimb_limits,
                 int n_levels=1,int volume_imbalance_upto_level=1, DTYPEf_t step=0.1,
                 ):
        if volume_imbalance_upto_level>n_levels:
            print('As per input: \n n_levels={}, volume_imbalance_upto_level={}'
                  .format(n_levels,volume_imbalance_upto_level))
            raise ValueError('volume_imbalance_upto_level must be smaller or equal than n_levels')
            
            
        self.n_levels=n_levels
        self.volume_imbalance_upto_level=volume_imbalance_upto_level
#         self.create_df_of_volume_encoding(step)
        try:
            self.store_volimb_limits(volimb_limits)
        except:
            pass
    def compute_volume_imbalance(self,np.ndarray[DTYPEf_t, ndim=2] liquidity_matrix, int upto_level=0):
        """
        liquidity matrix is supposed to have an instance every row. the columns are alternatively ask 
        and bid volumes from level 1 to level n
        """
        cdef np.ndarray[DTYPEf_t, ndim=2] matrix = liquidity_matrix
        if upto_level==0:
            upto_level=self.volume_imbalance_upto_level
        cdef int uplim=1+2*upto_level    
        cdef np.ndarray[DTYPEf_t, ndim=1] vol_ask = np.sum(matrix[:,0:uplim:2],axis=1)
        cdef np.ndarray[DTYPEf_t, ndim=1] vol_bid = np.sum(matrix[:,1:uplim:2],axis=1)
        cdef np.ndarray[DTYPEf_t, ndim=1] vol_imb = np.divide((vol_bid-vol_ask),np.maximum(1.0e-5,vol_bid+vol_ask))
        return vol_imb
    
    def compute_volume_imbalance_scalar(self,np.ndarray[DTYPEf_t, ndim=1] volumes, int upto_level=0):
        if upto_level==0:
            upto_level=self.volume_imbalance_upto_level     
        return compute_volume_imbalance_scalar(volumes,upto_level)    
       
    def classify_vol_imb_scalar(self, DTYPEf_t vol_imb):
        """
        volume imbalance is expected as a scalar with value between -1 and 1
        categories are sorted from the most negative volume imbalance to the most positive
        """
        return classify_vol_imb_scalar(vol_imb, self.volimb_limits)
    
    def classify_vol_imb_vector(self, np.ndarray[DTYPEf_t, ndim=1] vol_imb):
        """
        volume imbalance is expected as a one dimensional vector with values between -1 and 1
        categories are sorted from the most negative volume imbalance to the most positive according to self.volimb_limits
        """
        cdef Py_ssize_t k = 0
        cdef int len_vector = len(vol_imb)
        cdef np.ndarray[DTYPEi_t, ndim=1] classified_vi = np.zeros(len_vector, dtype=DTYPEi)
        for k in range(len_vector):
            classified_vi[k] =  -1+bisect.bisect_left(self.volimb_limits, vol_imb[k])
        assert np.all(classified_vi>=0)    
        return classified_vi    
    def store_volimb_limits(self,np.ndarray[DTYPEf_t, ndim=1] volimb_limits):
        self.volimb_limits = volimb_limits
        
        
cdef long convert_multidim_state_code(
    int num_of_states, DTYPEi_t [:,:] arr_state_enc, DTYPEi_t [:] state) nogil:
    cdef int i=0
    for i in range(num_of_states):
        if ( (arr_state_enc[i,1] == state[0]) & (arr_state_enc[i,2]==state[1]) ):
            return arr_state_enc[i,0]
    return 0         
        
        
cdef double compute_volume_imbalance_scalar(DTYPEf_t [:] volumes, int upto_level=2) nogil:
    cdef int n=0
    cdef DTYPEf_t vol_ask=0.0, vol_bid=0.0
    while n<upto_level:
        vol_ask+=volumes[2*n]
        vol_bid+=volumes[1+2*n]
        n+=1
    return ((vol_bid-vol_ask)/max(1.0e-10,vol_bid+vol_ask))

cdef int classify_vol_imb_scalar(DTYPEf_t vol_imb, DTYPEf_t [:] volimb_limits):
    """
    volume imbalance is expected as a scalar with value between -1 and 1
    categories are sorted from the most negative volume imbalance to the most positive
    """
    return int(max(0,-1+bisect.bisect_left(volimb_limits, vol_imb)))
