from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from numba import jit
import numpy as np

import multiprocessing as mp
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
core = 5
preMerge = {} #Define state vector.
S = [] #Define the collector S of all subsets.

alst = [[29,30,25,46,45,54,79,100,67,29,47,24,19,27,32,30,91,155,46,23,43,21,22,20,23,14,25,15,20,25],[0,1,0,0,2,3,35,20,12,11,2,3,2,7,7,6,52,44,4,2,1,3,2,4,12,6,9,10,9,11],[14,12,19,25,15,4,0,0,0,0,0,0,0,0,0,3,6,125,136,0,0,0,1,0,0,0,9,18,112,86],[4,3,8,11,12,10,56,98,51,13,16,16,19,15,20,11,60,50,0,0,1,0,7,8,8,12,10,15,11,18],[1,2,1,3,5,1,2,6,3,2,5,1,3,2,3,0,0,0,5,6,5,2,4,5,1,3,3,8,0,3],[14,12,19,25,15,4,0,0,0,0,0,0,0,0,0,5,12,36,28,0,0,0,1,0,0,2,3,6,5,8]]

arr = np.array(alst)

#Whether the time series is statioanry.
def isStat(x): 
    xr = adfuller(x)
    if xr[0] <= xr[4]['10%'] and xr[1] <= 0.05:
        return 1
    else:
        return 0

#Whether the two time series pass co-integration.
def isCi(x, y):
    xyi = coint(x, y)
    if xyi[0] <= xyi[2][2] and xyi[1] <= 0.05:
        return 1
    else:
        return 0    

#The logic of co-Integration test.
def cointPair(x, y):
    xi = isStat(x)
    yi = isStat(y)
    if xi == 1 and yi == 1:
        xyi = isCi(x, y)
        return xyi
    else:
        xd = np.diff(x)
        yd = np.diff(y)
        xii = isStat(xd)
        yii = isStat(yd)
        if xii == 1 and yii == 1:
            xyii = isCi(x, y)
            return xyii

def mRemove(pre, s):
    for i in s:
        if i in pre:
            del pre[i]
    return pre

def getS(name, mts):
    pre = {}
    for i in range(len(mts)):
        pre[mts[i]] = mts[i] 
        for j in range(mts[i]):
            if cointPair(arr[mts[i]],arr[mts[i] - j - 1]) == 1:
                pre[mts[i]] = mts[i] - j - 1
                break
    return pre

#get all subsets from original MTS by using parallel computing.
#def getS_parallel(MTS):
if __name__ == '__main__':
    pool = mp.Pool(core) #Define processor pool
    param_dict = {}
    part = int(arr.shape[0] / core)
    extr = arr.shape[0] % core   
    for i in range(core):
        param_dict[i] = list(range((i*part),((i+1) * part)))
    param_dict[core-1] = list(range(((core-1)*part),((core) * part + extr)))
    print (core,"Processors:", param_dict)    
    
    start_t = datetime.datetime.now()
    print ("Begin calculation!")
    results = [pool.apply_async(getS, args=(name, rng)) for name, rng in param_dict.items()]
    results = [p.get() for p in results]
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    
    preMerge = {}
    preMerge.update(dict(results[0]))
    for i in range(1, len(results)):
        preMerge.update(dict(results[i]))
    print ("pre status:", preMerge)

    preCopy = preMerge.copy()
    while len(preCopy) > 0:
        lst = [(ks,preCopy[ks]) for ks in sorted(preCopy.keys())]
        st = []
        sind = {}
        k = lst[-1][0]
        st.append(k)
        sind[k] = 1 
        while preCopy[k] != k:
            k = preCopy[k]
            st.append(k)
            sind[k] = 1 
        S.append(st)
        mRemove(preCopy, sind)
    print ("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
    print (S)
