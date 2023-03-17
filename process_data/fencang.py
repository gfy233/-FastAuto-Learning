#unicode = utf-8
import json
import numpy as np
from pandas import Series
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import sys
import csv
#sys.setrecursionlimit(1000000)

def data_manager():
    print("data_manager")
    data = {}
    data2={}
    warehouseList=[]
    warecodeList=[]
    with open("../galanz_data.json", 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            raw_data = json.loads(line)
            warehouse = raw_data['whSapName']
            warecode= raw_data['whSapCode']
            if(warehouse is None ):
                filename='../fencang/None.json'
            else:
                filename='../fencang/'+warehouse+'.json'
            
            
            #print(warehouse)
            if warehouse not in data.keys():
                data[warehouse] = 1
                warehouseList.append(raw_data['whSapName'])
                file = open(filename,'w')
                file.write(str(line))
                file.close()

              
            else:
                data[warehouse] += 1
                file = open(filename,'a')
                file.write(str(line))
                file.close()

            if warecode not in data2.keys():
                data2[warecode] = 1
                warecodeList.append(raw_data['whSapCode'])
            else :
                data2[warecode]+=1
  
                
  
    csvfile = open('./fencang_count.csv', 'w') 
    writer = csv.writer(csvfile)
    writer.writerow([ '仓库名', '销量统计'])
    csvfile.close()
    data_dict = {}
    for i in range(len(warehouseList)):
        data_dict[i]=data[warehouseList[i]]
        print(i,end=' ')
        print(warehouseList[i],end=' ')
        print(data[warehouseList[i]])
        csvfile = open('./fencang_count.csv', 'a') 
        writer = csv.writer(csvfile)
        datas = [( warehouseList[i], data[warehouseList[i]])  ]
        writer.writerows(datas)
        csvfile.close()
    f1.close()

    csvfile = open('./fencang_count_code.csv', 'w') 
    writer = csv.writer(csvfile)
    writer.writerow([ '仓库名', '销量统计'])
    csvfile.close()
    data_dict = {}
    for i in range(len(warecodeList)):
        data_dict[i]=data2[warecodeList[i]]
        
        csvfile = open('./fencang_count_code.csv', 'a') 
        writer = csv.writer(csvfile)
        datas = [( warecodeList[i], data2[warecodeList[i]])  ]
        writer.writerows(datas)
        csvfile.close()

    #print(data_dict)
    return data_dict



def show(data):
    series = Series(data)
    #series.plot()
    series.plot.pie(figsize=(8, 8))
    #plt.pie(X,labels=labels,autopct='%1.2f%%') 
    #plt.figure(figsize=(10, 10))
    plt.savefig('fencang_show.jpg')


if __name__ == "__main__":
    data = data_manager()
    #print(data)
    show(data)
    #autocorrelation(data)
