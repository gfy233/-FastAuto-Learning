# 将结果细表汇总成总表
import csv
import numpy as np
import os
import pandas as pd
import datetime

date_= '0329_0411'
#原始测试结果文件夹
global raw_test_result
raw_test_result = '/data1/lxt/galanz_test/stacking/FeatureResult/result_'+date_+'/'
#汇总测试结果文件夹
global outFilePath
outFilePath = '/data1/lxt/galanz_test/stacking/outResult/outresult_'+date_+'/'

# 训练测试汇总csv
global ALL_result_file
ALL_result_file = '/data1/lxt/galanz_test/stacking/allResult/ALL_'+date_+'成本.csv'


#原始预测结果文件夹
global raw_future_result
raw_future_result = '/data1/lxt/galanz_test/stacking/FeatureResult/result_future_'+date_+'/'
#预测结果文件夹
global outFilePath_future
outFilePath_future = '/data1/lxt/galanz_test/stacking/outResult/outresult_future_'+date_+'/'
#预测结果汇总csv
global ALL_result_file_future
ALL_result_file_future = '/data1/lxt/galanz_test/stacking/allResult/ALL_future_'+date_+'成本.csv'



#匹配预测结果的changedata时间序列文件 (需要在最新的接口匹配)
global chandata_file
chandata_file = '/data1/lxt/galanz_test/all_data/changeData_fencang2/'
#指数平滑加权平均，a2为指数平滑的权值
global a1
a1 = 0.1
global a2
a2 = 0.9

global zhishu_file
zhishu_file = '/data1/lxt/galanz_test/stacking/ES2/21.0301-21.0314-erci.csv'

global price_file
price_file = '/data1/lxt/galanz_test/stacking/galanz5_price.csv'

jiemi_file = './全仓库解密列表.csv'

#当月各仓库周转天数
arti_zhouzhuan_file='/data1/lxt/galanz_test/stacking/周转/out.csv'

#人工日结在库量（分组合）
arti_daily_stock_cost_file = '/data1/lxt/galanz_test/stacking/日结/'+date_+'_arti_out.csv'

#模型日结在库量（分组合）
model_daily_stock_cost_file = '/data1/lxt/galanz_test/stacking/日结/'+date_+'_model.csv'


#匹配预测窗口期间内的真实销量
def find_in_changeData(uID,uWname,beginDate,endDate):
    #分产品分仓的时间序列csv文件
	print("1");
	files= os.listdir(chandata_file)	
	for file in files: 
		
		filename = chandata_file + file
		Id = file.split("-")[1]
		Wname = file.split("-")[2].split(".")[0]
        #寻找id 和仓库名对应的时序csv文件
		sales_count = 0
		day_count = 0
		#print(Id,uID,Wname,uWname)
		if(str(Id) == str(uID) and str(Wname)==str(uWname)):
		#获取预测窗口内的销量
			print("！！！")
			sales_count = 0
			#记录预测窗口是否满14天
			csvfile = open(filename, 'r',encoding='utf-8') 
			reader = csv.reader(csvfile)
	
			beginDate = datetime.date(int(beginDate.split("-")[0]), int(beginDate.split("-")[1]), int(beginDate.split("-")[2]))
			endDate = datetime.date(int(endDate.split("-")[0]), int(endDate.split("-")[1]), int(endDate.split("-")[2]))
			#print(beginDate,endDate)
			for i,line in enumerate(reader):
				if(i==0):
					continue
				
				nowtime  = datetime.date(int(line[0].split("-")[0]), int(line[0].split("-")[1]), int(line[0].split("-")[2]))
				if(str(nowtime)>=str(beginDate) and str(nowtime) <= str(endDate)):
					sales_count += int(line[1])
					day_count += 1
			print(sales_count)
			return sales_count,day_count
	return 0,0


def dataManeger_future(filename):
	csvfile = open(filename, 'r',encoding='utf-8') 
	reader = csv.reader(csvfile)

    #写入csv outresult
	#print("filename",filename)
	outFileName = outFilePath_future + filename.split("/")[-1]
	#outFileName = outFilePath_future + filename.split("/")[2]
	writer = csv.writer(open(outFileName, 'w',newline='') )
	writer.writerow([filename])
	writer.writerow(['检索序号','id', '仓库','最优组合编号','最优模型预测值','预测窗口开始日期','预测窗口截止日期'])

	for i,raw in enumerate(reader):
		Id = raw[0]
		wname = raw[1]
		
		if(str(Id) == 'nan'):
			continue


		#计算双十一系数
		#writer.writerow(['id', '仓库','最优组合编号','最优模型预测值','真实值','预测窗口开始日期','预测窗口截止日期'])
		#对每一个产品

		# 读取ALL RESULT— 测试集结果，选择最优模型的预测值
		csvfile = open(ALL_result_file, 'r',encoding='utf-8') 
		reader = csv.reader(csvfile)
		for j,line in enumerate(reader):
			#匹配Id 仓库品
			#print("filename",filename.split("/")[-1].split(".")[0].split("_")[1])
			if(str(line[0])==str(Id) and filename.split("/")[-1].split(".")[0].split("_")[1] ==str(line[1]) ):
				#获取最优组合编号、预测值、日期
				
				model_count = line[2]
				pred = raw[int(model_count)*5+4]
				beginDate = raw[int(model_count)*5+2]
				endDate = raw[int(model_count)*5+3]

				#计算双十一系数
				#real = data[str(model_count)+'real'].mean()

				#写入csv]
				writer = csv.writer(open(outFileName, 'a',newline='') )
				writer.writerows([(str(wname)+str(Id),Id,wname,model_count,pred,beginDate,endDate)])
				#计算双十一系数
				#writer.writerows([(i, data['0future_whSapName'].max(),model_count,pred,real,beginDate,endDate)])

def dataManeger(filename):

	datas = pd.read_csv(filename)
	#列名列表
	colsList=datas.columns.tolist()

	idList=list(set(datas['0id']))

    #写入csv
	outFileName=outFilePath+filename.split("/")[-1]
	#outFileName=outFilePath+filename.split("/")[3]

	csvfile = open(outFileName, 'w',newline='') 
	writer = csv.writer(csvfile)
	writer.writerow([filename])
	writer.writerow(['id', '仓库','最优组合编号','基模型','元模型','真实销量平均值','最优模型预测平均值','最优模型mse',"最优模型mae","最优模型d_value","mae方差","窗口数量"])

    #对每一个产品
	for i in idList:
		if(str(i) == 'nan'):
			continue
		data=datas[datas['0id']==i]
		#print(data['0actual'])
		actucal_ave = data['0actual'].mean()

		min_mse=float('inf')	
		min_mae=float('inf')
		min_maeVar=float('inf')
		# 选取5组测试集平均MSE最小的组合
		for j in range(12):
			colName=str(j)+"mse_stacking"
		
			colName_mae=str(j)+"mae_stacking"
			if(data[colName].mean() < min_mse):
				min_mse = data[colName].mean()
				min_mae=data[colName_mae].mean()
				#求五个窗口MAE的方差
				count = data[colName].count()
				#print("count",count)
				min_maeVar=data[colName_mae].var()
				k=j
		

		#计算最优模型预测平均值
		pred_ave = data[str(k)+'pred'].mean()
		#获取最优模型
		min_metaModel=colsList[k*11+2]
		min_Model=colsList[k*11+3]
		writer.writerows([(i, list(set(data['0whSapName']))[0],k,min_metaModel,min_Model,actucal_ave,pred_ave,min_mse,min_mae,pred_ave-actucal_ave,min_maeVar,count)])

#大促节日系数
def get_cofficient(beginDate,endDate):
	date_10_20 =  str(datetime.date(2020, 10, 20))
	date_11_11 =  str(datetime.date(2020, 11, 11))
	date_6_15 =  str(datetime.date(2020, 6, 15))
	date_6_18 =  str(datetime.date(2020, 6, 18))
	date_9_28 =  str(datetime.date(2020, 9, 28))

	#双十一系数
	if((str(beginDate) < str(date_10_20) and str(endDate) >= str(date_10_20))
	or (str(beginDate) >= str(date_10_20) and str(endDate) <= str(date_11_11))
	or (str(beginDate) <= str(date_11_11) and str(endDate) > str(date_11_11))
	):
		#print("双十一系数")
		return 2.24
	# 618系数
	if((str(beginDate) < str(date_6_15) and str(endDate) >= str(date_6_15))
	or (str(beginDate) >= str(date_6_15) and str(endDate) <= str(date_6_18))
	or (str(beginDate) <= str(date_6_18) and str(endDate) > str(date_6_18))
	):
		return 1.91

	#9.28系数
	if(str(beginDate) <= str(date_9_28) and str(endDate) >= str(date_9_28)):
		#print("928系数")
		return 2.075	

	return 1

#根据仓库解密列表，返回wid对应的仓库名称
def get_warehouse_name(wid):
	print("wid",wid)
	with open(jiemi_file, 'r', encoding='utf-8') as f1:
		reader = csv.reader(f1)
		for i,line in enumerate(reader): 
			if(i==0):
				continue
		
			if(line[1]==wid):

				return line[0]

#根据顺丰仓大
# 小件划分，获取当前商品所属
def get_daxiao(wid):
	print("wid",wid)
	with open(jiemi_file, 'r', encoding='utf-8') as f1:
		reader = csv.reader(f1)
		for i,line in enumerate(reader): 
			if(i==0):
				continue
		
			if(line[1]==wid):

				return line[0]

# 汇总商品价格
def get_price(file):
    price_dic = {}
    with open(file, 'r') as price_f:
        reader = csv.reader(price_f)
        for i,line in enumerate(reader):
            if line[1] == 'id':
                continue
            id_store = '' + str(line[2]) + str(line[1])
            #print('id:', str(line[1]), '    store:', str(line[2]))
            price_dic[id_store] = float(line[3])
   
    return price_dic

def get_zhouzhuan(wid):
	
    with open(arti_zhouzhuan_file, 'r') as f:
        reader = csv.reader(f)
        for i,line in enumerate(reader):
            if i==0 :
                continue
            if(line[0] == wid):
				
                zhouzhuan = line[5]
                yueleiji = line[3]
                return zhouzhuan,yueleiji
    return 0,0

def get_arti_daily_stock(searchid):
    
    with open(arti_daily_stock_cost_file, 'r') as f:
        reader = csv.reader(f)
        for i,line in enumerate(reader):
            if i==0 :
                continue
            if(line[0] == searchid):
				#该组合的月累计每日结存量
                accu_daily = line[3]
                return accu_daily
    return 

def get_model_daily_stock(searchid):
    
    with open(model_daily_stock_cost_file, 'r') as f:
        reader = csv.reader(f)
        for i,line in enumerate(reader):
            if i==0 :
                continue
            if(line[0] == searchid):
				#该组合的月累计每日结存量
                accu_daily = line[3]
                return accu_daily
    return 


def CIC1(zhouzhuan):
	if(zhouzhuan <= 60):
		stock_price =0
	if(zhouzhuan > 60 and zhouzhuan <=90):
		stock_price = 4
	if(zhouzhuan > 90 and zhouzhuan <=120):
		stock_price =4.5
	if(zhouzhuan > 120 and zhouzhuan <=150):
		stock_price =5
	if(zhouzhuan > 150 and zhouzhuan <=180):
		stock_price =5.5
	if(zhouzhuan > 180 ):
		stock_price =6
	return stock_price

def CIC2(zhouzhuan):
	if(zhouzhuan <= 30):
		stock_price =0
	if(zhouzhuan > 30 and zhouzhuan <=40):
		stock_price = 1
	if(zhouzhuan > 40 and zhouzhuan <=60):
		stock_price =1.2
	if(zhouzhuan > 60 and zhouzhuan <=90):
		stock_price =3
	if(zhouzhuan > 90 and zhouzhuan <=180):
		stock_price =4
	if(zhouzhuan > 180 ):
		stock_price =6

	return stock_price

if __name__ == "__main__":
                         
    ''' ————————————————————汇总测试集结果————————————————————————————'''
    files= os.listdir(raw_test_result)
    for file in files: 
        #position = './fencang/'+ file   
        filename = raw_test_result+ file  
        #print("filename",filename)
        dataManeger(filename)
        #dataManeger()
    
    #汇总csv

    outFileName=ALL_result_file
    csvfile_out = open(outFileName, 'w',newline='') 
    writer = csv.writer(csvfile_out)
    writer.writerow(['id', '仓库','最优组合编号','基模型','元模型','真实销量平均值','最优模型预测平均值','最优模型mse',"最优模型mae","最优模型d_value","mae方差","窗口数量"])

    print("开始合并测试集文件")
    files= os.listdir(outFilePath)
    for file in files: 
        #print(file)
        filename=outFilePath+file
        csvfile = open(filename, 'r',encoding='utf-8') 
        reader = csv.reader(csvfile)
        for i,line in enumerate(reader):
            
            if(i<2):
                continue
     
            data=[(line[0],line[1],line[2],line[3],line[4],line[5],line[6],line[7],line[8],line[9],line[10],line[11])]
        
            writer.writerows(data)

    csvfile_out.close()
    print("finish step one")

    ''' ————————————————————汇总预测集结果————————————————————————————'''
	 #处理原始结果，获得每个组合最优模型得到的预测值
    files= os.listdir(raw_future_result)
    for file in files: 
        filename = raw_future_result+ file  
        print("filename",filename)
        dataManeger_future(filename)
   
    #预测结果汇总csv
    outFileName=ALL_result_file_future
    csvfile_out = open(outFileName, 'w',newline='') 
    writer = csv.writer(csvfile_out)
    writer.writerow(['检索序号','id', '仓库','仓库名称','仓库类别','最优组合编号','最优模型预测值','预测窗口开始日期',
	'预测窗口截止日期','预测窗口内有数据的天数','指数平滑结果','预测窗口真实值','模型加权预测值'
	,'d-value(预测-真实)','模型MAE','模型卸货总成本','模型月累积在库（日结之和）','模型仓储成本（月累计*仓储单价）','模型库存成本(卸货+仓储)','卸货单价','模型补少成本','商品原价','人工周转天数',
	'人工该仓月累计在库','仓储单价','分组合人工月累计日结数量','人工仓储成本（仓储单价*日结数量）'])
    #计算双十一系数
    #writer.writerow(['id', '仓库','最优组合编号','最优模型预测值','真实值','预测窗口开始日期','预测窗口截止日期'])
    
    #获取商品价格字典
    price_dic = get_price(price_file)

    print("开始合并预测文件")
    files = os.listdir(outFilePath_future)
    for file in files: 
        filename = outFilePath_future + file
        csvfile = open(filename, 'r',encoding='utf-8') 
        reader = csv.reader(csvfile)
        for i,line in enumerate(reader):
            if(i<2):
                continue
            #计算双十一系数
            #data=[(line[0],line[1],line[2],line[3],line[4],line[5],line[6])]  

            #匹配预测窗口对应的真实值
            print(line[1].strip())
            sales_count,day_count = find_in_changeData(line[1].strip(),line[2].strip(),line[5].strip(),line[6].strip())
            #写入id wname 最优模型编号,模型预测值，4预测窗口起始日期，5预测窗口终止日期，预测窗口真实销量，预测窗口内有数据的天数
			
			#对于处在大促期的预测窗口，乘以大促系数
			# 10.20-11.11 2.6791
			# 615 -618
			# 9.28
			#对预测值取整  
            search_id = line[0]
            id = line[1]
            wid = line[2]
           

            predictSale = float(line[4])
            beginDate = ( str(datetime.date(int(line[5].split("-")[0]), int(line[5].split("-")[1]), int(line[5].split("-")[2]))))
            endDate = ( str(datetime.date(int(line[6].split("-")[0]), int(line[6].split("-")[1]), int(line[6].split("-")[2]))))
            #判断预测窗口是否位于大促期间
            '''计算大促系数时需要屏蔽这段'''
            cofficient = get_cofficient(beginDate,endDate)
            predictSale = predictSale * cofficient 

            zhishu_result = 0
            zhishu_files = os.listdir('/data1/lxt/galanz_test/stacking/ES2/')
			#匹配指数平滑预测结果
            for zhishu_result_file in zhishu_files: 
                csvfile_zhisu = open('/data1/lxt/galanz_test/stacking/ES2/'+str(zhishu_result_file), 'r',encoding='utf-8') 
                reader_zhishu = csv.reader(csvfile_zhisu)
               
                for j in reader_zhishu:
				   #匹配检索序号 和预测窗口日期
				
                    if(str(j[0]) == str(line[0]) and str(j[-1]) == str(line[5])):
					    #指数平滑预测结果
                        print("!!!")
                        zhishu_result = float(j[-2])
                        break
            if(zhishu_result < 0):
                zhishu_result = 0

            csvfile_zhisu.close()
			
            #对于大销量产品，对预测值进行加权平均,否则为原预测值
            weight_result = round(predictSale)
            if(zhishu_result != 0):
                weight_result = round( predictSale * a1 + zhishu_result * a2)
		    
			#预测值小于0，则赋为0
            if(weight_result<0):
                weight_result = 0
            d_value = weight_result - sales_count
            model_mae = np.abs(weight_result - sales_count)
			# 计算模型预测的补少成本
            if d_value < 0:
                id_store =  str(line[0])
                price = float(price_dic[id_store])
                cost_less = price * (-d_value) * 0.3
            else:
                cost_less = 0
			
			# 获取商品原价
            originalPrice = float(price_dic[str(line[0])])

		  	#匹配原始仓库名称
            warehoust_name = str(get_warehouse_name(line[2]))
            zhouzhuan,yueleiji = get_zhouzhuan(wid)

            if(warehoust_name[:2] == '菜鸟'):
                waretype = '菜鸟'
            elif(warehoust_name[:2] == '顺丰'):
                waretype = '顺丰'
            elif(warehoust_name[:3] == '格兰仕'):
                waretype = '顺丰'
            else :
                waretype = '京东'
			
		 
            stock_price = 0
            xiehuo = 0
            xiehuo_unit_price = 0
			#京东：卸货+仓储  
            if(waretype == '京东'):
                xiehuo = weight_result * 4.1
                xiehuo_unit_price =4.1
				#仓储单价
                stock_price = CIC1(int(zhouzhuan))
			#菜鸟：仓储
            if(waretype == '菜鸟'):
				#仓储单价
                stock_price = CIC2(int(zhouzhuan))
            if(waretype == '顺丰'):
				#仓储单价
                stock_price = 4.180415
			
            '''____匹配人工日结部分'''
			#人工该组合的月累计日结量
            arti_accu_daily = get_arti_daily_stock(search_id)
            arti_stock_price = 0
            if(arti_accu_daily is not None):

                arti_stock_price = float(arti_accu_daily) * float(stock_price)
            '''匹配模型日结部分'''
            model_accu_daily = get_model_daily_stock(search_id)
            model_stock_price = 0
            if(model_accu_daily is not None):
                model_stock_price = float(model_accu_daily) * float(stock_price)
			
            model_all_stock = model_stock_price + xiehuo

            data=[(search_id,id,wid,warehoust_name,waretype,line[3],round(predictSale),line[5],line[6],
			day_count,zhishu_result,sales_count,weight_result,d_value,model_mae,xiehuo,
			model_accu_daily,model_stock_price,model_all_stock,xiehuo_unit_price
			,cost_less,originalPrice,zhouzhuan,yueleiji,stock_price,arti_accu_daily,arti_stock_price)]
            writer.writerows(data)
        csvfile.close()
    csvfile_out.close()
	
    print("finish step two!")



	
