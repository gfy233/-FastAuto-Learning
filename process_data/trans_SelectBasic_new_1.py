import json
import csv
import sys
import os
import datetime
import numpy as np
from scipy import stats
import pandas as pd
import process_data.text_feature
import process_data.picture_feature
import tensorflow
import torch
#记录各组特征是否有效
#导入数据增强包
from tsaug import (
    AddNoise,
    Convolve,
    Crop,
    Drift,
    Dropout,
    Pool,
    Quantize,
    Resize,
    Reverse,
    TimeWarp,
)

benchmark_file = './benchmark.csv'
inputDir = '/data1/lxt/Galanz-TimeSeries/gfy/fencang_selected/'
fileList = os.listdir(inputDir)
outputDir = '/data/gfy2021/gfy/stacking_DRANN/process_data/fencang_feature_selected_for_1214_1227_text+pic/'
normalDir = '/data/gfy2021/gfy/stacking_DRANN/process_data/fencang_feature_selected_normal_for_1214_1227_text+pic/'

flag = []
text_model = process_data.text_feature.text_feature()
pic_model = process_data.picture_feature.picture_feature()

#sentence_bert

#14天为窗口，获取特征及label
def feature_eng(in_file, csv_f,normal_f):
    if os.path.exists(normal_f.replace('.csv', '.json')):
        os.remove(normal_f.replace('.csv', '.json'))
    #读取仓库订单数据
    data_list = {}
    for line in open(in_file, 'r', encoding='utf-8'):
        raw_data = json.loads(line)
        if raw_data['orderBeginTime'] == None:
            continue

        now_date = datetime.date(int(raw_data['orderBeginTime'][0:4]),
                                 int(raw_data['orderBeginTime'][5:7]), int(raw_data['orderBeginTime'][8:10]))

        if raw_data['id'] not in data_list.keys():
            p = {}
            p['id'] = raw_data['id']
            p['goodsThirdName'] = raw_data['goodsThirdName']
            p['goodsClassifyName'] = raw_data['goodsClassifyName']
            p['whSapCode'] = raw_data['whSapCode']
            p['whSapName'] = raw_data['whSapName']
            p['sale_time_series'] = {}
            p['goodsDealPrice_time_series'] = {}
            p['goodsShopDiscountAmount_time_series'] = {}
            p['discountRate_time_series'] = {}
            p['orderPfTotalAmount_time_series'] = {}
            p['goodsShopReceiptPrice_time_series'] = {}
            data_list[raw_data['id']] = p
        if now_date not in data_list[raw_data['id']]['sale_time_series'].keys():
            data_list[raw_data['id']]['sale_time_series'][now_date] = 1
            data_list[raw_data['id']]['goodsDealPrice_time_series'][now_date] = raw_data['goodsDealPrice']
            data_list[raw_data['id']]['goodsShopDiscountAmount_time_series'][now_date] = raw_data['goodsShopDiscountAmount']
            if raw_data['goodsDealPrice'] > 0:
                data_list[raw_data['id']]['discountRate_time_series'][now_date] = raw_data['goodsShopDiscountAmount'] / raw_data['goodsDealPrice']
            else:
                data_list[raw_data['id']]['discountRate_time_series'][now_date] = 0
            data_list[raw_data['id']]['orderPfTotalAmount_time_series'][now_date] = float(raw_data['orderPfTotalAmount'])
            data_list[raw_data['id']]['goodsShopReceiptPrice_time_series'][now_date] = float(raw_data['goodsShopReceiptPrice'])
        else:
            data_list[raw_data['id']]['sale_time_series'][now_date] += 1
            data_list[raw_data['id']]['goodsDealPrice_time_series'][now_date] += raw_data['goodsDealPrice']
            data_list[raw_data['id']]['goodsShopDiscountAmount_time_series'][now_date] += raw_data['goodsShopDiscountAmount']
            if raw_data['goodsDealPrice'] > 0:
                data_list[raw_data['id']]['discountRate_time_series'][now_date] += raw_data['goodsShopDiscountAmount'] / raw_data['goodsDealPrice']
            else:
                data_list[raw_data['id']]['discountRate_time_series'][now_date] += 0
            data_list[raw_data['id']]['orderPfTotalAmount_time_series'][now_date] += float(raw_data['orderPfTotalAmount'])
            data_list[raw_data['id']]['goodsShopReceiptPrice_time_series'][now_date] += float(raw_data['goodsShopReceiptPrice'])

    #只添加有提升效果的特征
    sheet_title = ['watch', 'id', 'whSapName', 'input_start_date', 'input_end_date']
    if(flag[0]):
        sheet_title.extend(['pred_start_month', 'pred_end_month'])
    if(flag[1]):
        sheet_title.extend( ['max_gsda', 'min_gsda', 'std_gsda', 'var_gsda', 'mean_gsda', 'mode_gsda', 'median_gsda',
                   'max_gsdar', 'min_gsdar', 'std_gsdar', 'var_gsdar', 'mean_gsdar', 'mode_gsdar', 'median_gsdar'])
    if(flag[2]):
        sheet_title.extend( ['max_opta', 'min_opta', 'std_opta', 'var_opta', 'mean_opta', 'mode_opta', 'median_opta'])
    if(flag[3]):
        sheet_title.extend( ['max_gdp', 'min_gdp', 'std_gdp', 'var_gdp', 'mean_gdp', 'mode_gdp', 'median_gdp',
                   'max_gsrp', 'min_gsrp', 'std_gsrp', 'var_gsrp', 'mean_gsrp', 'mode_gsrp', 'median_gsrp'])
    if(flag[4]):
        sheet_title.extend( [  'max_sale', 'min_sale', 'std_sale', 'var_sale', 'mean_sale', 'mode_sale', 'mode_sale14', 'median_sale', 'median_sale14',
                   'sum_sale_1', 'sum_sale_2', 'sum_sale_3', 'sum_sale_4', 'sum_sale_7', 'sum_sale_9', 'sum_sale_11', 'sum_sale_14'])
    if(flag[5]):
        sheet_title.extend( [ 'max_sale_1', 'min_sale_1', 'std_sale_1', 'var_sale_1', 'mean_sale_1', 'mode_sale_1', 'mode_sale14_1', 'median_sale_1', 'median_sale14_1',
                   'sum_sale_1_1', 'sum_sale_2_1', 'sum_sale_3_1', 'sum_sale_4_1', 'sum_sale_7_1', 'sum_sale_9_1', 'sum_sale_11_1', 'sum_sale_14_1'])
    
    #增加文本特征
    sheet_title.extend(['good_name1','good_name2','good_name3','good_name4','good_name5','good_name6','good_name7','good_name8','good_name9','good_name10'])
    
    # 增加图片特征
    sheet_title.extend(['pic1','pic2','pic3','pic4','pic5','pic6','pic7','pic8','pic9','pic10'])
    

    

    sheet_title.append('label')
    

    csv_fp = open(csv_f, "w", encoding='utf-8')
    writer = csv.writer(csv_fp)
    writer.writerow(sheet_title)

    for k in data_list.keys():
        s = sorted(data_list[k]['sale_time_series'])[0]
        ''' 指定特征窗口尾端日期'''
        e0 = datetime.date(2020, 12,13 )
        e = datetime.date(2020, 12, 13)
        #e = sorted(data_list[k]['sale_time_series'])[-1]
         
        #watch_0是交给格兰仕的预测值的特征窗口
        begin = e0 - datetime.timedelta(days=14)
        end = e0
        watch = 0
        row, s_label, sale_list,aug_sales_list = window_feature(begin, end, data_list[k])
        row = [watch, ] + row
        if watch == 0:
            row[-1] = 0
        writer.writerow(row)

        sale_dict = {}
        all_sale = get_all_sale(begin, end, data_list)
        sale_dict['watch'] = watch
        sale_dict['X'] = all_sale.tolist()
        sale_dict['x'] = sale_list
        sale_dict['y'] = 0
        with open(normal_f.replace('.csv', '.json'), 'a') as fj:
            json.dump(sale_dict, fj)
            fj.write('\n')
        fj.close()


        #(begin, end)是特征窗口的区间，(end，end+14)则是预测窗口的区间
        begin = e - datetime.timedelta(days=28)
        end = e - datetime.timedelta(days=14)
        watch = 1
        while (begin.__sub__(s)) >= datetime.timedelta(days=28):
            row, s_label, sale_list,aug_sales_list = window_feature(begin, end, data_list[k])
            row = [watch, ] + row
            writer.writerow(row)
            sale_dict = {}
            all_sale = get_all_sale(begin, end, data_list)
            sale_dict['watch'] = watch
            sale_dict['X'] = all_sale.tolist()
            sale_dict['x'] = sale_list
            sale_dict['y'] = s_label
            with open(normal_f.replace('.csv', '.json'), 'a') as fj:
                json.dump(sale_dict, fj)
                fj.write('\n')
            fj.close()

            #更新窗口
            begin = begin - datetime.timedelta(days=14)
            end = end - datetime.timedelta(days=14)
            watch += 1
    csv_fp.close()

    #特征归一化
    data = pd.read_csv(csv_f)
    data_original = pd.DataFrame(data)
    column_num = data_original.shape[1]
    data_normal = pd.DataFrame()
    for column_name in sheet_title:
        feature_list = data_original[column_name].values.tolist()
      
        if column_name in ['watch', 'id', 'whSapName', 'input_start_date', 'input_end_date', 'pred_start_month', 'pred_end_month', 'label','good_name']:
            data_normal[column_name] = feature_list
        else:
            min_value = float(min(feature_list))
            max_value = float(max(feature_list))
            if (max_value-min_value) == 0:
                data_normal[column_name] = feature_list
            else:
                for j in range(len(feature_list)):
                    x = float(feature_list[j])
                    feature_list[j] = (x-min_value) / (max_value-min_value)
                data_normal[column_name] = feature_list
    data_normal.to_csv(normal_f, header=sheet_title, index=False)

#调用tsaug进行数据增强
def aug_x(X):

    aug_X = AddNoise().augment(X)
    for i in range(len(aug_X)):
        if(aug_X[i]<0):
            aug_X[i]=0
    return aug_X


def window_feature(begin, end, data):
    '''文本特征'''
    good_name = data['goodsThirdName']
    text_feature = text_model.get_sentence_feature(good_name)

    pic_feature = pic_model.get_picture_feature(data['id'])
  
    #产品ID及仓库
    p_feature = []
    p_feature.append(data['id'])
    p_feature.append(data['whSapName'])
    #特征窗口销量
    sale_list = []
    #预测窗口销量总数作为label
    sale_label = 0
    #特征窗口折扣
    dsc_list = []
    #特征窗口折扣率
    dscr_list = []
    #特征窗口平台垫付
    opta_list = []
    #特征窗口原价
    gdp_list = []
    #特征窗口价格
    gsrp_list = []
    for i in range(14):
        day = end - datetime.timedelta(days=i)
        day_pred = end + datetime.timedelta(days=14) - datetime.timedelta(days=i)
        if day in data['sale_time_series'].keys():
            sale_list.append(data['sale_time_series'][day])
            dsc_list.append(data['goodsShopDiscountAmount_time_series'][day]
                            / data['sale_time_series'][day])
            dscr_list.append(data['discountRate_time_series'][day]
                            / data['sale_time_series'][day])
            opta_list.append(data['orderPfTotalAmount_time_series'][day]
                            / data['sale_time_series'][day])
            gdp_list.append(data['goodsDealPrice_time_series'][day]
                            / data['sale_time_series'][day])               
            gsrp_list.append(data['goodsShopReceiptPrice_time_series'][day]
                            / data['sale_time_series'][day])
        else:
            sale_list.append(0)
            dsc_list.append(0)
            dscr_list.append(0)
            opta_list.append(0)
            gdp_list.append(0)
            gsrp_list.append(0)
        if day_pred in data['sale_time_series'].keys():
            sale_label += data['sale_time_series'][day_pred]

    sale_feature = []
    if(flag[4]):
       
        #特征窗口：最大销量，最小销量，销量的方差，销量的标准差，销量的平均值，销量众数，销量众数*14，销量中位数，销量中位数*14
        sale_feature.append(max(sale_list))
        sale_feature.append(min(sale_list))
        sale_feature.append(np.var(sale_list))
        sale_feature.append(np.std(sale_list))
        sale_feature.append(np.mean(sale_list))
        sale_feature.append(stats.mode(sale_list)[0][0])
        sale_feature.append(stats.mode(sale_list)[0][0]*14)
        sale_feature.append(np.median(sale_list))
        sale_feature.append(np.median(sale_list)*14)

        #特征窗口：滑动，前N天销量，N=1，2，3，4，7，9，11，14
        #sale_feature.append(sum(sale_list))
        sale_feature.append(sum(sale_list[13:]))
        sale_feature.append(sum(sale_list[12:]))
        sale_feature.append(sum(sale_list[11:]))
        sale_feature.append(sum(sale_list[10:]))
        sale_feature.append(sum(sale_list[7:]))
        sale_feature.append(sum(sale_list[5:]))
        sale_feature.append(sum(sale_list[3:]))
        sale_feature.append(sum(sale_list[:]))
          
    dsc_feature = []
    if(flag[1]):
        #特征窗口：最大折扣，最小折扣，折扣的方差，折扣的标准差，折扣平均值，折扣众数，折扣中位数
        dsc_feature.append(max(dsc_list))
        dsc_feature.append(min(dsc_list))
        dsc_feature.append(np.var(dsc_list))
        dsc_feature.append(np.std(dsc_list))
        dsc_feature.append(np.mean(dsc_list))
        dsc_feature.append(stats.mode(dsc_list)[0][0])
        dsc_feature.append(np.median(dsc_list))

        #特征窗口：最大折扣率，最小折扣率，折扣率的方差，折扣率的标准差，折扣率平均值，折扣率众数，折扣率中位数
        dsc_feature.append(max(dscr_list))
        dsc_feature.append(min(dscr_list))
        dsc_feature.append(np.var(dscr_list))
        dsc_feature.append(np.std(dscr_list))
        dsc_feature.append(np.mean(dscr_list))
        dsc_feature.append(stats.mode(dscr_list)[0][0])
        dsc_feature.append(np.median(dscr_list))

    if(flag[2]):
        #特征窗口：最大平台垫付，最小平台垫付，平台垫付的方差，平台垫付的标准差，平台垫付平均值，平台垫付众数，平台垫付中位数
        dsc_feature.append(max(opta_list))
        dsc_feature.append(min(opta_list))
        dsc_feature.append(np.var(opta_list))
        dsc_feature.append(np.std(opta_list))
        dsc_feature.append(np.mean(opta_list))
        dsc_feature.append(stats.mode(opta_list)[0][0])
        dsc_feature.append(np.median(opta_list))
    
    price_feature = []
    if(flag[3]):
        #print(gdp_list)
        #print(gsrp_list)
        #特征窗口：最大原价，最小原价，原价的方差，原价的标准差，原价平均值，原价众数，原价中位数
        gdp_sum = 0
        gdp_num = 0
        gdp_meam = 0
        for i in range(14):
            if gdp_list[i] != 0:
                gdp_sum += gdp_list[i]
                gdp_num += 1
        if gdp_num != 0:
            #print(gdp_sum,gdp_num)
            gdp_mean = float(gdp_sum/gdp_num)
            for j in range(14):
                if gdp_list[j] == 0:
                    gdp_list[j] = gdp_mean
        #print(gdp_list)

        price_feature.append(max(gdp_list))
        price_feature.append(min(gdp_list))
        price_feature.append(np.var(gdp_list))
        price_feature.append(np.std(gdp_list))
        price_feature.append(np.mean(gdp_list))
        price_feature.append(stats.mode(gdp_list)[0][0])
        price_feature.append(np.median(gdp_list))

        #特征窗口：最大价格，最小价格，价格的方差，价格的标准差，价格平均值，价格众数，价格中位数
        gsrp_sum = 0
        gsrp_num = 0
        gsrp_meam = 0
        for i in range(14):
            if gsrp_list[i] != 0:
                gsrp_sum += gsrp_list[i]
                gsrp_num += 1
        if gsrp_num != 0:
            gsrp_mean = float(gsrp_sum/gsrp_num)
            for j in range(14):
                if gsrp_list[j] == 0:
                    gsrp_list[j] = gsrp_mean
        #print(gsrp_list)

        price_feature.append(max(gsrp_list))
        price_feature.append(min(gsrp_list))
        price_feature.append(np.var(gsrp_list))
        price_feature.append(np.std(gsrp_list))
        price_feature.append(np.mean(gsrp_list))
        price_feature.append(stats.mode(gsrp_list)[0][0])
        price_feature.append(np.median(gsrp_list))


    #再取特征窗口前14天窗口的销量特征为输入特征
    sale_list_1 = []
    sale_feature_1 = []
    if(flag[5]):
        for i in range(14):
            day = end - datetime.timedelta(days=(14+i))
            if day in data['sale_time_series'].keys():
                sale_list_1.append(data['sale_time_series'][day])
            else:
                sale_list_1.append(0)
     
        # 特征窗口：最大销量，最小销量，销量的方差，销量的标准差，销量的平均值，销量众数，销量众数*14，销量中位数，销量中位数*14
        sale_feature_1.append(max(sale_list_1))
        sale_feature_1.append(min(sale_list_1))
        sale_feature_1.append(np.var(sale_list_1))
        sale_feature_1.append(np.std(sale_list_1))
        sale_feature_1.append(np.mean(sale_list_1))
        sale_feature_1.append(stats.mode(sale_list_1)[0][0])
        sale_feature_1.append(stats.mode(sale_list_1)[0][0] * 14)
        sale_feature_1.append(np.median(sale_list_1))
        sale_feature_1.append(np.median(sale_list_1) * 14)

        # 特征窗口：滑动，前N天销量，N=1，2，3，4，7，9，11，14
        # sale_feature.append(sum(sale_list))
        sale_feature_1.append(sum(sale_list_1[13:]))
        sale_feature_1.append(sum(sale_list_1[12:]))
        sale_feature_1.append(sum(sale_list_1[11:]))
        sale_feature_1.append(sum(sale_list_1[10:]))
        sale_feature_1.append(sum(sale_list_1[7:]))
        sale_feature_1.append(sum(sale_list_1[5:]))
        sale_feature_1.append(sum(sale_list_1[3:]))
        sale_feature_1.append(sum(sale_list_1[:]))

    sale_list_2 = []
    for i in range(14):
        day = end - datetime.timedelta(days=(14+i))
        if day in data['sale_time_series'].keys():
            sale_list_2.append(data['sale_time_series'][day])
        else:
            sale_list_2.append(0)
   
    # 特征窗口开始日期及终止日期，预测窗口开始月份和终止月份
    time_feature = []
    begin_end_feature = []
    begin_end_feature.append(begin)
    begin_end_feature.append(end)
    if(flag[0]):

        time_feature.append(int(end.month))
        end_pred = end + datetime.timedelta(days=14)
        time_feature.append(int(end_pred.month))

    all_list = []
    for i in sale_list_2[::-1]:
        all_list.append(i)
    for i in sale_list[::-1]:
        all_list.append(i)

    #合并特征为一行，返回
    #0time   1dsc    2opta   3gdp grsp   4sale   5sales
    
    row = p_feature + begin_end_feature + time_feature + dsc_feature + price_feature + sale_feature + sale_feature_1 
    if(not isinstance(text_feature,int)):
        for i in range(10):
            row = row + [str(text_feature[i])]
    else:
        for i in range(10):
            row = row + ['0']
  
    for i in range(10):
        row = row + [str(pic_feature[i])]
 
    row += [sale_label, ] 
    aug_sales_list = aug_x(np.array(sale_list))
    sum_sale = np.sum(np.array(sale_list))
    with open(benchmark_file, 'a') as fb:
     
        fb.write(str(data['id'])+","+data['whSapName']+","+str(sum_sale)+'\n')
    fb.close()

    return row, sale_label, all_list,aug_sales_list

def get_all_sale(begin, end, data_list):
    key_list = list(data_list.keys())
    all_sale = np.zeros((28, len(key_list)))
    for i, k in enumerate(key_list):
        data = data_list[k]
        for j in range(28):
            day = end - datetime.timedelta(days=28) + datetime.timedelta(days=j)
            if day in data['sale_time_series'].keys():
                all_sale[j, i] = data['sale_time_series'][day]
    return all_sale


if __name__ == "__main__":
    #outputDir_addTimeFeature = '../fencang_feature_selected_timeFeature_new/'
    with open(benchmark_file, 'w') as fb:
        print("create")
        fb.close()
    for f in fileList:
        print("file",f)
        with open("./select_basic_feature.csv",newline = '',encoding = 'utf-8') as f2:   
            reader = csv.reader(f2)   
            warehouseName = f.split(".")[0]
          
            for row in reader: #遍历reader对象的每一行
                if(str(row[0]).strip() == str(warehouseName)):             
                    flag = []
                    for i in range(6):
                        #有效特征
                        if(float(row[i+1]) < 0 ):
                            flag.append(True)
                        #无效特征
                        else:
                            flag.append(False)
                   
                    output_f = outputDir + f.replace('.json', '.csv')
                    normal_f = normalDir + f.replace('.json', '.csv')
                    feature_eng(inputDir+f, output_f,normal_f)
