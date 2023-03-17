# -FastAuto-Learning

Auto-Learning Mechanism
1. ensemble.py # Ensemble learning framework. 
2. main.py # main function.
3. model.py # candidate model set.
4. Optim.py # optimization strategy.

EA Mechanism
1. EAmain.py # main function.
2. EAmodel.py # EA model.
3. EAutils.py # EA utils. 

DK SR component
1. DKSRmain.py # main function to execute DK and SR.
2. DKSRmodels.py # the code of DK and SR.
3. DKSRmodel_runner.py # model train and test with parameter assignment.
4. DKSRoptimize.py # optimization strategies.
5. SAmodel.py # LSTM with attention mechanism.
6. coint.py and cointMP.py # co-integration function with mutli-cores.
7. utils document # util functions.
8. loss ducument # loss functions.

DP component
1. Dpcontrollers.py # train and test DP component with parameter assignment.
2. DpDataProcessers.py # input data processing.
3. DpModels.py # the code of dot processing.
4. DpOptimizers # optimization strategies of DP.
5. DpUtils.py # util functions of DP component.


# Run
Data acquisition module.
1. Code path：-FastAuto-Learning/spider/jd.py.
2. Function： Crawl Jingdong e-commerce page on the commodity information
3. Output： Commodity pictures, titles and other information on the e-commerce platform
4. Modify paths：
(1)Line 18 : csv_file = "./jd.csv"	，Specifies the address to save the text content
(2)Line 101:pname = product['name'].replace("\t","").replace(" ",""),Specify the location to save the image content
5. Run： python jd.py

Data pre-process
1. Code path：-FastAuto-Learning/process_data/fencang.py
2. Input： Historical sales data
3. Output： Sales data after warehouse division processing
4. Modify paths：
(1)Line 22 :  with open("../galanz_data.json", 'r', encoding='utf-8') as f1:，指定读入的历史销量数据
(2)line31:  filename='../fencang/'+warehouse+'.json'，指定输出路径
5. run：
python fencang.py


Get Picture Feature
1. Code path：-FastAuto-Learning/process_data/picture_feature.py
2. Function： Get figure embedding accroding to towhee 
3. Input： Figure
4. Output： Figure embedding
5. run：trans_SelectBasic_new_1.py call this function


Get Text Feature
1. Code path：-FastAuto-Learning/process_data/text_feature.py
2. Function： Get text embedding accroding to towhee 
3. Input： Text
4. Output： Text embedding
5. run：trans_SelectBasic_new_1.py call this function

Feature engineering
1. Code path：-FastAuto-Learning/process_data/trans_SelectBasic_new_1.py
2. Function： Generate a script for sorting warehouse features after cleaning, normalization, and feature screening
3. Input：Historical inventory data, pictures, text features
4. Output： Feature file
5. Modify paths
(1)Line 29 : inputDir = '/data1/lxt/Galanz-TimeSeries/gfy/fencang_selected/'，Enter historical inventory information
(2)Line 32:inputDir = '/data1/lxt/Galanz-TimeSeries/gfy/fencang_selected/'，Specifies the save address for the feature file
6. run：
python trans_SelectBasic_new_1.py


Time series tensor
1. Code path：-FastAuto-Learning/src/Time-Series-Tensor/models/change_tensor.py -
2. Function： Gets a tensor representation of a historical time series
3. Input： Time series of historical sales of products
4. Output： Time series tensor
5. Modify paths
      (1)Line128: data = '../galanz/0e117c1684b5ebd6093fc17b468455d1.json'，Specifies the historical sales data entry path
      (2)Line191：df.to_csv("./time_tensor.csv"), Specifies the tensor output path
6. Run：
python change_tensor.py 


Ensemble
1. Code path：-FastAuto-Learning/src/ensemble.py 
2. Input： Feature File
3. Output： Future sales forecast
4. Modify paths：
      （1）Line 44: Input_future_folder = '/data/gfy2021/gfy/KDD/process_data/fencang_feature_selected_normal_for_1214_1227_text+pic/'，Specifies the signature file input path
      （2）Line 52：Output_future_folder = '/data/gfy2021/gfy/KDD/stacking/stacking_result/
result_future_sim_0.75/' 指定预测结果输出路径
5. Run：
python ensemble.py 


STL COST
1. Code path：-FastAuto-Learning/src/changeCSV_for_compare.py
2. Function：According to the sales volume predicted by the model, the inventory cost caused by replenishing with the predicted value of the model is calculated
3. Input： Model prediction result
4. Output： Inventory costing results (STL COST)
5. Modify path：
(1)Line 11: raw_test_result = '/data1/lxt/galanz_test/stacking/FeatureResult
/result_'+date_+'/'，Specifies the prediction result file path
(2)Line 29：ALL_result_file_future = '/data1/lxt/galanz_test/stacking/allResult
/ALL_future_'+date_+'成本.csv'，Specify an output path for the costing results
6. Run：
python changeCSV_for_compare.py


