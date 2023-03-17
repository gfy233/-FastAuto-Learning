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
