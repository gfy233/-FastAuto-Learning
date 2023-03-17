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
· Output： Commodity pictures, titles and other information on the e-commerce platform
· Modify paths：
Line 18 : csv_file = "./jd.csv"	，Specifies the address to save the text content
Line 101:pname = product['name'].replace("\t","").replace(" ",""),Specify the location to save the image content
· Run：
python jd.py
