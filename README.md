首先执行code/文件夹下的（不分先后顺序）

gen_stack_age.py

gen_stack_age.py

gen_ac_stack_gender.py

gen_best_nn_gender.py

生成的stacking文件储存在../user_data/tmp/文件夹下

最后执行code/文件夹下的run.sh或run.py

sh run.sh 或 python run.py

最终生成的预测文件result.csv在../prediction_result/文件夹下

思路：先对gender进行预测分类，然后将预测的gender用于age预测中，两个模型均为lightgbm模型，有一部分公共特征，如下（均是针对每台设备）：

1、事件发生的总数统计以及唯一性统计

2、app_id的唯一性的统计

3、安装的总数统计

4、激活的总数、均值统计

5、事件发生日期的唯一天数、最大值、最小值

6、应用标签的均值、方差

7、以及以上部分特征组合

8、1-3日期发生的事件总数

9、每台使用应用次数最多的app_id

10、phone_brand、device_model进行target_encoder

11、Top10、Top50、Top100的应用使用个数

两个模型又分别构造了一些特征，如下（均是针对每台设备）：

i.gender模型特有特征：

1、利用每台设备的app列表与标签列表做TF-IDF生成特征然后用LR、SGD、PAC等8个分类器进行预测，得到各个模型的预测用于之后的stacking

2、利用每台设备激活的app列表做TF-IDF生成特征然后用LR、SGD、PAC等7个分类器进行预测，得到各个模型的预测用于之后的stacking

3、利用Word2Vec算法将每台设备的app用向量表示获取w2c特征，然后用神经网络进行预测，得到预测结果用于之后的stacking

ii.age模型特有特征:

1、利用每台设备的app列表与标签列表做TF-IDF生成特征然后用Ridge、PAR、SVR3个分类器进行预测，得到各个模型的预测用于之后的stacking

2、提取每台设备的应用列表和标签列表生成TF-IDF、CountVec特征






					
