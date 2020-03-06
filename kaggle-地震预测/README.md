# 检测图像是否有ps痕迹

- 任务描述: 
    - LANL Earthquake Prediction https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview
    - 模型目标，时间序列+回归问题: 预测地震的发生时间。训练集629145481个数据，大小近10G，两个字段，一个属性一个标签，测试集2624个序列，每个序列对应一个预测值
    - email: isbada@foxmail.com


- 实现细节:
    
    1. 数据概况分析:

        - 训练数据
        
            在原始训练数据中采样出1%的数据，可以得到如下分布的图形：
            ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck4lm8o2qj311s0m6wjb.jpg)
            从图中我们可以看出，acoustic_data数据呈现出类似于音波的震荡分布，而time_to_failure数据呈现出锯齿状分布，而且每次地震发生前（表现为time_to_failure突然大幅提升），都伴随着acoustic_data数据的震荡幅度加大，这证明acoustic_data和time_to_failure两者之间存在一定的相关性。

            为了具体的探究两者之间的关系，我们将注意力集中于其中一个震荡，仅取前1%的数据进行数据分布的统计，得到如下结果
            ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck4m88781j312w0lwq5k.jpg)

            从上图可以看出，声波与地震的发生并不是同一时刻，而是在声音异常震荡过后的一小段时间才真正开始。针对于其他震荡点的情况也是如此。

        - 测试数据
        通过pandas工具读取测试数据test文件夹中的每一个文件seg_******.csv ，并输出其长度。

        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck4ou9kn3j30bw0eeq5r.jpg)

        对测试数据的统计结果显示，共有2624个测试数据文件，每个测试文件的长度均为150000（标题占了一行）。由于每个测试数据长度都为150000，因此我们需要将训练数据处理为与测试数据相同维度的数据段。经过这种处理，我们可以得到新数据的数量为4194个。
    
    2. 特征工程。
        
        本项目只有一个属性，特征工程比较重要，要从一个属性中构造出各种维度的特征。

        在进行特征工程时，考虑到acoustic_data数据的特征，我们选取了如下指标构造特征，以充分挖掘震荡数据的数学特性。
        
        在原始数据的基础上，构造如下特征：

        name | 含义
        --- | ---
        mean    |    平均数
        median  |    中位数
        std     |    标准差
        max     |    最大值
        min     |    最小值
        abs_max |    绝对值最大值
        abs_min |    绝对值最小值
        ptp     |    最大值-最小值
        10p     |    10分位数
        25p     |    25分位数
        50p     |    50分位数
        75p     |    75分位数
        90p     |    90分位数

        同时为了发掘时序数据的特征，对原始数据进行滑动窗口操作，获取三个窗口长度 [10, 100, 1000]范围内的标准差数列x_roll_std和平均值数列x_roll_mean,使用这两者构建
        标准差窗口特征,经过处理后的训练数据的shape=(4194, 68)，也就是说新数据的数据量为4194个，特征有68个。

    3. 对于预测标签time_to_failure数据，我们将每个数据段最后一个时刻的time_to_failure值作为当前数据段对应的标签，这样我们得到的y_all数据shape=(4194, 1)。

    4. 使用SVM,RF,LR等模型对数据进行测试。评价标准取MSE值。部分预测效果可视化图：
        
        - LR模型预测效果可视化
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck5ah1utaj31340j8n10.jpg)


        - SVM模型预测效果可视化
        - ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck5byffq9j31360jk0y3.jpg)




- 结果展示
    Private Leaderboard 最终Rank 1204(Top 26%)
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck4z83lu7j30tp03f74p.jpg)

- 回顾分析
    1. 训练数据的选举采取连续切分，造成训练数据偏少，使用随机切分可以提高训练数据量，进一步提升效果。
    2. 时间序列的问题比较特殊。使用深度学习模型（RNN）可能会更合适。





