# 零件故障预警

- 任务描述: 
    - 有一批零件数据，代表该零件在发生故障前的1min内的指标数据(每个文件6000条数据，0.01s一条)，根据数据构建模型，预测零件是否会出现故障

    - 难点： 没有标签，需要自己构建训练标签。


- 实现细节:
    
    1. 原数据无标签，可以确定的是最后一个时间点是故障点。定义一个`TROUBLE_LEN`（故障标签长度）,将最后一小段时间标记为故障标签(1),其余时间点标记为无故障(0)

    2. 代码`sklearn评测.ipynb`：原数据比较庞大（无用特征比较多），需要选取特征进行模型构建，使用RF进行简单训练，输出特征重要程度排序，选取合适数量的特征.
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckc7dgudzj30m40h8ju3.jpg)


    选取排名靠前的特征进行可视化的结果
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckbuj9snyj30ul0u0gr2.jpg)

    使用传统机器学习模型进行处理时序问题，时序特征需要自己构建（窗口操作），因此选取深度学习模型进行处理

    3. 深度学习模型尝试
        
        - 使用`torch.utils.Dataset`构建数据集，采样连续的片段（片段长度可控`TROUBLE_LEN`）作为特征，比较方便取数据和随机选取训练数据
    
        - 模型参数 
        处理时序问题，基础模型选用RNN,尝试不同的参数进行测试。
        ```{python}
        EMBEDDING_DIM = len(FEAT_SELECT) # 输入数据维度
        RNN_HIDDEN = 10  # RNN隐含层维度
        TAGSET_SIZE = 2  # 目标维度
        RNN_LAYER = 2  # RNN结构的层数
        EPOCH = 1  # 训练EPOCH
        OUTPUT_HIDDEN = 20  # 输出层之前的隐含层维度
        OUTPUT_ACTIVATE = 'relu'  # 输出层之前的隐含层激活函数 relu/tanh
        FLATTEN = False  # 是否采用flatten训练方式 True/False
        RNN_TYPE = 'RNN'.lower()  # 模型类型 gru/lstm/rnn
        DROPOUT_RATE = 0.1  # DROPOUT的比例
        BI_DIRECT = False  # 网络是否采用双向模式
        ```

    4. 可视化选项：
        - GPU_0829_tensorboard.ipynb  tensorboard绘图（实时显示）
        - GPU_09250439_plot.ipynb  matplotlib绘图


