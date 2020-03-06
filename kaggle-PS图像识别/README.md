# 检测图像是否有ps痕迹

- 任务描述: 
    - kaggle链接(ADAMS Faces SS19) https://www.kaggle.com/c/adams-faces-ss19/data
    - 模型目标，图像二分类问题：检测图像是否经过ps。训练集图像fake标签(经过ps)463张，real标签(未经过ps)891张，测试集图像481张
    - gmail: hyau1121/hyau1121 


- 实现细节:
    1. 训练集 fake/real标签数量1:2，略有偏斜，需要采用CNN结构提取出有效的特征
    
    2. 使用迁移学习模型，加载预训练的resnet18模型，finetune最后的全连接层，可以更加有效的进行特征提取步骤，避免将大量的时间花费在特征提取上
    
    3. 实验数据记录：
        - 训练/验证 比例为9:1, GPU加速，3m 16s，11个epoch后，验证集 acc 达到1.0，模型已经充分训练
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck3whu2zdj30ja02kwep.jpg)
        
        - 参数优化方法：`SGD` + `optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)`


- 结果展示
    Rank 4(Top 10%), Score: 0.68184（AUC分数）
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gck40mms08j30pk06ndgp.jpg)

- 分析
    模型可能存在过拟合，可以尝试其他方法提取特征，或者引入噪声，提高模型鲁棒性。





