# IMDB数据集情感分类

- 任务描述: 
    - Bag of Words Meets Bags of Popcorn https://www.kaggle.com/c/word2vec-nlp-tutorial/overview
    - 模型目标，情感二分类问题:  IMDB数据集的文本情感分类。训练集25000条有标签数据，50000条无标签数据。测试集25000条无标签数据。
    - email: hyau1121/hyau1121

- 实现细节:
    
    1. 数据预处理（去标签，标点符号，正则化等）
        - 去掉网页标签
        - 具体的链接替换为字符'URL'，链接对文本分类干扰较大
        - 正则去除非英文字符，过滤符号项的干扰
    
    2. 语料库的选择方案:
        - 用labeledTrain+testData作为语料；
        - 用labeledTrain+unlabeledTrain+testData作为语料（更优，数据量大，字/词向量更准确）

    3. 词语向量化方案:
        - BOW (词频向量)
        - TFIDF (权重向量)
        - Word2Vec (无监督训练词向量)，gensim.word2vec实现

    4. 模型选择
        - SVM
        - naive bayes
        - RF
        - xgboost
        - LR

    5. 其他
        - KFOLD对比（cv=5）,最后要综合各个划分测出的TPR/FPR，使用`sklearn.metrics.auc`进行综合计算，其中TPR需要进行插值综合


- 结果展示
    
    1. 各个模型ROC图像
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckasvnjwjj30ru0mmtc3.jpg)
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckawu1dqhj30p40mgwhn.jpg)
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckax1h2llj30po0mo41s.jpg)
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckaxaoa9uj30qw0mcgou.jpg)
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckaxhfunjj30rg0mgtbq.jpg)

    2. 模型交叉对比
        
        - 训练时间
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckayi30cmj30gg0dfq3j.jpg)
        - F1 Score
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckazdb6mej30gj0dfq3y.jpg)
        - AUC
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckaz224thj30gj0df3zj.jpg)

    3. 结果提交 Rank 280
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckb1v29fpj30yu0cqjsw.jpg)

- 回顾分析
    1. Word2Vec特征的效果最佳,但是训练时间比较长
    2. 自测结果显示LR+Word2Vec的效果最佳
    3. 后续
        - 对非英文字符的过滤比较直接，可以考虑标点字符对文本情感的影响
        - 深度学习模型的引入，可能获取更好的结果





